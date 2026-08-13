#!/usr/bin/env bash
# Try a list of stage1 kernel names and report, for each, whether fused_moe ran
# it at all, whether the result was correct, and what the stage1 GEMM cost.
#
#   ./probe.sh <name> [<name> ...]
#
# Stage2 is pinned to the legacy reduce kernel with no knobs so nothing but
# stage1 moves.  A name that fused_moe cannot honour does not error -- it
# silently falls back -- so the reported kernel name is checked against what was
# asked for, and a mismatch is reported as MISS rather than a timing.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
GPU="${GPU:-4}"
K2="${K2:-flydsl_moe2_afp8_wfp8_bf16_t64x128x64_reduce_persist_bnt0}"
BM="${BM:-64}"

CASE_ARGS=(
  --token 32768 --model-dim 4096 --inter-dim 192 --expert 193 --topk 9
  --activation silu --dtype bf16 --use-g1u1 1 --doweight-stage1 0
  --quant fp8 --quant-type per_tensor
)

for k1 in "$@"; do
  cfg="$(mktemp /tmp/s1cfg.XXXX.csv)"
  bash "$HERE/mkcfg.sh" "$k1" "$K2" "$BM" 64 > "$cfg"
  log="$(mktemp)"
  ( cd "$REPO" && AITER_LOG_MORE=1 AITER_CONFIG_FMOE="$cfg" HIP_VISIBLE_DEVICES="$GPU" \
      timeout 3600 python test_qmoe_multi.py "${CASE_ARGS[@]}" ) > "$log" 2>&1
  rc=$?
  python3 - "$log" "$k1" "$rc" <<'PY'
import re, sys
path, want, rc = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(path, errors="replace").read()
got1 = re.search(r"kernelName1='([^']*)'", txt)
got1 = got1.group(1) if got1 else ""
e2e = re.search(r"e2e fused_moe: *([0-9.]+)", txt)
cos = re.search(r"cos=([0-9.]+)", txt)
ok = re.search(r"pass=(\w+)", txt)
# stage1 shows up as ck::kernel_moe_gemm (CK) or moe_gemm1_0 (flydsl legacy).
s1 = None
for line in txt.splitlines():
    f = line.split()
    if len(f) >= 8 and f[-2] == "CUDA":
        name = " ".join(f[1:-6])
        if "kernel_moe_gemm" in name or re.search(r"\bmoe_gemm1", name) or "gateup" in name:
            try:
                s1 = float(f[-3].replace(",", ""))
            except ValueError:
                pass
if got1 and got1 != want:
    print(f"  {want:<52} MISS  (fused_moe used {got1})")
elif e2e is None:
    tail = "\n".join(l for l in txt.splitlines() if l.strip())[-1200:]
    print(f"  {want:<52} FAIL  rc={rc}\n{tail}\n")
else:
    print(f"  {want:<52} stage1={s1 if s1 else '?':>9}  e2e={float(e2e.group(1)):9.1f}  "
          f"pass={ok.group(1) if ok else '?'} cos={cos.group(1) if cos else '?'}")
PY
  rm -f "$cfg" "$log"
done
