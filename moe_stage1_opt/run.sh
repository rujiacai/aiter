#!/usr/bin/env bash
# Walk the legacy FlyDSL stage1 kernel from `base` towards `target`, one feature
# at a time.
#
# Same shape as moe_stage2_opt/run.sh, with two deliberate differences:
#
#   * The headline number is the *stage1 GEMM kernel* time, not e2e.  Stage1 is
#     ~30% of this case, so an e2e headline would divide every delta by three
#     and bury it in run-to-run spread.  e2e is still recorded, as a check that
#     a stage1 win is not being paid for somewhere else.
#   * A stage varies `kernelName1` as well as env, because some of the ladder
#     is tile/codegen selection that lives in the kernel name rather than in a
#     knob.  Stage2 is pinned to one kernel with no knobs for the whole ladder,
#     so nothing downstream moves.
#
#   ./run.sh                        # every stage, 3 repeats
#   ./run.sh base f1                # only these stages
#   ./run.sh --repeats 5
#   ./run.sh --counters             # rocprofv3 pass per stage
#   ./run.sh --list
#
# Results land in results/*.csv (append-only, one session id per invocation) so
# a partial re-run never loses the rest.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RESULTS="$HERE/results"
GPU="${GPU:-4}"
RESERVED_GPUS="${RESERVED_GPUS-0,1,2,3}"

# Stage2 is held fixed for the whole ladder: the legacy reduce kernel with none
# of moe_stage2_opt's knobs set.  It is not the fast stage2 -- that is the
# point.  It has zero env knobs, so there is nothing that a stage1 knob can
# collide with, and it is the same row moe_stage2_opt/configs/old.csv uses, so
# the two studies' e2e numbers sit on the same base.
K2="flydsl_moe2_afp8_wfp8_bf16_t64x128x64_reduce_persist_bnt0"

# `--stage2-opt` adds moe_stage2_opt 那条梯子跑到头的 knob 集（它的 f8），同一个
# kernelName2，只是全部 knob 打开。默认不加，因为 stage1 的归因需要一个不动的
# stage2；加上之后测的是"两级都优化完"的 e2e。
S2_KNOBS=(
  AITER_FLYDSL_STAGE2_SORTED_PARTIAL=1 FLYDSL_MOE_STAGE2_FASTVALID=1
  FLYDSL_MOE_STAGE2_SCALAR_ASCALE=1 FLYDSL_MOE_STAGE2_VEC_SCALE=1
  FLYDSL_MOE_STAGE2_BUFSTORE=1
  FLYDSL_MOE_STAGE2_HOIST_PF=1 FLYDSL_MOE_STAGE2_HOIST_X=1
  FLYDSL_MOE_STAGE2_NO_MASK=1
  FLYDSL_MOE_STAGE2_BFIRST=1 FLYDSL_MOE_STAGE2_NLANE_FIT=1
  FLYDSL_MOE_STAGE2_EVEC=8 FLYDSL_MOE_STAGE2_LDSPAD=4
  FLYDSL_MOE_STAGE2_FASTIDX=1
  FLYDSL_MOE_STAGE2_SCALAR_WSCALE=1
  FLYDSL_MOE_STAGE2_LDSCHUNK=1
)

# block_m is pinned at 64 for every stage.  It feeds moe_sorting, so changing it
# changes how many padding rows stage2 and the reduction see -- a stage1 "win"
# from a different block_m would be partly those two getting less work.  64 is
# also what the CK reference and pr1x4 stage2 use.
BM=64

# ---------------------------------------------------------------------------
# Stages: id|kernelName1|description|env...
#
# Env accumulates down the ladder, so a feature only lists its own knobs;
# prefix with '!' to opt out of that accumulation.  kernelName1 does *not*
# accumulate -- each row states the whole name it runs.
#
# `ck` and `target` are reference points rather than rungs: they are what the
# ladder is climbing towards, not steps on it.
# ---------------------------------------------------------------------------
STAGES=(
  "base|flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16|旧 FlyDSL stage1，最好的合法 tile 配置，未做任何代码改动（起点）|"
  "f1|flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16|CShuffle 在 tile_n=64 上重新可用：nlane 随 e_vec 收窄，输出从 16 位标量存变 dwordx4|FLYDSL_MOE_STAGE1_NLANE_FIT=1 FLYDSL_MOE_STAGE1_EVEC=8"
  "f2|flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16|lds_tid 塞进 CShuffle 用不到的那半 X 区：LDS 16640->16384，每 CU 从 3 个 workgroup 变 4 个|FLYDSL_MOE_STAGE1_LDSTIGHT=1"
  "f3|flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16|per-tensor 激活 scale 提到入口读一次：epilogue 每 wave 少 16 次 buffer_load 同一个 float|FLYDSL_MOE_STAGE1_SCALAR_ASCALE=1"
  "f4|flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16_bnt0|去掉权重加载的 nt（非临时）标记：同一个专家的权重被约 25 个 M-block 复用，nt 让它每次都回 HBM。L2 命中率 13%->48%|"
  "f5|flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16_bnt0|B-first 累加器 + lds_out 行填充：Step1 从 16 次 ds_write_b16 变 4 次 ds_write_b64，LDS 指令数与 bank 冲突双双对齐 PR（时间不变，见 5b）|FLYDSL_MOE_STAGE1_BFIRST=1 FLYDSL_MOE_STAGE1_LDSPAD=8"
  "ck|moe_ck2stages_gemm1_256x64x64x128_1x4_MulABScale_v1_Nswizzle0_Quant1_MulRoutedWeight0_silu_F8_F8_B16|CK stage1（生产默认，参照点）|!"
)

# Every knob any stage uses.  Cleared before each run so an exported value in
# the parent shell cannot leak into a stage that is supposed to be without it --
# the failure mode there is a silently wrong number, not an error.
ALL_KNOBS=(
  FLYDSL_MOE_STAGE1_CSHUFFLE
  FLYDSL_MOE_STAGE1_NLANE_FIT
  FLYDSL_MOE_STAGE1_EVEC
  FLYDSL_MOE_STAGE1_LDSTIGHT
  FLYDSL_MOE_STAGE1_SCALAR_ASCALE
  FLYDSL_MOE_STAGE1_BFIRST
  FLYDSL_MOE_STAGE1_LDSPAD
  FLYDSL_CK_LDS128
  AITER_LOG_MORE
  AITER_FLYDSL_STAGE2_SORTED_PARTIAL
  FLYDSL_MOE_STAGE2_FASTVALID
  FLYDSL_MOE_STAGE2_SCALAR_ASCALE
  FLYDSL_MOE_STAGE2_VEC_SCALE
  FLYDSL_MOE_STAGE2_BUFSTORE
  FLYDSL_MOE_STAGE2_HOIST_PF
  FLYDSL_MOE_STAGE2_HOIST_X
  FLYDSL_MOE_STAGE2_NO_MASK
  FLYDSL_MOE_STAGE2_BFIRST
  FLYDSL_MOE_STAGE2_NLANE_FIT
  FLYDSL_MOE_STAGE2_EVEC
  FLYDSL_MOE_STAGE2_LDSPAD
  FLYDSL_MOE_STAGE2_FASTIDX
  FLYDSL_MOE_STAGE2_SCALAR_WSCALE
  FLYDSL_MOE_STAGE2_LDSCHUNK
)

COUNTER_GROUPS=(
  "SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR"
  "SQ_WAVES SQ_WAIT_ANY SQ_BUSY_CYCLES GRBM_GUI_ACTIVE SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE"
  "MfmaUtil MeanOccupancyPerCU VALUBusy"
  "MemUnitStalled LDSBankConflict SALUBusy"
)

CASE_ARGS=(
  --token 32768 --model-dim 4096 --inter-dim 192 --expert 193 --topk 9
  --activation silu --dtype bf16 --use-g1u1 1 --doweight-stage1 0
  --quant fp8 --quant-type per_tensor
)

field() { echo "$1" | cut -d'|' -f"$2"; }
stage_ids() { for s in "${STAGES[@]}"; do field "$s" 1; done; }
stage_row() {
  for s in "${STAGES[@]}"; do [ "$(field "$s" 1)" = "$1" ] && { echo "$s"; return 0; }; done
  return 1
}
env_upto() {
  local want="$1" acc="" e
  for s in "${STAGES[@]}"; do
    e="$(field "$s" 4)"
    if [ "${e#!}" != "$e" ]; then acc=" ${e#!}"; else acc="$acc $e"; fi
    [ "$(field "$s" 1)" = "$want" ] && break
  done
  echo "$acc"
}
# Which kernel the stage's stage1 GEMM shows up as in the trace.
kernel1_of() {
  case "$1" in
    moe_ck2stages_*) echo "kernel_moe_gemm" ;;
    *gateup*)        echo "moe_2stage_gateup_prefill_1x4" ;;
    *)               echo "moe_gemm1_0" ;;
  esac
}

usage() {
  echo "usage: $0 [--repeats N] [--counters] [--stage2-opt] [--no-gpu-check] [--list] [stage ...]"
  echo "stages: $(stage_ids | tr '\n' ' ')"
  echo "GPU=$GPU（保留卡 RESERVED_GPUS=$RESERVED_GPUS，另外会实测占用情况再决定）"
}

# PTL (Peak TOPS Limiter) is a machine-level setting worth ~22% on this case.
# On this node it reads back as on for GPU 0-3 and NOT_SUPPORTED for 4-7, and
# 4-7 measure ~23% slow -- i.e. they are the PTL-off half and cannot be made
# otherwise.  So unlike the stage2 script this one does not refuse to run; it
# prints what it found, and the number to remember is that nothing here is
# comparable with a PTL-on figure.
report_ptl() {
  python3 - "$GPU" <<'PY' 2>/dev/null || true
import sys
try:
    from amdsmi import (amdsmi_init, amdsmi_shut_down,
                        amdsmi_get_processor_handles, amdsmi_get_gpu_ptl_state)
    amdsmi_init()
    h = amdsmi_get_processor_handles()[int(sys.argv[1])]
    try:
        print(f"PTL on GPU {sys.argv[1]}: {amdsmi_get_gpu_ptl_state(h)}")
    except Exception:
        print(f"PTL on GPU {sys.argv[1]}: 查不到（NOT_SUPPORTED）——按关着算，绝对值比 PTL 开的机器慢约 23%")
    amdsmi_shut_down()
except Exception:
    print("PTL: amdsmi 不可用，状态未知")
PY
}

check_gpu_free() {
  case ",$RESERVED_GPUS," in
    *",$GPU,"*)
      echo "!! GPU $GPU 在保留列表里（RESERVED_GPUS=$RESERVED_GPUS），别人在用。" >&2
      echo "   换一张：GPU=<n> $0 ...   或临时放开：RESERVED_GPUS= $0 ..." >&2
      exit 4 ;;
  esac
  local state
  state="$(python3 - "$GPU" <<'PY' 2>/dev/null
import sys
try:
    from amdsmi import (amdsmi_init, amdsmi_shut_down, amdsmi_get_processor_handles,
                        amdsmi_get_gpu_activity, amdsmi_get_gpu_vram_usage)
    amdsmi_init()
    h = amdsmi_get_processor_handles()[int(sys.argv[1])]
    gfx = amdsmi_get_gpu_activity(h).get("gfx_activity", 0)
    used = amdsmi_get_gpu_vram_usage(h).get("vram_used", 0)
    amdsmi_shut_down()
    print(f"{gfx} {used}")
except Exception:
    print("unknown")
PY
)"
  [ "$state" = unknown ] && { echo "!! 查不到 GPU $GPU 的占用状态，继续跑，但请自行确认" >&2; return 0; }
  local gfx used
  read -r gfx used <<< "$state"
  # A run of ours that just exited still holds its VRAM for a second or two, and
  # reading that back as "someone else is on this card" costs a whole invocation.
  # Only a reading that survives a re-check counts.
  if [ "${gfx:-0}" -gt 5 ] || [ "${used:-0}" -gt 2000 ]; then
    sleep 5
    state="$(python3 - "$GPU" <<'PY' 2>/dev/null
import sys
try:
    from amdsmi import (amdsmi_init, amdsmi_shut_down, amdsmi_get_processor_handles,
                        amdsmi_get_gpu_activity, amdsmi_get_gpu_vram_usage)
    amdsmi_init()
    h = amdsmi_get_processor_handles()[int(sys.argv[1])]
    gfx = amdsmi_get_gpu_activity(h).get("gfx_activity", 0)
    used = amdsmi_get_gpu_vram_usage(h).get("vram_used", 0)
    amdsmi_shut_down()
    print(f"{gfx} {used}")
except Exception:
    print("unknown")
PY
)"
    read -r gfx used <<< "$state"
  fi
  if [ "${gfx:-0}" -gt 5 ] || [ "${used:-0}" -gt 2000 ]; then
    echo "!! GPU $GPU 正被占用：gfx ${gfx}%，显存 ${used} MB（空闲基线约 284 MB）。" >&2
    echo "   别人的负载会污染这批测量，我们的也会打扰他们。换一张空闲的卡。" >&2
    exit 4
  fi
}

REPEATS=3
WANT_COUNTERS=0
WANT_S2OPT=0
SKIP_GPU_CHECK=0
CHOSEN=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repeats) REPEATS="$2"; shift 2 ;;
    --counters) WANT_COUNTERS=1; shift ;;
    --stage2-opt) WANT_S2OPT=1; shift ;;
    --no-gpu-check) SKIP_GPU_CHECK=1; shift ;;
    --list)
      printf '%-8s %s\n' stage description
      for s in "${STAGES[@]}"; do
        printf '%-8s %s\n' "$(field "$s" 1)" "$(field "$s" 3)"
        printf '         k1:  %s\n' "$(field "$s" 2)"
        e="$(field "$s" 4)"; [ -n "${e//[ !]/}" ] && echo "         env: ${e#!}"
      done
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option $1" >&2; usage; exit 2 ;;
    *) CHOSEN+=("$1"); shift ;;
  esac
done
[ ${#CHOSEN[@]} -eq 0 ] && mapfile -t CHOSEN < <(stage_ids)
for id in "${CHOSEN[@]}"; do
  stage_row "$id" >/dev/null || { echo "unknown stage '$id'" >&2; usage; exit 2; }
done

[ "$SKIP_GPU_CHECK" = 1 ] || check_gpu_free

mkdir -p "$RESULTS"
SESSION="$(date +%Y%m%d-%H%M%S)"
MODE=$([ "$WANT_S2OPT" = 1 ] && echo s2opt || echo s2base)
E2E_CSV="$RESULTS/ladder.csv"
KRN_CSV="$RESULTS/kernels.csv"
CTR_CSV="$RESULTS/counters.csv"
[ -f "$E2E_CSV" ] || echo "session,gpu,mode,stage,repeat,stage1_us,stage2_us,e2e_us,passed,cos,kernel1" > "$E2E_CSV"
[ -f "$KRN_CSV" ] || echo "session,gpu,stage,kernel,us" > "$KRN_CSV"
[ -f "$CTR_CSV" ] || echo "session,gpu,stage,counter,value,dispatches" > "$CTR_CSV"

# Every pass runs with AITER_LOG_MORE, because the stage1 kernel time is the
# headline and that is where it comes from.  The tracer inflates e2e by ~0.6%
# but leaves device_time_avg alone, so the headline is clean and the e2e column
# is only good for comparing stages with each other, not with a tracer-free run.
run_case() {
  local id="$1" out="$2" extra_env="${3:-}"
  local row k1 cfg
  row="$(stage_row "$id")"; k1="$(field "$row" 2)"
  cfg="$(mktemp /tmp/s1cfg.XXXXXX.csv)"
  bash "$HERE/mkcfg.sh" "$k1" "$K2" "$BM" "$BM" > "$cfg"
  ( for k in "${ALL_KNOBS[@]}"; do unset "$k"; done
    export AITER_CONFIG_FMOE="$cfg" HIP_VISIBLE_DEVICES="$GPU" AITER_LOG_MORE=1
    [ "$WANT_S2OPT" = 1 ] && export "${S2_KNOBS[@]}"
    # shellcheck disable=SC2046,SC2086
    export $(env_upto "$id") $extra_env 2>/dev/null || true
    cd "$REPO" && timeout 3600 python test_qmoe_multi.py "${CASE_ARGS[@]}"
  ) > "$out" 2>&1
  rm -f "$cfg"
  python3 - "$out" "$(kernel1_of "$k1")" "$k1" <<'PY'
import re, sys
path, want_kernel, want_name = sys.argv[1:4]
txt = open(path, errors="replace").read()
e2e = re.search(r"e2e fused_moe: *([0-9.]+)", txt)
cos = re.search(r"cos=([0-9.]+)", txt)
ok = re.search(r"pass=(\w+)", txt)
got = re.search(r"kernelName1='([^']*)'", txt)
s1 = s2 = ""
for line in txt.splitlines():
    f = line.split()
    if len(f) >= 8 and f[-2] == "CUDA":
        name = " ".join(f[1:-6])
        try:
            avg = float(f[-3].replace(",", ""))
        except ValueError:
            continue
        if want_kernel in name:
            s1 = avg
        elif "moe_gemm2" in name or "down_prefill" in name:
            s2 = avg
print("|".join([
    str(s1), str(s2), e2e.group(1) if e2e else "",
    ok.group(1) if ok else "", cos.group(1) if cos else "",
    got.group(1) if got else "",
]))
PY
}

collect_counters() {
  local id="$1" row k1 cfg krn d gi=0
  row="$(stage_row "$id")"; k1="$(field "$row" 2)"
  krn="$(kernel1_of "$k1")"
  cfg="$(mktemp /tmp/s1cfg.XXXXXX.csv)"
  bash "$HERE/mkcfg.sh" "$k1" "$K2" "$BM" "$BM" > "$cfg"
  for grp in "${COUNTER_GROUPS[@]}"; do
    gi=$((gi + 1)); d="$(mktemp -d)"
    ( for k in "${ALL_KNOBS[@]}"; do unset "$k"; done
      export AITER_CONFIG_FMOE="$cfg" HIP_VISIBLE_DEVICES="$GPU"
      # shellcheck disable=SC2046,SC2086
      export $(env_upto "$id") 2>/dev/null || true
      # shellcheck disable=SC2086
      cd "$REPO" && timeout 900 rocprofv3 --pmc $grp \
        --kernel-include-regex "$krn" -d "$d" -o r -- \
        python test_qmoe_multi.py "${CASE_ARGS[@]}" --warmup 1 --iters 3 --run perf
    ) >/dev/null 2>&1
    python3 - "$d" "$SESSION" "$GPU" "$id" >> "$CTR_CSV" <<'PY'
import glob, sqlite3, sys
d, session, gpu, stage = sys.argv[1:5]
hits = glob.glob(f"{d}/**/*.db", recursive=True)
if not hits:
    sys.exit(0)
c = sqlite3.connect(hits[0])
t = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
info = next(x for x in t if x.startswith("rocpd_info_pmc"))
ev = next(x for x in t if x.startswith("rocpd_pmc_event"))
dsp = next(x for x in t if x.startswith("rocpd_kernel_dispatch"))
n = c.execute(f"select count(*) from {dsp}").fetchone()[0] or 1
for name, v in c.execute(
    f"select i.name, sum(e.value) from {ev} e join {info} i on i.id=e.pmc_id group by i.name"
):
    print(f"{session},{gpu},{stage},{name},{v / n},{n}")
PY
    rm -rf "$d"
  done
  rm -f "$cfg"
  local k
  k=$(awk -F, -v s="$SESSION" -v st="$id" '$1==s && $3==st' "$CTR_CSV" | wc -l)
  echo "           计数器 $k 项 -> results/counters.csv"
}

echo "session $SESSION   gpu $GPU   repeats $REPEATS"
echo "$(report_ptl)"
echo "stages: ${CHOSEN[*]}"
echo

declare -A MED MEDE MED2
for id in "${CHOSEN[@]}"; do
  row="$(stage_row "$id")"; want_k1="$(field "$row" 2)"
  vals=(); evals=(); v2=()
  for ((r = 0; r < REPEATS; r++)); do
    log="$(mktemp)"
    IFS='|' read -r s1 s2 e2e passed cos got <<< "$(run_case "$id" "$log")"
    if [ -z "$s1" ]; then
      echo "  $id rep$r  FAILED -- tail of $log:"; tail -25 "$log"; continue
    fi
    # A kernel name the config cannot honour does not error: fused_moe silently
    # falls back to another row.  This is the only way to notice.
    flag=""
    [ -n "$got" ] && [ "$got" != "$want_k1" ] && flag="  !! CONFIG MISS: got $got"
    [ -n "$passed" ] && [ "$passed" != "True" ] && flag="  !! FUNC FAIL cos=$cos"
    printf '  %-7s rep%-2d stage1 %9.1f  stage2 %9.1f   e2e %9.1f us  %s%s\n' \
      "$id" "$r" "$s1" "${s2:-0}" "${e2e:-0}" "${cos:+cos=$cos}" "$flag"
    vals+=("$s1"); evals+=("${e2e:-0}"); v2+=("${s2:-0}")
    echo "$SESSION,$GPU,$MODE,$id,$r,$s1,$s2,$e2e,$passed,$cos,$got" >> "$E2E_CSV"
    rm -f "$log"
  done
  [ ${#vals[@]} -eq 0 ] && continue
  med() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
  MED[$id]="$(med "${vals[@]}")"
  MEDE[$id]="$(med "${evals[@]}")"
  MED2[$id]="$(med "${v2[@]}")"
  [ "$WANT_COUNTERS" = 1 ] && collect_counters "$id"
done

echo
echo "mode: $MODE$([ "$WANT_S2OPT" = 1 ] && echo '  (stage2 全部 knob 打开 = moe_stage2_opt 的 f8)')"
printf '%-8s %11s %10s %11s %11s %10s   %s\n' stage stage1 "vs base" stage2 e2e "e2e Δbase" 说明
printf -- '-%.0s' {1..132}; echo
base="${MED[base]:-}"; ebase="${MEDE[base]:-}"
for id in "${CHOSEN[@]}"; do
  m="${MED[$id]:-}"; [ -z "$m" ] && continue
  db="-"; de="-"
  [ -n "$base" ]  && db="$(awk -v a="$m" -v b="$base" 'BEGIN{printf "%+.1f", a-b}')"
  [ -n "$ebase" ] && de="$(awk -v a="${MEDE[$id]}" -v b="$ebase" 'BEGIN{printf "%+.1f", a-b}')"
  printf '%-8s %11.1f %10s %11.1f %11.1f %10s   %s\n' \
    "$id" "$m" "$db" "${MED2[$id]:-0}" "${MEDE[$id]:-0}" "$de" "$(field "$(stage_row "$id")" 3)"
done

ref="${MED[ck]:-}"
if [ -n "$base" ] && [ -n "$ref" ]; then
  echo
  awk -v b="$base" -v t="$ref" 'BEGIN{
    printf "  stage1 起点 %.1f -> 参照 %.1f   总差距 %.1f us (%.3fx)\n", b, t, b-t, b/t}'
  prev="$base"; eprev="$ebase"
  for id in "${CHOSEN[@]}"; do
    case "$id" in base|ck) continue ;; esac
    m="${MED[$id]:-}"; [ -z "$m" ] && continue
    awk -v id="$id" -v p="$prev" -v m="$m" -v b="$base" -v t="$ref" \
        -v ep="$eprev" -v em="${MEDE[$id]:-0}" -v eb="$ebase" 'BEGIN{
      printf "  %-8s stage1 %+9.1f (累计 %+9.1f, 补上 %5.1f%%)   e2e %+9.1f (累计 %+9.1f)\n",
             id, m-p, m-b, (b-t)!=0 ? (b-m)/(b-t)*100 : 0, em-ep, em-eb}'
    prev="$m"; eprev="${MEDE[$id]:-0}"
  done
fi

echo
echo "wrote $E2E_CSV"
[ "$WANT_COUNTERS" = 1 ] && echo "wrote $CTR_CSV"
exit 0
