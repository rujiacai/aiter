#!/usr/bin/env bash
# Walk the legacy stage2 kernel from `base` towards `target`, one feature at a
# time.
#
# Every stage is cumulative: stage N runs with the env of stages 1..N, so the
# table reads as a ladder and each row's delta is what that one feature bought
# on top of everything before it.  `base` is the legacy kernel untouched and
# `target` is pr1x4 -- the gap between them is what there is left to close.
#
#   ./run.sh                        # every stage, 3 repeats
#   ./run.sh base f1                # only these stages
#   ./run.sh --repeats 5
#   ./run.sh --kernels              # also do an AITER_LOG_MORE pass per stage
#   ./run.sh --list
#
# Results land in results/e2e.csv and results/kernels.csv (append-only, one
# session id per invocation) so a partial re-run never loses the rest.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RESULTS="$HERE/results"
# Which half of the node is ours moves around, so the static list is only a
# reminder -- check_gpu_free() below is the guard that actually holds, since it
# looks at live occupancy and so also catches a card that frees up or gets taken
# after the list was last edited.  Note the `-` rather than `:-`: RESERVED_GPUS=
# has to mean "nothing is reserved", which is what the message below tells you
# to do, and `:-` would silently put the default back.
GPU="${GPU:-4}"
RESERVED_GPUS="${RESERVED_GPUS-0,1,2,3}"

# ---------------------------------------------------------------------------
# Stages.  Add a feature by appending a row: id|config|description|env...
#
# `config` picks the fused_moe row (which stage2 kernel runs); `env` is the
# knob set this feature adds, and stages accumulate, so a new feature only
# needs to list its own knobs.  Prefix the env with '!' to opt out of that
# accumulation.  Keep `target` last.
#
# A feature that is a code change rather than an env knob still gets a row:
# leave the env empty and gate the code on its own AITER_* variable, so `base`
# stays reproducible after the code lands.
# ---------------------------------------------------------------------------
STAGES=(
  "base|old|旧内核 reduce，未做任何改动（起点）|"
  "f1|old|partial 存储从 (token, slot) 改成 sorted 行序 + 去掉哨兵掩码|AITER_FLYDSL_STAGE2_SORTED_PARTIAL=1 FLYDSL_MOE_STAGE2_FASTVALID=1"
  "f2|old|epilogue 的 scale 与地址从逐元素重算改成提前算一次：per-tensor scale 提到入口 + 向量化缩放 + per-block buffer 存储|FLYDSL_MOE_STAGE2_SCALAR_ASCALE=1 FLYDSL_MOE_STAGE2_VEC_SCALE=1 FLYDSL_MOE_STAGE2_BUFSTORE=1"
  "f3|old|循环不变量外提（路由权重 / guard / X 的 LDS 读不再逐 N-tile 重做）+ 输出宽度 e_vec=4|FLYDSL_MOE_STAGE2_HOIST_PF=1 FLYDSL_MOE_STAGE2_HOIST_X=1 FLYDSL_MOE_STAGE2_EVEC=4"
  "f4|old|删掉掩码 epilogue：sorted 行布局下 padding 行的写无害，整条掩码路径是死代码|FLYDSL_MOE_STAGE2_NO_MASK=1"
  "f5|old|CShuffle 两端一起加宽：B-first 让 Step1 写 b64 + nlane 随 e_vec 收窄让 Step2 存 dwordx4，配 LDSPAD=4 解 bank 冲突|FLYDSL_MOE_STAGE2_BFIRST=1 FLYDSL_MOE_STAGE2_NLANE_FIT=1 FLYDSL_MOE_STAGE2_EVEC=8 FLYDSL_MOE_STAGE2_LDSPAD=4"
  "f6|old|去掉 B 下标拆分里的恒等取模：idx2crd 对非 2 幂的 experts*model_dim/16 会发 magic-number 取模，换成显式移位/掩码|FLYDSL_MOE_STAGE2_FASTIDX=1"
  "f7|old|per-tensor 权重 scale 塌缩成标量：整条 scale 链降成每行一个标量，epilogue 每个输出从两次向量乘变一次|FLYDSL_MOE_STAGE2_SCALAR_WSCALE=1"
  "f8|old|CShuffle 分块暂存：lds_out 只留一轮 16 行而不是整个 tile，LDS 29440->16768，occupancy 8->12 wave/CU|FLYDSL_MOE_STAGE2_LDSCHUNK=1"
  "target|new|新内核 pr1x4 + Triton 归约（目标）|!AITER_PR1X4_TRITON_REDUCE=1"
)

# Every knob any stage uses.  Cleared before each run so an exported value in
# the parent shell cannot leak into a stage that is supposed to be without it --
# the failure mode there is a silently wrong number, not an error.
ALL_KNOBS=(
  AITER_FLYDSL_STAGE2_SORTED_PARTIAL
  FLYDSL_MOE_STAGE2_FASTVALID
  FLYDSL_MOE_STAGE2_SCALAR_ASCALE
  FLYDSL_MOE_STAGE2_BUFSTORE
  FLYDSL_MOE_STAGE2_VEC_SCALE
  FLYDSL_MOE_STAGE2_HOIST_PF
  FLYDSL_MOE_STAGE2_HOIST_X
  FLYDSL_MOE_STAGE2_NO_MASK
  FLYDSL_MOE_STAGE2_EVEC
  FLYDSL_MOE_STAGE2_BFIRST
  FLYDSL_MOE_STAGE2_LDSPAD
  FLYDSL_MOE_STAGE2_NLANE_FIT
  FLYDSL_MOE_STAGE2_FASTIDX
  FLYDSL_MOE_STAGE2_SCALAR_WSCALE
  FLYDSL_MOE_STAGE2_LDSCHUNK
  AITER_PR1X4_TRITON_REDUCE
  AITER_LOG_MORE
)

# Hardware counters, in groups that each fit a single collection pass.  These are
# the *dynamic* numbers: static ISA counts cannot be compared between these two
# kernels at all, because the legacy one is fully unrolled behind 352
# s_cbranch_execz guards (static over-counts) while pr1x4 keeps a real loop
# (static under-counts by the trip count).
COUNTER_GROUPS=(
  "SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR"
  "SQ_WAVES SQ_WAIT_ANY SQ_BUSY_CYCLES GRBM_GUI_ACTIVE SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE"
  "MfmaUtil MeanOccupancyPerCU VALUBusy"
  "MemUnitStalled LDSBankConflict SALUBusy"
)
# Which kernel each config's stage2 GEMM shows up as.
kernel_of() { [ "$1" = new ] && echo moe_2stage_down_prefill_1x4_0 || echo moe_gemm2_0; }

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
# Cumulative env: everything from the first stage up to and including $1.
# A leading '!' means the stage stands alone -- used by `target`, which is a
# different kernel and so inherits nothing from the ladder.
env_upto() {
  local want="$1" acc="" e
  for s in "${STAGES[@]}"; do
    e="$(field "$s" 4)"
    if [ "${e#!}" != "$e" ]; then
      acc=" ${e#!}"
    else
      acc="$acc $e"
    fi
    [ "$(field "$s" 1)" = "$want" ] && break
  done
  echo "$acc"
}

usage() {
  echo "usage: $0 [--repeats N] [--kernels] [--counters]"
  echo "          [--no-ptl-check] [--no-gpu-check] [--list] [stage ...]"
  echo "stages: $(stage_ids | tr '\n' ' ')"
  echo "GPU=$GPU（保留卡 RESERVED_GPUS=$RESERVED_GPUS，另外会实测占用情况再决定）"
}

# PTL (Peak TOPS Limiter) is a machine-level setting that does not survive a
# reboot.  With it off every measurement here is ~22% slower -- and nothing else
# changes: kernel name, cos and the intra-run spread all look normal, only the
# absolute numbers move, and they move by different amounts per stage so the
# attribution shifts too.  One full batch of this project's data died that way.
# Refuse to run rather than append a silently wrong ladder to the csv.
check_ptl() {
  local bad
  # Queried three times: a single read has been observed to come back false for
  # every GPU right after a workload exits, while a direct query a second later
  # said all eight were on.  Only a reading that repeats is believed.
  bad="$(python3 - <<'PY' 2>/dev/null
import time
try:
    from amdsmi import amdsmi_init, amdsmi_shut_down
    from amdsmi import amdsmi_get_processor_handles, amdsmi_get_gpu_ptl_state
    amdsmi_init()
    hs = amdsmi_get_processor_handles()
    off = None
    for attempt in range(3):
        cur = {i for i, h in enumerate(hs) if not amdsmi_get_gpu_ptl_state(h)}
        off = cur if off is None else (off & cur)
        if not off:
            break
        time.sleep(0.5)
    amdsmi_shut_down()
    print(",".join(str(i) for i in sorted(off)))
except Exception:
    print("unknown")
PY
)"
  case "$bad" in
    "")       return 0 ;;
    unknown)  echo "!! 查不到 PTL 状态（amdsmi 不可用），继续跑，但请自行确认" >&2 ;;
    *)        echo "!! PTL 在 GPU $bad 上是关的。所有测量会偏慢约 22%，而且各 stage 偏得不一样多，" >&2
              echo "   归因也会跟着变——这批数字不能用。" >&2
              echo "   打开（整机设置，影响本节点所有人）：" >&2
              echo "     amd-smi set -g all --ptl-status 1" >&2
              echo "     amd-smi set -g all --ptl-format F8,BF16" >&2
              echo "   确实要带着关掉的 PTL 跑，加 --no-ptl-check。" >&2
              exit 3 ;;
  esac
}

# Running on a card someone else is using both disturbs them and gives us
# contaminated numbers, so refuse on both counts: reserved list, and live occupancy.
check_gpu_free() {
  case ",$RESERVED_GPUS," in
    *",$GPU,"*)
      echo "!! GPU $GPU 在保留列表里（RESERVED_GPUS=$RESERVED_GPUS），别人在用。" >&2
      echo "   换一张：GPU=<n> $0 ...   或临时放开：RESERVED_GPUS= $0 ..." >&2
      exit 4 ;;
  esac
  # An idle MI308X sits at ~284 MB VRAM and 0% gfx.  Anything clearly above that
  # is someone else's job -- their work perturbs our timing and ours perturbs theirs.
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
  if [ "${gfx:-0}" -gt 5 ] || [ "${used:-0}" -gt 2000 ]; then
    echo "!! GPU $GPU 正被占用：gfx ${gfx}%，显存 ${used} MB（空闲基线约 284 MB）。" >&2
    echo "   别人的负载会污染这批测量，我们的也会打扰他们。换一张空闲的卡。" >&2
    echo "   确实要挤着跑，加 --no-gpu-check。" >&2
    exit 4
  fi
}

REPEATS=3
WANT_KERNELS=0
WANT_COUNTERS=0
SKIP_PTL_CHECK=0
SKIP_GPU_CHECK=0
CHOSEN=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repeats) REPEATS="$2"; shift 2 ;;
    --kernels) WANT_KERNELS=1; shift ;;
    --counters) WANT_COUNTERS=1; shift ;;
    --no-ptl-check) SKIP_PTL_CHECK=1; shift ;;
    --no-gpu-check) SKIP_GPU_CHECK=1; shift ;;
    --list)
      printf '%-8s %-6s %s\n' stage config description
      for s in "${STAGES[@]}"; do
        printf '%-8s %-6s %s\n' "$(field "$s" 1)" "$(field "$s" 2)" "$(field "$s" 3)"
        e="$(field "$s" 4)"; [ -n "${e// /}" ] && echo "         env: $e"
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
[ "$SKIP_PTL_CHECK" = 1 ] || check_ptl

mkdir -p "$RESULTS"
SESSION="$(date +%Y%m%d-%H%M%S)"
E2E_CSV="$RESULTS/e2e.csv"
KRN_CSV="$RESULTS/kernels.csv"
CTR_CSV="$RESULTS/counters.csv"
[ -f "$E2E_CSV" ] || echo "session,gpu,stage,repeat,us,passed,cos,kernel2" > "$E2E_CSV"
[ -f "$KRN_CSV" ] || echo "session,gpu,stage,kernel,us" > "$KRN_CSV"
[ -f "$CTR_CSV" ] || echo "session,gpu,stage,counter,value,dispatches" > "$CTR_CSV"

# rocprofv3 over just this stage's stage2 GEMM, one pass per counter group.
# Timings from these runs are meaningless (collection perturbs every dispatch);
# only the counter values are.
collect_counters() {
  local id="$1" row cfg krn d gi=0
  row="$(stage_row "$id")"; cfg="$HERE/configs/$(field "$row" 2).csv"
  krn="$(kernel_of "$(field "$row" 2)")"
  for grp in "${COUNTER_GROUPS[@]}"; do
    gi=$((gi + 1)); d="$(mktemp -d)"
    ( for k in "${ALL_KNOBS[@]}"; do unset "$k"; done
      export AITER_CONFIG_FMOE="$cfg" HIP_VISIBLE_DEVICES="$GPU"
      # shellcheck disable=SC2046,SC2086
      export $(env_upto "$id") 2>/dev/null || true
      # shellcheck disable=SC2086
      cd "$REPO" && timeout 600 rocprofv3 --pmc $grp \
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
  local k
  k=$(awk -F, -v s="$SESSION" -v st="$id" '$1==s && $3==st' "$CTR_CSV" | wc -l)
  echo "           计数器 $k 项 -> results/counters.csv"
}

# Runs one stage once and echoes "us|passed|cos|kernel2"; stdout of the case
# goes to $2 so a failure can be read back.
run_case() {
  local id="$1" out="$2" extra_env="${3:-}"
  local row cfg
  row="$(stage_row "$id")"; cfg="$HERE/configs/$(field "$row" 2).csv"
  ( for k in "${ALL_KNOBS[@]}"; do unset "$k"; done
    export AITER_CONFIG_FMOE="$cfg" HIP_VISIBLE_DEVICES="$GPU"
    # shellcheck disable=SC2046,SC2086
    export $(env_upto "$id") $extra_env 2>/dev/null || true
    cd "$REPO" && timeout 1800 python test_qmoe_multi.py "${CASE_ARGS[@]}"
  ) > "$out" 2>&1
  awk '
    /e2e fused_moe:/ { if (match($0, /e2e fused_moe: *[0-9.]+/)) { s=substr($0,RSTART,RLENGTH); sub(/.*: */,"",s); us=s } }
    /\[FUNC\]/ { if (match($0,/pass=[A-Za-z]+/)) { p=substr($0,RSTART+5,RLENGTH-5) }
                 if (match($0,/cos=[0-9.]+/))    { c=substr($0,RSTART+4,RLENGTH-4) } }
    /kernelName2=/ { if (match($0,/kernelName2='"'"'[^'"'"']*/)) { k=substr($0,RSTART+13,RLENGTH-13) } }
    END { printf "%s|%s|%s|%s", us, p, c, k }
  ' "$out"
}

echo "session $SESSION   gpu $GPU   repeats $REPEATS"
echo "stages: ${CHOSEN[*]}"
echo

declare -A MED
for id in "${CHOSEN[@]}"; do
  row="$(stage_row "$id")"
  vals=()
  for ((r = 0; r < REPEATS; r++)); do
    log="$(mktemp)"
    IFS='|' read -r us passed cos krn <<< "$(run_case "$id" "$log")"
    if [ -z "$us" ]; then
      echo "  $id rep$r  FAILED -- tail of $log:"; tail -20 "$log"; continue
    fi
    # A bad kernel name in the csv does not error, fused_moe silently falls back
    # to another config; this is the only way to notice.
    want="$(grep -o 'flydsl_moe2[^,]*' "$HERE/configs/$(field "$row" 2).csv" | head -1)"
    flag=""
    [ -n "$krn" ] && [ "$krn" != "$want" ] && flag="  !! CONFIG MISS: got $krn want $want"
    [ -n "$passed" ] && [ "$passed" != "True" ] && flag="  !! FUNC FAIL cos=$cos"
    printf '  %-7s rep%-2d %9.1f us  %s%s\n' "$id" "$r" "$us" "${cos:+cos=$cos}" "$flag"
    vals+=("$us")
    echo "$SESSION,$GPU,$id,$r,$us,$passed,$cos,$krn" >> "$E2E_CSV"
    rm -f "$log"
  done
  [ ${#vals[@]} -eq 0 ] && continue
  MED[$id]="$(printf '%s\n' "${vals[@]}" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')"

  if [ "$WANT_KERNELS" = 1 ]; then
    log="$(mktemp)"
    run_case "$id" "$log" "AITER_LOG_MORE=1" >/dev/null
    # The pandas dump: rows between the header and the [avg us/iter] footer.
    # The kernel name contains spaces, so fields are taken from the right.
    python3 - "$log" "$SESSION" "$GPU" "$id" >> "$KRN_CSV" <<'PY'
import re, sys
path, session, gpu, stage = sys.argv[1:5]
rows, on = [], False
for line in open(path, errors="replace"):
    if "device_time_avg" in line and "host_time_sum" in line:
        on = True; continue
    if not on:
        continue
    if line.startswith("[avg us/iter]"):
        break
    f = line.split()
    if len(f) < 8 or f[-2] != "CUDA":
        continue
    avg = f[-3].replace(",", "")
    name = " ".join(f[1:-6])
    try:
        avg = float(avg)
    except ValueError:
        continue
    if avg > 0:
        rows.append((name, avg))
for name, avg in sorted(rows, key=lambda r: -r[1]):
    name = re.sub(r"\s+", " ", name).replace('"', "'")
    print(f'{session},{gpu},{stage},"{name}",{avg}')
PY
    n=$(awk -F, -v s="$SESSION" -v st="$id" '$1==s && $3==st' "$KRN_CSV" | wc -l)
    echo "           逐算子 $n 条 -> results/kernels.csv"
    rm -f "$log"
  fi

  if [ "$WANT_COUNTERS" = 1 ]; then
    collect_counters "$id"
  fi
done

echo
printf '%-8s %10s %10s %10s   %s\n' stage 中位数us "vs base" "vs target" 说明
printf -- '-%.0s' {1..104}; echo
base="${MED[base]:-}"; target="${MED[target]:-}"
for id in "${CHOSEN[@]}"; do
  m="${MED[$id]:-}"; [ -z "$m" ] && continue
  db="-"; dt="-"
  [ -n "$base" ]   && db="$(awk -v a="$m" -v b="$base"   'BEGIN{printf "%+.1f", a-b}')"
  [ -n "$target" ] && dt="$(awk -v a="$m" -v b="$target" 'BEGIN{printf "%+.1f", a-b}')"
  printf '%-8s %10.1f %10s %10s   %s\n' "$id" "$m" "$db" "$dt" "$(field "$(stage_row "$id")" 3)"
done

if [ -n "$base" ] && [ -n "$target" ]; then
  echo
  awk -v b="$base" -v t="$target" 'BEGIN{
    printf "  起点 %.1f -> 目标 %.1f   总差距 %.1f us (%.3fx)\n", b, t, b-t, b/t}'
  prev="$base"
  for id in "${CHOSEN[@]}"; do
    [ "$id" = base ] && continue
    m="${MED[$id]:-}"; [ -z "$m" ] && continue
    awk -v id="$id" -v p="$prev" -v m="$m" -v b="$base" -v t="$target" 'BEGIN{
      printf "  %-8s %+9.1f us   累计 %+9.1f us   已补上总差距的 %5.1f%%\n",
             id, m-p, m-b, (b-t)!=0 ? (b-m)/(b-t)*100 : 0}'
    prev="$m"
  done
fi

if [ "$WANT_COUNTERS" = 1 ]; then
  echo
  python3 - "$CTR_CSV" "$SESSION" "${CHOSEN[@]}" <<'PY'
import csv, collections, sys
path, session = sys.argv[1:3]
chosen = sys.argv[3:]
d = collections.defaultdict(dict)
for r in csv.DictReader(open(path)):
    if r["session"] == session:
        d[r["stage"]][r["counter"]] = float(r["value"])
have = [s for s in chosen if s in d]
if not have:
    sys.exit(0)
# Every stage runs the same GEMM, so SQ_INSTS_MFMA must be identical across them.
# When it is not, that collection pass was perturbed and every per-MFMA number
# derived from it is wrong -- this has happened, so check rather than trust.
mf = {s: d[s].get("SQ_INSTS_MFMA") for s in have if d[s].get("SQ_INSTS_MFMA")}
if len(set(mf.values())) > 1:
    print("!! SQ_INSTS_MFMA 在各 stage 间不一致，说明有采集被扰动，下表的每-MFMA 数不可信:")
    for s, v in mf.items():
        print(f"     {s:<10}{v:>16,.0f}")
    print("   重采那个 stage 的计数器再看。\n")
# Per-MFMA is the only fair unit: it cancels the padding blocks that skip work.
PER_MFMA = ["SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_LDS",
            "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR"]
RATE = ["MfmaUtil", "MeanOccupancyPerCU", "VALUBusy", "SALUBusy",
        "MemUnitStalled", "LDSBankConflict"]
RAW = ["SQ_INSTS_MFMA", "SQ_WAVES", "GRBM_GUI_ACTIVE", "SQ_WAIT_ANY"]

def cell(s, k):
    v = d[s].get(k)
    if v is None:
        return None
    if k in PER_MFMA:
        m = d[s].get("SQ_INSTS_MFMA")
        return v / m if m else None
    return v

w = 14
# Progress is measured for the newest legacy-side stage, i.e. the last one that
# is not `target`: how much of the f1 -> target gap it has closed.
cur_stage = next((s for s in reversed(have) if s != "target"), None)
show_prog = cur_stage not in (None, "f1") and "f1" in d and "target" in d
print("动态计数器（每条 MFMA 归一，除 rate/raw 外）")
hdr = f"{'counter':<24}" + "".join(f"{s:>{w}}" for s in have)
if show_prog:
    hdr += f"{cur_stage + ' 补上':>12}"
print(hdr)
print("-" * len(hdr))
for group, keys in (("每 MFMA", PER_MFMA), ("比率 %", RATE), ("原始值", RAW)):
    print(f"[{group}]")
    for k in keys:
        vals = [cell(s, k) for s in have]
        if all(v is None for v in vals):
            continue
        line = f"  {k:<22}"
        for v in vals:
            line += f"{v:>{w},.3f}" if v is not None else f"{'-':>{w}}"
        if show_prog:
            a, b, cur = cell("f1", k), cell("target", k), cell(cur_stage, k)
            if None not in (a, b, cur) and abs(a - b) > 1e-9:
                line += f"{(a - cur) / (a - b) * 100:>11.0f}%"
        print(line)
PY
fi

echo
echo "wrote $E2E_CSV"
[ "$WANT_KERNELS" = 1 ] && echo "wrote $KRN_CSV"
[ "$WANT_COUNTERS" = 1 ] && echo "wrote $CTR_CSV"
exit 0
