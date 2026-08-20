#!/usr/bin/env bash
# Blockwise-fp8 MoE sweep over op_tests/test_moe_2stage.py (-q 5 = per_128x128).
#
# Variants per shape (pick with VARIANTS, default all but `co`):
#   asm_ck        default dispatch (tuned CSV -> asm 1-stage / CK 2-stage)
#   flydsl        AITER_FLYDSL_BLKFP8=1 + AITER_BYPASS_TUNE_CONFIG=1
#   flydsl_fused  same, plus --swiglu-limit and --smooth-scale
#   co            AITER_MOE_BLK_CO=1: the same kernels as `flydsl`, loaded from
#                 the prebuilt code objects. Only shapes that hsa/flydsl_export.py
#                 has actually exported will run; anything else fails to find
#                 its .co.
#
# $OUT gets only variant/shape/us/us_stage1/us_stage2/cos_sim; the full run log
# goes to $OUT.log. Override anything from the environment, e.g.
#   SHAPES="6144,2048 6144,512" TOKENS="1 8 64" KERNEL=0 ./bench_moe_blockwise.sh
#   VARIANTS="flydsl co" E=256 TOPK=8 ./bench_moe_blockwise.sh
set -euo pipefail

SHAPES=${SHAPES:-"6144,256"}   # space-separated model_dim,inter_dim pairs
E=${E:-256}
TOPK=${TOPK:-8}
VARIANTS=${VARIANTS:-"asm_ck flydsl flydsl_fused"}
TOKENS=${TOKENS:-"1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768"}
LIMIT=${LIMIT:-10.0}
OUT=${OUT:-moe_blockwise_bench.csv}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-50}
# KERNEL=1 (default) adds --kernel: `us`/`tflops` still measure the whole
# fused_moe call, plus us_stage1/us_stage2 and their tflops from timing each
# kernel in isolation. The stages exclude host-side quant + moe_sorting, so they
# do not sum to `us` -- the gap is exactly that overhead. KERNEL=0 skips the
# per-stage passes (roughly 2x faster, end-to-end columns only).
KERNEL=${KERNEL:-1}

cd "$(dirname "$0")/../.."
: >"$OUT"
: >"$OUT.log"

KERNEL_ARG=()
[ "$KERNEL" = "1" ] && KERNEL_ARG=(--kernel)

run() {  # run <label> <model_dim,inter_dim> <env-assignments...>
    local label=$1 dim=$2
    shift 2
    echo "==> $label  dim=$dim"
    # Never pass --no-legacy: it disables the CLI sweep that -t/-dim/-q feed.
    env "$@" python op_tests/test_moe_2stage.py \
        -q 5 -t $TOKENS -dim "$dim" -e "$E" -k "$TOPK" \
        --no-flydsl-csv --csv "$OUT" --csv-tag "$label" \
        --warmup "$WARMUP" --iters "$ITERS" \
        "${KERNEL_ARG[@]}" "${EXTRA[@]}" >>"$OUT.log" 2>&1
}

for dim in $SHAPES; do
    for v in $VARIANTS; do
        case $v in
            asm_ck)       EXTRA=(); run "$v" "$dim" AITER_FLYDSL_BLKFP8=0 AITER_BYPASS_TUNE_CONFIG=0 ;;
            flydsl)       EXTRA=(); run "$v" "$dim" AITER_FLYDSL_BLKFP8=1 AITER_BYPASS_TUNE_CONFIG=1 ;;
            flydsl_fused) EXTRA=(--swiglu-limit "$LIMIT" --smooth-scale)
                          run "$v" "$dim" AITER_FLYDSL_BLKFP8=1 AITER_BYPASS_TUNE_CONFIG=1 ;;
            co)           EXTRA=(); run "$v" "$dim" AITER_MOE_BLK_CO=1 AITER_BYPASS_TUNE_CONFIG=1 ;;
            co_fused)     EXTRA=(--swiglu-limit "$LIMIT" --smooth-scale)
                          run "$v" "$dim" AITER_MOE_BLK_CO=1 AITER_BYPASS_TUNE_CONFIG=1 ;;
            *)            echo "unknown variant: $v" >&2; exit 1 ;;
        esac
    done
done

echo
# `column` is not in every container image; awk pads to the widest cell per field.
awk -F, '{for(i=1;i<=NF;i++){c[NR,i]=$i;if(length($i)>w[i])w[i]=length($i)};n=NF}
         END{for(r=1;r<=NR;r++){for(i=1;i<=n;i++)printf "%-*s ",w[i],c[r,i];print ""}}' "$OUT"
echo
echo "csv: $OUT   log: $OUT.log"
