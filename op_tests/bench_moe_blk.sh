#!/usr/bin/env bash
# Blockwise-fp8 MoE sweep over op_tests/test_moe_2stage.py (-q 5 = per_128x128).
#
# Variants (pick with VARIANTS, default `co co_fused`):
#   asm_ck    AITER_MOE_BLK_CO=0: stock dispatch (tuned CSV -> asm 1-stage / CK
#             2-stage). The baseline the code objects are meant to replace.
#   co        the prebuilt code objects in hsa/{arch}/moe_blk/
#   co_fused  same, plus the swiglu clamp and the per-expert smooth_scale
#
# AITER_BYPASS_TUNE_CONFIG=1 goes with both co variants on purpose: get_2stage_cfgs
# returns the asm 1-stage kernel before it ever reaches the code-object branch when
# a tuned_fmoe.csv row claims the shape, so without it the `co` rows would silently
# be asm timings and `co_fused` would fail outright on the clamp guard.
#
# Only shapes hsa/flydsl_export.py has published run on the code objects; anything
# else falls back to asm/CK with no error, so a `co` row for an unpublished shape
# is really an asm_ck row. aiter/configs/moe_blk_tuned.csv lists what exists --
# currently 6144,256 with E/k = 256/8 and 257/9, and 6144,2048 with 16/8 and 17/9.
#
# $OUT gets variant/shape/us/tflops plus two accuracy columns against the torch
# reference: logits_diff (||x-y||^2 / (||x||^2 + ||y||^2), 0.0 is bit-exact) and
# cos_sim. logits_diff is the one to gate on -- cosine cannot see a pure
# magnitude error, so a kernel that drops a scale factor still scores 1.0 there.
# The full run log goes to $OUT.log.
# Override anything from the environment, e.g.
#   SHAPES="6144,2048" E=16 TOPK=8 ./op_tests/bench_moe_blk.sh
#   VARIANTS="asm_ck co" TOKENS="1 8 64" KERNEL=0 ./op_tests/bench_moe_blk.sh
set -euo pipefail

SHAPES=${SHAPES:-"6144,256"}   # space-separated model_dim,inter_dim pairs
E=${E:-256}
TOPK=${TOPK:-8}
VARIANTS=${VARIANTS:-"co co_fused"}
TOKENS=${TOKENS:-"1 2 4 8 16 32 64 128 256"}
LIMIT=${LIMIT:-10.0}
OUT=${OUT:-moe_blk_bench.csv}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-50}
# KERNEL=1 (default) adds --kernel: `us`/`tflops` still measure the whole
# fused_moe call, plus us_stage1/us_stage2 and their tflops from timing each
# kernel launch in isolation. The stages exclude the host-side quant +
# moe_sorting, so they do not sum to `us` -- the gap is exactly that overhead.
# KERNEL=0 skips the per-stage passes (roughly 2x faster, end-to-end only).
KERNEL=${KERNEL:-1}

cd "$(dirname "$0")/.."
: >"$OUT"
: >"$OUT.log"

KERNEL_ARG=()
[ "$KERNEL" = "1" ] && KERNEL_ARG=(--kernel)

run() {  # run <label> <model_dim,inter_dim> <env-assignments...>
    local label=$1 dim=$2
    shift 2
    echo "==> $label  dim=$dim e=$E k=$TOPK"
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
            asm_ck)   EXTRA=(); run "$v" "$dim" AITER_MOE_BLK_CO=0 AITER_BYPASS_TUNE_CONFIG=0 ;;
            co)       EXTRA=(); run "$v" "$dim" AITER_MOE_BLK_CO=1 AITER_BYPASS_TUNE_CONFIG=1 ;;
            co_fused) EXTRA=(--swiglu-limit "$LIMIT" --smooth-scale)
                      run "$v" "$dim" AITER_MOE_BLK_CO=1 AITER_BYPASS_TUNE_CONFIG=1 ;;
            *)        echo "unknown variant: $v" >&2; exit 1 ;;
        esac
    done
done

echo
# `column` is not in every container image; awk pads to the widest cell per field.
# The two-space indent is a margin, not decoration: some terminals clip the first
# couple of columns of emitted lines, and this keeps the clip on padding instead
# of on the variant name. $OUT is the authoritative copy either way.
awk -F, '{for(i=1;i<=NF;i++){c[NR,i]=$i;if(length($i)>w[i])w[i]=length($i)};n=NF}
         END{for(r=1;r<=NR;r++){printf "  ";
              for(i=1;i<=n;i++)printf "%-*s%s",w[i],c[r,i],(i<n?" ":"\n")}}' "$OUT"
echo
echo "csv: $OUT   log: $OUT.log"
