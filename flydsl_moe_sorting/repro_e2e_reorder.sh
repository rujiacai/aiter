#!/usr/bin/env bash
# Root-cause control experiment: run the 32k e2e through moe_sorting with FlyDSL
# sorting, comparing scrambled (atomicAdd) vs token-ascending-within-expert
# (AITER_SORT_REORDER=1). Only the intra-expert token ORDER changes; everything
# else (padding / num_valid / segments / compute) is identical.
#
# Expect: REORDER=0 stage2(moe_gemm2_0) ~6500us, REORDER=1 ~4180us  -> order is
# the cause of the e2e regression (the +2340us is in stage2, not the sort).
#
#   bash flydsl_moe_sorting/repro_e2e_reorder.sh
set -e
cd "$(dirname "$0")/.."

ARGS="--token 32768 --model-dim 4096 --inter-dim 192 --expert 193 --topk 9 \
  --activation silu --dtype bf16 --use-g1u1 1 --doweight-stage1 0 \
  --quant fp8 --quant-type per_tensor"

for RO in 0 1; do
  echo "===================== AITER_SORT_REORDER=$RO ====================="
  AITER_USE_FLYDSL_MOE_SORTING=1 AITER_SORT_REORDER=$RO AITER_LOG_MORE=1 \
    python test_qmoe_multi.py $ARGS 2>&1 \
    | grep -E "moe_gemm2_0|kernel_moe_gemm|scatter_kernel|count_kernel|PERF\] e2e|^ PASS" \
    | grep -vE "import|override"
done

echo
echo "Baseline (CK sort, env off):"
AITER_LOG_MORE=1 python test_qmoe_multi.py $ARGS 2>&1 \
  | grep -E "moe_gemm2_0|kernel_moe_gemm|PERF\] e2e|^ PASS" | grep -vE "import|override"
