#!/usr/bin/env bash
# Micro-architectural evidence for the atomicAdd (scrambled-order) penalty.
# Collects single-pass RAW counters for stage2 (moe_gemm2_0) under scrambled
# (AITER_SORT_REORDER=0) vs ordered (=1):
#   - L2 (TCC): hit/miss/requests  -> identical (NOT a cache hit-rate problem)
#   - L1 (TCP): TCP_PENDING_STALL_CYCLES -> ~2.2x higher when scrambled
#               (== the metric that degrades: memory-stall / latency, MLP)
#
#   bash flydsl_moe_sorting/repro_counters.sh
#
# NOTE: only single-pass raw counters are used. Derived metrics
# (FETCH_SIZE / MemUnitStalled ...) need multiple replay passes, and each replay
# re-JIT-compiles the FlyDSL kernels (tens of seconds) -> impractically slow.
set -e
cd "$(dirname "$0")/.."

ARGS="--token 32768 --model-dim 4096 --inter-dim 192 --expert 193 --topk 9 \
  --activation silu --dtype bf16 --use-g1u1 1 --doweight-stage1 0 \
  --quant fp8 --quant-type per_tensor --warmup 1 --iters 2"

# group 1: L2 hit/miss ; group 2: L1 stall + access counts
for GRP in "TCC_HIT TCC_MISS TCC_REQ" "TCP_PENDING_STALL_CYCLES TCP_TCC_READ_REQ TCP_TOTAL_CACHE_ACCESSES"; do
  for RO in 0 1; do
    rm -rf "_cnt_${RO}"
    AITER_USE_FLYDSL_MOE_SORTING=1 AITER_SORT_REORDER=$RO \
      rocprofv3 --pmc $GRP --output-format csv -d "_cnt_${RO}" -- \
      python test_qmoe_multi.py $ARGS >/dev/null 2>&1
  done
  python flydsl_moe_sorting/parse_counters.py _cnt_0 _cnt_1
  rm -rf _cnt_0 _cnt_1
done
