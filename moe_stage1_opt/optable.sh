#!/bin/bash
# Per-op ROCTracer table for one full-knob config, which is how chapter 9's
# budget and before/after tables were made.  run.sh only records
# stage1/stage2/e2e; this dumps every kernel in the pipeline, which is what you
# need when the thing you changed is not one of the two GEMMs.
#
#   ./optable.sh f5      # every stage1+stage2 knob, quant knobs OFF
#   ./optable.sh f7      # ... plus AITER_QUANT_PT_BIGTILE + _FUSE_GUARD
#   GPU=5 ./optable.sh f7
#
# Output also lands in /tmp/optable_$1.txt.  The per-kernel column to read is
# `device_time_avg` (us per fused_moe call); the rows sum to e2e.
set -u
[ $# -eq 1 ] || { echo "usage: $0 f5|f7" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
K1=flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16_bnt0
K2=flydsl_moe2_afp8_wfp8_bf16_t64x128x64_reduce_persist_bnt0
cfg=$(mktemp /tmp/optcfg.XXXXXX.csv)
trap 'rm -f "$cfg"' EXIT
bash "$HERE/mkcfg.sh" "$K1" "$K2" 64 64 > "$cfg"

# env -i so an exported knob in the parent shell cannot silently join the run.
env -i PATH="$PATH" HOME="$HOME" \
  AITER_CONFIG_FMOE="$cfg" HIP_VISIBLE_DEVICES="${GPU:-4}" AITER_LOG_MORE=1 \
  FLYDSL_MOE_STAGE1_NLANE_FIT=1 FLYDSL_MOE_STAGE1_EVEC=8 \
  FLYDSL_MOE_STAGE1_LDSTIGHT=1 FLYDSL_MOE_STAGE1_SCALAR_ASCALE=1 \
  FLYDSL_MOE_STAGE1_BFIRST=1 FLYDSL_MOE_STAGE1_LDSPAD=8 \
  AITER_FLYDSL_STAGE2_SORTED_PARTIAL=1 FLYDSL_MOE_STAGE2_FASTVALID=1 \
  FLYDSL_MOE_STAGE2_SCALAR_ASCALE=1 FLYDSL_MOE_STAGE2_VEC_SCALE=1 \
  FLYDSL_MOE_STAGE2_BUFSTORE=1 FLYDSL_MOE_STAGE2_HOIST_PF=1 \
  FLYDSL_MOE_STAGE2_HOIST_X=1 FLYDSL_MOE_STAGE2_NO_MASK=1 \
  FLYDSL_MOE_STAGE2_BFIRST=1 FLYDSL_MOE_STAGE2_NLANE_FIT=1 \
  FLYDSL_MOE_STAGE2_EVEC=8 FLYDSL_MOE_STAGE2_LDSPAD=4 \
  FLYDSL_MOE_STAGE2_FASTIDX=1 FLYDSL_MOE_STAGE2_SCALAR_WSCALE=1 \
  FLYDSL_MOE_STAGE2_LDSCHUNK=1 \
  $( [ "$1" = f7 ] && echo "AITER_QUANT_PT_BIGTILE=1 AITER_QUANT_PT_FUSE_GUARD=1" ) \
  python "$HERE/../test_qmoe_multi.py" \
    --token 32768 --model-dim 4096 --inter-dim 192 --expert 193 --topk 9 \
    --activation silu --dtype bf16 --use-g1u1 1 --doweight-stage1 0 \
    --quant fp8 --quant-type per_tensor 2>&1 | tee "/tmp/optable_$1.txt" | tail -40
