#!/usr/bin/env bash
# Emit a one-row fused_moe config csv for the 32k Hunyuan3 case.
#
#   mkcfg.sh <kernelName1> <kernelName2> [block_m] [block_m2] > cfg.csv
#
# Everything except the two kernel names and the two block sizes is fixed by the
# case, so the ladder can vary stage1 alone and be sure nothing else moved.
set -euo pipefail
k1="$1"; k2="$2"; bm="${3:-64}"; bm2="${4:-64}"
echo "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,q_dtype_a2,q_dtype_w2,q_type2,block_m,block_m2,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw,_tag"
echo "80,32768,4096,192,193,9,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Tensor,1,0,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Tensor,${bm},${bm2},0,,${k1},,,${k2},,,0,,,"
