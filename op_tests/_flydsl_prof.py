# SPDX-License-Identifier: MIT
"""Minimal direct fused_moe driver for clean rocprofv3 kernel-trace profiling
of the dsv4 a4w4 (fp4/fp4, per_1x32) flydsl path. No torch.profiler, no graph.

Usage:
  AITER_CONFIG_FMOE=<tuned.csv> HIP_VISIBLE_DEVICES=N python op_tests/_flydsl_prof.py --token 8192 --iters 20 [--check]
"""
import argparse
import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk, fused_moe, torch_moe_stage1, torch_moe_stage2
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

p = argparse.ArgumentParser()
p.add_argument("--token", type=int, default=8192)
p.add_argument("--model-dim", type=int, default=7168)
p.add_argument("--inter-dim", type=int, default=384)
p.add_argument("-E", "--expert", type=int, default=384)
p.add_argument("-k", "--topk", type=int, default=6)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--check", action="store_true", help="run torch ref + cos check (mem heavy)")
args = p.parse_args()

dtype = dtypes.bf16
qType = QuantType.per_1x32
AQDType = dtypes.fp4x2
WQDType = dtypes.fp4x2
token, model_dim, inter_dim, E, topk = (
    args.token, args.model_dim, args.inter_dim, args.expert, args.topk,
)
torch_quant = aiter.get_torch_quant(qType)

torch.manual_seed(0)
inp = torch.randn((token, model_dim), dtype=dtype) / 10
w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
score = torch.randn((token, E), dtype=dtype)
topk_weights, topk_ids = fused_topk(inp, score, topk, True)

def _chunked_quant(w, chunk=8):
    """Per-expert-chunk MXFP4 quant to cap the f32 peak memory (the whole-tensor
    f32_to_mxfp4 materializes an E*d1*d2 f32 buffer ~8GB for E=384)."""
    qts, scs = [], []
    for i in range(0, w.shape[0], chunk):
        q, s = torch_quant(w[i : i + chunk], quant_dtype=WQDType)
        qts.append(q)
        scs.append(s)
    return torch.cat(qts, 0), torch.cat(scs, 0)


w1_qt, w1_scale = _chunked_quant(w1)
w2_qt, w2_scale = _chunked_quant(w2)
w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)

a1_qt, a1_scale = torch_quant(inp, quant_dtype=AQDType)

# Production a4w4 prep (matches op_tests/test_moe_2stage.py preshuffle path):
w1_qt_f = shuffle_weight(w1_qt, layout=(16, 16))
w2_qt_f = shuffle_weight(w2_qt, layout=(16, 16))
w1_scale_f = fp4_utils.e8m0_shuffle(w1_scale)
w2_scale_f = fp4_utils.e8m0_shuffle(w2_scale)
w1_qt_f.is_shuffled = True
w2_qt_f.is_shuffled = True


def call():
    return fused_moe(
        inp, w1_qt_f, w2_qt_f, topk_weights, topk_ids,
        w1_scale=w1_scale_f, w2_scale=w2_scale_f,
        quant_type=qType, activation=ActivationType.Silu, doweight_stage1=False,
    )


for _ in range(args.warmup):
    out = call()
torch.cuda.synchronize()

import time
t0 = time.perf_counter()
for _ in range(args.iters):
    out = call()
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"[prof] token={token} e2e avg = {(t1 - t0) / args.iters * 1e6:.2f} us/iter over {args.iters} iters")

if args.check:
    out1_ref = torch_moe_stage1(
        a1_qt, w1_qt, w2_qt, topk_weights, topk_ids,
        dtype=dtype, activation=ActivationType.Silu, quant_type=qType,
        a1_scale=a1_scale, w1_scale=w1_scale, doweight=False,
    )
    a2_qt, a2_scale = torch_quant(out1_ref, quant_dtype=AQDType)
    a2_qt = a2_qt.view(token, topk, -1)
    out2_ref = torch_moe_stage2(
        a2_qt, w1_qt, w2_qt, topk_weights, topk_ids,
        dtype=dtype, quant_type=qType, w2_scale=w2_scale, a2_scale=a2_scale,
        doweight=True,
    )
    x = out.double().flatten()
    y = out2_ref.double().flatten()
    cos = (torch.dot(x, y) / (x.norm() * y.norm() + 1e-12)).item()
    print(f"[prof] cos_sim = {cos:.6f}")
