#!/usr/bin/env python3
"""End-to-end fused_moe latency + MFU for the tp2 / ep2 shardings of one global MoE.

Global MoE:  model_dim=3584, inter_dim=1280, experts=384, topk=8, fp8 per-token, g1u1, silu.
Per-GPU shardings (equal per-GPU work):
  tp2 : inter_dim // 2 -> 640,  experts kept 384,  all local, topk=8.
  ep2 : experts // 2   -> 192 local (of 384 global), inter_dim kept 1280.
        Uses expert_mask (global 0/1, first half local). Each token is routed to
        exactly topk/2 = 4 LOCAL experts + 4 remote (masked) experts, so exactly
        half of each token's experts are masked out and the local load is balanced
        (local experts sampled uniformly).

MFU = achieved_tflops / peak_fp8.  FLOPs = tokens * eff_topk * 6 * model_dim * inter_dim
  (6 = stage1 g1u1 2*(2*inter*model) + stage2 2*(inter*model), per token-expert).
"""
import argparse
import re
import torch
from torch.profiler import profile, ProfilerActivity

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest

# MoE GEMM (stage1+stage2) kernels to attribute compute time to. Excludes
# moe_sorting, dynamic-quant, silu_and_mul, copy/index/elementwise kernels.
_GEMM_RE = re.compile(r"fmoe|moe_gemm|flydsl_moe|ck2stages_gemm|Cijk", re.IGNORECASE)


def gemm_kernel_us(run_fn, iters):
    """Sum GPU time of the MoE GEMM kernel(s) per iter via torch profiler."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run_fn()
        torch.cuda.synchronize()
    tot = 0.0
    for e in prof.key_averages():
        if _GEMM_RE.search(e.key) and getattr(e, "self_device_time_total", 0) > 0:
            tot += e.self_device_time_total  # microseconds
    return tot / iters

MODEL_DIM = 3584
TOPK = 8
PEAK_FP8_TFLOPS = 3567  # MI355X FP8 peak used for MFU (single GPU)

CONFIGS = {
    #        inter, local_E, global_E, eff_topk, mask
    "tp2": dict(inter=640,  local_E=384, global_E=384, eff_topk=8, mask=False),
    "ep2": dict(inter=1280, local_E=192, global_E=384, eff_topk=4, mask=True),
}


def build(cfg, token, device="cuda"):
    c = CONFIGS[cfg]
    inter, local_E, global_E = c["inter"], c["local_E"], c["global_E"]
    g = torch.Generator(device=device).manual_seed(0)

    x = torch.randn(token, MODEL_DIM, dtype=dtypes.bf16, device=device, generator=g) / 10
    w1 = torch.randn(local_E, inter * 2, MODEL_DIM, dtype=dtypes.bf16, device=device, generator=g) / 10
    w2 = torch.randn(local_E, MODEL_DIM, inter, dtype=dtypes.bf16, device=device, generator=g) / 10

    tq = aiter.get_torch_quant(QuantType.per_Token)
    w1_qt, w1_s = tq(w1, quant_dtype=dtypes.fp8)
    w2_qt, w2_s = tq(w2, quant_dtype=dtypes.fp8)
    w1_sh = shuffle_weight(w1_qt.view(w1.shape), layout=(16, 16))
    w2_sh = shuffle_weight(w2_qt.view(w2.shape), layout=(16, 16))

    topk_w = torch.rand(token, TOPK, dtype=torch.float32, device=device, generator=g)
    topk_w = topk_w / topk_w.sum(-1, keepdim=True)

    if c["mask"]:
        half = TOPK // 2
        loc = torch.stack(
            [torch.randperm(local_E, device=device, generator=g)[:half] for _ in range(token)]
        )
        rem = torch.stack(
            [local_E + torch.randperm(global_E - local_E, device=device, generator=g)[:half]
             for _ in range(token)]
        )
        topk_ids = torch.cat([loc, rem], dim=1).to(torch.int32)
        expert_mask = torch.zeros(global_E, dtype=dtypes.i32, device=device)
        expert_mask[:local_E] = 1
    else:
        topk_ids = torch.stack(
            [torch.randperm(global_E, device=device, generator=g)[:TOPK] for _ in range(token)]
        ).to(torch.int32)
        expert_mask = None

    return dict(x=x, w1=w1_sh, w2=w2_sh, w1_s=w1_s, w2_s=w2_s,
                topk_w=topk_w, topk_ids=topk_ids, mask=expert_mask,
                inter=inter, eff_topk=c["eff_topk"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default=None, help="tp2 / ep2 (default: both)")
    ap.add_argument("--tokens", type=int, nargs="*",
                    default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    cfgs = [args.config] if args.config else list(CONFIGS)
    print(f"model_dim={MODEL_DIM} topk={TOPK} peak_fp8={PEAK_FP8_TFLOPS} TFLOPS | fp8/per_Token g1u1 silu bf16")
    for cfg in cfgs:
        c = CONFIGS[cfg]
        print(f"\n== {cfg}: inter={c['inter']} local_E={c['local_E']} global_E={c['global_E']} "
              f"eff_topk={c['eff_topk']} mask={c['mask']} ==")
        print(f"{'token':>7} {'e2e_us':>10} {'gemm_us':>9} {'gemm_TFLOPS':>12} {'gemm_MFU%':>10}")
        for tok in args.tokens:
            d = build(cfg, tok)

            def _run():
                return fused_moe(
                    d["x"], d["w1"], d["w2"], d["topk_w"], d["topk_ids"], d["mask"],
                    activation=ActivationType.Silu, quant_type=QuantType.per_Token,
                    w1_scale=d["w1_s"], w2_scale=d["w2_s"],
                )

            # end-to-end wall time (full fused_moe: sorting + quant + gemms + act + reduce)
            _, e2e_us = run_perftest(
                fused_moe, d["x"], d["w1"], d["w2"], d["topk_w"], d["topk_ids"], d["mask"],
                activation=ActivationType.Silu, quant_type=QuantType.per_Token,
                w1_scale=d["w1_s"], w2_scale=d["w2_s"],
                num_iters=args.iters, num_warmup=args.warmup,
            )
            # MFU from the GEMM kernel(s) time only (stage1+stage2 compute)
            g_us = gemm_kernel_us(_run, args.iters)

            flops = tok * d["eff_topk"] * 6 * MODEL_DIM * d["inter"]
            tflops = flops / (g_us / 1e6) / 1e12
            mfu = tflops / PEAK_FP8_TFLOPS * 100
            print(f"{tok:>7} {e2e_us:>10.2f} {g_us:>9.2f} {tflops:>12.1f} {mfu:>9.1f}%")


if __name__ == "__main__":
    main()
