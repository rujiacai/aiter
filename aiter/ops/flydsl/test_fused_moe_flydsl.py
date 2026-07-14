# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""e2e latency comparison: FlyDSL a16w4 / a8w4 (THROUGH aiter.fused_moe) vs triton a16w4.

All three share ONE input+weight set per token and are timed with the SAME yardstick
(`run_perftest` -> get_trace_perf device time), so the e2e numbers are directly comparable.

  - a16w4 : bf16 activation x mxfp4 (e2m1) weight, via fused_moe (AITER_FLYDSL_A16W4=1).
  - a8w4  : fp8 activation x mxfp4->fp8-fold weight, via fused_moe (AITER_FLYDSL_A8W4=1).
  - triton: moe_gemm_a16w4 stage1+stage2 (standalone kernel, its own routing).

The AITER_FLYDSL_A16W4 / A8W4 env flags route fused_moe to the FlyDSL CDNA3 kernels;
they are read at dispatch (call) time, so this test sets them inside run_flydsl right
before the fused_moe call (keeps `import` side-effect-free).

Correctness: a16w4/a8w4 vs bf16 full-precision golden (FlyDSL routing); triton vs its
own moe_gemm_torch golden (triton routing). Perf uses run_perftest device time.

Reproduce (single command, 3-way compare at token=4096 DSV shape):
  PYTHONPATH=/data/aiter /opt/venv/bin/python test_fused_moe_flydsl.py \
      --method all -t 4096 --model-dim 4096 --inter-dim 512 -E 256 --topk 6 --perf
Add AITER_LOG_MORE=1 in front to also dump the per-kernel device-time table.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_flydsl_moe_a16w4 as T  # noqa: E402
from aiter import QuantType, ActivationType  # noqa: E402
from aiter.fused_moe import fused_moe, torch_moe_stage1, torch_moe_stage2  # noqa: E402
from aiter.test_common import run_perftest  # noqa: E402
from aiter.ops.flydsl.moe_kernels import (  # noqa: E402
    prep_a16w4_weight, prep_a16w4_scale, prep_a8w4_weight_scale,
)

torch.set_default_device("cuda")


def _enable_flydsl_dispatch():
    """Route fused_moe(per_1x32) to the FlyDSL a16w4/a8w4 CDNA3 kernels.

    The flags are read at fused_moe dispatch (call) time, so setting them here
    (before the call) is enough and keeps module import side-effect-free.
    """
    os.environ["AITER_FLYDSL_A16W4"] = "1"
    os.environ["AITER_FLYDSL_A8W4"] = "1"


def _flops(token, model_dim, inter_dim, topk):
    # stage1 (gate+up) + stage2 (down), 2 flops/MAC, topk-replicated rows.
    return token * model_dim * inter_dim * 3 * topk * 2


def _golden(d, model_dim, inter_dim, E):
    r1 = torch_moe_stage1(
        d["inp"], d["w1_dq"], d["w2_dq"], d["topk_weights"], d["topk_ids"],
        dtype=torch.bfloat16, activation=ActivationType.Silu, quant_type=QuantType.No,
    )
    return torch_moe_stage2(
        r1, torch.zeros((E, inter_dim * 2, model_dim), dtype=torch.bfloat16),
        d["w2_dq"], d["topk_weights"], d["topk_ids"],
        dtype=torch.bfloat16, quant_type=QuantType.No, doweight=True,
    )


def run_flydsl(method, d, model_dim, inter_dim, E, topk, perf):
    """a16w4/a8w4 through fused_moe. Returns dict(cos, us, tflops)."""
    _enable_flydsl_dispatch()  # route fused_moe(per_1x32) to FlyDSL kernels
    token = d["inp"].shape[0]
    tag = f"fused_moe-{method}"
    if method == "a16w4":
        w1 = prep_a16w4_weight(d["w1_qt"], inter_dim * 2, model_dim)
        w2 = prep_a16w4_weight(d["w2_qt"], model_dim, inter_dim)
        w1s = prep_a16w4_scale(d["w1_scale"], inter_dim * 2, model_dim)
        w2s = prep_a16w4_scale(d["w2_scale"], model_dim, inter_dim)
    else:  # a8w4
        w1, w1s = prep_a8w4_weight_scale(d["w1_qt"], d["w1_scale"], E, inter_dim * 2, model_dim)
        w2, w2s = prep_a8w4_weight_scale(d["w2_qt"], d["w2_scale"], E, model_dim, inter_dim)
    kw = dict(quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
              w1_scale=w1s, w2_scale=w2s)

    out = fused_moe(d["inp"], w1, w2, d["topk_weights"], d["topk_ids"], **kw)
    torch.cuda.synchronize()
    cos = T._check(_golden(d, model_dim, inter_dim, E), out, tag)

    us = tflops = None
    if perf:
        _, us = run_perftest(fused_moe, d["inp"], w1, w2, d["topk_weights"], d["topk_ids"],
                             num_iters=20, num_warmup=5, **kw)
        tflops = _flops(token, model_dim, inter_dim, topk) / us / 1e6
        print(f"  [{tag}] e2e: {us:9.2f} us   {tflops:7.2f} TFLOPS")
    return {"cos": cos, "us": us, "tflops": tflops}


def run_triton(d, model_dim, inter_dim, E, topk, perf):
    """triton moe_gemm_a16w4 e2e. Returns dict(cos, us, tflops). Uses its own routing."""
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4, moe_gemm_torch
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp
    from aiter.ops.triton.utils.types import str_to_torch_dtype

    token = d["inp"].shape[0]
    inp = d["inp"]
    logits = torch.randn((token, E), dtype=torch.float16, device="cuda")
    rdata, gindx, sindx = routing(logits, topk)
    gammas = rdata.gate_scal.to(torch.float32) if rdata.gate_scal is not None else None
    wdt = str_to_torch_dtype["mxfp4_e2m1"]
    w1_tri, w1s_tri = downcast_to_mxfp(d["w1"].transpose(1, 2).contiguous(), wdt, axis=1)
    w2_tri, w2s_tri = downcast_to_mxfp(d["w2"].transpose(1, 2).contiguous(), wdt, axis=1)
    b1 = torch.zeros((E, inter_dim * 2), dtype=torch.float32, device="cuda")
    b2 = torch.zeros((E, model_dim), dtype=torch.float32, device="cuda")

    def tri_e2e():
        h = moe_gemm_a16w4(inp, w1_tri, None, w1s_tri, None, None, b1,
                           rdata, gindx, None, None, None, torch.bfloat16, True)
        return moe_gemm_a16w4(h, w2_tri, None, w2s_tri, None, None, b2,
                              rdata, None, sindx, gammas, None, torch.bfloat16, False)

    out = tri_e2e()
    torch.cuda.synchronize()
    w1r = upcast_from_mxfp(w1_tri, w1s_tri, torch.bfloat16, axis=1)
    w2r = upcast_from_mxfp(w2_tri, w2s_tri, torch.bfloat16, axis=1)
    hg = moe_gemm_torch(inp, w1r, b1, rdata, gindx, None, None, True)
    trg = moe_gemm_torch(hg, w2r, b2, rdata, None, sindx, gammas, False)
    cos = T._check(trg, out, "triton-a16w4")

    us = tflops = None
    if perf:
        _, us = run_perftest(tri_e2e, num_iters=20, num_warmup=5)
        tflops = _flops(token, model_dim, inter_dim, topk) / us / 1e6
        print(f"  [triton-a16w4] e2e: {us:9.2f} us   {tflops:7.2f} TFLOPS")
    return {"cos": cos, "us": us, "tflops": tflops}


def compare(token, model_dim, inter_dim, E, topk, methods, perf):
    print(f"\n{'='*78}\n  token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk}\n{'='*78}")
    d = T._gen(token, model_dim, inter_dim, E, topk)  # shared input+weights
    res = {}
    for m in methods:
        if m == "triton":
            res[m] = run_triton(d, model_dim, inter_dim, E, topk, perf)
        else:
            res[m] = run_flydsl(m, d, model_dim, inter_dim, E, topk, perf)

    if perf and any(res[m]["us"] for m in methods):
        base = res.get("triton", {}).get("us")  # baseline = triton
        print(f"\n  {'method':<10} {'e2e(us)':>10} {'TFLOPS':>8} {'pass':>6} {'speedup vs triton':>18}")
        print(f"  {'-'*56}")
        for m in methods:
            r = res[m]
            sp = (f"{base/r['us']:.2f}x" if base and r["us"] else "-")
            print(f"  {m:<10} {r['us']:>10.1f} {r['tflops']:>8.1f} {str(r['cos']):>6} {sp:>18}")
        print("  (pass = cos-check vs golden; speedup vs triton = triton_us / this, >1 => faster than triton)")
        print("  (per-method corr printed above; a8w4~0.999 fp8-act, a16w4/triton=1.0)")
    return all(res[m]["cos"] for m in methods)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["a16w4", "a8w4", "triton", "both", "all"], default="all")
    ap.add_argument("-t", "--token", type=int, default=None)
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--inter-dim", type=int, default=512)
    ap.add_argument("-E", "--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--perf", action="store_true", help="run_perftest each method + print compare table")
    args = ap.parse_args()

    if args.method == "both":
        methods = ["a16w4", "a8w4"]
    elif args.method == "all":
        methods = ["triton", "a16w4", "a8w4"]  # triton first => speedup baseline
    else:
        methods = [args.method]
    tokens = [args.token] if args.token else [4096]
    ok = True
    for tok in tokens:
        ok &= compare(tok, args.model_dim, args.inter_dim, args.experts, args.topk, methods, args.perf)
    print("\nRESULT:", "PASS" if ok else "FAIL")
