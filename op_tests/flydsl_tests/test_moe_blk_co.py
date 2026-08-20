#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Parity check: prebuilt .co launcher vs the FlyDSL path it was exported from.

The two must agree bit for bit -- same binary, same arguments, only the launch
route differs. Export the code objects for the shape first:

    python hsa/flydsl_export.py --shape 1024,256,8,2 --token-bucket 128 --smooth 0 --waves 0
    python op_tests/flydsl_tests/test_moe_blk_co.py
"""

import argparse

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.moe_blk import co_name
from aiter.ops.moe_op import moe_blk_stage1, moe_blk_stage2
from aiter.ops.shuffle import shuffle_weight

BLK = 128


def block_quant(w):
    e, n, k = w.shape
    blocks = (
        w.view(e, n // BLK, BLK, k // BLK, BLK)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, -1, BLK * BLK)
    )
    q, s = aiter.pertoken_quant(blocks, quant_dtype=dtypes.fp8)
    q = (
        q.view(e, n // BLK, k // BLK, BLK, BLK)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, n, k)
        .contiguous()
    )
    return q, s.view(e, n // BLK, k // BLK).float().contiguous()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-t", "--token", type=int, default=128)
    p.add_argument("-dim", type=int, default=1024)
    p.add_argument("-idim", type=int, default=256)
    p.add_argument("-e", "--expert", type=int, default=8)
    p.add_argument("-k", "--topk", type=int, default=2)
    p.add_argument("--tile-m", type=int, default=16)
    p.add_argument("--waves", type=int, default=0)
    p.add_argument("--swiglu-limit", type=float, default=10.0)
    args = p.parse_args()

    dev, T, H, I = "cuda", args.token, args.dim, args.idim
    E, K, TM, W = args.expert, args.topk, args.tile_m, args.waves
    torch.manual_seed(0)

    x = torch.randn((T, H), dtype=dtypes.bf16, device=dev)
    w1 = torch.randn((E, 2 * I, H), dtype=dtypes.bf16, device=dev) / 10
    w2 = torch.randn((E, H, I), dtype=dtypes.bf16, device=dev) / 10
    w1_q, w1_s = block_quant(w1)
    w2_q, w2_s = block_quant(w2)
    a_q, a_s = aiter.pertoken_quant(x.view(T, -1, BLK), quant_dtype=dtypes.fp8)
    a_q, a_s = a_q.view(T, H), a_s.view(T, -1).float().contiguous()
    score = torch.randn((T, E), dtype=dtypes.bf16, device=dev)
    tw, ti = fused_topk(x, score, K, True)
    sid, sw, seid, nv, buf = moe_sorting(ti, tw, E, H, dtypes.bf16, TM)
    w1_sh = shuffle_weight(w1_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)
    w2_sh = shuffle_weight(w2_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)

    fails = 0

    # ---- stage1 ----
    ref1 = torch.zeros((T, K, I), dtype=dtypes.bf16, device=dev)
    flydsl_moe_stage1(
        a_q, w1_sh, sid, seid, nv, out=ref1, topk=K,
        tile_m=TM, tile_n=128, tile_k=128,
        a_dtype="fp8", b_dtype="fp8blk", out_dtype="bf16", act="silu",
        w1_scale=w1_s, a1_scale=a_s, waves_per_eu=W,
        swiglu_limit=args.swiglu_limit,
    )  # fmt: skip
    got1 = torch.zeros_like(ref1)
    moe_blk_stage1(
        got1, a_q, w1_sh, a_s, w1_s, sid, seid, None, nv, None,
        T, I, H, 128, float(args.swiglu_limit),
        co_name(1, H, I, E, K, TM, 128, 128, W),
    )  # fmt: skip
    ok = torch.equal(ref1, got1)
    fails += not ok
    print(f"  stage1  bitwise identical: {ok}"
          f"{'' if ok else f'   max|diff|={(ref1.float() - got1.float()).abs().max():.3e}'}")

    # ---- stage2 ----
    a2_q, a2_s = aiter.pertoken_quant(ref1.view(T, -1, BLK), quant_dtype=dtypes.fp8)
    a2_q, a2_s = a2_q.view(T, K, I), a2_s.view(T * K, -1).float().contiguous()
    ref2 = buf.clone().zero_()
    flydsl_moe_stage2(
        a2_q, w2_sh, sid, seid, nv, out=ref2, topk=K,
        tile_m=TM, tile_n=256, tile_k=256,
        a_dtype="fp8", b_dtype="fp8blk", out_dtype="bf16", mode="atomic",
        w2_scale=w2_s, a2_scale=a2_s, sorted_weights=sw, waves_per_eu=W,
    )  # fmt: skip
    got2 = torch.zeros_like(ref2)
    moe_blk_stage2(
        got2, a2_q, w2_sh, a2_s, w2_s, sid, seid, sw, nv,
        T, H, I, 256, co_name(2, H, I, E, K, TM, 256, 256, W),
    )  # fmt: skip
    # stage2 reduces with atomics, so the summation order (not the math) can
    # differ between two launches; compare with a tolerance, not bitwise.
    diff = (ref2.float() - got2.float()).abs().max().item()
    ok = diff < 1e-2
    fails += not ok
    print(f"  stage2  max|diff| = {diff:.3e}  ({'OK' if ok else 'FAIL'}, atomic order)")

    print("\nPASS" if not fails else f"\n{fails} check(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
