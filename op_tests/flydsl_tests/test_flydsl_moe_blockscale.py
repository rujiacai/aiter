# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone harness for the FlyDSL blockwise-fp8 (DeepSeek 128x128) MoE kernels.

Drives ``flydsl_moe_stage1`` / ``flydsl_moe_stage2`` per stage, bypassing the
``fused_moe`` dispatch so a kernel or tile config can be iterated on in isolation.
Golden values come from ``torch_moe_stage1`` / ``torch_moe_stage2`` with
``QuantType.per_1x128``.

  python op_tests/flydsl_tests/test_flydsl_moe_blockscale.py \
      -t 128 -dim 7168 -idim 512 -e 385 -k 7 --swiglu-limit 10.0

``--scales`` / ``--wscale-mode`` replace part of the quantization with unit or
single-axis scales, which isolates a scale-indexing bug from a dataflow bug.
``--in-dtype fp8`` runs the pre-existing per-token/per-row fp8 kernel through the
same harness as a control group.

Note: the whole weight tensor must stay under the 4 GiB that a buffer resource
can address, so E=385 with inter_dim=1536 does not fit on one GPU (that limit is
pre-existing and hits every in_dtype).
"""

import argparse

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose, run_perftest

BLK = 128


def block_quant_weight(w, blk_n=BLK, blk_k=BLK):
    """[E, N, K] bf16 -> fp8 codes + [E, N/blk_n, K/blk_k] f32 scales."""
    e, n, k = w.shape
    assert n % blk_n == 0 and k % blk_k == 0
    blocks = (
        w.view(e, n // blk_n, blk_n, k // blk_k, blk_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, -1, blk_n * blk_k)
    )
    q, s = aiter.pertoken_quant(blocks, quant_dtype=dtypes.fp8)
    q = (
        q.view(e, n // blk_n, k // blk_k, blk_n, blk_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, n, k)
    )
    return q.contiguous(), s.view(e, n // blk_n, k // blk_k).float().contiguous()


def quant_weight_with_scale(w, scale, blk_n=BLK, blk_k=BLK):
    """Quantize [E, N, K] with a caller-chosen [E, N/blk_n, K/blk_k] scale.

    Lets a probe vary the scale along only one axis, which pins down which half
    of the kernel's block index is wrong.
    """
    e, n, k = w.shape
    q = (
        w.view(e, n // blk_n, blk_n, k // blk_k, blk_k)
        / scale.view(e, n // blk_n, 1, k // blk_k, 1)
    ).to(dtypes.fp8)
    return q.view(e, n, k).contiguous(), scale.contiguous()


def make_w_scale(e, nb, kb, mode, dev):
    if mode == "unit":
        return torch.ones((e, nb, kb), dtype=torch.float32, device=dev)
    base = torch.rand((e, nb, kb), dtype=torch.float32, device=dev) + 0.5
    if mode == "n":  # constant along K -> only the N-block index matters
        return base[:, :, :1].expand(e, nb, kb).contiguous()
    if mode == "k":  # constant along N -> only the K-block index matters
        return base[:, :1, :].expand(e, nb, kb).contiguous()
    if mode == "e":  # constant within an expert -> only the expert stride matters
        return base[:, :1, :1].expand(e, nb, kb).contiguous()
    return base


def block_quant_act(a, blk_k=BLK):
    """[..., K] bf16 -> fp8 codes + [..., K/blk_k] f32 scales."""
    k = a.shape[-1]
    assert k % blk_k == 0
    q, s = aiter.pertoken_quant(a.view(-1, k // blk_k, blk_k), quant_dtype=dtypes.fp8)
    return q.view(a.shape).contiguous(), s.view(*a.shape[:-1], k // blk_k).float()


def _report(name, ref, got, us=None):
    diff = ((ref.float() - got.float()) ** 2).sum()
    denom = (ref.float() ** 2).sum() + (got.float() ** 2).sum()
    cos = 1 - (2 * (ref.float() * got.float()).sum() / denom).item()
    tail = "" if us is None else f"  ({us:.2f} us)"
    print(f"[{name}] logits_diff={cos:.3e} l2={diff.item():.3e}{tail}")
    return cos


def run_stage1(args):
    token, model_dim, inter_dim = args.token, args.dim, args.idim
    E, topk = args.expert, args.topk
    block_m = args.tile_m
    dev = "cuda"
    torch.manual_seed(args.seed)

    x = torch.randn((token, model_dim), dtype=dtypes.bf16, device=dev)
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16, device=dev) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device=dev) / 10
    score = torch.randn((token, E), dtype=dtypes.bf16, device=dev)
    topk_weights, topk_ids = fused_topk(x, score, topk, True)

    if args.in_dtype == "fp8":
        # Control group: the pre-existing per-token/per-row fp8 path, driven through
        # exactly the same harness. Tells apart harness bugs from blockwise bugs.
        a_q, a_s = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
        w1_q, w1_s = aiter.pertoken_quant(w1, quant_dtype=dtypes.fp8)
        ref = torch_moe_stage1(
            a_q,
            w1_q,
            w2,
            topk_weights,
            topk_ids,
            dtype=dtypes.bf16,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_Token,
            a1_scale=a_s,
            w1_scale=w1_s,
        )
        a_s = a_s.float().contiguous()
        w1_s = w1_s.float().contiguous()

    else:
        # Unit scales let a wrong scale index still produce the right answer, so
        # `--scales a`/`w`/`none` isolate which side of the indexing is broken.
        if args.scales in ("full", "a"):
            a_q, a_s = block_quant_act(x)
        else:
            a_q = x.to(dtypes.fp8)
            a_s = torch.ones(
                (token, model_dim // BLK), dtype=torch.float32, device=dev
            )
        if args.wscale_mode != "block":
            w1_s = make_w_scale(
                E, inter_dim * 2 // BLK, model_dim // BLK, args.wscale_mode, dev
            )
            w1_q, w1_s = quant_weight_with_scale(w1, w1_s)
        elif args.scales in ("full", "w"):
            w1_q, w1_s = block_quant_weight(w1)
        else:
            w1_q = w1.to(dtypes.fp8)
            w1_s = torch.ones(
                (E, inter_dim * 2 // BLK, model_dim // BLK),
                dtype=torch.float32,
                device=dev,
            )

        ref = torch_moe_stage1(
            a_q,
            w1_q,
            w2,
            topk_weights,
            topk_ids,
            dtype=dtypes.bf16,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_128x128,
            a1_scale=a_s,
            w1_scale=w1_s,
            swiglu_limit=args.swiglu_limit,
        )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtypes.bf16, block_m
    )

    out = torch.zeros((token, topk, inter_dim), dtype=dtypes.bf16, device=dev)
    w1_shuf = shuffle_weight(w1_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)

    def _launch():
        flydsl_moe_stage1(
            a_q,
            w1_shuf,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out=out,
            topk=topk,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            a_dtype="fp8",
            b_dtype="fp8blk" if args.in_dtype == "fp8_blk" else "fp8row",
            out_dtype="bf16",
            act="silu",
            w1_scale=w1_s,
            a1_scale=a_s,
            waves_per_eu=args.waves_per_eu,
            swiglu_limit=args.swiglu_limit,
        )

    _launch()
    torch.cuda.synchronize()
    for r in range(args.repeat - 1):
        prev = out.clone()
        out.zero_()
        _launch()
        torch.cuda.synchronize()
        same = torch.equal(prev, out)
        print(f"  [stage1 run {r + 2}] bitwise identical to run {r + 1}: {same}")

    us = None
    if args.bench:
        _, us = run_perftest(_launch, num_iters=20, num_warmup=5)
    cos = _report("stage1", ref, out, us)
    checkAllclose(ref.float(), out.float(), rtol=0.05, atol=0.05, msg="stage1")
    return cos, (x, w1, w2, topk_weights, topk_ids, ref)


def run_stage2(args, ctx):
    x, w1, w2, topk_weights, topk_ids, inter_ref = ctx
    token, model_dim, inter_dim = args.token, args.dim, args.idim
    E, topk = args.expert, args.topk
    block_m = args.tile_m
    dev = "cuda"

    a2_q, a2_s = block_quant_act(inter_ref)
    w2_q, w2_s = block_quant_weight(w2)

    ref = torch_moe_stage2(
        a2_q,
        w1,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=aiter.QuantType.per_128x128,
        w2_scale=w2_s,
        a2_scale=a2_s,
        doweight=True,
    )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtypes.bf16, block_m
    )
    moe_buf.zero_()
    w2_shuf = shuffle_weight(w2_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)

    def _launch2():
        flydsl_moe_stage2(
            a2_q,
            w2_shuf,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out=moe_buf,
            topk=topk,
            tile_m=args.tile_m,
            tile_n=args.tile_n2,
            tile_k=args.tile_k,
            a_dtype="fp8",
            b_dtype="fp8blk" if args.in_dtype == "fp8_blk" else "fp8row",
            out_dtype="bf16",
            mode="atomic",
            w2_scale=w2_s,
            a2_scale=a2_s,
            sorted_weights=sorted_weights,
            waves_per_eu=args.waves_per_eu,
        )

    _launch2()
    torch.cuda.synchronize()

    # Snapshot before benchmarking: stage2 reduces with atomics, so repeated
    # launches into the same buffer accumulate instead of overwriting.
    got = moe_buf.clone()
    us = None
    if args.bench:
        _, us = run_perftest(_launch2, num_iters=20, num_warmup=5)
    cos = _report("stage2", ref, got, us)
    checkAllclose(ref.float(), moe_buf.float(), rtol=0.05, atol=0.05, msg="stage2")
    return cos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-t", "--token", type=int, default=128)
    p.add_argument("-dim", type=int, default=7168)
    p.add_argument("-idim", type=int, default=512)
    p.add_argument("-e", "--expert", type=int, default=385)
    p.add_argument("-k", "--topk", type=int, default=7)
    p.add_argument("--tile-m", type=int, default=32)
    p.add_argument("--tile-n", type=int, default=128)
    p.add_argument("--tile-n2", type=int, default=128)
    p.add_argument("--tile-k", type=int, default=128)
    p.add_argument("--waves-per-eu", type=int, default=0)
    p.add_argument("--swiglu-limit", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--in-dtype", choices=["fp8_blk", "fp8"], default="fp8_blk")
    p.add_argument("--scales", choices=["full", "a", "w", "none"], default="full")
    p.add_argument(
        "--wscale-mode", choices=["block", "n", "k", "e", "unit"], default="block"
    )
    p.add_argument("--stage", choices=["1", "2", "both"], default="both")
    args = p.parse_args()

    ctx = None
    if args.stage in ("1", "both"):
        _, ctx = run_stage1(args)
    if args.stage in ("2", "both"):
        if ctx is None:
            _, ctx = run_stage1(args)
        run_stage2(args, ctx)


if __name__ == "__main__":
    main()
