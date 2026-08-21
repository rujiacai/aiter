#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check the prebuilt blockwise-fp8 MoE code objects against the stock asm/CK path.

Runs every (shape, token) the tuned config covers, both routes end to end through
fused_moe, and reports correctness plus the speed ratio.

The cos column compares against asm, which catches a code object that disagrees
with the kernel it replaced but not one where both are wrong the same way -- feed
either route weights that were never shuffled to (16,16) and both return garbage
that still matches. Cross-check against torch_moe on the pre-quantisation bf16
weights when the layout itself is in question; that scores ~0.998, the fp8
quantisation error.

    python op_tests/test_moe_blk_co.py
    python op_tests/test_moe_blk_co.py --tokens 1 64 --swiglu-limit 10 --smooth-scale
"""

from __future__ import annotations

import argparse
import csv
import os

import torch

import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.ops.moe_blk import TUNED_CSV
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest

BLK = 128


def tuned_points():
    """(model_dim, inter_dim, expert, topk) -> sorted tokens, from the shipped CSV.

    Driving the sweep off the config rather than a hardcoded list keeps this in
    step with whatever shapes were actually published.
    """
    shapes = {}
    with open(TUNED_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            key = tuple(
                int(r[c]) for c in ("model_dim", "inter_dim", "expert", "topk")
            )
            shapes.setdefault(key, set()).add(int(r["token"]))
    return {k: sorted(v) for k, v in shapes.items()}


def block_quant(w):
    """Per-128x128 fp8 weights, the layout the kernels expect."""
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


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def tflops(token, model_dim, inter_dim, topk, us):
    """Same accounting the fmoe tuner uses: gate+up, then down, both g1u1."""
    flop = (
        token * (inter_dim * 2) * model_dim * topk * 2
        + topk * token * model_dim * inter_dim * 2
    )
    return round(flop / (us * 1e6), 2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--tokens", type=int, nargs="+", default=None)
    p.add_argument("--min-cos", type=float, default=0.99)
    p.add_argument("--max-ratio", type=float, default=1.5)
    p.add_argument("--swiglu-limit", type=float, default=None)
    p.add_argument("--smooth-scale", action="store_true")
    p.add_argument(
        "--shape", action="append", metavar="MODEL_DIM,INTER_DIM,EXPERT,TOPK",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        help="repeatable; defaults to every shape in the tuned config",
    )  # fmt: skip
    p.add_argument("--csv", default=None, help="append the measurements here")
    p.add_argument("--tag", default="co", help="value for the csv variant column")
    p.add_argument(
        "--no-asm", dest="asm", action="store_false",
        help="skip the reference run; halves the sweep but drops ratio and cos_sim, "
             "which have nothing to compare against without it",
    )  # fmt: skip
    args = p.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    failures, ratios, rows = [], [], []

    points = tuned_points()
    if args.shape:
        points = {s: points.get(s, args.tokens or []) for s in args.shape}

    for (model_dim, inter_dim, E, topk), tokens in points.items():
        print(f"\n=== {model_dim}x{inter_dim} E{E} k{topk} ===", flush=True)
        w1 = torch.randn((E, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device=dev) / 10
        w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device=dev) / 10
        w1_q, w1_s = block_quant(w1)
        w2_q, w2_s = block_quant(w2)
        w1_sh = shuffle_weight(w1_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)
        w2_sh = shuffle_weight(w2_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)
        smooth = (
            torch.rand((E, inter_dim), dtype=torch.float32, device=dev) + 0.5
            if args.smooth_scale
            else None
        )

        for token in args.tokens or tokens:
            x = torch.randn((token, model_dim), dtype=dtypes.bf16, device=dev)
            score = torch.randn((token, E), dtype=dtypes.bf16, device=dev)
            tw, ti = fused_topk(x, score, topk, True)

            out = {}
            for name, env in ((("asm", "0"),) if args.asm else ()) + (("co", "1"),):
                os.environ["AITER_MOE_BLK_CO"] = env
                get_2stage_cfgs.cache_clear()
                # asm has no clamp or smooth operand; asking for them there is a
                # hard error by design, so only the code-object run gets them.
                extra = (
                    dict(swiglu_limit=args.swiglu_limit, smooth_scale=smooth)
                    if env == "1"
                    else {}
                )
                out[name] = run_perftest(
                    fused_moe, x, w1_sh, w2_sh, tw, ti,
                    quant_type=QuantType.per_128x128, w1_scale=w1_s, w2_scale=w2_s,
                    num_iters=args.iters, num_warmup=args.warmup, **extra,
                )  # fmt: skip

            co_us = out["co"][1]
            # Short names on purpose: these become the printed table's column
            # widths, and the full ones push it past 80 columns, where the wrap
            # overwrites the start of every line.
            row = {
                "variant": args.tag, "token": token, "dim": model_dim,
                "idim": inter_dim, "E": E, "topk": topk,
                "limit": args.swiglu_limit or "",
                "smooth": int(args.smooth_scale),
                "us": round(co_us, 3),
                "tflops": tflops(token, model_dim, inter_dim, topk, co_us),
            }  # fmt: skip
            line = f"  t={token:<6} us={co_us:8.1f}  {row['tflops']:7.2f} tflops"

            bad = []
            if args.asm:
                asm_us = out["asm"][1]
                ratio = co_us / asm_us
                ratios.append(ratio)
                # The clamp and the smooth factor change the math, so the outputs
                # are only comparable when neither is in play.
                cos = (
                    cos_sim(out["co"][0], out["asm"][0])
                    if smooth is None and not args.swiglu_limit
                    else float("nan")
                )
                row |= {
                    "asm_us": round(asm_us, 3), "ratio": round(ratio, 3),
                    "cos": round(cos, 7),
                }  # fmt: skip
                if cos == cos and cos < args.min_cos:
                    bad.append(f"cos={cos:.4f}")
                if ratio > args.max_ratio:
                    bad.append(f"ratio={ratio:.2f}")
                if bad:
                    failures.append(
                        f"{model_dim}x{inter_dim} E{E}k{topk} t={token}: {','.join(bad)}"
                    )
                line += (
                    f"  asm={asm_us:8.1f}  {ratio:.2f}x  cos={cos:.5f}"
                    f"  {'FAIL' if bad else 'ok'}"
                )
            rows.append(row)
            print(line, flush=True)
            torch.cuda.empty_cache()
        del w1, w2, w1_q, w2_q, w1_sh, w2_sh
        torch.cuda.empty_cache()

    if args.csv:
        # Appending lets one sweep collect several variants into one file.
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")

    if not ratios:
        print(f"\n{len(rows)} points measured")
        return 0
    print(f"\nworst ratio {max(ratios):.2f}x, mean {sum(ratios) / len(ratios):.2f}x")
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"all {len(ratios)} points pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
