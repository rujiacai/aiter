#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check the shipped .co set against asm/CK on every tuned (shape, token).

Guards the two ways the release can break: a code object the dispatch asks for
but the exporter never built (raises), and a tile that is correct but slow. The
timing here is end to end through fused_moe on both sides, so unlike the tuner's
kernel-only numbers the ratio is the one a customer sees.
"""

from __future__ import annotations

import argparse
import csv
import os

import torch

import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest

from tune_moe_blk import BLK, SHAPES, TOKENS, block_quant


def cos_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    p.add_argument("--min-cos", type=float, default=0.99)
    p.add_argument("--max-ratio", type=float, default=1.3)
    p.add_argument("--csv", default=None, help="write the measurements out")
    args = p.parse_args()
    measured = []

    dev = "cuda"
    torch.manual_seed(0)
    failures, ratios = [], []

    for model_dim, inter_dim, E, topk in SHAPES:
        print(f"\n=== {model_dim}x{inter_dim} E{E} k{topk} ===", flush=True)
        w1 = torch.randn((E, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device=dev) / 10
        w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device=dev) / 10
        w1_q, w1_s = block_quant(w1)
        w2_q, w2_s = block_quant(w2)
        w1_sh = shuffle_weight(w1_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)
        w2_sh = shuffle_weight(w2_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)

        for token in args.tokens:
            x = torch.randn((token, model_dim), dtype=dtypes.bf16, device=dev)
            score = torch.randn((token, E), dtype=dtypes.bf16, device=dev)
            tw, ti = fused_topk(x, score, topk, True)
            call = dict(
                quant_type=QuantType.per_128x128, w1_scale=w1_s, w2_scale=w2_s
            )  # fmt: skip

            out = {}
            for name, env in (("asm", "0"), ("co", "1")):
                os.environ["AITER_MOE_BLK_CO"] = env
                os.environ["AITER_FLYDSL_BLKFP8"] = "0"
                get_2stage_cfgs.cache_clear()
                out[name] = run_perftest(
                    fused_moe, x, w1_sh, w2_sh, tw, ti,
                    num_iters=args.iters, num_warmup=3, **call,
                )  # fmt: skip

            # asm is the reference: it is the production path today and it
            # quantizes identically, so any gap is the code object's own doing.
            cos = cos_sim(out["co"][0], out["asm"][0])
            asm_us, co_us = out["asm"][1], out["co"][1]
            ratio = co_us / asm_us
            ratios.append(ratio)
            measured.append(
                {
                    "model_dim": model_dim, "inter_dim": inter_dim, "expert": E,
                    "topk": topk, "token": token, "asm_us": round(asm_us, 3),
                    "co_us": round(co_us, 3), "ratio": round(ratio, 4),
                    "cos": round(cos, 6),
                }  # fmt: skip
            )
            bad = []
            if cos < args.min_cos:
                bad.append(f"cos={cos:.4f}")
            if ratio > args.max_ratio:
                bad.append(f"ratio={ratio:.2f}")
            if bad:
                failures.append(f"{model_dim}x{inter_dim} E{E}k{topk} t={token}: {','.join(bad)}")
            print(
                f"  t={token:<5} asm={asm_us:8.1f}  co={co_us:8.1f}  "
                f"{ratio:.2f}x  cos={cos:.5f}  {'FAIL' if bad else 'ok'}",
                flush=True,
            )
            torch.cuda.empty_cache()
        del w1, w2, w1_q, w2_q, w1_sh, w2_sh
        torch.cuda.empty_cache()

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(measured[0]))
            w.writeheader()
            w.writerows(measured)
        print(f"wrote {len(measured)} rows to {args.csv}")

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
