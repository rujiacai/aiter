#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune blockwise-fp8 MoE tiles per (shape, token) and emit the config CSV.

Weights are built once per shape and every candidate tile is compiled once, then
benchmarked across all tokens -- the naive loop would recompile the same kernel
for every token and spend most of its wall time in MLIR.

The asm/CK dispatch is timed alongside as the reference the result is judged
against; the goal is a usable tile, not the optimum.

    python op_tests/flydsl_tests/tune_moe_blk.py -o aiter/configs/moe_blk_tuned.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os

import torch

import moe_blk_config

import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs, moe_sorting
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.moe_blk import co_name
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest

BLK = 128
SHAPES = [
    # (model_dim, inter_dim, expert, topk)
    (6144, 256, 256, 8),
    (6144, 2048, 16, 8),
    (6144, 256, 257, 9),
    (6144, 2048, 17, 9),
]
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RAW_FIELDS = [
    "model_dim", "inter_dim", "expert", "topk", "token", "stage",
    "tile_m", "tile_n", "tile_k", "waves_per_eu", "us", "asm_us",
]  # fmt: skip


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


def candidates(stage: int, model_dim: int, inter_dim: int, wide: bool = False):
    """Tile grid, pruned to what can actually run and what measured competitive.

    The ping-pong tail consumes two K tiles, so K/tile_k has to be even; stage1's
    K is model_dim and stage2's is inter_dim. tile_k=256 lost to 128 on stage1 at
    both ends of the token range, so stage1 keeps 128 and only stage2 sweeps it.
    """
    tile_ms = (16, 32, 64)
    if stage == 1:
        k_dim, tile_ns, tile_ks = model_dim, (64, 128), (128,)
        if wide:
            tile_ns, tile_ks = (64, 128, 256, 512), (128, 256)
    else:
        k_dim, tile_ns, tile_ks = inter_dim, (128, 256), (128, 256)
        if wide:
            tile_ns = (128, 256, 512)
    out = []
    for tm, tn, tk in itertools.product(tile_ms, tile_ns, tile_ks):
        if k_dim % tk or (k_dim // tk) % 2:
            continue
        n_dim = inter_dim if stage == 1 else model_dim
        if n_dim % tn:
            continue
        out.append((tm, tn, tk))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--out", default="aiter/configs/moe_blk_tuned.csv")
    p.add_argument("--waves", type=int, default=2)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    p.add_argument("--shape", action="append",
                   type=lambda s: tuple(int(x) for x in s.split(",")))  # fmt: skip
    p.add_argument("--wide", action="store_true", help="expand the tile_n/tile_k grid")
    p.add_argument("--raw", default=None,
                   help="also dump every candidate measurement, for offline reselection")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if not args.verbose:
        aiter.logger.setLevel(logging.WARNING)

    dev = "cuda"
    torch.manual_seed(0)
    rows, raw = [], []
    cu_num = torch.cuda.get_device_properties(0).multi_processor_count

    for model_dim, inter_dim, E, topk in args.shape or SHAPES:
        tag = f"{model_dim}x{inter_dim} E{E} k{topk}"
        print(f"\n=== {tag} ===", flush=True)
        w1 = torch.randn((E, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device=dev) / 10
        w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device=dev) / 10
        w1_q, w1_s = block_quant(w1)
        w2_q, w2_s = block_quant(w2)
        del w1, w2
        torch.cuda.empty_cache()
        w1_sh = shuffle_weight(w1_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)
        w2_sh = shuffle_weight(w2_q.view(dtypes.i8), layout=(16, 16)).view(dtypes.fp8)

        for token in args.tokens:
            x = torch.randn((token, model_dim), dtype=dtypes.bf16, device=dev)
            score = torch.randn((token, E), dtype=dtypes.bf16, device=dev)
            tw, ti = fused_topk(x, score, topk, True)
            a_q, a_s = aiter.pertoken_quant(x.view(token, -1, BLK), quant_dtype=dtypes.fp8)
            a_q, a_s = a_q.view(token, model_dim), a_s.view(token, -1).float().contiguous()

            # asm/CK reference for the same shape, end to end.
            os.environ["AITER_FLYDSL_BLKFP8"] = "0"
            os.environ["AITER_MOE_BLK_CO"] = "0"
            get_2stage_cfgs.cache_clear()
            _, asm_us = run_perftest(
                fused_moe, x, w1_sh, w2_sh, tw, ti,
                quant_type=QuantType.per_128x128, w1_scale=w1_s, w2_scale=w2_s,
                num_iters=args.iters, num_warmup=3,
            )  # fmt: skip

            # Keyed by (stage, tile_m) so tile_m can be chosen jointly below.
            best = {}
            for stage in (1, 2):
                for tm, tn, tk in candidates(stage, model_dim, inter_dim, args.wide):
                    sid, sw, seid, nv, buf = moe_sorting(
                        ti, tw, E, model_dim, dtypes.bf16, tm
                    )
                    try:
                        if stage == 1:
                            out = torch.zeros(
                                (token, topk, inter_dim), dtype=dtypes.bf16, device=dev
                            )
                            fn = lambda: flydsl_moe_stage1(  # noqa: E731
                                a_q, w1_sh, sid, seid, nv, out=out, topk=topk,
                                tile_m=tm, tile_n=tn, tile_k=tk,
                                a_dtype="fp8", b_dtype="fp8blk", out_dtype="bf16",
                                act="silu", w1_scale=w1_s, a1_scale=a_s,
                                waves_per_eu=args.waves,
                            )  # fmt: skip
                        else:
                            a2 = torch.randn(
                                (token, topk, inter_dim), dtype=dtypes.bf16, device=dev
                            ).to(dtypes.fp8)
                            a2_s = torch.rand(
                                (token * topk, inter_dim // BLK),
                                dtype=torch.float32, device=dev,
                            ) + 0.5  # fmt: skip
                            fn = lambda: flydsl_moe_stage2(  # noqa: E731
                                a2, w2_sh, sid, seid, nv, out=buf, topk=topk,
                                tile_m=tm, tile_n=tn, tile_k=tk,
                                a_dtype="fp8", b_dtype="fp8blk", out_dtype="bf16",
                                mode="atomic", w2_scale=w2_s, a2_scale=a2_s.contiguous(),
                                sorted_weights=sw, waves_per_eu=args.waves,
                            )  # fmt: skip
                        _, us = run_perftest(fn, num_iters=args.iters, num_warmup=3)
                    except Exception as exc:  # a tile the kernel rejects
                        if args.verbose:
                            print(f"    skip s{stage} {tm}x{tn}x{tk}: {exc}"[:110])
                        continue
                    raw.append(
                        {
                            "model_dim": model_dim, "inter_dim": inter_dim,
                            "expert": E, "topk": topk, "token": token, "stage": stage,
                            "tile_m": tm, "tile_n": tn, "tile_k": tk,
                            "waves_per_eu": args.waves,
                            "us": round(us, 3), "asm_us": round(asm_us, 3),
                        }  # fmt: skip
                    )
                    if (stage, tm) not in best or us < best[(stage, tm)][0]:
                        best[(stage, tm)] = (us, tm, tn, tk)

            # Both stages read the one sorted_token_ids that moe_sorting built
            # for a single block_m, so their tile_m must agree -- picking each
            # stage's own optimum silently corrupts the output. Choose the tile_m
            # with the best combined time instead.
            shared = [
                tm
                for tm in {k[1] for k in best}
                if (1, tm) in best and (2, tm) in best
            ]
            if not shared:
                print(f"  t={token:<5} no tile_m works for both stages", flush=True)
                continue
            tile_m = min(shared, key=lambda m: best[(1, m)][0] + best[(2, m)][0])

            line = [f"  t={token:<5} asm={asm_us:8.1f}"]
            picked, names = [], []
            for stage in (1, 2):
                us, tm, tn, tk = best[(stage, tile_m)]
                picked.append(us)
                names.append(
                    co_name(
                        stage, model_dim, inter_dim, E, topk, tm, tn, tk, args.waves
                    )
                )
                line.append(f"  s{stage}={us:7.1f} ({tm}x{tn}x{tk})")
            total = sum(best[(s, tile_m)][0] for s in (1, 2))
            rows.append(
                moe_blk_config.row(
                    cu_num, token, model_dim, inter_dim, E, topk, tile_m,
                    picked[0], names[0], picked[1], names[1], total,
                )  # fmt: skip
            )
            line.append(f"  s1+s2={total:8.1f}  {total / asm_us:.2f}x asm")
            print("".join(line), flush=True)

            del x, score, tw, ti, a_q, a_s
            torch.cuda.empty_cache()
        del w1_q, w1_s, w2_q, w2_s, w1_sh, w2_sh
        torch.cuda.empty_cache()

    moe_blk_config.write(args.out, rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")
    if args.raw:
        with open(args.raw, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
            w.writeheader()
            w.writerows(raw)
        print(f"wrote {len(raw)} candidate measurements to {args.raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
