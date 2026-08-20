#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Re-pick tiles from a tuner sweep to land near a target ratio against asm.

The shipped configuration deliberately aims for "good enough" rather than the
fastest tile the sweep found, so this reselects from the raw candidate
measurements instead of re-running the sweep.

The sweep times kernels only while the target is stated end to end, and the
difference (host quant, sorting, two launches) is 25-45us depending on shape.
That cost does not change with the tile, so it is calibrated once per
(shape, token) from a verify run and added back before comparing to the target.
Predictions are only used to rank candidates; the verify pass afterwards is what
actually confirms the shipped numbers.

    python op_tests/flydsl_tests/select_moe_blk_tiles.py \
        --raw /tmp/raw.csv --current aiter/configs/moe_blk_tuned.csv \
        --verify /tmp/verify.csv -o aiter/configs/moe_blk_tuned.csv
"""

from __future__ import annotations

import argparse
import collections
import csv

import moe_blk_config
from aiter.ops.moe_blk import co_name

KEY = ("model_dim", "inter_dim", "expert", "topk", "token")


def key_of(row) -> tuple:
    return tuple(int(row[c]) for c in KEY)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, help="every candidate, from tune --raw")
    p.add_argument("--current", required=True, help="tuned CSV the verify run used")
    p.add_argument("--verify", required=True, help="end-to-end CSV from verify --csv")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--cu-num", type=int, default=80)
    p.add_argument("--target", type=float, default=1.2, help="desired ratio vs asm")
    p.add_argument("--cap", type=float, default=1.3, help="never predict above this")
    p.add_argument(
        "--force-fastest", action="append", default=[], metavar="MD,ID,E,K,TOKEN",
        help="keep the fastest tile here; for points a verify run showed the "
             "prediction overshot the cap on",
    )  # fmt: skip
    args = p.parse_args()
    forced = {tuple(int(x) for x in s.split(",")) for s in args.force_fastest}

    raw = collections.defaultdict(list)
    for r in csv.DictReader(open(args.raw)):
        raw[key_of(r)].append(r)

    # Kernel-only total of whatever tiles the verify run measured end to end.
    kernel_total = {
        key_of(r): float(r["us1"]) + float(r["us2"])
        for r in csv.DictReader(open(args.current))
    }

    overhead, asm_e2e = {}, {}
    for r in csv.DictReader(open(args.verify)):
        k = key_of(r)
        overhead[k] = float(r["co_us"]) - kernel_total[k]
        asm_e2e[k] = float(r["asm_us"])

    rows, report = [], []
    for k, cands in sorted(raw.items()):
        if k not in overhead:
            continue
        over, asm = overhead[k], asm_e2e[k]

        # Fastest candidate per (stage, tile_m); tile_m has to agree across the
        # two stages because they share one moe_sorting result.
        best = {}
        for c in cands:
            sk = (int(c["stage"]), int(c["tile_m"]))
            if sk not in best or float(c["us"]) < float(best[sk]["us"]):
                best[sk] = c
        combos = [
            (best[(1, m)], best[(2, m)])
            for m in {t[1] for t in best}
            if (1, m) in best and (2, m) in best
        ]
        if not combos:
            continue

        def predict(combo):
            return float(combo[0]["us"]) + float(combo[1]["us"]) + over

        fastest = min(combos, key=predict)
        floor = predict(fastest) / asm
        if k in forced:
            chosen, note = fastest, "forced"
        elif floor >= args.target:
            # Already at or past the target; nothing to give away here.
            chosen, note = fastest, "fastest"
        else:
            # Slowest candidate that still stays under the cap, but no further
            # from the target than the fastest one already is.
            allowed = [c for c in combos if predict(c) / asm <= args.cap]
            chosen = min(allowed, key=lambda c: abs(predict(c) / asm - args.target))
            note = "retargeted" if chosen is not fastest else "fastest"

        pred = predict(chosen) / asm
        report.append((k, floor, pred, note, chosen))
        md, idim, e, topk, token = k
        block_m = int(chosen[0]["tile_m"])
        names = [
            co_name(
                s, md, idim, e, topk, block_m,
                int(c["tile_n"]), int(c["tile_k"]), int(c["waves_per_eu"]),
            )  # fmt: skip
            for s, c in enumerate(chosen, start=1)
        ]
        rows.append(
            moe_blk_config.row(
                args.cu_num, token, md, idim, e, topk, block_m,
                float(chosen[0]["us"]), names[0],
                float(chosen[1]["us"]), names[1],
                predict(chosen),
            )  # fmt: skip
        )

    moe_blk_config.write(args.out, rows)

    print(f"{'shape':<24} {'token':>6} {'fastest':>9} {'chosen':>9}  tiles")
    for k, floor, pred, note, c in report:
        tiles = "  ".join(
            f"s{s['stage']}={s['tile_m']}x{s['tile_n']}x{s['tile_k']}" for s in c
        )
        print(
            f"d{k[0]}x{k[1]} E{k[2]}k{k[3]}".ljust(24)
            + f" {k[4]:>6} {floor:>8.2f}x {pred:>8.2f}x  {tiles}  {note}"
        )
    n = sum(1 for r in report if r[3] == "retargeted")
    print(f"\n{len(report)} points, {n} retargeted, wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
