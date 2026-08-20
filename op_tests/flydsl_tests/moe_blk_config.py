# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared writer for aiter/configs/moe_blk_tuned.csv.

The file follows aiter/configs/tuned_fmoe.csv column for column so it reads like
every other tuned config: one row per (shape, token), both stages side by side,
tiles carried in kernelName1/kernelName2. It stays a separate file rather than
joining the fmoe family, because that family already has rows for these shapes
from the asm/CK tuner and its key has no room for a second backend -- merging
would trip the duplicate-shape guard.
"""

from __future__ import annotations

import csv

COLUMNS = [
    "cu_num", "token", "model_dim", "inter_dim", "expert", "topk",
    "act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type",
    "use_g1u1", "doweight_stage1", "block_m", "ksplit",
    "us1", "kernelName1", "err1", "us2", "kernelName2", "err2",
    "us", "run_1stage", "tflops", "bw", "_tag",
]  # fmt: skip

# This path only ever builds silu g1u1 blockwise-fp8 kernels; the columns exist
# to match the schema, not because they vary.
FIXED = {
    "act_type": "ActivationType.Silu",
    "dtype": "torch.bfloat16",
    "q_dtype_a": "torch.float8_e4m3fnuz",
    "q_dtype_w": "torch.float8_e4m3fnuz",
    "q_type": "QuantType.per_1x128",
    "use_g1u1": 1,
    "doweight_stage1": 0,
    "ksplit": 0,
    "run_1stage": 0,
    "err1": "0.0%",
    "err2": "0.0%",
    "_tag": "moe_blk_co",
}
FP8_BYTES, BF16_BYTES = 1, 2


def perf(token, model_dim, inter_dim, expert, topk, us):
    """TFLOPS and GB/s, matching gemm_moe_tune.py's FmoeTuner.calculate."""
    n = inter_dim * 2  # use_g1u1
    flop = token * n * model_dim * topk * 2 + topk * token * model_dim * inter_dim * 2
    data_bytes = (
        token * model_dim * FP8_BYTES
        + n * model_dim * FP8_BYTES * expert
        + inter_dim * model_dim * FP8_BYTES * expert
        + token * model_dim * BF16_BYTES
    )
    return round(flop / (us * 1e6), 2), round(data_bytes / (us * 1e-6) / 1e9, 2)


def row(cu_num, token, model_dim, inter_dim, expert, topk, block_m,
        us1, name1, us2, name2, us):  # fmt: skip
    tflops, bw = perf(token, model_dim, inter_dim, expert, topk, us)
    return {
        **FIXED,
        "cu_num": cu_num, "token": token, "model_dim": model_dim,
        "inter_dim": inter_dim, "expert": expert, "topk": topk,
        "block_m": block_m, "us1": round(us1, 4), "kernelName1": name1,
        "us2": round(us2, 4), "kernelName2": name2, "us": round(us, 4),
        "tflops": tflops, "bw": bw,
    }  # fmt: skip


def write(path, rows):
    rows = sorted(
        rows,
        key=lambda r: (
            r["model_dim"], r["inter_dim"], r["expert"], r["topk"], r["token"]
        ),  # fmt: skip
    )
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
