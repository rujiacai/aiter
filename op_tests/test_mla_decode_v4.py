# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and timing for `aiter.mla_decode_v4_bf16`.

The cases vary how the CSR rows of neighbouring query tokens relate to each
other, because that relationship is the one thing the kernel must not assume
anything about. A DeepSeek-V4 verify step slides a window per token and, under
CSA, picks compressed slots per token as well, so "prefix" is the only shape a
shared-row kernel gets right and the only one that would hide such a bug.
"""

import argparse
import sys

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose, run_perftest

D = 512
MODES = ("slide", "disjoint", "prefix", "ragged")


def make_rows(mode, T, kv_len, num_slots, g):
    """One independent slot list per query token.

    slide     fixed-length window whose start moves by one per token, which is
              what a verify step's sliding-window section actually looks like.
    disjoint  every token samples its own slots, as CSA's per-token top-k does.
    prefix    nested prefixes; the plain MTP shape.
    ragged    independent lengths and slots, including empty rows.
    """
    if mode == "slide":
        pool = torch.randperm(num_slots, generator=g)[: kv_len + T].int()
        return [pool[t : t + kv_len] for t in range(T)]
    if mode == "disjoint":
        return [torch.randperm(num_slots, generator=g)[:kv_len].int() for _ in range(T)]
    if mode == "prefix":
        row = torch.randperm(num_slots, generator=g)[:kv_len].int()
        return [row[: max(kv_len - (3 - t % 4), 0)] for t in range(T)]
    if mode == "ragged":
        rows = []
        for _ in range(T):
            n = int(torch.randint(0, kv_len + 1, (1,), generator=g))
            rows.append(torch.randperm(num_slots, generator=g)[:n].int())
        return rows
    raise ValueError(mode)


def make_case(T, heads, kv_len, mode, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_slots = max(kv_len * 2, kv_len + T + 64)

    q = torch.randn(T, heads, D, generator=g).to(device).to(dtypes.bf16)
    unified_kv = torch.randn(num_slots, D, generator=g).to(device).to(dtypes.bf16)
    attn_sink = torch.randn(heads, generator=g).to(device).float()

    rows = make_rows(mode, T, kv_len, num_slots, g)
    kv_indices = torch.cat(rows).to(device)
    kv_indptr = torch.zeros(T + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.tensor(
        [len(r) for r in rows], dtype=torch.int32, device=device
    ).cumsum(0)
    return q, unified_kv, kv_indices, kv_indptr, attn_sink


def ref_mla_decode(q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale):
    """fp32 reference: each token reads its own CSR row and nothing else."""
    T, H, _ = q.shape
    out = torch.empty_like(q)
    qf = q.float()
    sink = attn_sink.float()
    for t in range(T):
        base = int(kv_indptr[t])
        n = int(kv_indptr[t + 1]) - base
        if n <= 0:
            out[t] = 0
            continue
        idx = kv_indices[base : base + n].long()
        kvt = unified_kv.index_select(0, idx).float()  # [n, D]
        scores = (qf[t] @ kvt.t()) * softmax_scale  # [H, n]
        m = torch.maximum(scores.max(dim=-1).values, sink)
        p = torch.exp(scores - m[:, None])
        denom = p.sum(dim=-1) + torch.exp(sink - m)
        out[t] = ((p @ kvt) / denom[:, None]).to(q.dtype)
    return out


def test_mla_decode_v4(T, heads, kv_len, kv_splits, mode):
    q, ukv, idx, indptr, sink = make_case(T, heads, kv_len, mode)
    scale = 1.0 / (D**0.5)

    ref = ref_mla_decode(q, ukv, idx, indptr, sink, scale)
    out, us = run_perftest(
        aiter.mla_decode_v4_bf16, q, ukv, idx, indptr, sink, scale, kv_splits
    )

    msg = f"T={T:<4} H={heads:<4} kv={kv_len:<6} splits={kv_splits} {mode:<8} {us:>8.2f} us"
    # bf16 accumulation of a 512-wide dot leaves ~2e-3 absolute; the ratio guard
    # is what actually catches a wrong row, since that moves whole heads.
    checkAllclose(ref.float(), out.float(), rtol=2e-2, atol=1e-2, msg=msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="one shape, all row modes")
    args = ap.parse_args()

    if get_gfx() != "gfx942":
        print(f"skipped: mla_decode_v4_bf16 is gfx942-only, running on {get_gfx()}")
        sys.exit(0)

    # H only has to be a multiple of 16. TP=8 on DeepSeek-V4-Pro's 128 heads
    # gives H=16, so the small counts are the deployed ones, not edge cases.
    shapes = [
        (4, 16, 1000, 1),
        (4, 32, 1000, 1),
        (4, 64, 100, 1),
        (4, 128, 256, 1),
        (8, 128, 1000, 1),
        (4, 128, 4096, 4),
        (12, 64, 777, 2),
        (16, 128, 2048, 8),
        # tile boundaries: BK=32, so exercise len % BK of 0, 1 and 3
        (4, 128, 32, 1),
        (4, 128, 33, 1),
        (4, 128, 35, 1),
        # single token, i.e. plain decode with no speculation at all
        (1, 128, 1000, 1),
        (1, 16, 50000, 8),
    ]
    if args.quick:
        shapes = shapes[:2]

    for shape in shapes:
        for mode in MODES:
            test_mla_decode_v4(*shape, mode)
