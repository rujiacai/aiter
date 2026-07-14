#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for the FlyDSL ``dyna_fused_topk`` router.

Compares :func:`aiter.ops.flydsl.flydsl_dyna_fused_topk` against the eager
torch reference :func:`aiter.fused_moe._dyna_fused_topk_torch` (which is also
the fallback used by :func:`aiter.fused_moe.dyna_fused_topk`).

Per token ``t`` the op computes ``softmax(gating) -> top-max_topk``, keeps the
first ``dyna_k[t]`` experts, renormalizes the kept weights to sum to 1, and
pads the tail (ids -> ``pad_id``, weights -> ``0``).

Usage:
    python op_tests/test_dyna_fused_topk.py
    python op_tests/test_dyna_fused_topk.py --tokens 256 --experts 64 --max-topk 20
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aiter.fused_moe import (
    _dyna_fused_topk_torch,
    dyna_fused_topk,
    fused_topk,
)
from aiter.ops.flydsl.utils import is_flydsl_available

try:
    from aiter.ops.flydsl import flydsl_dyna_fused_topk
except Exception:  # pragma: no cover - flydsl optional
    flydsl_dyna_fused_topk = None


def _compare(label, ref, got, atol):
    diff = (ref.float() - got.float()).abs()
    max_abs = diff.max().item()
    ok = max_abs <= atol
    print(f"  [{label}] max_abs={max_abs:.3e} atol={atol:.1e} -> {'OK' if ok else 'FAIL'}")
    return ok


def _check_ids(ref_id, got_id):
    # ids are integers; require exact match (selection / padding must agree).
    mism = (ref_id != got_id).sum().item()
    ok = mism == 0
    print(f"  [ids] mismatched={mism} -> {'OK' if ok else 'FAIL'}")
    return ok


def run_case(tokens, experts, max_topk, renormalize, pad_id, device, seed,
             scoring_func="softmax"):
    g = torch.Generator(device=device).manual_seed(seed)
    gating = torch.randn(
        tokens, experts, device=device, dtype=torch.float32, generator=g
    ).contiguous()
    dyna_k = torch.randint(
        1, max_topk + 1, (tokens,), device=device, dtype=torch.int32, generator=g
    )

    ref_w, ref_id = _dyna_fused_topk_torch(
        gating, dyna_k, max_topk, renormalize=renormalize, pad_id=pad_id,
        scoring_func=scoring_func,
    )

    print(
        f"\n=== [{scoring_func}] tokens={tokens} experts={experts} max_topk={max_topk} "
        f"renormalize={renormalize} pad_id={pad_id} device={device} ==="
    )

    # The eager fallback must always be self-consistent (sanity on the spec).
    fb_w, fb_id = dyna_fused_topk(
        gating, dyna_k, max_topk, renormalize=renormalize,
        pad_id=pad_id, scoring_func=scoring_func, use_flydsl=False,
    )
    ok = _compare("torch-fallback weights", ref_w, fb_w, 0.0)
    ok &= _check_ids(ref_id, fb_id)

    # renormalize invariant: each kept row sums to 1.
    if renormalize:
        got_sum = ref_w.sum(-1)
        ones = torch.ones_like(got_sum)
        ok &= _compare("renorm-sum==1 invariant", ones, got_sum, 1e-4)

    if is_flydsl_available() and device != "cpu":
        kw, kid = dyna_fused_topk(
            gating, dyna_k, max_topk, renormalize=renormalize,
            pad_id=pad_id, scoring_func=scoring_func, use_flydsl=True,
        )
        ok &= _compare("flydsl weights", ref_w, kw, 2e-3)
        ok &= _check_ids(ref_id, kid)

        # weight tail (j >= dyna_k) must be exactly 0 in the kernel output.
        ar = torch.arange(max_topk, device=device)
        kept = ar.unsqueeze(0) < dyna_k.to(torch.long).clamp(1, max_topk).unsqueeze(1)
        tail_zero = bool((torch.where(kept, torch.zeros_like(kw), kw) == 0).all())
        print(f"  [weight-tail-zero] {tail_zero} -> {'OK' if tail_zero else 'FAIL'}")
        ok &= tail_zero
    else:
        print("  [flydsl] skipped (flydsl unavailable or CPU device)")

    return ok


def run_dtype_case(tokens, experts, max_topk, dtype, device, seed,
                   scoring_func="softmax"):
    """bf16 / fp16 gating input. The wrapper up-casts to f32 on the host and the
    torch reference also up-casts internally, so BOTH paths see identical f32
    values -> weights match tightly. Coarse rounding can produce tied logits;
    when that happens torch.topk and the kernel may order/pick the tied experts
    differently, so an id mismatch is only a failure if the slot weights are
    *not* equal (i.e. it is not a genuine tie)."""
    name = {torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype]
    g = torch.Generator(device=device).manual_seed(seed)
    gating = torch.randn(
        tokens, experts, device=device, dtype=torch.float32, generator=g
    ).to(dtype).contiguous()
    dyna_k = torch.randint(
        1, max_topk + 1, (tokens,), device=device, dtype=torch.int32, generator=g
    )
    print(
        f"\n=== [dtype={name}][{scoring_func}] tokens={tokens} experts={experts} "
        f"max_topk={max_topk} device={device} ==="
    )
    ref_w, ref_id = _dyna_fused_topk_torch(gating, dyna_k, max_topk, scoring_func=scoring_func)
    if not (is_flydsl_available() and device != "cpu"):
        print("  [flydsl] skipped (flydsl unavailable or CPU device)")
        return True
    kw, kid = dyna_fused_topk(gating, dyna_k, max_topk, scoring_func=scoring_func, use_flydsl=True)

    ok = _compare("flydsl weights", ref_w, kw, 2e-3)

    # native low-precision (load 16-bit + widen in-register) must be bit-
    # identical to the host up-cast path (16-bit -> f32 widening is exact).
    if flydsl_dyna_fused_topk is not None:
        nw, nid = flydsl_dyna_fused_topk(
            gating, dyna_k, max_topk, native=True, scoring_func=scoring_func
        )
        hw, hid = flydsl_dyna_fused_topk(
            gating, dyna_k, max_topk, native=False, scoring_func=scoring_func
        )
        nat_ok = bool((nw == hw).all() and (nid == hid).all())
        print(f"  [native==host-upcast] {nat_ok} -> {'OK' if nat_ok else 'FAIL'}")
        ok &= nat_ok
    # id mismatches are acceptable only at slots where the weights are equal
    # (a tie); a mismatch with differing weights is a real selection error.
    id_diff = ref_id != kid
    n_mismatch = int(id_diff.sum().item())
    if n_mismatch == 0:
        print("  [ids] mismatched=0 -> OK")
    else:
        wdiff = (ref_w.float() - kw.float()).abs()
        bad = int((id_diff & (wdiff > 2e-3)).sum().item())
        tie_ok = bad == 0
        print(
            f"  [ids] mismatched={n_mismatch} (tie-induced={n_mismatch - bad}, "
            f"real_errors={bad}) -> {'OK' if tie_ok else 'FAIL'}"
        )
        ok &= tie_ok
    return ok


def run_clamp_case(tokens, experts, max_topk, device, seed, scoring_func="softmax"):
    """dyna_k out of [1, max_topk] (incl. 0 and > max_topk) must still match the
    torch reference -- exercises the kernel/reference clamp agreement."""
    g = torch.Generator(device=device).manual_seed(seed)
    gating = torch.randn(
        tokens, experts, device=device, dtype=torch.float32, generator=g
    ).contiguous()
    # mix in 0 and values larger than max_topk
    dyna_k = torch.randint(
        0, max_topk + 5, (tokens,), device=device, dtype=torch.int32, generator=g
    )
    print(
        f"\n=== [clamp][{scoring_func}] tokens={tokens} experts={experts} max_topk={max_topk} "
        f"k:{int(dyna_k.min())}..{int(dyna_k.max())} device={device} ==="
    )
    ref_w, ref_id = _dyna_fused_topk_torch(gating, dyna_k, max_topk, scoring_func=scoring_func)
    if not (is_flydsl_available() and device != "cpu"):
        print("  [flydsl] skipped (flydsl unavailable or CPU device)")
        return True
    kw, kid = dyna_fused_topk(
        gating, dyna_k, max_topk, scoring_func=scoring_func, use_flydsl=True
    )
    ok = _compare("flydsl weights", ref_w, kw, 2e-3)
    ok &= _check_ids(ref_id, kid)
    return ok


def run_large_batch_case(tokens, experts, max_topk, device, seed):
    """Exercise the large-batch layout (T >= LARGE_BATCH_TOKENS picks the
    throughput-tuned VPT for E >= 256) -- a different sub-warp split + grid
    sizing than the small/mid-batch layout, so it must still match torch."""
    g = torch.Generator(device=device).manual_seed(seed)
    gating = torch.randn(
        tokens, experts, device=device, dtype=torch.bfloat16, generator=g
    ).contiguous()
    dyna_k = torch.randint(
        1, max_topk + 1, (tokens,), device=device, dtype=torch.int32, generator=g
    )
    print(
        f"\n=== [large-batch] tokens={tokens} experts={experts} "
        f"max_topk={max_topk} dtype=bf16 device={device} ==="
    )
    ref_w, ref_id = _dyna_fused_topk_torch(gating, dyna_k, max_topk)
    if not (is_flydsl_available() and device != "cpu" and flydsl_dyna_fused_topk):
        print("  [flydsl] skipped (flydsl unavailable or CPU device)")
        return True
    kw, kid = flydsl_dyna_fused_topk(gating, dyna_k, max_topk)
    ok = _compare("flydsl weights", ref_w, kw, 2e-3)
    # ties from bf16 rounding are common at large T; only differing-weight id
    # mismatches are real errors.
    id_diff = ref_id != kid
    n = int(id_diff.sum().item())
    if n:
        wdiff = (ref_w.float() - kw.float()).abs()
        bad = int((id_diff & (wdiff > 2e-3)).sum().item())
        print(f"  [ids] mismatched={n} (real_errors={bad}) -> "
              f"{'OK' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    else:
        print("  [ids] mismatched=0 -> OK")
    return ok


def run_fused_topk_passthrough(tokens, experts, max_topk, device, seed):
    """``fused_topk(..., dyna_k=...)`` must delegate to the dynamic path and
    return exactly what ``dyna_fused_topk`` returns (same call site / contract)."""
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(tokens, 16, device=device, dtype=torch.float32, generator=g)
    gating = torch.randn(
        tokens, experts, device=device, dtype=torch.float32, generator=g
    ).contiguous()
    dyna_k = torch.randint(
        1, max_topk + 1, (tokens,), device=device, dtype=torch.int32, generator=g
    )
    print(
        f"\n=== [fused_topk passthrough] tokens={tokens} experts={experts} "
        f"max_topk={max_topk} device={device} ==="
    )
    # reference: call dyna_fused_topk directly
    ref_w, ref_id = dyna_fused_topk(gating, dyna_k, max_topk, renormalize=True)
    # under test: route through fused_topk with the dyna_k passthrough
    got_w, got_id = fused_topk(hidden, gating, max_topk, True, dyna_k=dyna_k)
    ok = _compare("passthrough weights", ref_w, got_w, 0.0 if device == "cpu" else 2e-3)
    ok &= _check_ids(ref_id, got_id)

    # static path must be untouched: dyna_k=None still returns max_topk routing.
    sw, sid = fused_topk(hidden, gating, max_topk, True)
    shape_ok = sw.shape == (tokens, max_topk) and sid.shape == (tokens, max_topk)
    print(f"  [static-path shape] {tuple(sw.shape)} -> {'OK' if shape_ok else 'FAIL'}")
    ok &= shape_ok
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument(
        "--experts", type=int, default=None,
        help="single expert count; default None sweeps a representative set "
        "(incl. 256 and non-multiples of the 64-lane wavefront)",
    )
    p.add_argument("--max-topk", type=int, default=20)
    p.add_argument(
        "--pad-id", type=int, default=None,
        help="dropped-tail id sentinel; default None -> num_experts (moe_sorting-skipped)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable -> running torch reference on CPU only")
        device = "cpu"

    # Sweep expert counts that exercise the wavefront design: <64 (tail lanes),
    # exactly 64 (one full wave), a multiple and a non-multiple of 64, and the
    # 256 case (CN = ceil(E/64) = 4).
    if args.experts is not None:
        expert_counts = [args.experts]
    else:
        # <64 (tail lanes), 64 (one wave), multiple/non-multiple of 64, the
        # 192 non-factoring (one-wave fallback) case, 256, and 512 (sub-warp).
        expert_counts = [8, 64, 128, 130, 192, 256, 512]

    all_ok = True
    # softmax (row-normalized) and sigmoid (per-expert) scoring share the same
    # selection / clamp / tail logic, so sweep both across the core grid.
    for scoring_func in ("softmax", "sigmoid"):
        for experts in expert_counts:
            max_topk = min(args.max_topk, experts)
            for renormalize in (True, False):
                all_ok &= run_case(
                    args.tokens, experts, max_topk, renormalize,
                    args.pad_id, device, args.seed, scoring_func=scoring_func,
                )

        # Token counts that are not multiples of WAVES_PER_BLOCK (=4): tail-wave
        # guard, single token, and a >256 token count.
        for tokens in (1, 3, 257, 300):
            all_ok &= run_case(
                tokens, 64, min(args.max_topk, 64), True,
                args.pad_id, device, args.seed, scoring_func=scoring_func,
            )

        # bf16 / fp16 gating input (host up-cast to f32), incl. large E.
        for dtype in (torch.bfloat16, torch.float16):
            all_ok &= run_dtype_case(
                128, 64, min(args.max_topk, 64), dtype, device, args.seed,
                scoring_func=scoring_func,
            )
            all_ok &= run_dtype_case(
                96, 256, min(args.max_topk, 256), dtype, device, args.seed,
                scoring_func=scoring_func,
            )

        # Out-of-range dyna_k (0 and > max_topk): kernel/reference clamp agreement.
        all_ok &= run_clamp_case(
            128, 64, min(args.max_topk, 64), device, args.seed, scoring_func=scoring_func
        )
        all_ok &= run_clamp_case(
            64, 256, min(args.max_topk, 256), device, args.seed, scoring_func=scoring_func
        )

    # Large-batch layout (T >= LARGE_BATCH_TOKENS=16384 -> throughput VPT for
    # E >= 256). Exercises the alternate sub-warp split + grid sizing.
    all_ok &= run_large_batch_case(20000, 256, min(args.max_topk, 256), device, args.seed)
    all_ok &= run_large_batch_case(16384, 512, min(args.max_topk, 512), device, args.seed)

    # fused_topk(..., dyna_k=...) passthrough into the dynamic router.
    all_ok &= run_fused_topk_passthrough(
        128, 64, min(args.max_topk, 64), device, args.seed
    )
    all_ok &= run_fused_topk_passthrough(
        96, 256, min(args.max_topk, 256), device, args.seed
    )

    print("\n=== RESULT:", "PASS" if all_ok else "FAIL", "===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
