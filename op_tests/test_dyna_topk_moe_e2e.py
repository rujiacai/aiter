#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end check: fused_topk(dyna_k, scoring_func) -> fused_moe (seedance).

Validates the dynamic top-k router output (carrying the ``pad_id == num_experts``
sentinel + zero-weight tail) flowing through ``moe_sorting`` + the fused MoE GEMM,
by comparing the HIP/CK ``fused_moe`` kernel against the eager ``torch_moe``
reference (which loops experts ``0..E-1`` and so drops the sentinel exactly like
moe_sorting must).

The kernel (CK 2-stage, QuantType.No bf16 g1u1) consumes shuffle_weight'd
weights; the torch reference uses the plain (unshuffled) weights.

Cases:
  A. static fused_topk softmax            -> harness/layout sanity (no dyna)
  B. fused_topk(dyna_k==topk, softmax)    -> full-k equivalence
  C. fused_topk(dyna_k var, softmax)      -> dynamic drop + pad skip
  D. fused_topk(dyna_k var, sigmoid)      -> sigmoid passthrough through whole pipe
  E. fused_topk(static + sigmoid) guard   -> must raise (sigmoid only on dyna path)
"""
import sys

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter.ops.shuffle import shuffle_weight


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-6)).item()


def run_case(name, input, w1, w2, w1s, w2s, tw, tid, E):
    ref = torch_moe(input, w1, w2, tw, tid)
    out = fused_moe(
        input, w1s, w2s, tw, tid,
        activation=ActivationType.Silu, quant_type=QuantType.No,
    )
    re = rel_err(out, ref)
    max_abs = (out.float() - ref.float()).abs().max().item()
    n_pad = int((tid == E).sum().item())
    ok = re < 0.02
    print(f"  [{name}] pad_slots={n_pad:4d} rel_err={re:.4e} max_abs={max_abs:.3e}"
          f" -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    dev, dtype = "cuda", dtypes.bf16
    M, D, I, E, topk = 128, 512, 256, 64, 6

    input = torch.randn((M, D), dtype=dtype, device=dev)
    w1 = torch.randn((E, I * 2, D), dtype=dtype, device=dev) / 10.0
    w2 = torch.randn((E, D, I), dtype=dtype, device=dev) / 10.0
    w1s, w2s = shuffle_weight(w1), shuffle_weight(w2)
    score = torch.randn((M, E), dtype=dtype, device=dev)
    dyna_full = torch.full((M,), topk, dtype=dtypes.i32, device=dev)
    dyna_var = torch.randint(1, topk + 1, (M,), dtype=dtypes.i32, device=dev)

    print(f"=== dyna_topk -> fused_moe e2e (M={M} D={D} I={I} E={E} topk={topk} bf16) ===")
    ok = True

    tw, tid = fused_topk(input, score, topk, True)
    ok &= run_case("A static softmax          ", input, w1, w2, w1s, w2s, tw, tid, E)

    tw, tid = fused_topk(input, score, topk, True, dyna_k=dyna_full, scoring_func="softmax")
    ok &= run_case("B fused_topk dyna k=topk  ", input, w1, w2, w1s, w2s, tw, tid, E)

    tw, tid = fused_topk(input, score, topk, True, dyna_k=dyna_var, scoring_func="softmax")
    ok &= run_case("C fused_topk dyna var soft", input, w1, w2, w1s, w2s, tw, tid, E)

    tw, tid = fused_topk(input, score, topk, True, dyna_k=dyna_var, scoring_func="sigmoid")
    ok &= run_case("D fused_topk dyna var sig ", input, w1, w2, w1s, w2s, tw, tid, E)

    # E. guard: sigmoid on the static path (no dyna_k) must raise.
    try:
        fused_topk(input, score, topk, True, scoring_func="sigmoid")
        print("  [E guard static+sigmoid    ] no error -> FAIL")
        ok = False
    except (ValueError, RuntimeError) as e:
        print(f"  [E guard static+sigmoid    ] raised ({type(e).__name__}) -> OK")

    print("\n=== RESULT:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
