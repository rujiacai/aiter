# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""a8w4 Phase-1 FOLD (real 4-bit): fp8 activation x PACKED 4-bit mxfp4 weight.

Weight is stored PACKED 4-bit e2m1 (0.5B, HALF the HBM of the mxfp8/Phase-0 fp8-recast)
and unpacked e2m1->fp8 IN-KERNEL with a per-pair ratio-fold (b_dtype="mxfp4").
Native fp8 MFMA; per-32 E8M0 base scale applied post-MFMA.

Stage2 down-proj + E2E (stage1 -> quant fp8 -> stage2). Correctness vs a bf16 gold and
a quant-matched (dequant) stage2 reference (isolates kernel arithmetic); FlyDSL-vs-triton
perf. Expect stage2 cos>=0.999; e2e-vs-gold cos limited by 2x fp8 activation quant.
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_flydsl_moe_a16w4 as T  # noqa: E402
from aiter import QuantType, ActivationType, dtypes  # noqa: E402
from aiter.fused_moe import moe_sorting, torch_moe_stage1, torch_moe_stage2  # noqa: E402
from aiter.ops.quant import get_hip_quant  # noqa: E402  (optimized HIP per-token fp8 quant)
from aiter.ops.flydsl.moe_kernels import (  # noqa: E402
    flydsl_moe_stage1,
    flydsl_moe_stage2,
    prep_a8w4_w4,
    prep_a8w4_w4_aligned,
)

FP8 = torch.float8_e4m3fnuz
torch.set_default_device("cuda")

# The fold and aligned kernels read *different* weight layouts and neither validates
# the other's, so a host/kernel mismatch is silently wrong output rather than an
# error -- derive the host prep from the same env var the kernel dispatches on.
A8W4_ALIGNED = os.environ.get("AITER_A8W4_ALIGNED", "0") == "1"


def _prep_a8w4(w_qt, w_scale, E, N, K, G=None):
    """mxfp4 weight -> PACKED 4-bit e2m1 (0.5B) + raw per-32 E8M0 bf16 scale.

    Thin wrapper over ``moe_kernels.prep_a8w4_w4`` (E and G are inferred internally);
    keeps the ``(E, N, K, G)`` call sites below unchanged. The kernel unpacks e2m1->fp8
    and does the per-pair ratio-fold in-kernel (paired with b_dtype="mxfp4").

    Under ``AITER_A8W4_ALIGNED=1`` the weight instead goes out in the
    shuffle_weight_NK(16,32) layout, where one K32 MFMA operand covers exactly one
    per-32 scale block and the kernel skips the in-kernel fold.
    """
    prep = prep_a8w4_w4_aligned if A8W4_ALIGNED else prep_a8w4_w4
    return prep(w_qt, w_scale, N, K)
_HIPQ = get_hip_quant(QuantType.per_Token)  # bf16 -> fp8 e4m3fnuz, per-token scale=amax/240


def _hipq_tokens(x):
    """existing HIP quant: x (token, model) bf16 -> (x_fp8, scale_flat[token] f32)."""
    y, s = _HIPQ(x, quant_dtype=dtypes.fp8)
    return y, s.view(-1).contiguous().float()


def _hipq_slots(x):
    """existing HIP quant: x (token, topk, inter) bf16 -> (x_fp8, scale_flat[token*topk] f32)."""
    t, k, i = x.shape
    y, s = _HIPQ(x.reshape(t * k, i), quant_dtype=dtypes.fp8)
    return y.view(t, k, i), s.view(-1).contiguous().float()


def _quant_fp8_perslot(x):
    """x (token, topk, inter) bf16 -> (x_fp8, scale_flat[token*topk], x_dq bf16). (torch ref)"""
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 240.0
    x_fp8 = (x.float() / scale).to(FP8)
    x_dq = (x_fp8.float() * scale).to(torch.bfloat16)
    return x_fp8, scale.view(-1).contiguous().float(), x_dq


def _quant_fp8_pertoken(x):
    """x (token, model) bf16 -> (x_fp8, scale_flat[token], x_dq bf16). (torch ref)"""
    amax = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / 240.0
    x_fp8 = (x.float() / scale).to(FP8)
    x_dq = (x_fp8.float() * scale).to(torch.bfloat16)
    return x_fp8, scale.view(-1).contiguous().float(), x_dq


def test_stage2(token=64, model_dim=512, inter_dim=256, E=8, topk=2,
                block_m=32, tile_n=128, tile_k=128):
    print(f"\n[a8w4-stage2] token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk} "
          f"tile=({block_m},{tile_n},{tile_k})")
    d = T._gen(token, model_dim, inter_dim, E, topk)
    a2 = torch.randn((token, topk, inter_dim), dtype=torch.bfloat16) / 10
    a2_fp8, a2_scale, a2_dq = _quant_fp8_perslot(a2)

    # reference: fp8-dequant activation x mxfp4-dequant weight, with topk weight.
    ref = torch_moe_stage2(
        a2_dq, torch.zeros((E, inter_dim * 2, model_dim), dtype=torch.bfloat16),
        d["w2_dq"], d["topk_weights"], d["topk_ids"],
        dtype=torch.bfloat16, quant_type=QuantType.No, doweight=True,
    )

    G = inter_dim // 32
    w2_fp8_shuf, w2_scale_shuf = _prep_a8w4(
        d["w2_qt"], d["w2_scale"], E, model_dim, inter_dim, G
    )

    sorted_ids, sw, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, block_m
    )
    out = flydsl_moe_stage2(
        a2_fp8, w2_fp8_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="fp8", b_dtype="mxfp4", out_dtype="bf16",
        w2_scale=w2_scale_shuf, a2_scale=a2_scale, sorted_weights=sw,
    )
    torch.cuda.synchronize()
    return T._check(ref, out, "a8w4-stage2")


def test_e2e(token=64, model_dim=512, inter_dim=256, E=8, topk=2,
             block_m=32, tile_n=128, tile_k=128):
    print(f"\n[a8w4-e2e] token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk} "
          f"tile=({block_m},{tile_n},{tile_k})")
    d = T._gen(token, model_dim, inter_dim, E, topk)

    # ---- host prep: stage1 (a1 fp8 + w1 fold), stage2 (w2 fold) ----
    a1_fp8, a1_scale, a1_dq = _quant_fp8_pertoken(d["inp"])
    Ns1 = inter_dim * 2
    Gs1 = model_dim // 32
    w1_fp8_shuf, w1_scale_shuf = _prep_a8w4(d["w1_qt"], d["w1_scale"], E, Ns1, model_dim, Gs1)
    Gs2 = inter_dim // 32
    w2_fp8_shuf, w2_scale_shuf = _prep_a8w4(d["w2_qt"], d["w2_scale"], E, model_dim, inter_dim, Gs2)

    sorted_ids, sw, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, block_m
    )

    # ---- kernel e2e: stage1 -> quant fp8 -> stage2 ----
    s1 = flydsl_moe_stage1(
        a1_fp8, w1_fp8_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="fp8", b_dtype="mxfp4", out_dtype="bf16", act="silu",
        w1_scale=w1_scale_shuf, a1_scale=a1_scale,
    )
    a2_fp8, a2_scale, a2_dq = _quant_fp8_perslot(s1)
    out = flydsl_moe_stage2(
        a2_fp8, w2_fp8_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="fp8", b_dtype="mxfp4", out_dtype="bf16",
        w2_scale=w2_scale_shuf, a2_scale=a2_scale, sorted_weights=sw,
    )
    torch.cuda.synchronize()

    # ---- ref A: bf16 gold (no fp8 quant anywhere) ----
    r1 = torch_moe_stage1(
        d["inp"], d["w1_dq"], d["w2_dq"], d["topk_weights"], d["topk_ids"],
        dtype=torch.bfloat16, activation=ActivationType.Silu, quant_type=QuantType.No,
    )
    gold = torch_moe_stage2(
        r1, torch.zeros((E, inter_dim * 2, model_dim), dtype=torch.bfloat16),
        d["w2_dq"], d["topk_weights"], d["topk_ids"],
        dtype=torch.bfloat16, quant_type=QuantType.No, doweight=True,
    )
    # ---- ref B: quant-matched stage2 (feed kernel s1's dequant-fp8 into torch stage2) ----
    refB = torch_moe_stage2(
        a2_dq, torch.zeros((E, inter_dim * 2, model_dim), dtype=torch.bfloat16),
        d["w2_dq"], d["topk_weights"], d["topk_ids"],
        dtype=torch.bfloat16, quant_type=QuantType.No, doweight=True,
    )
    p_gold = T._check(gold, out, "a8w4-e2e vs bf16-gold")
    p_km = T._check(refB, out, "a8w4-e2e vs quant-matched-stage2")
    return p_gold, p_km


def bench(token=8192, model_dim=7168, inter_dim=384, E=384, topk=6,
          block_m=128, tile_n=128, tile_k=128, iters=50):
    """Time a8w4 (fp8 MFMA) stage2 vs a16w4 (bf16 MFMA) stage2. Timing valid regardless of cos."""
    print(f"\n[bench-stage2] token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk} "
          f"tile=({block_m},{tile_n},{tile_k})")
    d = T._gen(token, model_dim, inter_dim, E, topk)
    a2 = torch.randn((token, topk, inter_dim), dtype=torch.bfloat16) / 10
    a2_fp8, a2_scale, _ = _quant_fp8_perslot(a2)
    G = inter_dim // 32
    w2_fp8_shuf, w2_scale_shuf = _prep_a8w4(d["w2_qt"], d["w2_scale"], E, model_dim, inter_dim, G)
    # a16w4 setup
    w2_w4 = T._prep_weight_for_kernel(d["w2_qt"], model_dim, inter_dim)
    w2s_a16 = T._prep_scale_for_kernel(d["w2_scale"], model_dim, inter_dim, 1.0)
    sorted_ids, sw, seid, nvi, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, block_m
    )

    def a8w4():
        return flydsl_moe_stage2(a2_fp8, w2_fp8_shuf, sorted_ids, seid, nvi, topk=topk,
                                 tile_m=block_m, tile_n=tile_n, tile_k=tile_k, a_dtype="fp8",
                                 b_dtype="mxfp4", out_dtype="bf16", w2_scale=w2_scale_shuf,
                                 a2_scale=a2_scale, sorted_weights=sw)

    def a16w4():
        return flydsl_moe_stage2(a2, w2_w4, sorted_ids, seid, nvi, topk=topk,
                                 tile_m=block_m, tile_n=tile_n, tile_k=tile_k, a_dtype="bf16",
                                 b_dtype="mxfp4", out_dtype="bf16", w2_scale=w2s_a16,
                                 sorted_weights=sw)

    a8w4(); a16w4(); torch.cuda.synchronize()
    t_a8 = T._time_cuda(a8w4, iters=iters)
    t_a16 = T._time_cuda(a16w4, iters=iters)
    print(f"  a8w4  (fp8 MFMA) stage2 : {t_a8:8.1f} us")
    print(f"  a16w4 (bf16 MFMA) stage2: {t_a16:8.1f} us")
    print(f"  speedup (a16w4/a8w4)    : {t_a16 / t_a8:.2f}x  (>1 => fp8 faster)")


def _fmt(us):
    return f"{us:8.1f}" if us is not None else "     n/a"


def _cos(ref, test):
    r = ref.float().reshape(-1)
    t = test.float().reshape(-1)
    m = r.abs() > 1e-4
    if not m.any():
        return float("nan")
    return torch.corrcoef(torch.stack([r[m], t[m]]))[0, 1].item()


def sweep(tokens, model_dim=4096, inter_dim=512, E=256, topk=6, iters=50,
          with_triton=True, acc_max_token=32768):
    """e2e sweep: ONE shared bf16 input for a8w4 / a16w4 / triton.

    Fair comparison (all start from the same bf16 activation):
      - a8w4 : (HIP per-token fp8 quant of inp) -> stage1 -> (HIP fp8 requant of s1) -> stage2.
               The fp8 quant is the EXISTING optimized aiter kernel and is INCLUDED in a8w4's time.
      - a16w4: stage1(bf16) -> stage2(bf16). No quant.
      - triton: moe_gemm_a16w4 x2 (bf16). No quant. NOTE: triton uses its OWN routing, so its
               output is NOT element-wise comparable to FlyDSL; triton cos is vs its OWN bf16
               golden (moe_gemm_torch), FlyDSL cos is vs the bf16 full-precision torch golden.
    Golden = bf16 full precision (dequantized weights, bf16 activation) -- the ground truth.
    """
    tri_ok = with_triton
    if with_triton:
        try:
            from aiter.ops.triton.moe.moe_routing.routing import routing
            from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4, moe_gemm_torch
            from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp
            from aiter.ops.triton.utils.types import str_to_torch_dtype
        except Exception as e:  # noqa: BLE001
            print(f"  [triton unavailable: {e}]")
            tri_ok = False

    print(f"\n{'='*118}")
    print(f"e2e sweep  dim=({model_dim},{inter_dim}) E={E} topk={topk}  "
          f"a8w4={'aligned' if A8W4_ALIGNED else 'fold'}  "
          f"[shared bf16 input; a8w4 time INCLUDES HIP fp8 quant]")
    print(f"{'='*118}")
    hdr = (f"{'token':>7} {'tile':>9} | {'a8w4_e2e':>9} {'a16_e2e':>8} {'tri_e2e':>8} | "
           f"{'vs_a16':>6} {'vs_tri':>6} | {'a8_cos':>7} {'a16_cos':>7} {'tri_cos':>7}")
    print(hdr)
    print("-" * 118)

    rows = []
    for token in tokens:
        tile_m, tile_k = T._adaptive_tile_a16w4(token, topk, E)
        d = T._gen(token, model_dim, inter_dim, E, topk)
        inp = d["inp"]
        sorted_ids, sw, seid, nvi, _ = moe_sorting(
            d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, tile_m
        )

        # ---- a8w4 setup (weights pre-folded; activation quant is INSIDE the timed chain) ----
        w1f8, w1s8 = _prep_a8w4(d["w1_qt"], d["w1_scale"], E, inter_dim * 2, model_dim, model_dim // 32)
        w2f8, w2s8 = _prep_a8w4(d["w2_qt"], d["w2_scale"], E, model_dim, inter_dim, inter_dim // 32)

        def a8_e2e():
            a1_fp8, a1_scale = _hipq_tokens(inp)                    # existing HIP quant (bf16->fp8)
            s1 = flydsl_moe_stage1(a1_fp8, w1f8, sorted_ids, seid, nvi, topk=topk,
                                   tile_m=tile_m, tile_n=128, tile_k=128, a_dtype="fp8",
                                   b_dtype="mxfp4", out_dtype="bf16", act="silu",
                                   w1_scale=w1s8, a1_scale=a1_scale)
            a2_fp8, a2_scale = _hipq_slots(s1)                      # existing HIP requant (bf16->fp8)
            return flydsl_moe_stage2(a2_fp8, w2f8, sorted_ids, seid, nvi, topk=topk,
                                     tile_m=tile_m, tile_n=256, tile_k=128, a_dtype="fp8",
                                     b_dtype="mxfp4", out_dtype="bf16", w2_scale=w2s8,
                                     a2_scale=a2_scale, sorted_weights=sw)

        # ---- a16w4 setup (bf16 activation, no quant) ----
        w1w4 = T._prep_weight_for_kernel(d["w1_qt"], inter_dim * 2, model_dim)
        w1s4 = T._prep_scale_for_kernel(d["w1_scale"], inter_dim * 2, model_dim, 1.0)
        w2w4 = T._prep_weight_for_kernel(d["w2_qt"], model_dim, inter_dim)
        w2s4 = T._prep_scale_for_kernel(d["w2_scale"], model_dim, inter_dim, 1.0)

        def a16_e2e():
            s1 = flydsl_moe_stage1(inp, w1w4, sorted_ids, seid, nvi, topk=topk,
                                   tile_m=tile_m, tile_n=128, tile_k=tile_k, a_dtype="bf16",
                                   b_dtype="mxfp4", out_dtype="bf16", act="silu", w1_scale=w1s4)
            return flydsl_moe_stage2(s1, w2w4, sorted_ids, seid, nvi, topk=topk,
                                     tile_m=tile_m, tile_n=256, tile_k=tile_k, a_dtype="bf16",
                                     b_dtype="mxfp4", out_dtype="bf16", w2_scale=w2s4,
                                     sorted_weights=sw)

        # warmup + time flydsl
        for _ in range(3):
            a8_e2e(); a16_e2e()
        torch.cuda.synchronize()
        t_a8 = T._time_cuda(a8_e2e, iters=iters)
        t_a16 = T._time_cuda(a16_e2e, iters=iters)

        # ---- triton (bf16 input, own routing) ----
        t_tri = tri_cos = None
        tri_e2e = None
        if tri_ok:
            try:
                dev = "cuda"
                logits = torch.randn((token, E), dtype=torch.float16, device=dev)
                rdata, gindx, sindx = routing(logits, topk)
                gammas = rdata.gate_scal.to(torch.float32) if rdata.gate_scal is not None else None
                wdt = str_to_torch_dtype["mxfp4_e2m1"]
                w1t = d["w1"].transpose(1, 2).contiguous()
                w2t = d["w2"].transpose(1, 2).contiguous()
                w1_tri, w1s_tri = downcast_to_mxfp(w1t, wdt, axis=1)
                w2_tri, w2s_tri = downcast_to_mxfp(w2t, wdt, axis=1)
                b1 = torch.zeros((E, inter_dim * 2), dtype=torch.float32, device=dev)
                b2 = torch.zeros((E, model_dim), dtype=torch.float32, device=dev)

                def tri_e2e():
                    h = moe_gemm_a16w4(inp, w1_tri, None, w1s_tri, None, None, b1,
                                       rdata, gindx, None, None, None, torch.bfloat16, True)
                    return moe_gemm_a16w4(h, w2_tri, None, w2s_tri, None, None, b2,
                                          rdata, None, sindx, gammas, None, torch.bfloat16, False)

                for _ in range(3):
                    tri_e2e()
                torch.cuda.synchronize()
                t_tri = T._time_cuda(tri_e2e, iters=iters)
            except Exception as e:  # noqa: BLE001
                print(f"  [triton token={token} failed: {e}]")
                tri_e2e = None

        # ---- accuracy vs bf16 full-precision golden (only up to acc_max_token; heavy) ----
        a8_cos = a16_cos = None
        if token <= acc_max_token:
          try:
            r1 = torch_moe_stage1(inp, d["w1_dq"], d["w2_dq"], d["topk_weights"], d["topk_ids"],
                                  dtype=torch.bfloat16, activation=ActivationType.Silu,
                                  quant_type=QuantType.No)
            gold = torch_moe_stage2(
                r1, torch.zeros((E, inter_dim * 2, model_dim), dtype=torch.bfloat16),
                d["w2_dq"], d["topk_weights"], d["topk_ids"],
                dtype=torch.bfloat16, quant_type=QuantType.No, doweight=True)
            a8_cos = _cos(gold, a8_e2e())
            a16_cos = _cos(gold, a16_e2e())
            del r1, gold
            torch.cuda.empty_cache()
          except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
            print(f"  [golden token={token} OOM -> cos n/a: {e}]")
            torch.cuda.empty_cache()
          if token <= acc_max_token and tri_e2e is not None:
                # triton vs its OWN routing's bf16 golden (moe_gemm_torch, dequant weights)
                try:
                    w1r = upcast_from_mxfp(w1_tri, w1s_tri, torch.bfloat16, axis=1)
                    w2r = upcast_from_mxfp(w2_tri, w2s_tri, torch.bfloat16, axis=1)
                    hg = moe_gemm_torch(inp, w1r, b1, rdata, gindx, None, None, True)
                    trg = moe_gemm_torch(hg, w2r, b2, rdata, None, sindx, gammas, False)
                    tri_cos = _cos(trg, tri_e2e())
                except Exception as e:  # noqa: BLE001
                    print(f"  [triton golden token={token} failed: {e}]")

        vs_a16 = t_a16 / t_a8
        vs_tri = (t_tri / t_a8) if t_tri else None
        print(f"{token:>7} {f'{tile_m}x{tile_k}':>9} | {_fmt(t_a8)} {_fmt(t_a16)} {_fmt(t_tri)} | "
              f"{vs_a16:5.2f}x {(f'{vs_tri:4.2f}x' if vs_tri else '  n/a'):>6} | "
              f"{(f'{a8_cos:.4f}' if a8_cos is not None else '   n/a'):>7} "
              f"{(f'{a16_cos:.4f}' if a16_cos is not None else '   n/a'):>7} "
              f"{(f'{tri_cos:.4f}' if tri_cos is not None else '   n/a'):>7}")
        rows.append(dict(token=token, a8_e2e=t_a8, a16_e2e=t_a16, tri_e2e=t_tri,
                         vs_a16=vs_a16, vs_tri=vs_tri, a8_cos=a8_cos, a16_cos=a16_cos, tri_cos=tri_cos))
    print("-" * 118)
    print("e2e us/iter (lower=better). vs_a16/vs_tri = a8w4 speedup (>1 => a8w4 faster; a8w4 time INCLUDES fp8 quant).")
    print("a8_cos/a16_cos vs bf16 full-precision golden (FlyDSL routing). tri_cos vs triton's own bf16 golden (own routing).")
    return rows


def compare_e2e(token, model_dim, inter_dim, E, topk, block_m=None, iters=50):
    """FlyDSL a8w4 (fp8 act × mxfp4->fp8 weight) vs triton a16w4 (bf16 act): stage1 / stage2 / e2e.

    Mirror of test_flydsl_moe_a16w4.compare_e2e, but the FlyDSL side is the a8w4
    (fp8-activation) path. FlyDSL's time INCLUDES the HIP per-token fp8 activation
    quant (triton bf16 has none) -- the fair e2e cost. triton has no fp8-act MoE, so
    it runs bf16-act moe_gemm_a16w4 on the same mxfp4 weights with its OWN routing
    (latency comparable; outputs NOT element-wise comparable). Reports per-iter us +
    e2e speedup, in the same format as the a16w4 test.
    """
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
    from aiter.ops.triton.utils.types import str_to_torch_dtype

    tile_m = block_m or T._adaptive_tile_a16w4(token, topk, E)[0]
    tile_k = 128  # a8w4 fp8 path uses tile_k=128
    print(f"\n[a8w4-e2e] token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk}  "
          f"tile_m={tile_m} tile_k={tile_k}")
    d = T._gen(token, model_dim, inter_dim, E, topk)

    # ---- FlyDSL a8w4 (fp8 act; weights mxfp4->fp8 fold) ----
    w1f8, w1s8 = _prep_a8w4(d["w1_qt"], d["w1_scale"], E, inter_dim * 2, model_dim, model_dim // 32)
    w2f8, w2s8 = _prep_a8w4(d["w2_qt"], d["w2_scale"], E, model_dim, inter_dim, inter_dim // 32)
    sorted_ids, sw, seid, nvi, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, tile_m
    )
    s1_kw = dict(topk=topk, tile_m=tile_m, tile_n=128, tile_k=tile_k, a_dtype="fp8",
                 b_dtype="mxfp4", out_dtype="bf16", act="silu", w1_scale=w1s8)
    s2_kw = dict(topk=topk, tile_m=tile_m, tile_n=256, tile_k=tile_k, a_dtype="fp8",
                 b_dtype="mxfp4", out_dtype="bf16", w2_scale=w2s8, sorted_weights=sw)
    a2 = torch.randn((token, topk, inter_dim), dtype=torch.bfloat16) / 10  # standalone stage2 input

    def fly_s1():
        a1_fp8, a1_scale = _hipq_tokens(d["inp"])
        return flydsl_moe_stage1(a1_fp8, w1f8, sorted_ids, seid, nvi, a1_scale=a1_scale, **s1_kw)

    def fly_s2():
        a2_fp8, a2_scale = _hipq_slots(a2)
        return flydsl_moe_stage2(a2_fp8, w2f8, sorted_ids, seid, nvi, a2_scale=a2_scale, **s2_kw)

    def fly_e2e():
        a1_fp8, a1_scale = _hipq_tokens(d["inp"])
        s1 = flydsl_moe_stage1(a1_fp8, w1f8, sorted_ids, seid, nvi, a1_scale=a1_scale, **s1_kw)
        a2_fp8, a2_scale = _hipq_slots(s1)
        return flydsl_moe_stage2(a2_fp8, w2f8, sorted_ids, seid, nvi, a2_scale=a2_scale, **s2_kw)

    # ---- triton (both stages = moe_gemm_a16w4, bf16 act, own routing) ----
    dev = "cuda"
    logits = torch.randn((token, E), dtype=torch.float16, device=dev)
    rdata, gindx, sindx = routing(logits, topk)
    gammas = rdata.gate_scal.to(torch.float32) if rdata.gate_scal is not None else None
    x_tri = d["inp"]
    wdt = str_to_torch_dtype["mxfp4_e2m1"]
    w1_tri, w1s_tri = downcast_to_mxfp(d["w1"].transpose(1, 2).contiguous(), wdt, axis=1)
    w2_tri, w2s_tri = downcast_to_mxfp(d["w2"].transpose(1, 2).contiguous(), wdt, axis=1)
    b1 = torch.zeros((E, inter_dim * 2), dtype=torch.float32, device=dev)
    b2 = torch.zeros((E, model_dim), dtype=torch.float32, device=dev)
    h_tri = torch.randn((token * topk, inter_dim), dtype=torch.bfloat16, device=dev)

    def tri_s1():
        return moe_gemm_a16w4(x_tri, w1_tri, None, w1s_tri, None, None, b1,
                              rdata, gindx, None, None, None, torch.bfloat16, True)

    def tri_s2():
        return moe_gemm_a16w4(h_tri, w2_tri, None, w2s_tri, None, None, b2,
                              rdata, None, sindx, gammas, None, torch.bfloat16, False)

    def tri_e2e():
        h = moe_gemm_a16w4(x_tri, w1_tri, None, w1s_tri, None, None, b1,
                           rdata, gindx, None, None, None, torch.bfloat16, True)
        return moe_gemm_a16w4(h, w2_tri, None, w2s_tri, None, None, b2,
                              rdata, None, sindx, gammas, None, torch.bfloat16, False)

    fo = fly_e2e()
    to = tri_e2e()
    torch.cuda.synchronize()
    assert fo.shape == (token, model_dim), f"flydsl e2e {tuple(fo.shape)}"
    assert to.shape == (token, model_dim), f"triton e2e {tuple(to.shape)}"
    assert torch.isfinite(fo).all() and torch.isfinite(to).all(), "non-finite e2e output"

    fly = {k: T._time_cuda(fn, iters=iters) for k, fn in
           (("s1", fly_s1), ("s2", fly_s2), ("e2e", fly_e2e))}
    tri = {k: T._time_cuda(fn, iters=iters) for k, fn in
           (("s1", tri_s1), ("s2", tri_s2), ("e2e", tri_e2e))}
    e2e_speedup = tri["e2e"] / fly["e2e"]
    print(f"  stage1  FlyDSL {fly['s1']:8.1f} us | triton {tri['s1']:8.1f} us")
    print(f"  stage2  FlyDSL {fly['s2']:8.1f} us | triton {tri['s2']:8.1f} us")
    print(f"  e2e     FlyDSL {fly['e2e']:8.1f} us | triton {tri['e2e']:8.1f} us "
          f"| speedup(triton/flydsl) {e2e_speedup:.2f}x")
    return fly, tri


def main():
    # CLI mirrors test_flydsl_moe_a16w4.main (same flags / default shape) so the
    # a8w4-vs-triton and a16w4-vs-triton comparisons are driven identically.
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage2", "e2e", "all"], default="all")
    ap.add_argument("-t", "--tokens", type=int, action="append", default=None)
    ap.add_argument("--model-dim", type=int, default=7168)
    ap.add_argument("--inter-dim", type=int, default=384)
    ap.add_argument("-E", "--experts", type=int, default=384)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--compare-triton", action="store_true",
                    help="a8w4(fp8) vs triton a16w4, stage1 only")
    ap.add_argument("--compare-e2e", action="store_true",
                    help="a8w4(fp8) vs triton a16w4, stage1 / stage2 / end-to-end")
    ap.add_argument("--bench", action="store_true", help="a8w4 vs a16w4 stage2 timing")
    ap.add_argument("--sweep", action="store_true", help="a8w4/a16w4/triton e2e sweep")
    ap.add_argument("--full", action="store_true", help="sweep 1..32768")
    args = ap.parse_args()
    tokens = args.tokens or [16, 128, 1024]

    if args.compare_e2e:
        print("\nFlyDSL a8w4 vs triton a16w4 stage1 / stage2 / e2e performance:")
        rows = []
        for tk in tokens:
            fly, tri = compare_e2e(tk, args.model_dim, args.inter_dim, args.experts, args.topk)
            rows.append((tk, fly, tri))
        print("\n" + "=" * 96)
        print(f"  {'token':>6} | {'fly_s1':>9} {'tri_s1':>9} | {'fly_s2':>9} {'tri_s2':>9} "
              f"| {'fly_e2e':>9} {'tri_e2e':>9} {'e2e spd':>8}")
        print("  " + "-" * 92)
        for tk, fly, tri in rows:
            spd = tri["e2e"] / fly["e2e"]
            print(f"  {tk:>6} | {fly['s1']:>9.1f} {tri['s1']:>9.1f} | "
                  f"{fly['s2']:>9.1f} {tri['s2']:>9.1f} | "
                  f"{fly['e2e']:>9.1f} {tri['e2e']:>9.1f} {spd:>7.2f}x")
        print("=" * 96)
        sys.exit(0)

    if args.compare_triton:
        print("\nFlyDSL a8w4 vs triton a16w4 (stage1) performance:")
        rows = []
        for tk in tokens:
            fly, tri = compare_e2e(tk, args.model_dim, args.inter_dim, args.experts, args.topk)
            rows.append((tk, fly["s1"], tri["s1"], tri["s1"] / fly["s1"]))
        print("\n" + "=" * 60)
        print(f"  {'token':>6} {'flydsl(us)':>12} {'triton(us)':>12} {'speedup':>9}")
        for tk, f, t, s in rows:
            print(f"  {tk:>6} {f:>12.1f} {t:>12.1f} {s:>8.2f}x")
        print("=" * 60)
        sys.exit(0)

    if args.sweep:
        toks = args.tokens or [1, 16, 64, 256, 1024, 4096, 16384]
        if args.full:
            toks = [2 ** i for i in range(16)]  # 1..32768
        sweep(toks, args.model_dim, args.inter_dim, args.experts, args.topk)
        sys.exit(0)

    if args.bench:
        bench(model_dim=args.model_dim, inter_dim=args.inter_dim, E=args.experts, topk=args.topk)
        sys.exit(0)

    # correctness (cos)
    stages = ["stage2", "e2e"] if args.stage == "all" else [args.stage]
    results = []
    for tk in tokens:
        for st in stages:
            if st == "stage2":
                ok = test_stage2(tk, args.model_dim, args.inter_dim, args.experts, args.topk)
            else:
                pg, pk = test_e2e(tk, args.model_dim, args.inter_dim, args.experts, args.topk)
                ok = pg and pk
            results.append((tk, st, ok))
    print("\n" + "=" * 60 + "\nSUMMARY")
    for tk, st, ok in results:
        print(f"  token={tk:5d} {st:8s} {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, _, ok in results)
    print("=" * 60)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
