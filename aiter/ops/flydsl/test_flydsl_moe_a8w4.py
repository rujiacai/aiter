# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""a8w4 stage2 + e2e correctness/perf: mxfp8 (fp8 recast weight + E8M0 per-32 scale).

Stage2 down-proj:  a2(fp8) x w2(mxfp4->fp8 fold) with native fp8 MFMA + per-32 scale.
E2E:  stage1_a8w4 -> quant fp8 -> stage2_a8w4, compared to a bf16 gold reference and
      to a quant-matched (dequant-fp8) stage2 reference (isolates kernel arithmetic).

Weight recast is lossless (E8M0 scales are powers of 2, folded into fp8 exponent);
only fp8-activation quant differs, and that error is already in the dequant reference.
Expect stage2 cos>=0.999; e2e-vs-gold cos limited by 2x fp8 activation quant.
"""
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
    prep_a8w4_weight_scale,
)

FP8 = torch.float8_e4m3fnuz
torch.set_default_device("cuda")


def _prep_a8w4(w_qt, w_scale, E, N, K, G=None):
    """mxfp4 weight -> fp8 (per-group-pair base fold) + per-pair-equal E8M0 scale.

    Thin wrapper over the public ``moe_kernels.prep_a8w4_weight_scale`` (G = K//32
    is computed internally); keeps the ``(E, N, K, G)`` call sites below unchanged.
    """
    return prep_a8w4_weight_scale(w_qt, w_scale, E, N, K)
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
        a_dtype="fp8", b_dtype="mxfp8", out_dtype="bf16",
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
        a_dtype="fp8", b_dtype="mxfp8", out_dtype="bf16", act="silu",
        w1_scale=w1_scale_shuf, a1_scale=a1_scale,
    )
    a2_fp8, a2_scale, a2_dq = _quant_fp8_perslot(s1)
    out = flydsl_moe_stage2(
        a2_fp8, w2_fp8_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="fp8", b_dtype="mxfp8", out_dtype="bf16",
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


def bench(token=8192, model_dim=4096, inter_dim=512, E=256, topk=6,
          block_m=128, tile_n=128, tile_k=128, iters=30):
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
                                 b_dtype="mxfp8", out_dtype="bf16", w2_scale=w2_scale_shuf,
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
                                   b_dtype="mxfp8", out_dtype="bf16", act="silu",
                                   w1_scale=w1s8, a1_scale=a1_scale)
            a2_fp8, a2_scale = _hipq_slots(s1)                      # existing HIP requant (bf16->fp8)
            return flydsl_moe_stage2(a2_fp8, w2f8, sorted_ids, seid, nvi, topk=topk,
                                     tile_m=tile_m, tile_n=256, tile_k=128, a_dtype="fp8",
                                     b_dtype="mxfp8", out_dtype="bf16", w2_scale=w2s8,
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


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    elif "--e2e" in sys.argv:
        test_e2e()
    elif "--sweep" in sys.argv:
        toks = [1, 16, 64, 256, 1024, 4096, 16384]
        if "--full" in sys.argv:
            toks = [2 ** i for i in range(16)]  # 1..32768
        sweep(toks)
    else:
        test_stage2()
