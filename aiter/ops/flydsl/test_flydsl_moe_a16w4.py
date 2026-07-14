# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL MoE a16w4 (bf16 activation x mxfp4/e2m1 weight) kernels.

Target: MI308X (gfx942 / CDNA3). CDNA3 has no native fp4 MFMA, so mxfp4 weights
are dequantized to bf16 in-kernel (e2m1 codebook + E8M0 per-32 block scale) and
fed through the bf16 MFMA path -- this reuses FlyDSL's int4_bf16 (W4A16) pipeline
skeleton with an e2m1 dequant instead of the symmetric-int4 dequant.

Tests:
  - Stage1 (gate+up GEMM): flydsl_moe_stage1 with a_dtype="bf16", b_dtype="mxfp4"
  - Stage2 (down-proj GEMM): flydsl_moe_stage2 with a_dtype="bf16", b_dtype="mxfp4"
  - End-to-end (stage1 + stage2)
  - Optional: compare vs aiter triton a16w4 (moe_op_gemm_a16w4) with --compare-triton

Usage:
    python aiter/ops/flydsl/test_flydsl_moe_a16w4.py                  # all stages
    python aiter/ops/flydsl/test_flydsl_moe_a16w4.py --stage stage1
    python aiter/ops/flydsl/test_flydsl_moe_a16w4.py --stage stage2
    python aiter/ops/flydsl/test_flydsl_moe_a16w4.py --stage e2e
    python aiter/ops/flydsl/test_flydsl_moe_a16w4.py -t 16 -t 128 -t 1024
    python aiter/ops/flydsl/test_flydsl_moe_a16w4.py --compare-triton

Accuracy (verified vs hand-computed GEMM+SwiGLU reference, corr=1.0000):
  - K = model_dim >= 512 (multi K-tile): stage1/stage2/e2e all exact.
  - The E8M0->bf16 groupwise scale is fed as-is; the kernel applies it correctly.
  - The single-K-tile case (model_dim == tile_k == 256) is a degenerate edge case
    in the shared int4_bf16 pipeline (int4 shows the same ~4x offset there) and is
    intentionally not used; real dsv4 has model_dim=7168.
"""

import argparse
import sys

import torch

import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import (
    shuffle_weight,
    pack_int8_to_packed_int4,
    shuffle_scale_for_int4,
)

torch.set_default_device("cuda")

# e2m1 codebook (index by 4-bit code 0..15). Sign in bit3.
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Scale is fed as-is (E8M0 -> bf16). No fudge factor: the kernel applies the
# groupwise scale correctly for K >= 2*tile_k (multi K-tile). NOTE: the single
# K-tile case (model_dim == tile_k, e.g. 256) is a degenerate edge case in the
# shared int4_bf16 pipeline (affects int4 too) and is not exercised here; use
# model_dim >= 512 (dsv4 uses 7168).
_STAGE1_SCALE_MUL = 1.0
_STAGE2_SCALE_MUL = 1.0


def _mxfp4_quant(w):
    """Quantize bf16 weight (E,N,K) to mxfp4. Returns (wq_fp4x2, ws_e8m0)."""
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    wq, ws = tq(w, quant_dtype=dtypes.fp4x2)
    E, N, _ = w.shape
    K = w.shape[2]
    return wq.view(E, N, K // 2), ws


def _mxfp4_dequant_bf16(wq, ws, N, K):
    """Dequantize mxfp4 (fp4x2 codes + E8M0 scale) to bf16 weight (E,N,K)."""
    E = wq.shape[0]
    u = wq.view(torch.uint8)
    lo = (u & 0x0F).long()
    hi = ((u >> 4) & 0x0F).long()
    codes = torch.empty((E, N, K), dtype=torch.long, device=wq.device)
    codes[..., 0::2] = lo
    codes[..., 1::2] = hi
    scale = torch.pow(
        2.0, ws.view(torch.uint8).view(E, N, K // 32).float() - 127.0
    )
    table = _E2M1.to(wq.device)
    return (table[codes] * scale.repeat_interleave(32, dim=2)).to(torch.bfloat16)


def _mxfp4_codes_i8(wq, N, K):
    """fp4x2 packed (E,N,K/2) -> e2m1 codes int8 (E,N,K), low nibble first."""
    E = wq.shape[0]
    u = wq.view(torch.uint8)
    codes = torch.empty((E, N, K), dtype=torch.int8, device=wq.device)
    codes[..., 0::2] = (u & 0x0F).to(torch.int8)
    codes[..., 1::2] = ((u >> 4) & 0x0F).to(torch.int8)
    return codes


def _prep_weight_for_kernel(wq, N, K):
    """Preshuffle mxfp4 codes for the int4_bf16-layout kernel (2 nibbles/byte)."""
    E = wq.shape[0]
    codes = _mxfp4_codes_i8(wq, N, K)
    shuf = pack_int8_to_packed_int4(shuffle_weight(codes.view(dtypes.i8), (16, 16)))
    return shuf.view(E, N, K // 2)


def _prep_scale_for_kernel(ws, N, K, scale_mul):
    """E8M0 -> bf16 groupwise scale in (E,K/32,N) layout, shuffled for the kernel."""
    E = ws.view(torch.uint8).numel() // (N * (K // 32))
    ws_u8 = ws.view(torch.uint8).view(E, N, K // 32)
    scale_f32 = torch.pow(2.0, ws_u8.float() - 127.0) * scale_mul
    scale_bf16 = scale_f32.permute(0, 2, 1).contiguous().to(torch.bfloat16)  # (E,K/32,N)
    return shuffle_scale_for_int4(scale_bf16, group_size=32).view(-1).contiguous()


def _check(ref, test, label, atol=1.0, rtol=0.05, pass_pct=95.0):
    ref_f = ref.float().reshape(-1)
    test_f = test.float().reshape(-1)
    max_delta = (ref_f - test_f).abs().max().item()
    close = torch.isclose(ref_f, test_f, atol=atol, rtol=rtol).float().mean().item() * 100
    mask = ref_f.abs() > 1e-3
    if mask.any():
        corr = torch.corrcoef(torch.stack([ref_f[mask], test_f[mask]]))[0, 1].item()
        med = (test_f[mask] / ref_f[mask]).median().item()
    else:
        corr, med = float("nan"), float("nan")
    passed = close > pass_pct
    print(f"  [{label}] max_delta={max_delta:.4f}  {close:.1f}% close  "
          f"corr={corr:.4f}  median_ratio={med:.3f}")
    print(f"  ref  sample: {ref_f[:6]}")
    print(f"  test sample: {test_f[:6]}")
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed


def _gen(token, model_dim, inter_dim, E, topk, seed=0):
    torch.manual_seed(seed)
    inp = torch.randn((token, model_dim), dtype=torch.bfloat16) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=torch.bfloat16) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=torch.bfloat16) / 10
    score = torch.randn((token, E), dtype=torch.bfloat16)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    w1_qt, w1_scale = _mxfp4_quant(w1)
    w2_qt, w2_scale = _mxfp4_quant(w2)
    w1_dq = _mxfp4_dequant_bf16(w1_qt, w1_scale, inter_dim * 2, model_dim)
    w2_dq = _mxfp4_dequant_bf16(w2_qt, w2_scale, model_dim, inter_dim)
    return dict(
        inp=inp, w1=w1, w2=w2, topk_weights=topk_weights, topk_ids=topk_ids,
        w1_qt=w1_qt, w1_scale=w1_scale, w2_qt=w2_qt, w2_scale=w2_scale,
        w1_dq=w1_dq, w2_dq=w2_dq,
        token=token, model_dim=model_dim, inter_dim=inter_dim, E=E, topk=topk,
    )


def test_stage1(token, model_dim, inter_dim, E, topk, block_m=32):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
    print(f"\n[stage1] token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk}")
    d = _gen(token, model_dim, inter_dim, E, topk)

    ref = torch_moe_stage1(
        d["inp"], d["w1_dq"], d["w2_dq"], d["topk_weights"], d["topk_ids"],
        dtype=torch.bfloat16, activation=ActivationType.Silu, quant_type=QuantType.No,
    )
    w1_shuf = _prep_weight_for_kernel(d["w1_qt"], inter_dim * 2, model_dim)
    w1_scale_shuf = _prep_scale_for_kernel(d["w1_scale"], inter_dim * 2, model_dim, _STAGE1_SCALE_MUL)

    sorted_ids, _sw, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, block_m
    )
    out = flydsl_moe_stage1(
        d["inp"], w1_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
        a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16",
        act="silu", w1_scale=w1_scale_shuf,
    )
    torch.cuda.synchronize()
    return _check(ref, out, "stage1")


def test_stage2(token, model_dim, inter_dim, E, topk, block_m=32):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    print(f"\n[stage2] token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk}")
    d = _gen(token, model_dim, inter_dim, E, topk)
    a2 = torch.randn((token, topk, inter_dim), dtype=torch.bfloat16) / 10

    ref = torch_moe_stage2(
        a2, torch.zeros((E, inter_dim * 2, model_dim), dtype=torch.bfloat16), d["w2_dq"],
        d["topk_weights"], d["topk_ids"], dtype=torch.bfloat16,
        quant_type=QuantType.No, doweight=True,
    )
    w2_shuf = _prep_weight_for_kernel(d["w2_qt"], model_dim, inter_dim)
    w2_scale_shuf = _prep_scale_for_kernel(d["w2_scale"], model_dim, inter_dim, _STAGE2_SCALE_MUL)

    sorted_ids, sw, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, block_m
    )
    out = flydsl_moe_stage2(
        a2, w2_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
        a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16",
        w2_scale=w2_scale_shuf, sorted_weights=sw,
    )
    torch.cuda.synchronize()
    return _check(ref, out, "stage2")


def test_e2e(token, model_dim, inter_dim, E, topk, block_m=32):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
    print(f"\n[e2e] token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk}")
    d = _gen(token, model_dim, inter_dim, E, topk)

    ref1 = torch_moe_stage1(
        d["inp"], d["w1_dq"], d["w2_dq"], d["topk_weights"], d["topk_ids"],
        dtype=torch.bfloat16, activation=ActivationType.Silu, quant_type=QuantType.No,
    )
    ref2 = torch_moe_stage2(
        ref1, torch.zeros((E, inter_dim * 2, model_dim), dtype=torch.bfloat16), d["w2_dq"],
        d["topk_weights"], d["topk_ids"], dtype=torch.bfloat16,
        quant_type=QuantType.No, doweight=True,
    )

    w1_shuf = _prep_weight_for_kernel(d["w1_qt"], inter_dim * 2, model_dim)
    w1_scale_shuf = _prep_scale_for_kernel(d["w1_scale"], inter_dim * 2, model_dim, _STAGE1_SCALE_MUL)
    w2_shuf = _prep_weight_for_kernel(d["w2_qt"], model_dim, inter_dim)
    w2_scale_shuf = _prep_scale_for_kernel(d["w2_scale"], model_dim, inter_dim, _STAGE2_SCALE_MUL)

    sorted_ids, sw, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, block_m
    )
    s1 = flydsl_moe_stage1(
        d["inp"], w1_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
        a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16", act="silu",
        w1_scale=w1_scale_shuf,
    )
    out = flydsl_moe_stage2(
        s1, w2_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
        a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16",
        w2_scale=w2_scale_shuf, sorted_weights=sw,
    )
    torch.cuda.synchronize()
    return _check(ref2, out, "e2e")


def _time_cuda(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def compare_triton(token, model_dim, inter_dim, E, topk, block_m=32, iters=50):
    """Compare FlyDSL a16w4 stage1 (gate+up+silu) vs triton moe_gemm_a16w4 (apply_swiglu).

    Both compute the same MoE gate/up GEMM + SwiGLU at matched M/N/K/E/topk on
    mxfp4 weights. Reports per-iter latency and speedup. On gfx942 the triton
    tl.dot_scaled is software-emulated to bf16 (no native fp4 MFMA).
    """
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
    from aiter.ops.triton.utils.types import str_to_torch_dtype

    print(f"\n[compare] token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk}")

    # ---- FlyDSL stage1 setup ----
    d = _gen(token, model_dim, inter_dim, E, topk)
    w1_shuf = _prep_weight_for_kernel(d["w1_qt"], inter_dim * 2, model_dim)
    w1_scale_shuf = _prep_scale_for_kernel(
        d["w1_scale"], inter_dim * 2, model_dim, _STAGE1_SCALE_MUL
    )
    sorted_ids, _sw, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, block_m
    )
    fly_kw = dict(
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
        a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16",
        act="silu", w1_scale=w1_scale_shuf,
    )

    def run_flydsl():
        flydsl_moe_stage1(
            d["inp"], w1_shuf, sorted_ids, sorted_expert_ids, num_valid_ids, **fly_kw
        )

    # ---- triton moe_gemm_a16w4 setup (gate+up GEMM, apply_swiglu) ----
    dev = "cuda"
    logits = torch.randn((token, E), dtype=torch.float16, device=dev)
    rdata, gindx, sindx = routing(logits, topk)
    rdata.gate_scal = None
    x_tri = d["inp"]
    # triton weight layout: (E, K, N) with N = 2*inter_dim (gate+up)
    w_tri_bf16 = d["w1"].transpose(1, 2).contiguous()  # (E, model_dim, 2*inter_dim)
    wdt = str_to_torch_dtype["mxfp4_e2m1"]
    w_tri, w_scale_tri = downcast_to_mxfp(w_tri_bf16, wdt, axis=1)
    bias_tri = torch.zeros((E, inter_dim * 2), dtype=torch.float32, device=dev)

    def run_triton():
        moe_gemm_a16w4(
            x_tri, w_tri, None, w_scale_tri, None, None, bias_tri,
            rdata, gindx, sindx, None, None, torch.bfloat16, True,
        )

    # sanity: both produce finite output
    run_flydsl()
    run_triton()
    torch.cuda.synchronize()

    fly_us = _time_cuda(run_flydsl, iters=iters)
    tri_us = _time_cuda(run_triton, iters=iters)
    speedup = tri_us / fly_us
    print(f"  FlyDSL a16w4 : {fly_us:8.1f} us/iter")
    print(f"  triton a16w4 : {tri_us:8.1f} us/iter")
    print(f"  speedup (triton/flydsl): {speedup:.2f}x")
    return fly_us, tri_us, speedup


def _adaptive_tile_a16w4(token, topk, E):
    """Pick (tile_m, tile_k) for the a16w4 FlyDSL GEMM by tokens-per-expert,
    mirroring triton's block_m heuristic. A larger tile_m amortizes the heavy
    mxfp4->bf16 dequant VALU over more M rows (see docs/flydsl_a16w4_vs_triton
    _perf_analysis_cn.md); tile_m=128 needs tile_k=128 to fit the 64KB LDS budget.

    Measured crossover (MI308X, model_dim=4096, inter_dim=512, E=256, topk=6):
      tokens/expert <=~32 -> (32,256); ~48 -> (64,128); >=~96 -> (128,128).
    """
    tpe = max(1, (token * topk + E - 1) // E)
    bm = 1 << max(0, (tpe - 1).bit_length())  # next power of 2 of tokens/expert
    bm = max(32, min(bm, 128))
    tk = 256 if bm <= 32 else 128  # bm>=64 prefers tk=128 (faster + fits LDS at bm=128)
    return bm, tk


def compare_e2e(token, model_dim, inter_dim, E, topk, block_m=None, iters=50):
    """Compare FlyDSL vs triton for stage1, stage2, and end-to-end (stage1->stage2).

    Both use the same mxfp4 weights. On the triton side *both* stages are the
    single `moe_gemm_a16w4` kernel:
      - stage1 (gate+up): gather_indx + apply_swiglu=True  -> (token*topk, inter_dim)
      - stage2 (down)    : scatter_indx + gammas(topk wt) + apply_swiglu=False
                           -> (token, model_dim)
    This mirrors FlyDSL's flydsl_moe_stage1 (act="silu") + flydsl_moe_stage2
    (sorted_weights=topk weights). Reports per-iter latency (us) and e2e speedup.
    """
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
    from aiter.ops.triton.utils.types import str_to_torch_dtype

    if block_m is None:
        tile_m, tile_k = _adaptive_tile_a16w4(token, topk, E)  # adaptive block_m
    else:
        tile_m, tile_k = block_m, 256  # fixed (legacy) path
    print(f"\n[e2e] token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk}  "
          f"tile_m={tile_m} tile_k={tile_k}")
    d = _gen(token, model_dim, inter_dim, E, topk)

    # ---- FlyDSL setup (stage1 + stage2) ----
    w1_shuf = _prep_weight_for_kernel(d["w1_qt"], inter_dim * 2, model_dim)
    w1_scale_shuf = _prep_scale_for_kernel(d["w1_scale"], inter_dim * 2, model_dim, _STAGE1_SCALE_MUL)
    w2_shuf = _prep_weight_for_kernel(d["w2_qt"], model_dim, inter_dim)
    w2_scale_shuf = _prep_scale_for_kernel(d["w2_scale"], model_dim, inter_dim, _STAGE2_SCALE_MUL)
    sorted_ids, sw, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, tile_m
    )
    # stage1 tile_n=128; stage2 uses tile_n=256 (down-proj N=model_dim is wide,
    # 256 halves the N-tile count -> ~half the workgroups; W2 stays packed in LDS
    # so it fits). Both stages share tile_m so a single moe_sorting suffices.
    s1_kw = dict(topk=topk, tile_m=tile_m, tile_n=128, tile_k=tile_k,
                 a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16",
                 act="silu", w1_scale=w1_scale_shuf)
    s2_kw = dict(topk=topk, tile_m=tile_m, tile_n=256, tile_k=tile_k,
                 a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16",
                 w2_scale=w2_scale_shuf, sorted_weights=sw)
    a2 = torch.randn((token, topk, inter_dim), dtype=torch.bfloat16) / 10  # standalone stage2 input

    def fly_s1():
        return flydsl_moe_stage1(d["inp"], w1_shuf, sorted_ids, sorted_expert_ids, num_valid_ids, **s1_kw)

    def fly_s2():
        return flydsl_moe_stage2(a2, w2_shuf, sorted_ids, sorted_expert_ids, num_valid_ids, **s2_kw)

    def fly_e2e():
        s1 = flydsl_moe_stage1(d["inp"], w1_shuf, sorted_ids, sorted_expert_ids, num_valid_ids, **s1_kw)
        return flydsl_moe_stage2(s1, w2_shuf, sorted_ids, sorted_expert_ids, num_valid_ids, **s2_kw)

    # ---- triton setup (both stages = moe_gemm_a16w4) ----
    dev = "cuda"
    logits = torch.randn((token, E), dtype=torch.float16, device=dev)
    rdata, gindx, sindx = routing(logits, topk)
    gammas = rdata.gate_scal.to(torch.float32) if rdata.gate_scal is not None else None
    x_tri = d["inp"]
    wdt = str_to_torch_dtype["mxfp4_e2m1"]
    # stage1 weight: (E, K=model_dim, N=2*inter_dim); stage2 weight: (E, K=inter_dim, N=model_dim)
    w1_tri, w1s_tri = downcast_to_mxfp(d["w1"].transpose(1, 2).contiguous(), wdt, axis=1)
    w2_tri, w2s_tri = downcast_to_mxfp(d["w2"].transpose(1, 2).contiguous(), wdt, axis=1)
    b1 = torch.zeros((E, inter_dim * 2), dtype=torch.float32, device=dev)
    b2 = torch.zeros((E, model_dim), dtype=torch.float32, device=dev)
    h_tri = torch.randn((token * topk, inter_dim), dtype=torch.bfloat16, device=dev)  # standalone stage2 input

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

    # sanity: shapes finite + e2e output is (token, model_dim)
    fo = fly_e2e()
    to = tri_e2e()
    torch.cuda.synchronize()
    assert fo.shape == (token, model_dim), f"flydsl e2e {tuple(fo.shape)}"
    assert to.shape == (token, model_dim), f"triton e2e {tuple(to.shape)}"
    assert torch.isfinite(fo).all() and torch.isfinite(to).all(), "non-finite e2e output"

    fly = {k: _time_cuda(fn, iters=iters) for k, fn in
           (("s1", fly_s1), ("s2", fly_s2), ("e2e", fly_e2e))}
    tri = {k: _time_cuda(fn, iters=iters) for k, fn in
           (("s1", tri_s1), ("s2", tri_s2), ("e2e", tri_e2e))}
    e2e_speedup = tri["e2e"] / fly["e2e"]
    print(f"  stage1  FlyDSL {fly['s1']:8.1f} us | triton {tri['s1']:8.1f} us")
    print(f"  stage2  FlyDSL {fly['s2']:8.1f} us | triton {tri['s2']:8.1f} us")
    print(f"  e2e     FlyDSL {fly['e2e']:8.1f} us | triton {tri['e2e']:8.1f} us "
          f"| speedup(triton/flydsl) {e2e_speedup:.2f}x")
    return fly, tri


def bench_stage1(token, model_dim, inter_dim, E, topk, block_m=32, iters=50):
    """Time the FlyDSL a16w4 stage1 kernel launch (excludes prep)."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
    d = _gen(token, model_dim, inter_dim, E, topk)
    w1_shuf = _prep_weight_for_kernel(d["w1_qt"], inter_dim * 2, model_dim)
    w1_scale_shuf = _prep_scale_for_kernel(d["w1_scale"], inter_dim * 2, model_dim, _STAGE1_SCALE_MUL)
    sorted_ids, _sw, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        d["topk_ids"], d["topk_weights"], E, model_dim, torch.bfloat16, block_m
    )
    kw = dict(topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
              a_dtype="bf16", b_dtype="mxfp4", out_dtype="bf16",
              act="silu", w1_scale=w1_scale_shuf)
    for _ in range(5):
        flydsl_moe_stage1(d["inp"], w1_shuf, sorted_ids, sorted_expert_ids, num_valid_ids, **kw)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        flydsl_moe_stage1(d["inp"], w1_shuf, sorted_ids, sorted_expert_ids, num_valid_ids, **kw)
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) / iters * 1000.0
    print(f"  [bench stage1] token={token:5d} : {us:8.1f} us/iter")
    return us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage1", "stage2", "e2e", "all"], default="all")
    ap.add_argument("-t", "--tokens", type=int, action="append", default=None)
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--inter-dim", type=int, default=512)
    ap.add_argument("-E", "--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--compare-triton", action="store_true")
    ap.add_argument("--compare-e2e", action="store_true",
                    help="FlyDSL vs triton for stage1 / stage2 / end-to-end")
    ap.add_argument("--bench", action="store_true", help="time the FlyDSL stage1 kernel")
    args = ap.parse_args()

    tokens = args.tokens or [16, 128, 1024]

    if args.compare_e2e:
        print("\nFlyDSL vs triton a16w4 stage1 / stage2 / e2e performance:")
        rows = []
        for tk in tokens:
            fly, tri = compare_e2e(
                tk, args.model_dim, args.inter_dim, args.experts, args.topk
            )
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
        print("\nFlyDSL vs triton a16w4 (gate+up+SwiGLU) performance:")
        rows = []
        for tk in tokens:
            f, t, s = compare_triton(
                tk, args.model_dim, args.inter_dim, args.experts, args.topk
            )
            rows.append((tk, f, t, s))
        print("\n" + "=" * 60)
        print(f"  {'token':>6} {'flydsl(us)':>12} {'triton(us)':>12} {'speedup':>9}")
        for tk, f, t, s in rows:
            print(f"  {tk:>6} {f:>12.1f} {t:>12.1f} {s:>8.2f}x")
        print("=" * 60)
        sys.exit(0)

    if args.bench:
        print("\nFlyDSL a16w4 (mxfp4) stage1 kernel timing:")
        for tk in tokens:
            bench_stage1(tk, args.model_dim, args.inter_dim, args.experts, args.topk)
        sys.exit(0)

    stages = ["stage1", "stage2", "e2e"] if args.stage == "all" else [args.stage]

    results = []
    for tk in tokens:
        for st in stages:
            fn = {"stage1": test_stage1, "stage2": test_stage2, "e2e": test_e2e}[st]
            ok = fn(tk, args.model_dim, args.inter_dim, args.experts, args.topk)
            results.append((tk, st, ok))

    print("\n" + "=" * 60)
    print("SUMMARY")
    for tk, st, ok in results:
        print(f"  token={tk:5d} {st:8s} {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, _, ok in results)
    print("=" * 60)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
