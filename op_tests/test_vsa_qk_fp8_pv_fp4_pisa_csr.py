# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""PISA CSR op: usage sample, ρ=0 vs VSA dropB, ρ>0 vs customer golden.

Call sequence (CSR → plan → dropB), matching the VSA CSR test's usage block:

    exact_col, exact_meta = padded_to_csr(exact_idx, exact_num)
    approx_col, approx_meta = padded_to_csr(approx_idx, approx_num)
    plan = build_l2_aware_lim_vsa_qk_fp8_pv_fp4_pisa_csr(...)
    out, lse = vsa_qk_fp8_pv_fp4_pisa_csr_dropB(..., plan=plan)

Checks:
  1. ρ=0 (all exact) — bitwise equal to vsa_qk_fp8_pv_fp4_dropB on the same routes.
  2. ρ>0 with >8 approx blocks/row — cosine vs pisa_attention_reference.

Usage:
    PYTHONPATH=. python3 op_tests/test_vsa_qk_fp8_pv_fp4_pisa_csr.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pisa_attention_golden import pisa_attention_reference  # noqa: E402
from test_vsa_qk_fp8_pv_fp4 import (  # noqa: E402
    _dequantize_fp4_per_channel_kblock,
    _dequantize_fp8_e8m0,
    make_synthetic_data,
)

from aiter.ops.vsa_qk_fp8_pv_fp4 import vsa_qk_fp8_pv_fp4_dropB  # noqa: E402
from aiter.ops.vsa_qk_fp8_pv_fp4_pisa import (  # noqa: E402
    APX_PROMOTE_EXACT,
    build_l2_aware_lim_vsa_qk_fp8_pv_fp4_pisa_csr,
    vsa_qk_fp8_pv_fp4_pisa_csr_dropB,
)

SPARSE_BLK = 128
_HEAD_DIM = 128


def padded_to_csr(q2k_idx: torch.Tensor, q2k_num: torch.Tensor):
    """Test-only: flatten a padded (n_tasks, width) descriptor to global CSR."""
    q2k_idx = q2k_idx.view(-1, q2k_idx.shape[-1])
    q2k_num = q2k_num.view(-1).to(torch.int32)
    n, width = q2k_idx.shape
    keep = torch.arange(width, device=q2k_idx.device).unsqueeze(0) < q2k_num.unsqueeze(1)
    col_indices = q2k_idx[keep].to(torch.int32).contiguous()
    offsets = torch.cumsum(q2k_num.to(torch.int64), 0) - q2k_num
    meta = torch.stack((offsets.to(torch.int32), q2k_num), dim=-1).contiguous()
    assert meta.shape[0] == n
    return col_indices, meta


def attention_pisa_csr(*, q, k, v, qs, ks, vs, vbs,
                       exact_index, exact_meta,
                       approx_index=None, approx_meta=None,
                       B, T, num_q_blks, plan=None, out=None, lse=None):
    """Production call sequence: CSR streams → plan → dropB."""
    if plan is None:
        BH = q.shape[0] if q.ndim == 3 else q.shape[0] * q.shape[1]
        heads = BH // B
        plan = build_l2_aware_lim_vsa_qk_fp8_pv_fp4_pisa_csr(
            exact_index=exact_index,
            exact_meta=exact_meta,
            approx_index=approx_index,
            approx_meta=approx_meta,
            batch=B,
            heads=heads,
            blocks=num_q_blks,
            device=q.device,
        )
    return vsa_qk_fp8_pv_fp4_pisa_csr_dropB(
        q=q, k=k, v=v, qs=qs, ks=ks, vs=vs, vbs=vbs,
        exact_index=exact_index, exact_meta=exact_meta,
        approx_index=approx_index, approx_meta=approx_meta,
        plan=plan, B=B, T=T, num_q_blks=num_q_blks, out=out, lse=lse,
    )


def _split_padded(q2k_idx, q2k_num, n_approx):
    """Move the last n_approx valid columns of each row to the approx stream."""
    q2k_idx = q2k_idx.view(-1, q2k_idx.shape[-1]).contiguous()
    q2k_num = q2k_num.view(-1).to(torch.int32).contiguous()
    n, width = q2k_idx.shape
    n_approx = int(n_approx)
    exact_num = (q2k_num - n_approx).clamp(min=0)
    approx_num = (q2k_num - exact_num).to(torch.int32)
    exact_idx = torch.full_like(q2k_idx, -1)
    approx_idx = torch.full_like(q2k_idx, -1)
    lane = torch.arange(width, device=q2k_idx.device)
    exact_idx[lane.unsqueeze(0) < exact_num.unsqueeze(1)] = q2k_idx[
        lane.unsqueeze(0) < exact_num.unsqueeze(1)
    ]
    for row in range(n):
        e = int(exact_num[row].item())
        a = int(approx_num[row].item())
        if a:
            approx_idx[row, :a] = q2k_idx[row, e : e + a]
    return exact_idx, exact_num, approx_idx, approx_num


def _dequant_bh(data):
    BH, T, D = data["q"].shape
    q = _dequantize_fp8_e8m0(data["q"], data["qs"]).to(torch.bfloat16)
    k = _dequantize_fp8_e8m0(data["k"], data["ks"]).to(torch.bfloat16)
    vs = data["vs"].permute(0, 1, 3, 2).reshape(BH, -1, D).contiguous()
    v = _dequantize_fp4_per_channel_kblock(data["v"], vs, 32).to(torch.bfloat16)
    return q, k, v


def _to_4d(x, B, H):
    BH = B * H
    assert x.shape[0] == BH
    return x.view(B, H, *x.shape[1:])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def test_rho0_bitwise_vs_vsa_dropB():
    data = make_synthetic_data(seed=0, num_q_blks=32, B=1, H=4)
    B, T, num_q_blks = data["B"], data["T"], data["num_q_blks"]
    exact_col, exact_meta = padded_to_csr(data["q2k_idx"], data["q2k_num"])

    from aiter.ops.vsa_qk_fp8_pv_fp4 import build_l2_aware_lim_vsa_qk_fp8_pv_fp4

    lim, n_dense = build_l2_aware_lim_vsa_qk_fp8_pv_fp4(
        data["q2k_idx"], data["q2k_num"], data["max_kv"]
    )
    vsa_out, vsa_lse = vsa_qk_fp8_pv_fp4_dropB(
        q=data["q"], k=data["k"], v=data["v"],
        qs=data["qs"], ks=data["ks"], vs=data["vs"],
        q2k_idx=data["q2k_idx"], q2k_num=data["q2k_num"],
        vbs=data["vbs"], lim=lim, n_dense=n_dense,
        B=B, T=T, num_q_blks=num_q_blks, max_kv=data["max_kv"],
    )
    pisa_out, pisa_lse = attention_pisa_csr(
        q=data["q"], k=data["k"], v=data["v"],
        qs=data["qs"], ks=data["ks"], vs=data["vs"], vbs=data["vbs"],
        exact_index=exact_col, exact_meta=exact_meta,
        B=B, T=T, num_q_blks=num_q_blks,
    )
    assert torch.equal(pisa_out.view(torch.int16), vsa_out.view(torch.int16)), (
        "ρ=0 PISA out is not bitwise-identical to VSA dropB"
    )
    assert torch.equal(pisa_lse.view(torch.int32), vsa_lse.view(torch.int32)), (
        "ρ=0 PISA lse is not bitwise-identical to VSA dropB"
    )
    print("ρ=0 bitwise vs vsa_qk_fp8_pv_fp4_dropB: PASS")

    H = data["H"]
    q4 = data["q"].view(B, H, T, _HEAD_DIM)
    k4 = data["k"].view(B, H, T, _HEAD_DIM)
    v4 = data["v"].view(B, H, T, _HEAD_DIM // 2)
    qs4 = data["qs"].view(B, H, T, 4)
    ks4 = data["ks"].view(B, H, T, 4)
    vs4 = data["vs"].view(B, H, *data["vs"].shape[1:])
    pisa4, _ = attention_pisa_csr(
        q=q4, k=k4, v=v4, qs=qs4, ks=ks4, vs=vs4, vbs=data["vbs"],
        exact_index=exact_col, exact_meta=exact_meta,
        B=B, T=T, num_q_blks=num_q_blks,
    )
    assert pisa4.ndim == 4
    assert torch.equal(pisa4.view(torch.int16), vsa_out.view(B, H, T, _HEAD_DIM).view(torch.int16))
    print("ρ=0 4D layout bitwise: PASS")


def test_rho_gt0_cosine_vs_golden():
    n_approx = APX_PROMOTE_EXACT + 1
    data = make_synthetic_data(
        seed=1, num_q_blks=16, B=1, H=4, sparsity=12 / 16, dense_frac=0.0
    )
    assert int(data["q2k_num"].min().item()) >= n_approx + 1
    exact_idx, exact_num, approx_idx, approx_num = _split_padded(
        data["q2k_idx"], data["q2k_num"], n_approx
    )
    assert int(approx_num.min().item()) > APX_PROMOTE_EXACT

    exact_col, exact_meta = padded_to_csr(exact_idx, exact_num)
    approx_col, approx_meta = padded_to_csr(approx_idx, approx_num)
    B, T, num_q_blks = data["B"], data["T"], data["num_q_blks"]
    H = data["H"]

    plan = build_l2_aware_lim_vsa_qk_fp8_pv_fp4_pisa_csr(
        exact_index=exact_col,
        exact_meta=exact_meta,
        approx_index=approx_col,
        approx_meta=approx_meta,
        batch=B,
        heads=H,
        blocks=num_q_blks,
        device=data["q"].device,
    )
    assert plan.need_stats, "geometry must run the centroid path (approx/row > 8)"

    pisa_out, _ = vsa_qk_fp8_pv_fp4_pisa_csr_dropB(
        q=data["q"], k=data["k"], v=data["v"],
        qs=data["qs"], ks=data["ks"], vs=data["vs"], vbs=data["vbs"],
        plan=plan, B=B, T=T, num_q_blks=num_q_blks,
    )

    q_bf, k_bf, v_bf = _dequant_bh(data)
    width = exact_idx.shape[-1]
    golden = pisa_attention_reference(
        _to_4d(q_bf, B, H),
        _to_4d(k_bf, B, H),
        _to_4d(v_bf, B, H),
        exact_idx.view(B, H, num_q_blks, width),
        exact_num.view(B, H, num_q_blks),
        approx_idx.view(B, H, num_q_blks, width),
        approx_num.view(B, H, num_q_blks),
        data["vbs"],
    )
    cos = _cosine(pisa_out, golden.view_as(pisa_out))
    print(f"ρ>0 cosine vs pisa_attention_reference: {cos:.6f}")
    # The exact route quantises softmax P to FP4, so ~0.99 against the BF16
    # equation is the kernel's floor with no approximate route at all; the
    # all-exact case below measures it on this same fixture.
    assert cos > 0.985, f"cosine {cos} below 0.985 vs customer golden"

    exact_only_col, exact_only_meta = padded_to_csr(
        data["q2k_idx"], data["q2k_num"]
    )
    exact_only_out, _ = attention_pisa_csr(
        q=data["q"], k=data["k"], v=data["v"],
        qs=data["qs"], ks=data["ks"], vs=data["vs"], vbs=data["vbs"],
        exact_index=exact_only_col, exact_meta=exact_only_meta,
        B=B, T=T, num_q_blks=num_q_blks,
    )
    empty_idx = torch.full_like(data["q2k_idx"].view(-1, width), -1)
    empty_num = torch.zeros(
        empty_idx.shape[0], dtype=torch.int32, device=empty_idx.device
    )
    exact_only_golden = pisa_attention_reference(
        _to_4d(q_bf, B, H),
        _to_4d(k_bf, B, H),
        _to_4d(v_bf, B, H),
        data["q2k_idx"].view(B, H, num_q_blks, width),
        data["q2k_num"].view(B, H, num_q_blks),
        empty_idx.view(B, H, num_q_blks, width),
        empty_num.view(B, H, num_q_blks),
        data["vbs"],
    )
    cos_exact = _cosine(exact_only_out, exact_only_golden.view_as(exact_only_out))
    print(f"    (all-exact FP4 floor on same fixture: {cos_exact:.6f})")
    assert cos > cos_exact - 0.005, (
        f"approx route lost accuracy: {cos} vs all-exact floor {cos_exact}"
    )


if __name__ == "__main__":
    test_rho0_bitwise_vs_vsa_dropB()
    test_rho_gt0_cosine_vs_golden()
    print("all PISA CSR op_tests passed")
