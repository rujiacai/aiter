# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused PISA CSR attention: exact + approximate routes on gfx950.

Public API:
  build_l2_aware_lim_vsa_qk_fp8_pv_fp4_pisa_csr  # CSR → PisaPlan
  vsa_qk_fp8_pv_fp4_pisa_csr_dropB               # 3D/4D convenience wrapper

Inputs are exact + approx CSR only.  HIP lives in VSA_CSR; this module loads
``AITER_ASM_DIR/gfx950/vsa/vsa_qk_fp8_pv_fp4_pisa_csr.co``.

op_tests/test_vsa_qk_fp8_pv_fp4_pisa_csr.py opens with a runnable call sequence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..jit.core import compile_ops

BLOCK = 128
HEAD_DIM = 128

# Must match APX_PROMOTE_EXACT in vsa_qk_fp8_pv_fp4_pisa.hip.
APX_PROMOTE_EXACT = 8

__all__ = [
    "PisaPlan",
    "build_l2_aware_lim_vsa_qk_fp8_pv_fp4_pisa_csr",
    "vsa_qk_fp8_pv_fp4_pisa_csr_dropB",
]


@compile_ops("module_vsa_qk_fp8_pv_fp4_pisa")
def vsa_qk_fp8_pv_fp4_pisa_stats(
    k: torch.Tensor,
    v: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    vbs: torch.Tensor,
    k_center: torch.Tensor,
    value_center: torch.Tensor,
    used_ids: torch.Tensor,
    n_used: int,
    BH: int,
    T: int,
) -> None: ...


@compile_ops("module_vsa_qk_fp8_pv_fp4_pisa")
def vsa_qk_fp8_pv_fp4_pisa_csr(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    exact_col_indices: torch.Tensor,
    exact_row_meta: torch.Tensor,
    approx_col_indices: torch.Tensor,
    approx_row_meta: torch.Tensor,
    vbs: torch.Tensor,
    lim: torch.Tensor,
    k_center: torch.Tensor,
    value_center: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    counters: torch.Tensor,
    B: int,
    T: int,
    num_q_blks: int,
    n_dense: int,
    centroids_ready: int,
) -> None: ...


def _validate_csr(
    name: str,
    col_indices: torch.Tensor,
    meta: torch.Tensor,
    rows: int,
    blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    col_indices = col_indices.reshape(-1).contiguous()
    meta = meta.reshape(-1, 2).contiguous()
    if col_indices.dtype != torch.int32 or meta.dtype != torch.int32:
        raise ValueError(f"{name} CSR tensors must be INT32")
    if col_indices.device != device or meta.device != device:
        raise ValueError(f"{name} CSR tensors must be on {device}")
    if meta.shape[0] != rows:
        raise ValueError(f"{name}_meta must contain {rows} rows")
    counts = meta[:, 1].to(torch.int64)
    offsets = meta[:, 0].to(torch.int64)
    expected = torch.cumsum(counts, 0) - counts
    if bool((counts < 0).any().item()) or not torch.equal(offsets, expected):
        raise ValueError(f"{name}_meta must be canonical global [offset,count] CSR")
    if int(counts.sum().item()) != col_indices.numel():
        raise ValueError(f"{name} CSR count sum does not match col_indices")
    if col_indices.numel() and bool(
        ((col_indices < 0) | (col_indices >= blocks)).any().item()
    ):
        raise ValueError(f"{name}_col_indices contains an invalid KV block id")
    return col_indices, meta


def _normalize_csr_layout(
    name: str,
    col_indices: torch.Tensor,
    meta: torch.Tensor,
    batch: int,
    heads: int,
    query_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept either flat global CSR or per-(B,H) streams."""
    if col_indices.ndim == 1:
        return col_indices, meta
    if col_indices.ndim != 3 or col_indices.shape[:2] != (batch, heads):
        raise ValueError(
            f"{name}_index must be [nnz] or [B,H,C], got {tuple(col_indices.shape)}"
        )
    if meta.shape != (batch, heads, query_blocks, 2):
        raise ValueError(f"{name}_meta must be [B,H,QB,2]")

    pieces: list[torch.Tensor] = []
    adjusted_meta: list[torch.Tensor] = []
    global_base = 0
    flat_index = col_indices.reshape(batch * heads, col_indices.shape[-1])
    flat_meta = meta.reshape(batch * heads, query_blocks, 2)
    for bh in range(batch * heads):
        local_counts = flat_meta[bh, :, 1].to(torch.int64)
        local_offsets = flat_meta[bh, :, 0].to(torch.int64)
        expected = torch.cumsum(local_counts, 0) - local_counts
        if not torch.equal(local_offsets, expected):
            raise ValueError(
                f"{name}_meta offsets must be local row-major offsets per (B,H)"
            )
        used = int(local_counts.sum().item())
        if used > flat_index.shape[1]:
            raise ValueError(f"{name}_index capacity is smaller than CSR counts")
        pieces.append(flat_index[bh, :used])
        adjusted = torch.stack(
            (local_offsets + global_base, local_counts), dim=-1
        )
        adjusted_meta.append(adjusted)
        global_base += used
    return (
        torch.cat(pieces).to(torch.int32).contiguous()
        if pieces
        else torch.empty(0, dtype=torch.int32, device=col_indices.device),
        torch.cat(adjusted_meta).to(torch.int32).view(
            batch, heads, query_blocks, 2
        ).contiguous(),
    )


def _build_fused_schedule(
    exact_index: torch.Tensor,
    exact_meta: torch.Tensor,
    approx_index: torch.Tensor,
    approx_meta: torch.Tensor,
    blocks: int,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Build one task order and reorder both CSR metadata streams to match."""
    exact_meta = exact_meta.reshape(-1, 2)
    approx_meta = approx_meta.reshape(-1, 2)
    exact_count = exact_meta[:, 1].to(torch.int64)
    approx_count = approx_meta[:, 1].to(torch.int64)

    def first_key(
        index: torch.Tensor, meta: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        if index.numel() == 0:
            return torch.full_like(counts, 1 << 30)
        starts = meta[:, 0].to(torch.int64)
        safe = torch.where(counts > 0, starts, torch.zeros_like(starts))
        first = index.to(torch.int64).gather(0, safe)
        return torch.where(counts > 0, first, torch.full_like(first, 1 << 30))

    exact_first = first_key(exact_index, exact_meta, exact_count)
    approx_first = first_key(approx_index, approx_meta, approx_count)
    first = torch.minimum(exact_first, approx_first)
    band = first // 512
    work = exact_count * BLOCK + approx_count
    radix = blocks * BLOCK * 2 + blocks + 1
    key = band * radix + (blocks * BLOCK + blocks - work)
    base_order = torch.argsort(key, stable=True)
    dense = exact_count >= int(blocks * 7 / 8)
    ordered_dense = dense[base_order]
    schedule = torch.cat(
        (base_order[ordered_dense], base_order[~ordered_dense])
    )
    n_dense = int(ordered_dense.sum().item())

    def packed(
        meta: torch.Tensor, counts: torch.Tensor, first_value: torch.Tensor
    ) -> torch.Tensor:
        records = torch.stack(
            (
                counts,
                meta[:, 0].to(torch.int64),
                first_value,
                torch.zeros_like(counts),
            ),
            dim=1,
        )
        return records[schedule].to(torch.int32).contiguous()

    return (
        schedule.to(torch.int32).contiguous(),
        n_dense,
        packed(exact_meta, exact_count, exact_first),
        packed(approx_meta, approx_count, approx_first),
    )


class PisaPlan:
    """Route-dependent launch state: CSR descriptors, task order, scratch."""

    __slots__ = (
        "exact_index",
        "exact_row_meta",
        "approx_index",
        "approx_row_meta",
        "schedule",
        "n_dense",
        "rows",
        "batch",
        "heads",
        "blocks",
        "dim",
        "device",
        "approx_nnz",
        "need_stats",
        "stats_ids",
        "k_center",
        "value_center",
        "counters",
    )

    def __init__(self, **fields) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


def _empty_csr(
    batch: int, heads: int, blocks: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(0, dtype=torch.int32, device=device),
        torch.zeros((batch, heads, blocks, 2), dtype=torch.int32, device=device),
    )


@torch.no_grad()
def build_l2_aware_lim_vsa_qk_fp8_pv_fp4_pisa_csr(
    *,
    exact_index: Optional[torch.Tensor],
    exact_meta: Optional[torch.Tensor],
    approx_index: Optional[torch.Tensor],
    approx_meta: Optional[torch.Tensor],
    batch: int,
    heads: int,
    blocks: int,
    dim: int = HEAD_DIM,
    device: Optional[torch.device] = None,
) -> PisaPlan:
    """Validate CSR routes and precompute everything the launch reuses."""
    rows = batch * heads * blocks
    if device is None:
        reference = exact_meta if exact_meta is not None else approx_meta
        if reference is None:
            raise ValueError(
                "pass at least one CSR route stream, or an explicit device"
            )
        device = reference.device
    device = torch.device(device)
    if exact_index is None or exact_meta is None:
        exact_index, exact_meta = _empty_csr(batch, heads, blocks, device)
    if approx_index is None or approx_meta is None:
        approx_index, approx_meta = _empty_csr(batch, heads, blocks, device)
    exact_index, exact_meta = _normalize_csr_layout(
        "exact", exact_index, exact_meta, batch, heads, blocks
    )
    approx_index, approx_meta = _normalize_csr_layout(
        "approx", approx_index, approx_meta, batch, heads, blocks
    )
    exact_index, exact_meta = _validate_csr(
        "exact", exact_index, exact_meta, rows, blocks, device
    )
    approx_index, approx_meta = _validate_csr(
        "approx", approx_index, approx_meta, rows, blocks, device
    )
    schedule, n_dense, exact_row_meta, approx_row_meta = _build_fused_schedule(
        exact_index, exact_meta, approx_index, approx_meta, blocks
    )
    approx_nnz = int(approx_index.numel())
    stats_ids = torch.empty(0, dtype=torch.int32, device=device)
    max_approx = 0
    if approx_nnz:
        max_approx = int(approx_meta.reshape(-1, 2)[:, 1].max().item())
    need_stats = max_approx > APX_PROMOTE_EXACT
    if need_stats and approx_nnz <= 8_000_000:
        counts = approx_meta.reshape(-1, 2)[:, 1]
        row = torch.repeat_interleave(
            torch.arange(rows, device=device, dtype=torch.int32), counts
        )
        linear = torch.div(row, blocks, rounding_mode="floor") * blocks + (
            approx_index
        )
        unique_ids = torch.unique(linear)
        if int(unique_ids.numel()) < rows:
            stats_ids = unique_ids.to(torch.int32).contiguous()
    if need_stats:
        stats_shape = (batch * heads, blocks, dim)
        k_center = torch.empty(stats_shape, dtype=torch.bfloat16, device=device)
        value_center = torch.empty(stats_shape, dtype=torch.bfloat16, device=device)
    else:
        k_center = torch.zeros(1, dtype=torch.bfloat16, device=device)
        value_center = torch.zeros(1, dtype=torch.bfloat16, device=device)
    return PisaPlan(
        exact_index=exact_index,
        exact_row_meta=exact_row_meta,
        approx_index=approx_index,
        approx_row_meta=approx_row_meta,
        schedule=schedule,
        n_dense=n_dense,
        rows=rows,
        batch=batch,
        heads=heads,
        blocks=blocks,
        dim=dim,
        device=device,
        approx_nnz=approx_nnz,
        need_stats=need_stats,
        stats_ids=stats_ids,
        k_center=k_center,
        value_center=value_center,
        counters=torch.zeros(2, dtype=torch.int32, device=device),
    )


def _flatten_qk_v(q, k, v, qs, ks, vs, B):
    is_4d = q.ndim == 4
    if is_4d:
        B_in, H_in = q.shape[0], q.shape[1]
        assert B == B_in, (
            f"4D layout: q.shape[0]={B_in} but caller passed B={B}"
        )
        BH = B_in * H_in
        q = q.view(BH, *q.shape[2:])
        k = k.view(BH, *k.shape[2:])
        v = v.view(BH, *v.shape[2:])
        qs = qs.view(BH, *qs.shape[2:])
        ks = ks.view(BH, *ks.shape[2:])
        vs = vs.view(BH, *vs.shape[2:])
        return q, k, v, qs, ks, vs, BH, B_in, H_in, True
    BH = q.shape[0]
    assert BH % B == 0, f"BH={BH} must be divisible by B={B}"
    return q, k, v, qs, ks, vs, BH, B, BH // B, False


def vsa_qk_fp8_pv_fp4_pisa_csr_dropB(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    vbs: torch.Tensor,
    B: int,
    T: int,
    num_q_blks: int,
    exact_index: Optional[torch.Tensor] = None,
    exact_meta: Optional[torch.Tensor] = None,
    approx_index: Optional[torch.Tensor] = None,
    approx_meta: Optional[torch.Tensor] = None,
    plan: Optional[PisaPlan] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CSR PISA counterpart of :func:`vsa_qk_fp8_pv_fp4_csr_dropB`.

    Q/K/V follow the VSA 3D ``(BH,T,D)`` or 4D ``(B,H,T,D)`` contract.
    Connectivity is two CSR streams (exact + approx).  Pass a :class:`PisaPlan`
    to reuse schedule and centroid buffers; otherwise the four CSR arguments
    are converted via :func:`build_l2_aware_lim_vsa_qk_fp8_pv_fp4_pisa_csr`.
    Either stream may be omitted (empty / ``None``).
    """
    assert q.ndim in (3, 4), (
        f"q must be 3D (BH,T,D) or 4D (B,H,T,D); got shape={tuple(q.shape)}"
    )
    for _name, _t in (("q", q), ("k", k), ("v", v),
                      ("qs", qs), ("ks", ks), ("vs", vs), ("vbs", vbs)):
        assert _t.is_contiguous(), (
            f"vsa_qk_fp8_pv_fp4_pisa_csr_dropB: `{_name}` must be contiguous"
        )

    q, k, v, qs, ks, vs, BH, B_in, H_in, is_4d = _flatten_qk_v(
        q, k, v, qs, ks, vs, B
    )
    device = q.device
    if plan is None:
        plan = build_l2_aware_lim_vsa_qk_fp8_pv_fp4_pisa_csr(
            exact_index=exact_index,
            exact_meta=exact_meta,
            approx_index=approx_index,
            approx_meta=approx_meta,
            batch=B_in,
            heads=H_in,
            blocks=num_q_blks,
            dim=HEAD_DIM,
            device=device,
        )
    elif (
        plan.batch != B_in
        or plan.heads != H_in
        or plan.blocks != num_q_blks
        or plan.device != device
    ):
        raise ValueError("plan was built for a different route geometry or device")

    if out is None:
        out = torch.empty((BH, T, HEAD_DIM), dtype=torch.bfloat16, device=device)
    if lse is None:
        lse = torch.empty((BH, T), dtype=torch.float32, device=device)

    out_kernel = out if out.ndim == 3 else out.view(BH, T, HEAD_DIM)
    lse_kernel = lse if lse.ndim == 2 else lse.view(BH, T)

    if plan.need_stats:
        vsa_qk_fp8_pv_fp4_pisa_stats(
            k, v, ks, vs, vbs,
            plan.k_center, plan.value_center, plan.stats_ids,
            int(plan.stats_ids.numel()), BH, T,
        )

    vsa_qk_fp8_pv_fp4_pisa_csr(
        q, k, v, qs, ks, vs,
        plan.exact_index, plan.exact_row_meta,
        plan.approx_index, plan.approx_row_meta,
        vbs, plan.schedule, plan.k_center, plan.value_center,
        out_kernel, lse_kernel, plan.counters,
        B, T, num_q_blks, plan.n_dense, int(plan.need_stats),
    )

    if is_4d:
        return (out_kernel.view(B_in, H_in, T, HEAD_DIM),
                lse_kernel.view(B_in, H_in, T))
    return out_kernel, lse_kernel
