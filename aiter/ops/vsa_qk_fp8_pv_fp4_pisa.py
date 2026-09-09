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
import triton
import triton.language as tl

from ..jit.core import compile_ops
from .vsa_qk_fp8_pv_fp4 import (
    build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr,
    vsa_qk_fp8_pv_fp4_csr_dropB,
)

BLOCK = 128
HEAD_DIM = 128
_LIM_BLOCK = 256

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


def _shape_check_csr(
    name: str,
    col_indices: torch.Tensor,
    meta: torch.Tensor,
    rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten to the kernel's global CSR layout.

    Only metadata is inspected (dtype / device / shape), so this costs no
    device synchronisation.  Payload checks live in :func:`_validate_csr`.
    """
    col_indices = col_indices.reshape(-1).contiguous()
    meta = meta.reshape(-1, 2).contiguous()
    if col_indices.dtype != torch.int32 or meta.dtype != torch.int32:
        raise ValueError(f"{name} CSR tensors must be INT32")
    if col_indices.device != device or meta.device != device:
        raise ValueError(f"{name} CSR tensors must be on {device}")
    if meta.shape[0] != rows:
        raise ValueError(f"{name}_meta must contain {rows} rows")
    return col_indices, meta


def _validate_csr(
    name: str,
    col_indices: torch.Tensor,
    meta: torch.Tensor,
    blocks: int,
) -> None:
    """Assert the CSR payload is canonical.  Reads back from the device."""
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


def _normalize_csr_layout(
    name: str,
    col_indices: torch.Tensor,
    meta: torch.Tensor,
    batch: int,
    heads: int,
    query_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept either flat global CSR or the per-(B,H) padded streams.

    Flat CSR is the fast path and returns untouched; the ``[B,H,C]`` form is
    restitched into one global stream without any device synchronisation.
    """
    if col_indices.ndim == 1:
        return col_indices, meta
    if col_indices.ndim != 3 or col_indices.shape[:2] != (batch, heads):
        raise ValueError(
            f"{name}_index must be [nnz] or [B,H,C], got {tuple(col_indices.shape)}"
        )
    if meta.shape != (batch, heads, query_blocks, 2):
        raise ValueError(f"{name}_meta must be [B,H,QB,2]")

    capacity = col_indices.shape[-1]
    flat_index = col_indices.reshape(batch * heads, capacity)
    flat_meta = meta.reshape(batch * heads, query_blocks, 2)
    counts = flat_meta[..., 1].to(torch.int64)
    used = counts.sum(1)
    lane = torch.arange(capacity, device=col_indices.device)
    # Boolean-mask selection on a contiguous 2D tensor yields row-major order,
    # which is exactly the global CSR order the kernel expects.
    packed = flat_index[lane.unsqueeze(0) < used.unsqueeze(1)]
    base = torch.cumsum(used, 0) - used
    offsets = flat_meta[..., 0].to(torch.int64) + base.unsqueeze(1)
    adjusted = torch.stack((offsets, counts), dim=-1)
    return (
        packed.to(torch.int32).contiguous(),
        adjusted.to(torch.int32).view(batch, heads, query_blocks, 2).contiguous(),
    )


@triton.jit
def _pisa_csr_meta_kernel(
    exact_meta_ptr,       # *int32 (n_tasks, 2): {start, nnz}
    exact_col_ptr,        # *int32 (exact_nnz,)
    approx_meta_ptr,      # *int32 (n_tasks, 2): {start, nnz}
    approx_col_ptr,       # *int32 (approx_nnz,)
    exact_row_meta_ptr,   # *int32 (n_tasks, 4), logical order
    approx_row_meta_ptr,  # *int32 (n_tasks, 4), logical order
    is_dense_ptr,         # *int32 (n_tasks,)
    key_ptr,              # *int64 (n_tasks,)
    n_tasks: tl.int32,
    exact_nnz: tl.int64,
    approx_nnz: tl.int64,
    max_work: tl.int64,
    radix: tl.int64,
    dense_threshold: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    FKV_LOG2: tl.constexpr,
):
    """Fused counterpart of VSA's ``_csr_meta_kernel`` for two CSR streams."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_tasks

    exact_start = tl.load(exact_meta_ptr + offs * 2, mask=mask, other=0).to(
        tl.int64
    )
    exact_count = tl.load(
        exact_meta_ptr + offs * 2 + 1, mask=mask, other=0
    ).to(tl.int64)
    approx_start = tl.load(approx_meta_ptr + offs * 2, mask=mask, other=0).to(
        tl.int64
    )
    approx_count = tl.load(
        approx_meta_ptr + offs * 2 + 1, mask=mask, other=0
    ).to(tl.int64)

    exact_live = mask & (exact_count > 0) & (exact_start < exact_nnz)
    approx_live = mask & (approx_count > 0) & (approx_start < approx_nnz)
    exact_first = tl.load(
        exact_col_ptr + exact_start, mask=exact_live, other=1 << 30
    ).to(tl.int64)
    approx_first = tl.load(
        approx_col_ptr + approx_start, mask=approx_live, other=1 << 30
    ).to(tl.int64)
    first = tl.minimum(exact_first, approx_first)

    # Exact work is one 128-token tile; approximate work is one rank-1 update.
    work = exact_count * 128 + approx_count
    key = (first >> FKV_LOG2) * radix + (max_work - work)
    tl.store(key_ptr + offs, key, mask=mask)
    # Density is a property of the row's connectivity, not of which stream
    # carries it: moving routes to the approximate side leaves the row just as
    # wide, and it stays the heaviest tile in the launch.  Counting only the
    # exact stream drops these rows out of the dense partition partway up the
    # rho sweep and costs more than the approximate route saves.
    tl.store(
        is_dense_ptr + offs,
        tl.where(exact_count + approx_count >= dense_threshold, 1, 0),
        mask=mask,
    )

    exact_rec = exact_row_meta_ptr + offs * 4
    tl.store(exact_rec, exact_count.to(tl.int32), mask=mask)
    tl.store(exact_rec + 1, exact_start.to(tl.int32), mask=mask)
    tl.store(exact_rec + 2, exact_first.to(tl.int32), mask=mask)
    tl.store(exact_rec + 3, 0, mask=mask)

    approx_rec = approx_row_meta_ptr + offs * 4
    tl.store(approx_rec, approx_count.to(tl.int32), mask=mask)
    tl.store(approx_rec + 1, approx_start.to(tl.int32), mask=mask)
    tl.store(approx_rec + 2, approx_first.to(tl.int32), mask=mask)
    tl.store(approx_rec + 3, 0, mask=mask)


def _build_fused_schedule(
    exact_index: torch.Tensor,
    exact_meta: torch.Tensor,
    approx_index: torch.Tensor,
    approx_meta: torch.Tensor,
    blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VSA-style one-kernel key/meta build, then radix sort and partition."""
    n = exact_meta.shape[0]
    dev = exact_meta.device
    i32 = dict(dtype=torch.int32, device=dev)
    exact_row_meta = torch.empty((n, 4), **i32)
    approx_row_meta = torch.empty((n, 4), **i32)
    is_dense = torch.empty(n, **i32)
    key = torch.empty(n, dtype=torch.int64, device=dev)

    radix = blocks * BLOCK * 2 + blocks + 1
    grid = ((n + _LIM_BLOCK - 1) // _LIM_BLOCK,)
    # A zero-length tensor may carry a null data pointer.  The loads are
    # masked when nnz=0, but HIP still requires a valid kernel argument.
    exact_col_ptr = exact_index if exact_index.numel() else exact_meta
    approx_col_ptr = approx_index if approx_index.numel() else approx_meta
    _pisa_csr_meta_kernel[grid](
        exact_meta,
        exact_col_ptr,
        approx_meta,
        approx_col_ptr,
        exact_row_meta,
        approx_row_meta,
        is_dense,
        key,
        n,
        exact_index.numel(),
        approx_index.numel(),
        blocks * BLOCK + blocks,
        radix,
        int(blocks * 7 / 8),
        BLOCK_SIZE=_LIM_BLOCK,
        FKV_LOG2=9,  # 512-KV-block L2 band, matching VSA.
    )

    order = torch.argsort(key, stable=True).to(torch.int32)
    dense_mask = is_dense[order.long()] != 0
    schedule = torch.cat((order[dense_mask], order[~dense_mask])).contiguous()
    return (
        schedule,
        dense_mask.sum(),
        exact_row_meta[schedule.long()].contiguous(),
        approx_row_meta[schedule.long()].contiguous(),
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
        "vsa_fallback",
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
    validate: bool = False,
) -> PisaPlan:
    """Order the CSR routes and precompute everything the launch reuses.

    Counterpart of ``build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr``, and priced
    like it: the CSR payload is trusted, and the only device readback is one
    batched fetch of the scalars the launch ABI needs as ints (``n_dense``,
    plus the approximate row width that decides whether centroids run).

    ``validate=True`` additionally proves the two streams are canonical
    ``[offset, count]`` CSR with in-range block ids.  That costs several
    synchronising reductions per stream, so keep it for bring-up and tests
    rather than the per-step path.  A malformed stream without it is a
    kernel-side out-of-bounds read, not a Python exception.
    """
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
    exact_index, exact_meta = _shape_check_csr(
        "exact", exact_index, exact_meta, rows, device
    )
    approx_index, approx_meta = _shape_check_csr(
        "approx", approx_index, approx_meta, rows, device
    )
    if validate:
        _validate_csr("exact", exact_index, exact_meta, blocks)
        _validate_csr("approx", approx_index, approx_meta, blocks)
    approx_nnz = int(approx_index.numel())

    # With no approximate route the fused kernel computes exactly what the VSA
    # CSR kernel does, bitwise, but pays for the centroid path it never enters:
    # both paths are inlined into one loop body, so their live ranges share a
    # register budget and the exact loop reloads 9 spilled values per KV block
    # against VSA's 4.  Hand those launches to the VSA kernel instead, and take
    # its schedule rather than building one the launch would not read.
    vsa_fallback = None
    if approx_nnz == 0 and exact_index.numel():
        row_ptr = torch.cat(
            (
                exact_meta[:, 0],
                torch.full((1,), exact_index.numel(), dtype=torch.int32,
                           device=device),
            )
        )
        vsa_fallback = build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr(
            row_ptr, exact_index, blocks
        )
        schedule, n_dense, exact_row_meta = vsa_fallback
        approx_row_meta = torch.empty(0, dtype=torch.int32, device=device)
        max_approx = 0
    else:
        schedule, n_dense_dev, exact_row_meta, approx_row_meta = (
            _build_fused_schedule(
                exact_index, exact_meta, approx_index, approx_meta, blocks
            )
        )
        # Both scalars have to reach the launcher as Python ints, so fetch
        # them together: one synchronisation instead of one apiece.
        max_approx_dev = (
            approx_meta[:, 1].max()
            if approx_nnz
            else torch.zeros((), dtype=torch.int32, device=device)
        )
        n_dense, max_approx = (
            torch.stack(
                (n_dense_dev.to(torch.int64), max_approx_dev.to(torch.int64))
            ).tolist()
        )

    stats_ids = torch.empty(0, dtype=torch.int32, device=device)
    need_stats = max_approx > APX_PROMOTE_EXACT
    if need_stats and approx_nnz <= 8_000_000:
        row = torch.repeat_interleave(
            torch.arange(rows, device=device, dtype=torch.int32),
            approx_meta[:, 1],
        )
        linear = torch.div(row, blocks, rounding_mode="floor") * blocks + (
            approx_index
        )
        # linear indexes [0, rows), so flag-and-compact beats torch.unique:
        # no sort of the nnz-long list, and nnz far exceeds rows here.
        seen = torch.zeros(rows, dtype=torch.bool, device=device)
        seen[linear.long()] = True
        unique_ids = seen.nonzero().flatten()
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
        vsa_fallback=vsa_fallback,
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
    validate: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CSR PISA counterpart of :func:`vsa_qk_fp8_pv_fp4_csr_dropB`.

    Q/K/V follow the VSA 3D ``(BH,T,D)`` or 4D ``(B,H,T,D)`` contract.
    Connectivity is two CSR streams (exact + approx).  Either stream may be
    omitted (empty / ``None``).

    The schedule and centroid buffers depend only on the routing pattern, so
    build a :class:`PisaPlan` once per pattern and pass it back in; without one
    this rebuilds the plan on every call, which for sparse routes costs more
    than the kernel itself.  ``validate`` is forwarded to the builder and is
    ignored when ``plan`` is supplied.
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
            validate=validate,
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

    if plan.vsa_fallback is not None:
        lim, vsa_n_dense, vsa_row_meta = plan.vsa_fallback
        vsa_qk_fp8_pv_fp4_csr_dropB(
            q=q, k=k, v=v, qs=qs, ks=ks, vs=vs,
            q2k_col_indices=plan.exact_index, q2k_row_meta=vsa_row_meta,
            vbs=vbs, lim=lim, n_dense=vsa_n_dense,
            B=B, T=T, num_q_blks=num_q_blks,
            out=out_kernel, lse=lse_kernel, counters=plan.counters,
        )
        if is_4d:
            return (out_kernel.view(B_in, H_in, T, HEAD_DIM),
                    lse_kernel.view(B_in, H_in, T))
        return out_kernel, lse_kernel

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
