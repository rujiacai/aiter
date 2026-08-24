# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""QK=FP8 / PV=FP4 mixed-precision VSA block-sparse attention (gfx950).

Public API:
  vsa_qk_fp8_pv_fp4_dropB           # high-level convenience wrapper
  vsa_qk_fp8_pv_fp4                 # raw C++ binding (advanced use)
  build_l2_aware_lim_vsa_qk_fp8_pv_fp4  # GPU helper for task ordering

The same three with CSR connectivity — ``(row_ptr, col_indices)`` in place of
the dense ``(BH*num_q_blks, max_kv)`` rectangle, O(nnz + rows) instead of
O(BH * num_q_blks * max_kv), bitwise-identical results:
  vsa_qk_fp8_pv_fp4_csr_dropB
  vsa_qk_fp8_pv_fp4_csr
  build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr

Emit CSR straight from the sparsity selector; the rectangle is the cost CSR
exists to avoid, so nothing here builds or consumes one.

op_tests/test_vsa_qk_fp8_pv_fp4_csr.py opens with a runnable call sequence for
each, side by side.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ..jit.core import compile_ops

__all__ = [
    "vsa_qk_fp8_pv_fp4_dropB",
    "vsa_qk_fp8_pv_fp4",
    "build_l2_aware_lim_vsa_qk_fp8_pv_fp4",
    "vsa_qk_fp8_pv_fp4_csr_dropB",
    "vsa_qk_fp8_pv_fp4_csr",
    "build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr",
]


# --------------------------------------------------------------------------- #
# Raw binding to the .co launcher (loaded from
# /opt/aiter/hsa/gfx950/vsa/vsa_qk_fp8_pv_fp4.co at first call).
# --------------------------------------------------------------------------- #
@compile_ops("module_vsa_qk_fp8_pv_fp4")
def vsa_qk_fp8_pv_fp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    vbs: torch.Tensor,
    lim: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    counters: torch.Tensor,
    B: int,
    T: int,
    num_q_blks: int,
    max_kv: int,
    n_dense: int,
) -> None: ...


# CSR connectivity ABI (vsa/vsa_qk_fp8_pv_fp4_csr.co).
@compile_ops("module_vsa_qk_fp8_pv_fp4")
def vsa_qk_fp8_pv_fp4_csr(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    q2k_col_indices: torch.Tensor,
    q2k_row_meta: torch.Tensor,
    vbs: torch.Tensor,
    lim: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    counters: torch.Tensor,
    B: int,
    T: int,
    num_q_blks: int,
    n_dense: int,
) -> None: ...


# --------------------------------------------------------------------------- #
# L2-aware task ordering — Triton-fused composite key + single GPU radix sort.
#
# Sort tasks by (first_kv_block // FKV_BAND_SIZE   ASC,   -- HBM/L2 band
#                -q2k_num                          ASC)   -- longest-job-first
# packed into a single int32 composite key:
#   high (32-QN_BITS) bits = (first_kv_block >> FKV_LOG2)  -- HBM-row band
#   low  QN_BITS      bits = (2*max_kv - q2k_num) clipped  -- LJF within band
#
# The kernel partitions the schedule into a "dense" head and a "sparse" tail
# using the host-side `n_dense` count (threshold = max_kv * 7/8 by default);
# inside each partition tiles are L2-banded so adjacent waves share HBM rows.
# --------------------------------------------------------------------------- #
_LIM_BLOCK = 256


@triton.jit
def _vsa_qk_fp8_pv_fp4_composite_key_kernel(
    q2k_num_ptr,        # *int32   (n_tasks,)
    q2k_idx_ptr,        # *int32   (n_tasks, idx_stride)  first col = first KV blk
    is_dense_ptr,       # *int32   (n_tasks,)             1 if q2k_num >= threshold
    out_key_ptr,        # *int32   (n_tasks,)
    n_tasks: tl.int32,
    max_kv_x2: tl.int32,
    threshold: tl.int32,
    idx_stride0: tl.int64,   # int64 to survive long contexts where
                             #   max_kv * (n_tasks-1) > INT32_MAX
                             # (e.g. Seedance 5.9M tokens => max_kv = n_tasks = 46_425,
                             #  product 2.155e9 overflows int32 by 7.8M).
    BLOCK: tl.constexpr,
    FKV_LOG2: tl.constexpr,
    QN_BITS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_tasks
    qn = tl.load(q2k_num_ptr + offs, mask=mask, other=0)
    # Promote offs to int64 BEFORE the multiply so the address arithmetic
    # cannot wrap at huge num_q_blks; without this Triton would do an
    # int32 multiply and then sign-extend the wrapped result, jumping to
    # a wild GPU address (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION).
    fkv = tl.load(q2k_idx_ptr + offs.to(tl.int64) * idx_stride0, mask=mask, other=0)
    band = fkv >> FKV_LOG2
    qn_neg = (max_kv_x2 - qn) & ((1 << QN_BITS) - 1)
    key = (band << QN_BITS) | qn_neg
    tl.store(out_key_ptr + offs, key, mask=mask)
    tl.store(is_dense_ptr + offs,
             tl.where(qn >= threshold, 1, 0), mask=mask)


def build_l2_aware_lim_vsa_qk_fp8_pv_fp4(
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    max_kv: int,
    fkv_band_size: int = 512,
    dense_ratio: float = 7.0 / 8.0,
) -> Tuple[torch.Tensor, int]:
    """Return ``(lim, n_dense)`` — L2-cache-aware task ordering for the
    kernel's outer scheduler.

    Two-level sort packed into a single int32 composite key:
      1. ``first_kv_block // fkv_band_size`` ASC  — adjacent tasks share an HBM row
      2. ``-q2k_num``                        ASC  — longest job first within band

    A leading **dense partition** of length ``n_dense`` is concatenated before
    the sparse tail (``q2k_num >= max_kv * dense_ratio`` -> dense).  The kernel
    consumes that partition through its dedicated dense-tile path which keeps
    per-K loop body tight when every q-row touches every KV block.

    Cost: O(n_tasks) Triton key build + O(n) radix sort + 2 small splits.

    Accepts ``q2k_idx`` / ``q2k_num`` of any rank; they are internally
    flattened to ``(n_tasks, max_kv)`` and ``(n_tasks,)``, matching what
    the caller will subsequently feed to ``vsa_qk_fp8_pv_fp4_dropB`` after
    its own layout normalisation.  Both inputs must be contiguous; we use
    ``.view`` for the flatten (0-copy, raises on non-contiguous) instead of
    ``.reshape().contiguous()`` to avoid a silent copy on the perf path.
    """
    q2k_idx = q2k_idx.view(-1, q2k_idx.shape[-1])
    q2k_num = q2k_num.view(-1)
    n = q2k_num.numel()
    threshold = int(max_kv * dense_ratio)

    key_buf      = torch.empty(n, dtype=torch.int32, device=q2k_num.device)
    is_dense_buf = torch.empty(n, dtype=torch.int32, device=q2k_num.device)
    grid = ((n + _LIM_BLOCK - 1) // _LIM_BLOCK,)
    _vsa_qk_fp8_pv_fp4_composite_key_kernel[grid](
        q2k_num, q2k_idx, is_dense_buf, key_buf,
        n, 2 * max_kv, threshold, q2k_idx.stride(0),
        BLOCK=_LIM_BLOCK,
        FKV_LOG2=fkv_band_size.bit_length() - 1,
        QN_BITS=13,  # max_kv < 4096 fits in 13 bits; with negation max_kv < 8192
    )

    # Sort the full task list by the composite key.
    order = torch.argsort(key_buf).to(torch.int32)
    # Partition that ordered list into dense vs sparse, preserving the
    # composite-key ordering within each partition.
    is_dense_ord = is_dense_buf[order.long()]
    d_mask = is_dense_ord != 0
    d_order = order[d_mask]
    s_order = order[~d_mask]
    lim = torch.cat([d_order, s_order]).contiguous()
    n_dense = int(d_order.numel())
    return lim, n_dense


# --------------------------------------------------------------------------- #
# CSR task ordering.
#
# Same two-level sort as the indexed builder, but reading connectivity from
# (row_ptr, col_indices) and emitting, in the same pass, the packed per-row
# metadata record the CSR kernel ABI expects.  Fusing them costs nothing: nnz
# and start are the loop bounds the sort already needs, and first_kv is the
# band key it already computes.
#
# The composite key is int64 here rather than the indexed path's packed int32.
# That packing is only sound while max_kv < 8192; past it the low QN_BITS field
# wraps and longest-job-first degrades into noise.  Ordering is a throughput
# property, not a correctness one — the kernel's result is invariant to the
# permutation in `lim` — but at the context lengths CSR exists to serve, the
# indexed key is always in the wrapped regime.
# --------------------------------------------------------------------------- #
_EMPTY_BAND = 1 << 40   # sorts empty rows last; kernel early-outs before use


@triton.jit
def _vsa_qk_fp8_pv_fp4_csr_meta_kernel(
    row_ptr_ptr,        # *int32/int64 (n_tasks+1,)
    col_ind_ptr,        # *int32       (nnz,)
    row_meta_ptr,       # *int32       (n_tasks, 4)  OUT {nnz, start, first_kv, _}
    is_dense_ptr,       # *int32       (n_tasks,)    OUT
    out_key_ptr,        # *int64       (n_tasks,)    OUT
    n_tasks: tl.int32,
    nnz_total: tl.int64,
    num_q_blks_x2: tl.int64,
    radix: tl.int64,
    empty_band: tl.int64,
    threshold: tl.int32,
    BLOCK: tl.constexpr,
    FKV_LOG2: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_tasks

    start = tl.load(row_ptr_ptr + offs, mask=mask, other=0).to(tl.int64)
    end = tl.load(row_ptr_ptr + offs + 1, mask=mask, other=0).to(tl.int64)
    nnz = end - start

    # A trailing empty row has start == nnz_total, one past the payload, so the
    # gather is predicated off rather than clamped.
    live = mask & (nnz > 0) & (start < nnz_total)
    fkv = tl.load(col_ind_ptr + start, mask=live, other=0).to(tl.int64)

    band = tl.where(nnz > 0, fkv >> FKV_LOG2, empty_band)
    tl.store(out_key_ptr + offs, band * radix + (num_q_blks_x2 - nnz), mask=mask)

    rec = row_meta_ptr + offs * 4
    tl.store(rec + 0, nnz.to(tl.int32), mask=mask)
    tl.store(rec + 1, start.to(tl.int32), mask=mask)
    tl.store(rec + 2, fkv.to(tl.int32), mask=mask)
    tl.store(rec + 3, tl.zeros_like(nnz).to(tl.int32), mask=mask)
    tl.store(is_dense_ptr + offs, tl.where(nnz >= threshold, 1, 0), mask=mask)


def build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr(
    q2k_row_ptr: torch.Tensor,
    q2k_col_indices: torch.Tensor,
    num_q_blks: int,
    fkv_band_size: int = 512,
    dense_ratio: float = 7.0 / 8.0,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """Return ``(lim, n_dense, row_meta)`` for the CSR ABI.

    ``q2k_row_ptr`` is the usual CSR row pointer of length ``n_tasks + 1``
    (int32 or int64); ``q2k_col_indices`` holds ``row_ptr[-1]`` KV block ids,
    with each row's ids in the order the kernel should consume them.

    ``row_meta`` is ``(n_tasks, 4)`` int32 holding ``{nnz, start, first_kv, 0}``
    in **schedule order** — ``row_meta[t]`` describes the tile ``lim[t]``, not
    logical row ``t``.  It and ``q2k_col_indices`` are the connectivity
    arguments of ``vsa_qk_fp8_pv_fp4_csr``.
    """
    assert q2k_row_ptr.is_contiguous() and q2k_col_indices.is_contiguous(), (
        "build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr: row_ptr and col_indices "
        "must be contiguous"
    )
    assert q2k_col_indices.dtype == torch.int32, (
        f"q2k_col_indices must be int32, got {q2k_col_indices.dtype}"
    )

    row_ptr = q2k_row_ptr.view(-1)
    n = row_ptr.numel() - 1
    dev = row_ptr.device
    nnz_total = q2k_col_indices.numel()
    assert nnz_total <= 0x7FFFFFFF, (
        f"nnz={nnz_total} exceeds the int32 row_start range the kernel ABI uses"
    )

    i32 = dict(dtype=torch.int32, device=dev)
    row_meta = torch.empty((n, 4), **i32)
    is_dense_buf = torch.empty(n, **i32)
    key_buf = torch.empty(n, dtype=torch.int64, device=dev)

    grid = ((n + _LIM_BLOCK - 1) // _LIM_BLOCK,)
    _vsa_qk_fp8_pv_fp4_csr_meta_kernel[grid](
        row_ptr, q2k_col_indices,
        row_meta, is_dense_buf, key_buf,
        n,
        nnz_total,
        2 * num_q_blks,
        2 * num_q_blks + 1,      # radix: leaves room for nnz in [0, 2*num_q_blks]
        _EMPTY_BAND,
        int(num_q_blks * dense_ratio),
        BLOCK=_LIM_BLOCK,
        FKV_LOG2=fkv_band_size.bit_length() - 1,
    )

    order = torch.argsort(key_buf).to(torch.int32)
    d_mask = is_dense_buf[order.long()] != 0
    lim = torch.cat([order[d_mask], order[~d_mask]]).contiguous()
    # Emit the records through the schedule, so row_meta[t] describes the tile
    # lim[t].  The kernel can then address it by task index and issue it in
    # parallel with the lim load instead of chaining behind it.
    row_meta = row_meta[lim.long()].contiguous()
    return lim, int(d_mask.sum().item()), row_meta


# --------------------------------------------------------------------------- #
# High-level wrapper — auto-allocates out / lse / counters and zeros the
# atomic dispatch counters before each launch.  Mirrors the style of the
# stand-alone /home/vsa_qk_fp8_pv_fp4_hip/vsa_hybrid.py:vsa_qk_fp8_pv_fp4().
# --------------------------------------------------------------------------- #
_HEAD_DIM = 128


def vsa_qk_fp8_pv_fp4_dropB(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    vbs: torch.Tensor,
    lim: torch.Tensor,
    n_dense: int,
    B: int,
    T: int,
    num_q_blks: int,
    max_kv: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    counters: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the QK=FP8 / PV=FP4 mixed-precision VSA kernel.

    Accepts EITHER input layout; ``out`` / ``lse`` are returned in the same
    rank as the input so callers don't need to reshape on either side:

      * **Flat ("BH")** — ``q.ndim == 3``::
          q, k:    (BH, T, 128)            float8_e4m3fn
          v:       (BH, T, 64)             uint8   (FP4, 2 nibbles/byte)
          qs, ks:  (BH, T, 4)              uint8   (E8M0 per 32-elem group)
          vs:      (BH, num_q_blks, 128, 4) uint8  (E8M0 per K-block)
          q2k_idx: (BH * num_q_blks, max_kv) int32
          q2k_num: (BH * num_q_blks,)        int32
          out:     (BH, T, 128)              bfloat16
          lse:     (BH, T)                   float32

      * **Batched ("B,H")** — ``q.ndim == 4``::
          q, k:    (B, H, T, 128)              float8_e4m3fn
          v:       (B, H, T, 64)               uint8
          qs, ks:  (B, H, T, 4)                uint8
          vs:      (B, H, num_q_blks, 128, 4)  uint8
          q2k_idx: (..., num_q_blks, max_kv)   int32   (any leading shape)
          q2k_num: (..., num_q_blks)           int32   (any leading shape)
          out:     (B, H, T, 128)              bfloat16  (returned in this shape)
          lse:     (B, H, T)                   float32   (returned in this shape)

    Always-flat regardless of layout::
      vbs: (num_q_blks,)              int32
      lim: (BH * num_q_blks,)         int32   (produced by build_l2_aware_lim_...)

    Numerical contract (sparsity 0.0846, seed-independent, T = 50k..1M):
      - cosine similarity vs FP32 ref: 0.9826 .. 0.9833
      - cos(LSE)                     : 1.000000
      - max |diff|                   : 6e-3 (T=1M) .. 2.4e-2 (T=50k)
      - no NaN / Inf at any tested size
    """
    assert q.ndim in (3, 4), (
        f"q must be 3D (BH,T,D) or 4D (B,H,T,D); got shape={tuple(q.shape)}"
    )
    # Kernel ABI is row-major over (BH, T, D) -- silently calling .contiguous()
    # on a transposed view would copy hundreds of MB on every attention step.
    # Fail-fast instead so the caller can fix it once at construction time.
    for _name, _t in (("q", q), ("k", k), ("v", v),
                      ("qs", qs), ("ks", ks), ("vs", vs),
                      ("q2k_idx", q2k_idx), ("q2k_num", q2k_num), ("lim", lim)):
        assert _t.is_contiguous(), (
            f"vsa_qk_fp8_pv_fp4_dropB: `{_name}` must be contiguous "
            f"(shape={tuple(_t.shape)}, strides={tuple(_t.stride())}); "
            f"call .contiguous() at allocation time"
        )

    is_4d = q.ndim == 4
    if is_4d:
        B_in, H_in = q.shape[0], q.shape[1]
        assert B == B_in, (
            f"4D layout: q.shape[0]={B_in} but caller passed B={B}; "
            f"these must match (B is the batch dim of q in 4D mode)"
        )
        BH = B_in * H_in
        # All inputs verified contiguous above, so .view is 0-copy and safe.
        # Contiguous (B, H, T, D) is byte-identical to (B*H, T, D) -- the
        # flatten is purely a metadata change (shape + stride), no DMA.
        q  = q .view(BH, *q .shape[2:])
        k  = k .view(BH, *k .shape[2:])
        v  = v .view(BH, *v .shape[2:])
        qs = qs.view(BH, *qs.shape[2:])
        ks = ks.view(BH, *ks.shape[2:])
        vs = vs.view(BH, *vs.shape[2:])
        q2k_idx = q2k_idx.view(BH * num_q_blks, -1)
        q2k_num = q2k_num.view(BH * num_q_blks)
        if lim.ndim > 1:
            lim = lim.view(-1)
    else:
        BH = q.shape[0]

    assert BH % B == 0, f"BH={BH} must be divisible by B={B}"
    assert q2k_num.numel() == BH * num_q_blks, (
        f"q2k_num.numel()={q2k_num.numel()} != BH*num_q_blks={BH*num_q_blks}; "
        f"in 3D mode flatten q2k_num to (BH*num_q_blks,) before calling"
    )
    assert lim.numel() == BH * num_q_blks, (
        f"lim.numel()={lim.numel()} != BH*num_q_blks={BH*num_q_blks}; "
        f"lim must be flat 1D of length BH*num_q_blks"
    )

    if out is None:
        out = torch.empty((BH, T, _HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    if lse is None:
        lse = torch.empty((BH, T), dtype=torch.float32, device=q.device)
    if counters is None:
        counters = torch.zeros(2, dtype=torch.int32, device=q.device)

    # Caller may hand us pre-allocated 4D out / 3D lse to match input layout;
    # kernel ABI is (BH,T,D) / (BH,T) so 0-copy view-flatten before the launch.
    out_kernel = out if out.ndim == 3 else out.view(BH, T, _HEAD_DIM)
    lse_kernel = lse if lse.ndim == 2 else lse.view(BH, T)

    vsa_qk_fp8_pv_fp4(
        q, k, v,
        qs, ks, vs,
        q2k_idx, q2k_num, vbs,
        lim, out_kernel, lse_kernel, counters,
        B, T, num_q_blks, max_kv, n_dense,
    )

    if is_4d:
        out = out_kernel.view(B_in, H_in, T, _HEAD_DIM)
        lse = lse_kernel.view(B_in, H_in, T)
    else:
        out = out_kernel
        lse = lse_kernel
    return out, lse


def vsa_qk_fp8_pv_fp4_csr_dropB(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    q2k_col_indices: torch.Tensor,
    q2k_row_meta: torch.Tensor,
    vbs: torch.Tensor,
    lim: torch.Tensor,
    n_dense: int,
    B: int,
    T: int,
    num_q_blks: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    counters: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CSR-connectivity counterpart of :func:`vsa_qk_fp8_pv_fp4_dropB`.

    Q/K/V, ``vbs``, ``out`` and ``lse`` are exactly as documented there,
    including the 3D-or-4D contract and the rank-preserving return.  Only the
    connectivity differs: ``q2k_idx`` / ``q2k_num`` / ``max_kv`` give way to::

      q2k_col_indices: (nnz,)               int32  CSR payload, each row's KV
                                                   block ids contiguous
      q2k_row_meta:    (BH*num_q_blks, 4)   int32  {nnz, row_start, first_kv, _}
      lim:             (BH*num_q_blks,)     int32

    All three come from :func:`build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr`.
    ``row_meta`` is in schedule order — record ``t`` describes tile ``lim[t]``,
    not logical row ``t`` — so it must come from the same builder call as the
    ``lim`` passed with it.  Both depend on the connectivity alone, so build
    them once per sparsity pattern and reuse across forwards.

    Given the same connectivity this is bitwise-identical to the indexed launch
    and carries the same numerical contract.  ``row_start`` is int32, capping
    one launch at 2**31 CSR entries.
    """
    assert q.ndim in (3, 4), (
        f"q must be 3D (BH,T,D) or 4D (B,H,T,D); got shape={tuple(q.shape)}"
    )
    for _name, _t in (("q", q), ("k", k), ("v", v),
                      ("qs", qs), ("ks", ks), ("vs", vs),
                      ("q2k_col_indices", q2k_col_indices),
                      ("q2k_row_meta", q2k_row_meta), ("lim", lim)):
        assert _t.is_contiguous(), (
            f"vsa_qk_fp8_pv_fp4_csr_dropB: `{_name}` must be contiguous "
            f"(shape={tuple(_t.shape)}, strides={tuple(_t.stride())}); "
            f"call .contiguous() at allocation time"
        )

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
        if lim.ndim > 1:
            lim = lim.view(-1)
    else:
        BH = q.shape[0]

    n_rows = BH * num_q_blks
    assert BH % B == 0, f"BH={BH} must be divisible by B={B}"
    assert lim.numel() == n_rows, (
        f"lim.numel()={lim.numel()} != BH*num_q_blks={n_rows}"
    )
    assert tuple(q2k_row_meta.shape) == (n_rows, 4), (
        f"q2k_row_meta must be ({n_rows}, 4); got {tuple(q2k_row_meta.shape)}"
    )

    if out is None:
        out = torch.empty((BH, T, _HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    if lse is None:
        lse = torch.empty((BH, T), dtype=torch.float32, device=q.device)
    if counters is None:
        counters = torch.zeros(2, dtype=torch.int32, device=q.device)

    out_kernel = out if out.ndim == 3 else out.view(BH, T, _HEAD_DIM)
    lse_kernel = lse if lse.ndim == 2 else lse.view(BH, T)

    vsa_qk_fp8_pv_fp4_csr(
        q, k, v,
        qs, ks, vs,
        q2k_col_indices, q2k_row_meta, vbs,
        lim, out_kernel, lse_kernel, counters,
        B, T, num_q_blks, n_dense,
    )

    if is_4d:
        return (out_kernel.view(B_in, H_in, T, _HEAD_DIM),
                lse_kernel.view(B_in, H_in, T))
    return out_kernel, lse_kernel
