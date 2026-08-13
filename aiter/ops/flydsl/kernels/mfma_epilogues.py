"""Reusable epilogue helpers for MFMA 16x16-based kernels.

This module provides:

- `mfma_epilog(...)`
  A single entrypoint that dispatches to either the default row-epilogue or the
  LDS CShuffle epilogue based on input parameters.

- `default_epilog(...)` (implementation helper)
  A lightweight row-iterator for the common MFMA accumulator-to-output mapping
  (mi in [0,m_repeat), ii in [0,4), row = bx_m + mi*16 + lane_div_16*4 + ii).
  The caller supplies `body_row(...)` that performs the per-row epilogue work
  (e.g. loads scales once, loops over ni, stores).

- `c_shuffle_epilog(...)` (implementation helper)
  A LDS CShuffle epilogue skeleton:
    1) call `write_row_to_lds(...)` for each MFMA output row to populate `lds_out`
       in row-major [tile_m, tile_n] order
    2) barrier
    3) remap threads into (MLane, NLane) = (8,32) and read half2 from LDS,
       then call `store_pair(...)` to emit the final global store/atomic.

These helpers are intentionally *dialect-agnostic*: callers pass the dialect
modules (`arith`, `vector`, `gpu`) and the `range_constexpr` iterator.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from flydsl._mlir import ir
import flydsl.expr as fx
from flydsl.expr.typing import T


@contextmanager
def _if_then(if_op, scf):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def default_epilog(
    *,
    arith,
    range_constexpr,
    m_repeat: int,
    lane_div_16,
    bx_m,
    body_row: Callable,
):
    """Iterate the standard MFMA 16x16 row mapping and call `body_row(...)`.

    The mapping matches the common MFMA fragment layout used across kernels in this repo.

    Args:
      arith: flydsl arith ext module.
      range_constexpr: compile-time unrolled range helper.
      m_repeat: tile_m // 16 (python int).
      lane_div_16: index Value (0..3).
      bx_m: base row (index Value). For MoE, this is the base sorted-row for the tile.
      body_row: callback invoked as:
        body_row(mi=<int>, ii=<int>, row_in_tile=<index>, row=<index>)
    """
    bx_m_v = bx_m
    lane_div_16_mul4 = lane_div_16 * 4
    ii_idx_list = [fx.Index(ii) for ii in range(4)]

    for mi in range_constexpr(m_repeat):
        mi_base = arith.constant(mi * 16, index=True)
        for ii in range_constexpr(4):
            row_off = lane_div_16_mul4 + ii_idx_list[ii]
            row_in_tile = mi_base + row_off
            row = bx_m_v + row_in_tile
            body_row(mi=mi, ii=ii, row_in_tile=row_in_tile, row=row)


def c_shuffle_epilog(
    *,
    arith,
    vector,
    gpu,
    scf=None,
    range_constexpr,
    # Tile params
    tile_m: int,
    tile_n: int,
    e_vec: int = 2,
    cshuffle_nlane: int = 32,
    block_size: int = 256,
    m_repeat: int,
    num_acc_n: int,
    # B-first accumulator orientation: only Step 1's thread->element mapping
    # changes; Step 2 reads lds_out through its own mapping and is unaffected.
    bfirst: bool = False,
    # Row stride of lds_out in elements; defaults to tile_n (unpadded).
    lds_out_stride: int | None = None,
    # Interleave Step 1 and Step 2 in chunks of `chunk_m` rows instead of staging
    # the whole tile.  Step 1's `mi` writes rows [mi*16, mi*16+16) and Step 2's
    # `mr` reads rows [mr*CShuffleMLane, +CShuffleMLane), so when those two spans
    # coincide each chunk can be written and read back before the next one starts
    # and lds_out only ever holds one chunk.  Costs one extra barrier pair per
    # chunk; buys back the LDS that caps occupancy.  B-first only.
    chunk_m: int | None = None,
    # Thread mapping inputs
    tx,
    lane_div_16,
    lane_mod_16,
    bx_m,
    by_n,
    n_tile_base,
    # LDS buffer (f16 view, row-major [tile_m, tile_n] flattened)
    lds_out,
    # Element type for LDS loads (defaults to f16). Pass bf16 to support bf16 epilogues.
    frag_elem_type: ir.Type | None = None,
    # Callbacks
    write_row_to_lds: Callable,
    precompute_row: Callable | None = None,
    store_pair: Callable,
):
    """LDS CShuffle epilogue skeleton.

    Call pattern:
      - `write_row_to_lds(...)` is called once per MFMA row produced by this thread.
        It is responsible for writing all ni columns for that row into `lds_out`.
      - `store_pair(...)` is called for each (row_local, col_pair0) half2 after shuffle.

    `store_pair` can implement either global stores or atomics.
    """
    if int(block_size) <= 0 or (int(block_size) % int(cshuffle_nlane)) != 0:
        raise ValueError(
            f"block_size ({block_size}) must be divisible by cshuffle_nlane ({cshuffle_nlane})"
        )
    cshuffle_mlane = int(block_size) // int(cshuffle_nlane)
    if (int(tile_m) % cshuffle_mlane) != 0:
        raise ValueError(
            f"tile_m must be divisible by CShuffleMLane ({cshuffle_mlane}), got tile_m={tile_m}"
        )
    if int(e_vec) <= 0:
        raise ValueError(f"e_vec must be positive, got {e_vec}")
    if (int(tile_n) % (int(cshuffle_nlane) * int(e_vec))) != 0:
        raise ValueError(
            f"tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_vec}, got tile_n={tile_n}"
        )

    # ---------------- Step 1: write C tile to LDS (row-major, fp16) ----------------
    # Row stride may exceed tile_n when lds_out is padded to break bank aliasing;
    # columns still live in [0, tile_n).
    tile_n_idx = arith.constant(int(lds_out_stride or tile_n), index=True)
    n_tile_base_v = n_tile_base
    # A-first: a lane owns one channel (lane%16) across 4 rows (lane/16*4 + ii).
    # B-first: it owns one row (lane%16) across 4 channels (lane/16*4 + 0..3).
    col_base_local = n_tile_base_v + (
        lane_div_16 * 4 if bfirst else lane_mod_16
    )  # index within [0,tile_n)

    def _write_row(mi: int, ii: int, row_in_tile, row, lds_row=None):
        # row_base_lds = row_in_tile * tile_n; chunked keeps only one chunk of
        # rows resident, so the row index is taken modulo the chunk.
        row_base_lds = (row_in_tile if lds_row is None else lds_row) * tile_n_idx
        write_row_to_lds(
            mi=mi,
            ii=ii,
            row_in_tile=row_in_tile,
            row=row,
            row_base_lds=row_base_lds,
            col_base_local=col_base_local,
            num_acc_n=num_acc_n,
            lds_out=lds_out,
        )

    def _step1(mi: int):
        # One call per `mi`: the callback emits all 4 channels as a single store,
        # so there is no `ii` axis to iterate here.
        row_in_tile = arith.constant(mi * 16, index=True) + lane_mod_16
        _write_row(
            mi,
            0,
            row_in_tile,
            bx_m + row_in_tile,
            lds_row=lane_mod_16 if chunk_m is not None else None,
        )

    # ---------------- Step 2: shuffle mapping + half2 store/atomic ----------------
    CShuffleNLane = int(cshuffle_nlane)
    CShuffleMLane = int(cshuffle_mlane)
    EVec = int(e_vec)

    m_reps_shuffle = int(tile_m) // CShuffleMLane
    n_reps_shuffle = int(tile_n) // (CShuffleNLane * EVec)

    c_nlane = fx.Index(CShuffleNLane)
    m_lane = tx // c_nlane
    n_lane = tx % c_nlane
    c_evec = fx.Index(EVec)

    if frag_elem_type is None:
        frag_elem_type = T.f16
    vec_frag = T.vec(EVec, frag_elem_type)
    bx_m_v = bx_m
    by_n_v = by_n

    def _step2(mr: int):
        row_base_m = arith.constant(mr * CShuffleMLane, index=True)
        row_local = row_base_m + m_lane
        row = bx_m_v + row_local
        lds_row = m_lane if chunk_m is not None else row_local

        row_ctx_raw = (
            precompute_row(row_local=row_local, row=row)
            if precompute_row is not None
            else None
        )

        # Optional row-level predicate: if `precompute_row` returns `(ctx, pred_i1)` and `scf`
        # is provided, we can skip the entire N-loop for invalid rows (cheaper than per-store checks).
        row_ctx = row_ctx_raw
        row_pred = None
        if (
            scf is not None
            and row_ctx_raw is not None
            and isinstance(row_ctx_raw, tuple)
            and len(row_ctx_raw) == 2
        ):
            row_ctx, row_pred = row_ctx_raw

        def _do_store_row():
            row_base_lds = lds_row * tile_n_idx
            # Hoist *all* LDS (CShuffle) reads for this row ahead of the stores so
            # the backend can batch them under a single lgkmcnt wait and issue the
            # subsequent atomics back-to-back, instead of the serial
            # "ds_read -> s_waitcnt lgkmcnt(0) -> atomic" chain (one full LDS wait
            # per atomic). This directly targets the vL1D-bound atomic epilogue.
            loaded = []
            for nr in range_constexpr(n_reps_shuffle):
                col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
                col_pair0 = col_base_nr + (n_lane * c_evec)  # even col within tile

                lds_idx_pair = row_base_lds + col_pair0
                frag = vector.load_op(vec_frag, lds_out, [lds_idx_pair])
                loaded.append((col_pair0, frag))

            for col_pair0, frag in loaded:
                store_pair(
                    row_local=row_local,
                    row=row,
                    row_ctx=row_ctx,
                    col_pair0=col_pair0,
                    col_g0=by_n_v + col_pair0,
                    frag=frag,
                )

        if row_pred is not None:
            _if_row = scf.IfOp(row_pred)
            with _if_then(_if_row, scf):
                _do_store_row()
        else:
            _do_store_row()

    if chunk_m is None:
        # Ensure all LDS reads finished before the lds write.
        gpu.barrier()
        if bfirst:
            for mi in range_constexpr(m_repeat):
                _step1(mi)
        else:
            default_epilog(
                arith=arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=_write_row,
            )
        # Ensure all LDS writes are visible before the shuffle-read.
        gpu.barrier()
        for mr in range_constexpr(m_reps_shuffle):
            _step2(mr)
    else:
        if not bfirst:
            raise ValueError("chunk_m requires the B-first Step-1 mapping")
        if m_repeat != m_reps_shuffle:
            raise ValueError(
                f"chunk_m needs Step 1 and Step 2 to walk the same row spans, but "
                f"m_repeat={m_repeat} and m_reps_shuffle={m_reps_shuffle}"
            )
        for c in range_constexpr(m_repeat):
            gpu.barrier()
            _step1(c)
            gpu.barrier()
            _step2(c)


def mfma_epilog(
    *,
    use_cshuffle: bool,
    # Common (always required)
    arith,
    range_constexpr,
    m_repeat: int,
    lane_div_16,
    bx_m,
    # Default epilog (required when use_cshuffle=False)
    body_row: Callable | None = None,
    # CShuffle epilog (required when use_cshuffle=True)
    vector=None,
    gpu=None,
    scf=None,
    tile_m: int | None = None,
    tile_n: int | None = None,
    e_vec: int = 2,
    cshuffle_nlane: int = 32,
    block_size: int = 256,
    num_acc_n: int | None = None,
    tx=None,
    lane_mod_16=None,
    by_n=None,
    n_tile_base=None,
    lds_out=None,
    write_row_to_lds: Callable | None = None,
    precompute_row: Callable | None = None,
    store_pair: Callable | None = None,
    frag_elem_type: ir.Type | None = None,
):
    if not use_cshuffle:
        if body_row is None:
            raise ValueError("mfma_epilog(use_cshuffle=False) requires `body_row`.")
        return default_epilog(
            arith=arith,
            range_constexpr=range_constexpr,
            m_repeat=m_repeat,
            lane_div_16=lane_div_16,
            bx_m=bx_m,
            body_row=body_row,
        )

    return c_shuffle_epilog(
        arith=arith,
        vector=vector,
        gpu=gpu,
        scf=scf,
        range_constexpr=range_constexpr,
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        e_vec=int(e_vec),
        cshuffle_nlane=int(cshuffle_nlane),
        block_size=int(block_size),
        m_repeat=m_repeat,
        num_acc_n=int(num_acc_n),
        tx=tx,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        bx_m=bx_m,
        by_n=by_n,
        n_tile_base=n_tile_base,
        lds_out=lds_out,
        frag_elem_type=frag_elem_type,
        write_row_to_lds=write_row_to_lds,
        precompute_row=precompute_row,
        store_pair=store_pair,
    )
