# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL StreamK schedule precompute.

The smallest unit of MoE-GEMM work is one ``(m_block, n_tile, k_tile)`` ==
``tile_m x tile_n x tile_k`` MAC step. This tiny kernel reads the *dynamic* valid
token count (``num_valid_ids[0]`` == the sorted valid length, which ``moe_sorting``
block-aligns to the sorting block) and the *static* per-stage tiling and emits an
even-split boundary array so a fixed grid of persistent workgroups can be launched:

    valid_m_blocks = ceil(num_valid_ids[0] / tile_m)
    total_units    = valid_m_blocks * n_tiles * k_tiles          # n_tiles=N/tile_n
    wg_unit_start[i] = i * total_units / num_wg   for i in [0, num_wg]

Workgroup ``i`` then owns the flat unit range ``[wg_unit_start[i], wg_unit_start[i+1])``
and decodes each unit -> ``(gx=n_tile, gy=m_block, gz=k_tile)`` + expert. Computing
this on-device avoids a host<->device sync of the dynamic valid count.
"""
from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf as _scf

__all__ = [
    "build_streamk_schedule_module",
    "flydsl_streamk_schedule",
    "streamk_default_num_wg",
]


@functools.lru_cache(maxsize=None)
def streamk_default_num_wg(occupancy: int = 4) -> int:
    """Default persistent-grid size for StreamK: ``num_cu * occupancy``.

    occupancy defaults to 4 (4 persistent WGs/CU). The persistent kernel is
    memory-latency-bound; more WGs/CU => more in-flight waves => better latency
    hiding. Measured: on the concurrency-limited stage1 shape (N=inter_dim, few
    n-tiles) raising num_wg from num_cu*2 to num_cu*8 improved mfocus from
    ~0.88x -> ~0.72x of baseline. Stage2 (N=model_dim, huge natural grid)
    plateaus regardless (~1.31x), so this helps stage1 and is ~neutral for
    stage2. The autotuner sweeps the exact num_wg per shape via
    AITER_TUNE_MOE_STREAMK_WG (best stage1 point is num_cu*8), so this default
    only affects untuned runs.
    """
    from aiter.jit.utils.chip_info import get_cu_num

    return int(get_cu_num()) * max(1, int(occupancy))


@functools.lru_cache(maxsize=None)
def build_streamk_schedule_module(num_wg: int, block_threads: int = 256):
    """Build (and cache) the StreamK schedule launcher for a fixed ``num_wg``.

    Launcher signature:
        ``(num_valid_ids, wg_unit_start, tile_m, n_tiles, k_tiles, stream)``
    writes ``wg_unit_start[num_wg+1]`` (int32).
    """
    NWG1 = int(num_wg) + 1
    bt = min(int(block_threads), 1024)
    n_strides = (NWG1 + bt - 1) // bt

    @flyc.kernel
    def streamk_schedule_kernel(
        arg_num_valid_ids: fx.Tensor,   # (>=1,) int32; [0] == valid sorted length
        arg_wg_start: fx.Tensor,        # (num_wg+1,) int32, output
        i32_tile_m: fx.Int32,
        i32_n_tiles: fx.Int32,
        i32_k_tiles: fx.Int32,
    ):
        i32 = T.i32
        idx = T.index
        c0_i32 = arith.constant(0, type=i32)
        c_nwg1_i32 = arith.constant(NWG1, type=i32)
        c1_idx = arith.constant(1, index=True)
        c_numwg_idx = arith.constant(int(num_wg), index=True)

        tid_i32 = ArithValue(fx.thread_idx.x)
        tid = arith.index_cast(idx, tid_i32)

        nvi_rsrc = buffer_ops.create_buffer_resource(arg_num_valid_ids, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(
            arg_wg_start, max_size=False, num_records_bytes=fx.Index(NWG1 * 4)
        )
        max_tok_i32 = buffer_ops.buffer_load(nvi_rsrc, c0_i32, vec_width=1, dtype=i32)
        # All work-size math in `index` (64-bit) so w*total_units never overflows.
        max_tok = arith.index_cast(idx, max_tok_i32)
        tile_m = arith.index_cast(idx, ArithValue(i32_tile_m))
        n_tiles = arith.index_cast(idx, ArithValue(i32_n_tiles))
        k_tiles = arith.index_cast(idx, ArithValue(i32_k_tiles))
        vmb = (max_tok + tile_m - c1_idx) // tile_m          # ceil(max_tok/tile_m)
        total = vmb * n_tiles * k_tiles                       # total_units

        for st in range_constexpr(n_strides):
            w = tid + arith.constant(st * bt, index=True)
            w_i32 = arith.index_cast(i32, w)
            in_range = arith.cmpi(CmpIPredicate.ult, w_i32, c_nwg1_i32)
            _if = _scf.IfOp(in_range)
            with ir.InsertionPoint(_if.then_block):
                start = (w * total) // c_numwg_idx            # even split
                start_i32 = arith.index_cast(i32, start)
                buffer_ops.buffer_store(start_i32, out_rsrc, w_i32)
                _scf.YieldOp([])

    @flyc.jit
    def launch_streamk_schedule(
        arg_num_valid_ids: fx.Tensor,
        arg_wg_start: fx.Tensor,
        i32_tile_m: fx.Int32,
        i32_n_tiles: fx.Int32,
        i32_k_tiles: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        streamk_schedule_kernel(
            arg_num_valid_ids, arg_wg_start,
            i32_tile_m, i32_n_tiles, i32_k_tiles,
        ).launch(grid=(1, 1, 1), block=(bt, 1, 1), stream=stream)

    return launch_streamk_schedule


def flydsl_streamk_schedule(
    num_valid_ids, num_wg, tile_m, n_tiles, k_tiles, stream=None
):
    """Even-split StreamK unit ranges.

    Returns ``wg_unit_start`` (int32 ``[num_wg+1]``, device): workgroup ``i`` owns
    flat units ``[wg_unit_start[i], wg_unit_start[i+1])`` over the
    ``valid_m_blocks * n_tiles * k_tiles`` unit space.
    """
    dev = num_valid_ids.device
    wg_start = torch.empty(int(num_wg) + 1, dtype=torch.int32, device=dev)
    launcher = build_streamk_schedule_module(int(num_wg))
    if stream is None:
        stream = torch.cuda.current_stream()
    launcher(
        num_valid_ids, wg_start,
        int(tile_m), int(n_tiles), int(k_tiles), stream,
    )
    return wg_start
