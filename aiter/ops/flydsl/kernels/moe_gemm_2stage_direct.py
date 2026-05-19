"""Small-M direct MoE stage2 kernel.

This path is intentionally narrow: fp8 A2/W2, per-tensor/per-expert f32
scales, bf16 output.  It avoids sorted stage2 padding and global atomics by
assigning one workgroup to (token, N-tile) and reducing topk in-kernel.
"""

import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T

from .mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    load_b_pack_k32,
    make_preshuffle_b_layout,
)


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _idx_to_llvm_ptr(idx_val, addr_space=1):
    idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
    i64_v = arith.index_cast(T.i64, idx_v)
    i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
    return llvm.inttoptr(ir.Type.parse(f"!llvm.ptr<{addr_space}>"), i64_raw)


def _value(v):
    return v._value if hasattr(v, "_value") else v


def _s_nop(count=1):
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string=f"s_nop {count}",
        constraints="",
        has_side_effects=True,
    )


@functools.lru_cache(maxsize=256)
def compile_moe_gemm1_direct_smallm(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 64,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    a_scale_scalar: bool = True,
    w_scale_per_expert: bool = True,
    routes_per_block: int = 1,
    num_waves_override: int = 0,
):
    """Compile direct small-M stage1 for fp8/fp8 + silu(gate)*up.

    The grid is (N tile, token, topk slot). It bypasses expert padded
    `moe_sorting` for stage1 and writes [token, topk, inter_dim] directly.
    """

    if in_dtype != "fp8":
        raise ValueError(f"direct small-M stage1 supports only fp8, got {in_dtype!r}")
    if out_dtype != "bf16":
        raise ValueError(f"direct small-M stage1 supports only bf16, got {out_dtype!r}")
    if tile_m != 16:
        raise ValueError("direct MFMA stage1 currently requires tile_m=16")
    if tile_n < 16 or tile_n % 16 != 0:
        raise ValueError("direct MFMA stage1 requires tile_n divisible by 16")
    if tile_k % 64 != 0:
        raise ValueError("direct MFMA stage1 requires tile_k to be a multiple of 64")
    if model_dim % tile_k != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_k={tile_k}")
    if inter_dim % tile_n != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_n={tile_n}")
    if routes_per_block < 1 or topk % routes_per_block != 0:
        raise ValueError(
            f"routes_per_block={routes_per_block} must evenly divide topk={topk}"
        )

    num_waves = num_waves_override if num_waves_override > 0 else _ceil_div(tile_n, 64)
    if num_waves < 1:
        raise ValueError("direct MFMA stage1 requires at least one wave")
    if tile_n % num_waves != 0:
        raise ValueError(
            f"direct MFMA stage1 requires tile_n divisible by num_waves={num_waves}"
        )
    total_threads = num_waves * 64
    if total_threads > 1024:
        raise ValueError(
            f"direct MFMA stage1 block size exceeds HIP limit: {total_threads}"
        )
    n_per_wave = tile_n // num_waves
    if n_per_wave % 16 != 0:
        raise ValueError(
            f"direct MFMA stage1 requires n_per_wave={n_per_wave} divisible by 16"
        )
    num_acc_n = n_per_wave // 16
    block_threads = total_threads
    scale_tag = ("ass" if a_scale_scalar else "asr") + (
        "_wse" if w_scale_per_expert else "_wsn"
    )
    module_name = (
        f"direct_moe1_{in_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
        f"_abi9_mfma_nolds_w{num_waves}_{scale_tag}"
        f"{'_rpb' + str(routes_per_block) if routes_per_block != 1 else ''}"
    )

    def out_elem():
        return T.bf16() if callable(T.bf16) else T.bf16

    def silu(x):
        t = x * (-1.4426950408889634)
        emu = rocdl.exp2(T.f32, t)
        den = 1.0 + emu
        sig = rocdl.rcp(T.f32, den)
        return x * sig

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_gemm1_direct(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_topk_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        n_in = arith.index_cast(T.index, i32_n_in)
        k_in = arith.index_cast(T.index, i32_k_in)

        bx_n = gpu.block_id("x")
        by_tok = gpu.block_id("y")
        bz_slot = gpu.block_id("z")
        tx = gpu.thread_id("x")

        out_nbytes = tokens_in * fx.Index(topk) * n_in * fx.Index(2)
        x_nbytes = tokens_in * k_in
        w_nbytes = fx.Index(experts) * fx.Index(2) * n_in * k_in
        scale_x_nbytes = (
            fx.Index(1) if a_scale_scalar else tokens_in
        ) * fx.Index(4)
        scale_w_nbytes = (
            fx.Index(experts)
            if w_scale_per_expert
            else fx.Index(experts) * fx.Index(2) * n_in
        ) * fx.Index(4)
        topk_nbytes = tokens_in * fx.Index(topk) * fx.Index(4)

        out_rsrc = buffer_ops.create_buffer_resource(
            arg_out, max_size=False, num_records_bytes=out_nbytes
        )
        x_rsrc = buffer_ops.create_buffer_resource(
            arg_x, max_size=False, num_records_bytes=x_nbytes
        )
        w_rsrc = buffer_ops.create_buffer_resource(
            arg_w, max_size=False, num_records_bytes=w_nbytes
        )
        sx_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_x, max_size=False, num_records_bytes=scale_x_nbytes
        )
        sw_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_w, max_size=False, num_records_bytes=scale_w_nbytes
        )
        tid_rsrc = buffer_ops.create_buffer_resource(
            arg_topk_ids, max_size=False, num_records_bytes=topk_nbytes
        )

        b_layout = make_preshuffle_b_layout(
            arith,
            c_n=arith.index(experts * inter_dim * 2),
            c_k=k_in,
            kpack_bytes=16,
            elem_bytes=1,
        )
        layout_b = b_layout.layout_b
        layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        layout_n_blk_intra = fx.make_layout(
            (experts * inter_dim * 2 // 16, 16), stride=(16, 1)
        )

        coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
        wave_id = fx.get(coord_wl, 0)
        lane_id = fx.get(coord_wl, 1)
        coord_l16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_l16, 0)
        lane_mod_16 = fx.get(coord_l16, 1)

        by_n = bx_n * fx.Index(tile_n)
        n_tile_base = (wave_id % fx.Index(num_waves)) * fx.Index(n_per_wave)
        col_offset_base_bytes = lane_div_16 * fx.Index(16)
        acc_init = arith.constant_vector(0.0, T.f32x4)

        col_g_list = []
        gate_blk_list = []
        gate_intra_list = []
        up_blk_list = []
        up_intra_list = []
        for ni in range_constexpr(num_acc_n):
            col_g = by_n + n_tile_base + fx.Index(ni * 16) + lane_mod_16
            col_g_list.append(col_g)
            coord_gate = fx.idx2crd(col_g, layout_n_blk_intra)
            gate_blk_list.append(fx.get(coord_gate, 0))
            gate_intra_list.append(fx.get(coord_gate, 1))
            coord_up = fx.idx2crd(fx.Index(inter_dim) + col_g, layout_n_blk_intra)
            up_blk_list.append(fx.get(coord_up, 0))
            up_intra_list.append(fx.get(coord_up, 1))

        route_group_base = by_tok * fx.Index(topk) + bz_slot * fx.Index(routes_per_block)
        route_idx_list = []
        expert_i32_list = []
        expert_idx_list = []
        for rb in range_constexpr(routes_per_block):
            route_idx = route_group_base + fx.Index(rb)
            route_idx_list.append(route_idx)
            expert_i32 = buffer_ops.buffer_load(
                tid_rsrc, arith.index_cast(T.i32, route_idx), vec_width=1, dtype=T.i32
            )
            expert_i32_list.append(expert_i32)
            expert_idx_list.append(arith.index_cast(T.index, expert_i32))

        def load_a_packs_k64(base_k):
            idx_elem = (by_tok * k_in + base_k + col_offset_base_bytes) // fx.Index(4)
            loaded_a16 = buffer_copy_gmem16_dwordx4(
                buffer_ops,
                vector,
                elem_type=T.f8,
                idx_i32=idx_elem,
                rsrc=x_rsrc,
                vec_elems=16,
                elem_bytes=1,
            )
            a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        def load_b_pair(expert_idx, base_k):
            expert_base_blk = expert_idx * fx.Index((2 * inter_dim) // 16)
            gate0 = []
            gate1 = []
            up0 = []
            up1 = []
            for ni in range_constexpr(num_acc_n):
                gate_blk = expert_base_blk + gate_blk_list[ni]
                up_blk = expert_base_blk + up_blk_list[ni]
                gate0.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=0,
                        n_blk=gate_blk,
                        n_intra=gate_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
                gate1.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=1,
                        n_blk=gate_blk,
                        n_intra=gate_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
                up0.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=0,
                        n_blk=up_blk,
                        n_intra=up_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
                up1.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=1,
                        n_blk=up_blk,
                        n_intra=up_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
            return gate0, gate1, up0, up1

        def mfma_k64(acc0, a0, a1, b0, b1):
            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4, [a0, b0, acc0, 0, 0, 0]
            )
            return rocdl.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4, [a1, b1, acc1, 0, 0, 0]
            )

        acc_gate = [[acc_init] * num_acc_n for _ in range_constexpr(routes_per_block)]
        acc_up = [[acc_init] * num_acc_n for _ in range_constexpr(routes_per_block)]
        for kt in range_constexpr(model_dim // tile_k):
            base_k = fx.Index(kt * tile_k)
            for kk in range_constexpr(tile_k // 64):
                k_base = base_k + fx.Index(kk * 64)
                a0, a1 = load_a_packs_k64(k_base)
                for rb in range_constexpr(routes_per_block):
                    gate0, gate1, up0, up1 = load_b_pair(expert_idx_list[rb], k_base)
                    for ni in range_constexpr(num_acc_n):
                        acc_gate[rb][ni] = mfma_k64(
                            acc_gate[rb][ni], a0, a1, gate0[ni], gate1[ni]
                        )
                        acc_up[rb][ni] = mfma_k64(
                            acc_up[rb][ni], a0, a1, up0[ni], up1[ni]
                        )

        x_scale_idx = fx.Index(0) if a_scale_scalar else by_tok
        x_scale = buffer_ops.buffer_load(
            sx_rsrc,
            arith.index_cast(T.i32, x_scale_idx),
            vec_width=1,
            dtype=T.f32,
        )
        row0_lane = arith.cmpi(
            arith.CmpIPredicate.eq,
            arith.index_cast(T.i32, lane_div_16),
            arith.constant(0, type=T.i32),
        )
        _if_row0 = scf.IfOp(row0_lane)
        with _if_then(_if_row0):
            for rb in range_constexpr(routes_per_block):
                if w_scale_per_expert:
                    w_scale = buffer_ops.buffer_load(
                        sw_rsrc,
                        expert_i32_list[rb],
                        vec_width=1,
                        dtype=T.f32,
                    )
                else:
                    w_scale = arith.constant(1.0, type=T.f32)
                route_scale = x_scale * w_scale
                for ni in range_constexpr(num_acc_n):
                    gate_v = vector.extract(
                        acc_gate[rb][ni], static_position=[0], dynamic_position=[]
                    ) * route_scale
                    up_v = vector.extract(
                        acc_up[rb][ni], static_position=[0], dynamic_position=[]
                    ) * route_scale
                    if not w_scale_per_expert:
                        sw_gate_idx = (
                            expert_idx_list[rb] * fx.Index(2 * inter_dim)
                            + col_g_list[ni]
                        )
                        sw_up_idx = sw_gate_idx + fx.Index(inter_dim)
                        sw_gate = buffer_ops.buffer_load(
                            sw_rsrc,
                            arith.index_cast(T.i32, sw_gate_idx),
                            vec_width=1,
                            dtype=T.f32,
                        )
                        sw_up = buffer_ops.buffer_load(
                            sw_rsrc,
                            arith.index_cast(T.i32, sw_up_idx),
                            vec_width=1,
                            dtype=T.f32,
                        )
                        gate_v = gate_v * sw_gate
                        up_v = up_v * sw_up
                    out_v = arith.trunc_f(out_elem(), silu(gate_v) * up_v)
                    out_idx = route_idx_list[rb] * n_in + col_g_list[ni]
                    buffer_ops.buffer_store(
                        out_v, out_rsrc, arith.index_cast(T.i32, out_idx)
                    )

    @flyc.jit
    def launch_moe_gemm1_direct(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_topk_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        stream: fx.Stream,
    ):
        gx = fx.Index(_ceil_div(inter_dim, tile_n))
        gy = arith.index_cast(T.index, i32_tokens_in)
        moe_gemm1_direct(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_topk_ids,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
        ).launch(
            grid=(gx, gy, fx.Index(topk // routes_per_block)),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_moe_gemm1_direct.__name__ = module_name
    return launch_moe_gemm1_direct


@functools.lru_cache(maxsize=256)
def compile_moe_gemm2_direct_smallm(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 64,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    a_scale_scalar: bool = True,
    w_scale_per_expert: bool = True,
    split_reduce: bool = False,
):
    """Compile direct small-M stage2 using MFMA fragments.

    `tile_m` is a dummy 16-row MFMA tile. For each token/topk route we fill the
    tile with the same A row, multiply by that route's expert W2 tile, and keep
    one row of the fragment. This preserves the direct single-write contract
    while avoiding scalar fp8 decode/dot in the hot loop.
    """

    if in_dtype != "fp8":
        raise ValueError(f"direct small-M stage2 supports only fp8, got {in_dtype!r}")
    if out_dtype != "bf16":
        raise ValueError(f"direct small-M stage2 supports only bf16, got {out_dtype!r}")
    if tile_m < topk:
        raise ValueError(f"tile_m={tile_m} must cover topk={topk}")
    if inter_dim % tile_k != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if model_dim % tile_n != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_n={tile_n}")

    if tile_m != 16:
        raise ValueError("direct MFMA stage2 currently requires tile_m=16")
    if tile_n % 32 != 0:
        raise ValueError("direct MFMA stage2 requires tile_n divisible by 32")
    if tile_k != 64:
        raise ValueError("direct MFMA stage2 currently requires tile_k=64")

    total_threads = (tile_n // 32) * 64
    if total_threads > 1024:
        raise ValueError(
            f"direct MFMA stage2 block size exceeds HIP limit: {total_threads}"
        )
    num_waves = total_threads // 64
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16

    block_threads = total_threads
    scale_tag = ("ass" if a_scale_scalar else "asr") + (
        "_wse" if w_scale_per_expert else "_wsn"
    )
    module_name = (
        f"direct_moe2_{in_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
        f"_abi8_mfma_nolds_{scale_tag}"
        f"{'_splitreduce' if split_reduce else ''}"
    )

    def out_elem():
        return T.bf16() if callable(T.bf16) else T.bf16

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_gemm2_direct(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_topk_ids: fx.Tensor,
        arg_topk_weights: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        n_in = arith.index_cast(T.index, i32_n_in)
        k_in = arith.index_cast(T.index, i32_k_in)

        bx_n = gpu.block_id("x")
        by_tok = gpu.block_id("y")
        tx = gpu.thread_id("x")

        out_rows = tokens_in * (fx.Index(topk) if split_reduce else fx.Index(1))
        out_nbytes = out_rows * n_in * fx.Index(2)
        x_nbytes = tokens_in * fx.Index(topk) * k_in
        w_nbytes = fx.Index(experts) * n_in * k_in
        scale_x_nbytes = (
            fx.Index(1) if a_scale_scalar else tokens_in * fx.Index(topk)
        ) * fx.Index(4)
        scale_w_nbytes = (
            fx.Index(experts) if w_scale_per_expert else fx.Index(experts) * n_in
        ) * fx.Index(4)
        topk_nbytes = tokens_in * fx.Index(topk) * fx.Index(4)

        out_rsrc = buffer_ops.create_buffer_resource(
            arg_out, max_size=False, num_records_bytes=out_nbytes
        )
        x_rsrc = buffer_ops.create_buffer_resource(
            arg_x, max_size=False, num_records_bytes=x_nbytes
        )
        w_rsrc = buffer_ops.create_buffer_resource(
            arg_w, max_size=False, num_records_bytes=w_nbytes
        )
        sx_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_x, max_size=False, num_records_bytes=scale_x_nbytes
        )
        sw_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_w, max_size=False, num_records_bytes=scale_w_nbytes
        )
        tid_rsrc = buffer_ops.create_buffer_resource(
            arg_topk_ids, max_size=False, num_records_bytes=topk_nbytes
        )
        tw_rsrc = buffer_ops.create_buffer_resource(
            arg_topk_weights, max_size=False, num_records_bytes=topk_nbytes
        )
        b_layout = make_preshuffle_b_layout(
            arith,
            c_n=arith.index(experts * model_dim),
            c_k=k_in,
            kpack_bytes=16,
            elem_bytes=1,
        )
        layout_b = b_layout.layout_b
        layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        layout_n_blk_intra = fx.make_layout(
            (experts * model_dim // 16, 16), stride=(16, 1)
        )

        coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
        wave_id = fx.get(coord_wl, 0)
        lane_id = fx.get(coord_wl, 1)
        coord_l16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_l16, 0)
        lane_mod_16 = fx.get(coord_l16, 1)

        by_n = bx_n * fx.Index(tile_n)
        n_tile_base = (wave_id % fx.Index(num_waves)) * fx.Index(n_per_wave)
        col_offset_base_bytes = lane_div_16 * fx.Index(16)
        acc_init = arith.constant_vector(0.0, T.f32x4)

        col_g_list = []
        n_blk_list = []
        n_intra_list = []
        for ni in range_constexpr(num_acc_n):
            col_g = by_n + n_tile_base + fx.Index(ni * 16) + lane_mod_16
            col_g_list.append(col_g)
            coord_w = fx.idx2crd(col_g, layout_n_blk_intra)
            n_blk_list.append(fx.get(coord_w, 0))
            n_intra_list.append(fx.get(coord_w, 1))

        def load_a_packs_k64(route_idx, base_k):
            idx_elem = (route_idx * k_in + base_k + col_offset_base_bytes) // fx.Index(4)
            loaded_a16 = buffer_copy_gmem16_dwordx4(
                buffer_ops,
                vector,
                elem_type=T.f8,
                idx_i32=idx_elem,
                rsrc=x_rsrc,
                vec_elems=16,
                elem_bytes=1,
            )
            a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        def load_b_tile(expert_idx, base_k):
            expert_off_idx = expert_idx * n_in
            packs0 = []
            packs1 = []
            for ni in range_constexpr(num_acc_n):
                row_w = expert_off_idx + col_g_list[ni]
                coord_w = fx.idx2crd(row_w, layout_n_blk_intra)
                n_blk = fx.get(coord_w, 0)
                n_intra = fx.get(coord_w, 1)
                packs0.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=0,
                        n_blk=n_blk,
                        n_intra=n_intra,
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
                packs1.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=1,
                        n_blk=n_blk,
                        n_intra=n_intra,
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
            return packs0, packs1

        def mfma_k64(acc0, a0, a1, b0, b1):
            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4, [a0, b0, acc0, 0, 0, 0]
            )
            return rocdl.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4, [a1, b1, acc1, 0, 0, 0]
            )

        out_acc = [arith.constant(0.0, type=T.f32)] * num_acc_n
        route_base = by_tok * fx.Index(topk)
        route_slot = gpu.block_id("z") if split_reduce else fx.Index(0)
        row0_lane = arith.cmpi(
            arith.CmpIPredicate.eq,
            arith.index_cast(T.i32, lane_div_16),
            arith.constant(0, type=T.i32),
        )

        for slot in range_constexpr(1 if split_reduce else topk):
            route_idx = route_base + (route_slot if split_reduce else fx.Index(slot))
            route_i32 = arith.index_cast(T.i32, route_idx)
            expert_i32 = buffer_ops.buffer_load(
                tid_rsrc, route_i32, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(T.index, expert_i32)
            x_scale_idx = fx.Index(0) if a_scale_scalar else route_idx
            x_scale = buffer_ops.buffer_load(
                sx_rsrc,
                arith.index_cast(T.i32, x_scale_idx),
                vec_width=1,
                dtype=T.f32,
            )
            route_weight = buffer_ops.buffer_load(
                tw_rsrc, route_i32, vec_width=1, dtype=T.f32
            )
            route_scale = x_scale * route_weight
            if w_scale_per_expert:
                sw_expert = buffer_ops.buffer_load(
                    sw_rsrc,
                    expert_i32,
                    vec_width=1,
                    dtype=T.f32,
                )
                route_scale = route_scale * sw_expert
            acc_slot = [acc_init] * num_acc_n

            for kt in range_constexpr(inter_dim // tile_k):
                base_k = fx.Index(kt * tile_k)
                b0, b1 = load_b_tile(expert_idx, base_k)
                a0, a1 = load_a_packs_k64(route_idx, base_k)
                for ni in range_constexpr(num_acc_n):
                    acc_slot[ni] = mfma_k64(acc_slot[ni], a0, a1, b0[ni], b1[ni])

            for ni in range_constexpr(num_acc_n):
                v = vector.extract(
                    acc_slot[ni], static_position=[0], dynamic_position=[]
                )
                if w_scale_per_expert:
                    out_acc[ni] = out_acc[ni] + (v * route_scale)
                else:
                    sw_idx = expert_idx * n_in + col_g_list[ni]
                    sw = buffer_ops.buffer_load(
                        sw_rsrc,
                        arith.index_cast(T.i32, sw_idx),
                        vec_width=1,
                        dtype=T.f32,
                    )
                    out_acc[ni] = out_acc[ni] + (v * route_scale * sw)

        _if_row0 = scf.IfOp(row0_lane)
        with _if_then(_if_row0):
            for ni in range_constexpr(num_acc_n):
                out_v = arith.trunc_f(out_elem(), out_acc[ni])
                out_idx = (
                    route_idx * n_in + col_g_list[ni]
                    if split_reduce
                    else by_tok * n_in + col_g_list[ni]
                )
                out_i32 = arith.index_cast(T.i32, out_idx)
                buffer_ops.buffer_store(out_v, out_rsrc, out_i32)

    if split_reduce:
        reduce_block_threads = 256
        reduce_vec_elems = 8
        reduce_tile_cols = reduce_block_threads * reduce_vec_elems

        @flyc.kernel(known_block_size=[reduce_block_threads, 1, 1])
        def moe_topk_reduce_direct(
            arg_tmp: fx.Tensor,
            arg_final: fx.Tensor,
            i32_tokens_in: fx.Int32,
        ):
            from flydsl._mlir.dialects import fly as _fly

            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            token = gpu.block_id("x")
            tile = gpu.block_id("y")
            tid = gpu.thread_id("x")
            vec_i32 = T.vec(reduce_vec_elems // 2, T.i32)
            vec_bf16 = T.vec(reduce_vec_elems, T.bf16)
            vec_f32 = T.vec(reduce_vec_elems, T.f32)

            ptr_ty = ir.Type.parse("!llvm.ptr")
            tmp_base_ptr = _fly.extract_aligned_pointer_as_index(ptr_ty, arg_tmp)
            final_base_ptr = _fly.extract_aligned_pointer_as_index(ptr_ty, arg_final)
            tmp_base_idx = arith.index_cast(
                T.index, llvm.ptrtoint(T.i64, tmp_base_ptr)
            )
            final_base_idx = arith.index_cast(
                T.index, llvm.ptrtoint(T.i64, final_base_ptr)
            )

            col_base = (
                tile * fx.Index(reduce_tile_cols) + tid * fx.Index(reduce_vec_elems)
            )
            # The tuned small-M shapes use full 8-column vectors; keep a scalar tail
            # only for robustness on other model_dim values.
            col_ok = arith.cmpi(
                arith.CmpIPredicate.ult,
                arith.index_cast(T.i32, col_base),
                arith.constant(model_dim, type=T.i32),
            )
            full_ok = arith.cmpi(
                arith.CmpIPredicate.ule,
                arith.index_cast(T.i32, col_base + fx.Index(reduce_vec_elems)),
                arith.constant(model_dim, type=T.i32),
            )
            _if_col = scf.IfOp(col_ok)
            with _if_then(_if_col):
                _if_full = scf.IfOp(full_ok, has_else=True)
                with _if_then(_if_full):
                    acc = arith.constant_vector(0.0, vec_f32)
                    token_route_base = token * fx.Index(topk * model_dim)
                    for slot in range_constexpr(topk):
                        elem_idx = (
                            token_route_base + fx.Index(slot * model_dim) + col_base
                        )
                        byte_idx = tmp_base_idx + elem_idx * fx.Index(2)
                        raw = llvm.LoadOp(
                            vec_i32,
                            _idx_to_llvm_ptr(byte_idx),
                            alignment=16,
                        ).res
                        vals = vector.bitcast(vec_bf16, raw)
                        acc = acc + arith.extf(vec_f32, vals)
                    out_vec = arith.trunc_f(vec_bf16, acc)
                    out_elem_idx = token * fx.Index(model_dim) + col_base
                    out_byte_idx = final_base_idx + out_elem_idx * fx.Index(2)
                    llvm.StoreOp(
                        _value(out_vec),
                        _idx_to_llvm_ptr(out_byte_idx),
                        alignment=16,
                    )

                with _if_else(_if_full):
                    tmp_nbytes = tokens_in * fx.Index(topk * model_dim * 2)
                    final_nbytes = tokens_in * fx.Index(model_dim * 2)
                    tmp_rsrc = buffer_ops.create_buffer_resource(
                        arg_tmp, max_size=False, num_records_bytes=tmp_nbytes
                    )
                    final_rsrc = buffer_ops.create_buffer_resource(
                        arg_final, max_size=False, num_records_bytes=final_nbytes
                    )
                    token_route_base = token * fx.Index(topk * model_dim)
                    for lane in range_constexpr(reduce_vec_elems):
                        col = col_base + fx.Index(lane)
                        lane_ok = arith.cmpi(
                            arith.CmpIPredicate.ult,
                            arith.index_cast(T.i32, col),
                            arith.constant(model_dim, type=T.i32),
                        )
                        _if_lane = scf.IfOp(lane_ok)
                        with _if_then(_if_lane):
                            acc = arith.constant(0.0, type=T.f32)
                            for slot in range_constexpr(topk):
                                elem_idx = (
                                    token_route_base
                                    + fx.Index(slot * model_dim)
                                    + col
                                )
                                v = buffer_ops.buffer_load(
                                    tmp_rsrc,
                                    arith.index_cast(T.i32, elem_idx),
                                    vec_width=1,
                                    dtype=out_elem(),
                                )
                                acc = acc + arith.extf(T.f32, v)
                            out_v = arith.trunc_f(out_elem(), acc)
                            out_idx = token * fx.Index(model_dim) + col
                            buffer_ops.buffer_store(
                                out_v,
                                final_rsrc,
                                arith.index_cast(T.i32, out_idx),
                            )

        @flyc.jit
        def launch_moe_gemm2_direct(
            arg_final: fx.Tensor,
            arg_tmp: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_topk_ids: fx.Tensor,
            arg_topk_weights: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            stream: fx.Stream,
        ):
            gx = fx.Index(_ceil_div(model_dim, tile_n))
            gy = arith.index_cast(T.index, i32_tokens_in)
            moe_gemm2_direct(
                arg_tmp,
                arg_x,
                arg_w,
                arg_scale_x,
                arg_scale_w,
                arg_topk_ids,
                arg_topk_weights,
                i32_tokens_in,
                i32_n_in,
                i32_k_in,
            ).launch(grid=(gx, gy, fx.Index(topk)), block=(block_threads, 1, 1), stream=stream)
            moe_topk_reduce_direct(arg_tmp, arg_final, i32_tokens_in).launch(
                grid=(
                    gy,
                    fx.Index(_ceil_div(model_dim, reduce_tile_cols)),
                    fx.Index(1),
                ),
                block=(reduce_block_threads, 1, 1),
                stream=stream,
            )
    else:

        @flyc.jit
        def launch_moe_gemm2_direct(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_topk_ids: fx.Tensor,
            arg_topk_weights: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            stream: fx.Stream,
        ):
            gx = fx.Index(_ceil_div(model_dim, tile_n))
            gy = arith.index_cast(T.index, i32_tokens_in)
            moe_gemm2_direct(
                arg_out,
                arg_x,
                arg_w,
                arg_scale_x,
                arg_scale_w,
                arg_topk_ids,
                arg_topk_weights,
                i32_tokens_in,
                i32_n_in,
                i32_k_in,
            ).launch(grid=(gx, gy, 1), block=(block_threads, 1, 1), stream=stream)

    launch_moe_gemm2_direct.__name__ = module_name
    return launch_moe_gemm2_direct
