# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared MFMA preshuffle helpers for preshuffle GEMM kernels.

Key primitives:
- B preshuffle layout builder (supports byte-packed element types, incl. packed int4)
- B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)
"""

from __future__ import annotations
from dataclasses import dataclass
from flydsl._mlir import ir
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr.typing import T
from flydsl.expr import arith as _arith
import flydsl.expr as fx


def crd2idx(crd, layout):
    """crd2idx returning an index-typed ir.Value (unwraps fly.int_tuple)."""
    scalar = fx.get_scalar(fx.crd2idx(crd, layout)).ir_value()
    if isinstance(scalar.type, ir.IndexType):
        return scalar
    return _arith.IndexCastOp(T.index, scalar).result


def swizzle_xor16(row, col, k_blocks16):
    """XOR-with-row swizzle on the K dimension at 16B granularity.

    Computes: col XOR ((row & (k_blocks16 - 1)) * 16)

    k_blocks16 is always a power of 2 (tile_k_bytes / 16), so use
    bitwise AND instead of remui to save ~10 VALU cycles on CDNA.
    """
    from flydsl.expr import arith as _swz_arith

    mask = k_blocks16 - _swz_arith.index(1)
    rem = _swz_arith.andi(row, mask)
    return col ^ (rem * 16)


def lds_row_major_idx(row, col, row_stride, base=None):
    """Linearize a 2D LDS coordinate with explicit index arithmetic."""
    idx = row * row_stride + col
    return idx if base is None else idx + base


def split_row_major_2d(index, minor_extent):
    """Split a linear row-major index into (major, minor)."""
    return index // minor_extent, index % minor_extent


def _buffer_load_vec(
    buffer_ops,
    vector,
    rsrc,
    idx,
    *,
    elem_type,
    vec_elems,
    elem_bytes,
    offset_in_bytes,
    cache_modifier=0,
):
    """Load vec_elems elements via buffer_load dwordx[1,2,4] + bitcast."""
    from flydsl.expr import arith as _ld_arith

    elem_size = int(elem_bytes)
    load_bytes = int(vec_elems) * elem_size
    vec_width = load_bytes // 4

    if offset_in_bytes:
        idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(2))
    elif elem_bytes == 2:
        idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(1))
    else:
        idx_i32 = idx

    i32_val = buffer_ops.buffer_load(
        rsrc,
        idx_i32,
        vec_width=vec_width,
        dtype=T.i32,
        cache_modifier=cache_modifier,
    )
    if vec_width == 1:
        i32_vec = vector.from_elements(T.vec(1, T.i32), [i32_val])
    else:
        i32_vec = i32_val
    return vector.bitcast(T.vec(int(vec_elems), elem_type), i32_vec)


@dataclass(frozen=True)
class PreshuffleScaleLayout:
    """Container returned by `make_preshuffle_scale_layout`.

    The scale layout is ``(c_mn1, c_k1, 4, 16) : (stride_n0, stride_k0, stride_klane, 1)``.
    Callers compute flat index directly with plain arith::

        idx = mni * stride_n0 + ku * stride_k0 + k_lane * stride_klane + n_lane
    """

    layout_scale: object
    stride_n0: object
    stride_k0: object
    stride_klane: object


def make_preshuffle_scale_layout(
    arith,
    *,
    c_mn: ir.Value,
    c_k: ir.Value,
    mn_pack: int = 2,
    k_pack: int = 2,
    elem_bytes: int = 4,
    scale_block_size: int = 32,
) -> PreshuffleScaleLayout:
    """Build scale layout matching aiter/CK preshuffle for FP4/FP8 microscale.

    Layout shape: ``(c_mn1, c_k1, 4, 16)`` where
    ``c_mn1 = c_mn / 16 / mn_pack`` and ``c_k1 = (c_k / scale_block_size) / 4 / k_pack``.
    """
    c16 = fx.Index(16)
    c4 = fx.Index(4)
    c_k_scale = c_k // fx.Index(scale_block_size)

    c_mn1 = (c_mn // c16) // fx.Index(mn_pack)
    c_k1 = (c_k_scale // c4) // fx.Index(k_pack)
    if elem_bytes != mn_pack * k_pack:
        raise ValueError(
            f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}"
        )

    stride_klane = c16
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k1 * stride_k0

    c_mn1_i32 = arith.index_cast(T.i32, c_mn1)
    c_k1_i32 = arith.index_cast(T.i32, c_k1)
    stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
    stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
    stride_klane_i32 = arith.index_cast(T.i32, stride_klane)

    layout_scale = fx.make_layout(
        (c_mn1_i32, c_k1_i32, 4, 16),
        stride=(stride_n0_i32, stride_k0_i32, stride_klane_i32, 1),
    )

    return PreshuffleScaleLayout(
        layout_scale=layout_scale,
        stride_n0=stride_n0,
        stride_k0=stride_k0,
        stride_klane=stride_klane,
    )


@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""

    layout_b: object
    kpack_bytes: int


def make_preshuffle_b_layout(
    arith,
    *,
    c_n: ir.Value,
    c_k: ir.Value,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    k_major: bool = False,
) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels.

    When *k_major* is True the block-level order is K-major (``k_blk`` outermost),
    matching the ``(0,3,1,4,2,5)`` shuffle permutation.  The default N-major
    order (``k_major=False``) matches the legacy ``(0,1,3,4,2,5)`` permutation.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

    c16 = fx.Index(16)
    c_kpack = fx.Index(kpack_bytes)

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
    n0 = c_n // c16

    c_kpack_elems = (
        c_kpack
        if elem_bytes == 1
        else (c_kpack // arith.constant(int(elem_bytes), index=True))
    )

    stride_nlane = c_kpack_elems

    if k_major:
        c32 = fx.Index(32)
        c2 = fx.Index(2)
        c_k0 = c_k_bytes // c32
        klane_dim = 2
        stride_klane = c16 * stride_nlane
        stride_n0 = c2 * stride_klane
        stride_k0 = n0 * stride_n0
    else:
        c64 = fx.Index(64)
        c4 = fx.Index(4)
        c_k0 = c_k_bytes // c64
        klane_dim = 4
        stride_klane = c16 * stride_nlane
        stride_k0 = c4 * stride_klane
        stride_n0 = c_k0 * stride_k0

    kpack_elems_static = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
    n0_i32 = arith.index_cast(T.i32, n0)
    c_k0_i32 = arith.index_cast(T.i32, c_k0)
    stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
    stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
    stride_klane_i32 = arith.index_cast(T.i32, stride_klane)
    stride_nlane_i32 = arith.index_cast(T.i32, stride_nlane)

    stride_b = (stride_n0_i32, stride_k0_i32, stride_klane_i32, stride_nlane_i32, 1)
    layout_b = fx.make_layout(
        (n0_i32, c_k0_i32, klane_dim, 16, kpack_elems_static), stride_b
    )
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


def _unpack_int4_to_int8_pair(packed32):
    """Split packed int4 dword into two int8 dwords (even/odd nibbles).

    7-op bit manipulation shared by all int4 unpack paths (W4A8, W4A16, W4A_FP8).
    """
    c_08 = fx.Int32(0x08080808)
    c_0f = fx.Int32(0x0F0F0F0F)
    c_1e = fx.Int32(0x1E)
    c_4 = fx.Int32(4)
    s0 = (packed32 & c_08) * c_1e
    even = (packed32 & c_0f) | s0
    t = packed32 >> c_4
    s1 = (t & c_08) * c_1e
    odd = (t & c_0f) | s1
    return even, odd


def _pack_i32_pair_to_i64(lo, hi, vector):
    """Pack two i32 values into one i64 via vector bitcast."""
    v2 = vector.from_elements(T.vec(2, T.i32), [lo, hi])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def _i8x4_in_i32_to_bf16x4_i64(val_i32, arith, vector, scale_val=None):
    """Convert one i32 (4 signed int8 bytes) to 4 bf16 packed as i64.

    Uses shift-based f32->bf16 truncation (lshr 16) instead of arith.truncf
    which on gfx942 expands to ~5 VALU per element. The shift is exact for
    unscaled int8 values and introduces <0.5 ULP error for scaled values.
    """
    vec1_i32_t = T.vec(1, T.i32)
    vec2_i32 = T.i32x2
    vec4_i8 = T.i8x4
    vec1_i64 = T.vec(1, T.i64)

    v1 = vector.from_elements(vec1_i32_t, [val_i32])
    i8x4 = vector.bitcast(vec4_i8, v1)

    f32_vals = []
    for i in range(4):
        val_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        v = arith.sitofp(T.f32, val_i8)
        if scale_val is not None:
            v = v * scale_val
        f32_vals.append(v)

    c16 = fx.Int32(16)
    c_ffff0000 = fx.Int32(0xFFFF0000)
    bits0 = arith.bitcast(T.i32, f32_vals[0])
    bits1 = arith.bitcast(T.i32, f32_vals[1])
    bits2 = arith.bitcast(T.i32, f32_vals[2])
    bits3 = arith.bitcast(T.i32, f32_vals[3])
    i32_lo = (bits0 >> c16) | (bits1 & c_ffff0000)
    i32_hi = (bits2 >> c16) | (bits3 & c_ffff0000)

    v2 = vector.from_elements(vec2_i32, [i32_lo, i32_hi])
    v64 = vector.bitcast(vec1_i64, v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def _e2m1_byte_to_bf16_bits(code_i32, arith):
    """Convert one i32 holding an e2m1 nibble code (0..15) to bf16 hi-bits (i32).

    Pure integer bit-ops (validated against the e2m1 codebook
    {0,±.5,±1,±1.5,±2,±3,±4,±6}). Layout of the 4-bit code: [sign|exp2|mant1].

    normal (exp>=1):   bf16 = (sign<<15) | ((exp-1+127)<<7) | (mant<<6)
    subnormal(exp==0): value = mant*0.5 -> (sign<<15) | (mant * 0x3F00)

    Returns the low 16 bits (in an i32) representing the bf16 value.
    """
    c1 = fx.Int32(1)
    c3 = fx.Int32(3)
    c6 = fx.Int32(6)
    c7 = fx.Int32(7)
    c126 = fx.Int32(126)

    s = (code_i32 >> c3) & c1
    e = (code_i32 >> c1) & c3
    m = code_i32 & c1

    hi_normal = ((e + c126) << c7) | (m << c6)
    hi_sub = m * fx.Int32(0x3F00)

    is_e0 = _arith.cmpi(
        _arith.CmpIPredicate.eq, _arith._to_raw(e), _arith._to_raw(fx.Int32(0))
    )
    hi = fx.Int32(
        _arith.select(is_e0, _arith._to_raw(hi_sub), _arith._to_raw(hi_normal))
    )
    return (s << fx.Int32(15)) | hi


def _e2m1x4_in_i32_to_bf16x4_i64(val_i32, arith, vector, scale_val=None):
    """Convert one i32 (4 e2m1 nibble codes as 4 bytes) to 4 bf16 packed as i64.

    Mirrors :func:`_i8x4_in_i32_to_bf16x4_i64` but decodes e2m1 (mxfp4) codes
    instead of signed int8. When ``scale_val`` (an f32, e.g. decoded E8M0) is
    provided, applies it via f32 multiply + shift-truncate to bf16.
    """
    v1 = vector.from_elements(T.vec(1, T.i32), [val_i32])
    i8x4 = vector.bitcast(T.i8x4, v1)

    if scale_val is None:
        # No scale: build bf16 bits directly and pack.
        bf16_his = []
        for i in range(4):
            byte_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
            # zero-extend byte to i32 code (values 0..15 fit in low nibble)
            code_i32 = _arith.extui(T.i32, _arith._to_raw(byte_i8))
            bf16_his.append(_e2m1_byte_to_bf16_bits(fx.Int32(code_i32), arith))
        c16 = fx.Int32(16)
        c_ffff = fx.Int32(0xFFFF)
        i32_lo = (bf16_his[0] & c_ffff) | (bf16_his[1] << c16)
        i32_hi = (bf16_his[2] & c_ffff) | (bf16_his[3] << c16)
        return _pack_i32_pair_to_i64(i32_lo, i32_hi, vector)

    # Scaled path: e2m1 code -> bf16 bits -> f32 -> * scale -> bf16 (shift-trunc).
    # Mirrors _i8x4_in_i32_to_bf16x4_i64's scaling structure (raw arith values).
    f32_vals = []
    for i in range(4):
        byte_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        code_i32 = _arith.extui(T.i32, _arith._to_raw(byte_i8))
        bf16_hi = _e2m1_byte_to_bf16_bits(fx.Int32(code_i32), arith)
        # Widen bf16 bits into the high half of an i32, then bitcast to f32.
        f32_bits = _arith._to_raw((bf16_hi << fx.Int32(16)))
        v = arith.bitcast(T.f32, f32_bits)
        if scale_val is not None:
            v = v * scale_val
        f32_vals.append(v)

    c16 = fx.Int32(16)
    c_ffff0000 = fx.Int32(0xFFFF0000)
    # Match _i8x4_in_i32_to_bf16x4_i64 exactly: keep raw bitcast values so the
    # right-shift lowers to a *logical* shift (shrui). Wrapping in fx.Int32 would
    # emit an arithmetic shift and sign-extend negative bf16 values into the high
    # half, corrupting the packed result (NaN/Inf).
    bits = [arith.bitcast(T.i32, f) for f in f32_vals]
    i32_lo = (bits[0] >> c16) | (bits[1] & c_ffff0000)
    i32_hi = (bits[2] >> c16) | (bits[3] & c_ffff0000)
    return _pack_i32_pair_to_i64(i32_lo, i32_hi, vector)


def load_b_raw_w4a16(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ku: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 8,
):
    """Phase 1 of W4A16 B load: issue buffer_load_dword, return raw packed i32.

    Same address calculation as the int4 unpack path in load_b_pack_k32
    but using ku-based indexing for 2-phase latency hiding.
    """
    if kpack_bytes != 8:
        raise ValueError(f"W4A16 requires kpack_bytes=8, got {kpack_bytes!r}")

    c64 = fx.Index(64)
    half_bytes = kpack_bytes // 2
    c2_idx = fx.Index(2)
    c4_idx = fx.Index(4)

    k0_base = base_k // c64

    k1_layout_offset = ku * 2
    lane_div_32 = lane_div_16 // c2_idx
    total_k1 = fx.Index(k1_layout_offset) + lane_div_32
    k0 = k0_base + (total_k1 // c4_idx)
    k1_local = total_k1 % c4_idx
    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * fx.Index(half_bytes)

    coord_pack = (n_blk, k0, k1_local, n_intra, fx.Index(0))
    idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)
    idx_bytes = idx_pack + k2_base

    b4 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_bytes,
        elem_type=elem_type,
        vec_elems=4,
        elem_bytes=1,
        offset_in_bytes=True,
    )
    packed32 = vector.extract(
        vector.bitcast(T.vec(1, T.i32), b4),
        static_position=[0],
        dynamic_position=[],
    )
    return packed32


def _int4_to_bf16x4_i64_gfx950(
    packed32, nibble_offsets, arith, vector, scale_val=None, defer_scale16=False
):
    """Convert 4 int4 nibbles to 4 bf16 packed as i64 using gfx950 instructions.

    Uses v_cvt_off_f32_i4_sdwa with byte_sel to avoid per-nibble shifts.
    Even nibbles (0,2,4,6) → SDWA BYTE_0/1/2/3 on original src.
    Odd nibbles (1,3,5,7)  → SDWA BYTE_0/1/2/3 on (src >> 4).
    Only 1 shift total instead of 7.

    When defer_scale16=True, the ×16 correction factor for v_cvt_off_f32_i4 is
    omitted and must be applied later (e.g. in the epilogue).  This saves VALU
    in the hot loop and uses v_cvt_pk_bf16_f32 for proper f32→bf16 conversion.
    """
    from flydsl.expr import rocdl
    from flydsl._mlir.dialects._arith_ops_gen import MulFOp as _MulFOp

    _uw = _arith._to_raw
    _av = _arith.ArithValue

    src_even = packed32
    src_odd = packed32 >> fx.Int32(4)

    f32_vals = []
    for nib in nibble_offsets:
        byte_idx = nib // 2
        src = src_odd if (nib % 2) else src_even
        v = rocdl.cvt_off_f32_i4(src, byte_sel=byte_idx)
        f32_vals.append(v)

    if defer_scale16:
        # Skip ×16; multiply by scale_val only if groupwise.
        if scale_val is not None:
            raw_scale = _uw(scale_val)
            f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
        # Use v_cvt_pk_bf16_f32 for proper f32→bf16 (no bit-shift trick needed).
        i32_lo = rocdl.cvt_pk_bf16_f32(f32_vals[0], f32_vals[1])
        i32_hi = rocdl.cvt_pk_bf16_f32(f32_vals[2], f32_vals[3])
    else:
        c16 = fx.Float32(16.0)
        if scale_val is not None:
            effective_scale = scale_val * c16
        else:
            effective_scale = c16
        raw_scale = _uw(effective_scale)
        f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
        # Truncate f32→bf16 via bit-shift (exact for scaled int values).
        c16_shift = fx.Int32(16)
        c_ffff0000 = fx.Int32(0xFFFF0000)
        bf16_vals = [arith.bitcast(T.i32, _av(v)) for v in f32_vals]
        i32_lo = (bf16_vals[0] >> c16_shift) | (bf16_vals[1] & c_ffff0000)
        i32_hi = (bf16_vals[2] >> c16_shift) | (bf16_vals[3] & c_ffff0000)

    v2 = vector.from_elements(T.vec(2, T.i32), [i32_lo, i32_hi])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def unpack_b_w4a16(
    packed32, arith, vector, scale_val=None, use_gfx950_cvt=False, defer_scale16=False
):
    """Phase 2 of W4A16 B load: unpack int4->int8 + convert int8->bf16.

    Takes raw packed32 from load_b_raw_w4a16 and produces (b0, b1) --
    two i64 values each containing 4 bf16 for one MFMA.

    When use_gfx950_cvt=True, uses v_cvt_off_f32_i4 + v_cvt_pk_bf16_f32
    for ~2x fewer VALU instructions.

    When defer_scale16=True (requires use_gfx950_cvt=True), the ×16
    correction for v_cvt_off_f32_i4 is omitted; caller must apply it
    in the epilogue.
    """
    if use_gfx950_cvt:
        b0 = _int4_to_bf16x4_i64_gfx950(
            packed32,
            [0, 2, 4, 6],
            arith,
            vector,
            scale_val,
            defer_scale16=defer_scale16,
        )
        b1 = _int4_to_bf16x4_i64_gfx950(
            packed32,
            [1, 3, 5, 7],
            arith,
            vector,
            scale_val,
            defer_scale16=defer_scale16,
        )
        return (b0, b1)
    even, odd = _unpack_int4_to_int8_pair(packed32)
    b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)
    return (b0, b1)


def _unpack_mxfp4_nibble_pair(packed32):
    """Split packed mxfp4 (fp4x2) dword into two i32s of 4 nibble-codes each.

    Unlike int4 (which sign-extends), mxfp4 codes are raw 4-bit patterns
    (0..15) interpreted via the e2m1 codebook, so we only mask — no
    sign extension.
    """
    c_0f = fx.Int32(0x0F0F0F0F)
    c_4 = fx.Int32(4)
    even = packed32 & c_0f
    odd = (packed32 >> c_4) & c_0f
    return even, odd


def unpack_b_w4a16_mxfp4(packed32, arith, vector, scale_val=None):
    """Phase 2 of W4A16 mxfp4 B load: unpack fp4x2 + e2m1 dequant to bf16.

    Takes raw packed32 from :func:`load_b_raw_w4a16` and produces (b0, b1) --
    two i64 values each containing 4 bf16 for one MFMA. ``scale_val`` (f32,
    decoded from E8M0) is applied as a per-group multiply when provided.
    """
    even, odd = _unpack_mxfp4_nibble_pair(packed32)
    b0 = _e2m1x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _e2m1x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)
    return (b0, b1)


def _e2m1x4_in_i32_to_fp8x4_i32(val_i32, arith, vector, ratios=None):
    """Convert one i32 (4 e2m1 nibble codes as 4 bytes) to 4 fp8 (e4m3fnuz) in an i32.

    e2m1 code -> bf16 bits -> f32 -> (optional * ratio, a power-of-2 fold factor) ->
    v_cvt_pk_fp8_f32. ``ratios`` is None (no fold, a8w4 Phase-1 UNIFORM/PAIR) or a
    length-4 list of f32 fold factors (2^(exp_g - base)) for the a8w4 in-kernel fold.

    The e2m1 code 8 is -0.0; e4m3fnuz encodes -0 as 0x80 (NaN), so we add +0.0
    (IEEE: -0.0 + 0.0 = +0.0) to normalize -0 -> +0 before the cvt (matches torch
    ``.to(float8_e4m3fnuz)`` which the Phase-1 proto validated byte-exact).
    """
    from flydsl.expr import rocdl

    v1 = vector.from_elements(T.vec(1, T.i32), [val_i32])
    i8x4 = vector.bitcast(T.i8x4, v1)
    c_pzero = fx.Float32(0.0)
    f32_vals = []
    for i in range(4):
        byte_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        code_i32 = _arith.extui(T.i32, _arith._to_raw(byte_i8))
        bf16_hi = _e2m1_byte_to_bf16_bits(fx.Int32(code_i32), arith)
        v = arith.bitcast(T.f32, _arith._to_raw(bf16_hi << fx.Int32(16)))
        if ratios is not None:
            v = v * ratios[i]
        # normalize -0.0 -> +0.0 (e4m3fnuz -0 == 0x80 NaN)
        v = v + c_pzero
        f32_vals.append(_arith._to_raw(v))
    p = _arith._to_raw(fx.Int32(0))
    p = rocdl.cvt_pk_fp8_f32(T.i32, f32_vals[0], f32_vals[1], p, 0)
    p = rocdl.cvt_pk_fp8_f32(T.i32, f32_vals[2], f32_vals[3], p, 1)
    return p


def unpack_b_w4a16_mxfp4_to_fp8(packed32, arith, vector, ratios_even=None, ratios_odd=None):
    """Unpack packed32 (8 e2m1 codes) -> i64 (8 fp8) for one K32 fp8 MFMA operand.

    Mirrors the int4 W4A8 pack layout (:func:`load_b_pack_k32` unpack_int4 path):
    even nibbles -> low 4 fp8 bytes, odd nibbles -> high 4 fp8 bytes of the i64.
    ``ratios_even``/``ratios_odd`` are None (no fold) or length-4 f32 fold factors
    for the a8w4 in-kernel per-element ratio-fold.
    """
    even, odd = _unpack_mxfp4_nibble_pair(packed32)
    fe = _e2m1x4_in_i32_to_fp8x4_i32(even, arith, vector, ratios=ratios_even)
    fo = _e2m1x4_in_i32_to_fp8x4_i32(odd, arith, vector, ratios=ratios_odd)
    return _pack_i32_pair_to_i64(fe, fo, vector)


def load_b_pack_k32_pair_raw(
    buffer_ops,
    arith,
    vector,
    *,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ku: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 8,
    elem_bytes: int = 1,
):
    """Load BOTH K32 operands (r0, r1) of one K64 micro-step in ONE wide buffer
    load, matching the mxfp8 path's ``dwordx4`` load.

    The two operands of a K64 step are the two halves of one 8-byte kpack
    (r0 = bytes[0:4], r1 = bytes[4:8]); the packed-int4 raw path used to load
    them as two separate ``dword`` loads (2x the B-load instructions of the
    mxfp8 ``dwordx4``). Loading the whole kpack once (``dwordx2``) halves the
    B-load instruction count -- the real a8w4 Phase-1 bottleneck (the unpack ALU
    is largely hidden by MFMA). Returns (packed32_r0, packed32_r1).
    """
    c64 = fx.Index(64)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0 = base_k_bytes // c64 + arith.constant(int(ku), index=True)
    k1 = lane_div_16
    coord_pack = (n_blk, k0, k1, n_intra, fx.Index(0))
    idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)
    # one dwordx2 (8 bytes) = the full kpack = both K32 operands.
    b8 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_pack,
        elem_type=elem_type,
        vec_elems=8,
        elem_bytes=1,
        offset_in_bytes=True,
    )
    b_i32x2 = vector.bitcast(T.vec(2, T.i32), b8)
    r0 = vector.extract(b_i32x2, static_position=[0], dynamic_position=[])
    r1 = vector.extract(b_i32x2, static_position=[1], dynamic_position=[])
    return r0, r1


def _e2m1_code_to_fp8_byte_fold(code_i32, r_i32, arith):
    """e2m1 nibble code (i32, 0..15) + integer exp-shift r (i32, <=0) -> fp8
    e4m3fnuz byte (i32), via PURE integer bit-ops (no bf16/f32 detour, no cvt).

    Reproduces ``(e2m1_value * 2^r).to(float8_e4m3fnuz)``: handles normal,
    subnormal (round-to-nearest-even), flush-to-0 and fnuz -0->+0. Validated
    byte-exact for all (code, r) by aiter_logs/proto_a8w4_fp8bits.py.

    Optimization for a8w4 Phase-1: e2m1 (mant<={0,1}) -> fp8 e4m3fnuz is exact in
    the normal range, and the per-pair ratio (a power of 2) folds into the fp8
    exponent as an integer add -- so the whole unpack+fold is integer bit-ops,
    replacing the e2m1->bf16->f32->(*ratio f32)->cvt_pk_fp8 chain.
    """
    _r = _arith._to_raw
    P = _arith.CmpIPredicate
    c0 = fx.Int32(0)
    c1 = fx.Int32(1)
    c2 = fx.Int32(2)
    c3 = fx.Int32(3)
    c7 = fx.Int32(7)
    c8 = fx.Int32(8)
    c31 = fx.Int32(31)
    s = (code_i32 >> c3) & c1
    e = (code_i32 >> c1) & c3
    m = code_i32 & c1
    is_e0 = _arith.cmpi(P.eq, _r(e), _r(c0))
    is_zero = _arith.andi(is_e0, _arith.cmpi(P.eq, _r(m), _r(c0)))
    # e2m1 magnitude as mant4 (=1.MMM*8, so {8,12}) and unbiased exponent uexp.
    mant4 = fx.Int32(_arith.select(is_e0, _r(c8), _r(c8 + (m << c2))))
    uexp = fx.Int32(_arith.select(is_e0, _r(fx.Int32(-1)), _r(e - c1)))
    # fold ratio 2^r into the exponent; fp8 normal exp field E = uexp + r + 8.
    E = uexp + r_i32 + c8
    byte_n = (s << c7) | (E << c3) | (mant4 - c8)
    # underflow (E<1) -> subnormal mant_sub = round_even(mant4 >> (1-E)).
    shift0 = c1 - E
    shift = fx.Int32(
        _arith.select(
            _arith.cmpi(P.slt, _r(shift0), _r(c1)),
            _r(c1),
            _arith.select(
                _arith.cmpi(P.sgt, _r(shift0), _r(c31)), _r(c31), _r(shift0)
            ),
        )
    )
    floor = mant4 >> shift
    rem = mant4 - (floor << shift)
    half = c1 << (shift - c1)
    round_up = _arith.ori(
        _arith.cmpi(P.sgt, _r(rem), _r(half)),
        _arith.andi(
            _arith.cmpi(P.eq, _r(rem), _r(half)),
            _arith.cmpi(P.ne, _r(floor & c1), _r(c0)),
        ),
    )
    msub = floor + fx.Int32(_arith.select(round_up, _r(c1), _r(c0)))
    # e2m1 mant4<=12, shift>=1 -> msub<=6 <8, so no subnormal->normal overflow.
    byte_s = (s << c7) | msub
    byte_s = fx.Int32(
        _arith.select(_arith.cmpi(P.eq, _r(msub), _r(c0)), _r(c0), _r(byte_s))
    )
    byte = fx.Int32(
        _arith.select(_arith.cmpi(P.sge, _r(E), _r(c1)), _r(byte_n), _r(byte_s))
    )
    byte = fx.Int32(_arith.select(is_zero, _r(c0), _r(byte)))
    return byte & fx.Int32(0xFF)


def _e2m1x4_in_i32_to_fp8x4_i32_bitfold(val_i32, r_i32, arith, vector):
    """4 e2m1 codes (as 4 bytes in an i32) + integer exp-shift r -> 4 fp8 in an i32.

    Pure integer path (see :func:`_e2m1_code_to_fp8_byte_fold`); all 4 share r
    (a8w4 Phase-1 per-lane fold). Packs 4 fp8 bytes little-endian into the i32.
    """
    v1 = vector.from_elements(T.vec(1, T.i32), [val_i32])
    i8x4 = vector.bitcast(T.i8x4, v1)
    c8b = fx.Int32(8)
    c16 = fx.Int32(16)
    c24 = fx.Int32(24)
    b = []
    for i in range(4):
        byte_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        code = fx.Int32(_arith.extui(T.i32, _arith._to_raw(byte_i8)))
        b.append(_e2m1_code_to_fp8_byte_fold(code, r_i32, arith))
    return b[0] | (b[1] << c8b) | (b[2] << c16) | (b[3] << c24)


def unpack_b_w4a16_mxfp4_to_fp8_bitfold(packed32, r_i32, arith, vector):
    """Integer-only variant of :func:`unpack_b_w4a16_mxfp4_to_fp8`: packed32
    (8 e2m1 codes) + per-lane integer exp-shift r -> i64 (8 fp8) for one K32 fp8
    MFMA operand. No f32 detour / no cvt (see _e2m1_code_to_fp8_byte_fold)."""
    even, odd = _unpack_mxfp4_nibble_pair(packed32)
    fe = _e2m1x4_in_i32_to_fp8x4_i32_bitfold(even, r_i32, arith, vector)
    fo = _e2m1x4_in_i32_to_fp8x4_i32_bitfold(odd, r_i32, arith, vector)
    return _pack_i32_pair_to_i64(fe, fo, vector)


def load_b_pack_k32(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ki_step: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    unpack_int4: bool = False,
    raw_packed: bool = False,
) -> ir.Value:
    """Load one B pack for one MFMA(x32) micro-step.

    Returns an i64 Value containing 8 bytes consumed by MFMA.
    When ``raw_packed`` (packed-int4 weight, e.g. a8w4 Phase-1 mxfp4_fp8), returns
    the raw i32 ``packed32`` (8 nibble codes) instead — the caller unpacks + folds
    e2m1->fp8 with the per-group scale in the B-load/compute stage.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")
    if unpack_int4 and kpack_bytes != 8:
        raise ValueError("unpack_int4 requires kpack_bytes=8 (packed int4 layout)")
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")

    c64 = fx.Index(64)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0_base = base_k_bytes // c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16
    half_bytes = kpack_bytes // 2
    k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)

    coord_pack = (n_blk, k0, k1, n_intra, fx.Index(0))
    idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)

    if unpack_int4 or raw_packed:
        idx_bytes = idx_pack + k2_base
        b4 = _buffer_load_vec(
            buffer_ops,
            vector,
            b_rsrc,
            idx_bytes,
            elem_type=elem_type,
            vec_elems=4,
            elem_bytes=1,
            offset_in_bytes=True,
        )
        packed32 = vector.extract(
            vector.bitcast(T.vec(1, T.i32), b4),
            static_position=[0],
            dynamic_position=[],
        )
        if raw_packed:
            return packed32
        even, odd = _unpack_int4_to_int8_pair(packed32)
        return _pack_i32_pair_to_i64(even, odd, vector)

    vec_elems = kpack_bytes // int(elem_bytes)
    b16 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_pack,
        elem_type=elem_type,
        vec_elems=vec_elems,
        elem_bytes=elem_bytes,
        offset_in_bytes=(elem_bytes == 1),
    )

    b_i32x4 = vector.bitcast(T.i32x4, b16)

    half = ki_step % 2
    if half == 0:
        d0 = vector.extract(b_i32x4, static_position=[0], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[1], dynamic_position=[])
    else:
        d0 = vector.extract(b_i32x4, static_position=[2], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[3], dynamic_position=[])

    v2 = vector.from_elements(T.vec(2, T.i32), [d0, d1])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def tile_chunk_coord_i32(
    arith,
    *,
    tx_i32_base: ir.Value,
    i: int,
    total_threads: int,
    layout_tile_div4,
    chunk_i32: int = 4,
):
    """Map (thread, chunk_id) -> (row_local, col_local_i32) for X/A loads."""
    if chunk_i32 not in (1, 2, 4):
        raise ValueError(f"chunk_i32 must be one of (1,2,4), got {chunk_i32!r}")
    chunk_off_i32 = arith.constant(i * total_threads * chunk_i32, index=True)
    tile_idx_i32 = tx_i32_base + chunk_off_i32
    coord_local = fx.idx2crd(fx.Int32(tile_idx_i32), layout_tile_div4)
    row_local = fx.get(coord_local, 0)
    col_local_i32 = fx.get(coord_local, 1)
    return row_local, col_local_i32


def buffer_copy_gmem16_dwordx4(
    buffer_ops,
    vector,
    *,
    elem_type,
    idx_i32: ir.Value,
    rsrc,
    vec_elems: int = 16,
    elem_bytes: int = 1,
):
    """Copy 16 bytes from global memory into regs via buffer-load dwordx4 lowering."""
    if int(vec_elems) <= 0:
        raise ValueError(f"vec_elems must be > 0, got {vec_elems!r}")
    return _buffer_load_vec(
        buffer_ops,
        vector,
        rsrc,
        idx_i32,
        elem_type=elem_type,
        vec_elems=vec_elems,
        elem_bytes=elem_bytes,
        offset_in_bytes=False,
    )


def lds_store_16b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec16_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x4: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 16B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(tuple(fx.Int32(c) for c in coord_store), layout_lds) + lds_base
    v16 = vector.bitcast(vec16_ty, vec_part_i32x4)
    vector.store(v16, lds_memref, [idx0])


def lds_store_8b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec8_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x2: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 8B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(tuple(fx.Int32(c) for c in coord_store), layout_lds) + lds_base
    v8 = vector.bitcast(vec8_ty, vec_part_i32x2)
    vector.store(v8, lds_memref, [idx0])


def lds_store_4b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec4_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x1: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 4B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(tuple(fx.Int32(c) for c in coord_store), layout_lds) + lds_base
    v4 = vector.bitcast(vec4_ty, vec_part_i32x1)
    vector.store(v4, lds_memref, [idx0])


def lds_load_pack_k32(
    arith,
    vector,
    *,
    lds_memref,
    layout_lds,
    k_blocks16: ir.Value,
    curr_row_a_lds: ir.Value,
    col_base: ir.Value,
    half: int,
    lds_base: ir.Value,
    ck_lds128: bool,
    vec16_ty,
    vec8_ty,
    vec2_i64_ty,
    vec1_i64_ty,
):
    """Load one i64 A-pack for an MFMA K32 micro-step from LDS."""
    col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
    if ck_lds128:
        coord_a16 = (curr_row_a_lds, col_base_swz)
        idx_a16 = crd2idx(tuple(fx.Int32(c) for c in coord_a16), layout_lds) + lds_base
        loaded_a16 = vector.load_op(vec16_ty, lds_memref, [idx_a16])
        a_vec128 = vector.bitcast(vec2_i64_ty, loaded_a16)
        return vector.extract(a_vec128, static_position=[half], dynamic_position=[])
    else:
        col_swizzled = col_base_swz + (half * 8)
        coord_a = (curr_row_a_lds, col_swizzled)
        idx_a = crd2idx(tuple(fx.Int32(c) for c in coord_a), layout_lds) + lds_base
        loaded_a8 = vector.load_op(vec8_ty, lds_memref, [idx_a])
        a_vec64 = vector.bitcast(vec1_i64_ty, loaded_a8)
        return vector.extract(a_vec64, static_position=[0], dynamic_position=[])


def xcd_remap_bx_by(
    bx,
    by,
    c_m,
    *,
    tile_m: int,
    tile_n: int,
    N: int,
    xcd_swizzle: int,
    num_xcds: int = 8,
):
    """Remap (bx, by) for L2-cache reuse via XCD swizzle.

    No-op when ``xcd_swizzle <= 0``. Otherwise:
      1. Linearize the original (bx, by) grid round-robin across ``num_xcds``
         XCDs so that contiguous workgroup ids stay on the same XCD.
      2. Re-tile that 1-D order with an M-major group of size ``xcd_swizzle``,
         folding the tail group when ``gy`` does not divide evenly.

    Designed to be called inside a ``@flyc.kernel`` immediately after::

        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bx, by = xcd_remap_bx_by(bx, by, c_m, tile_m=..., tile_n=..., N=...,
                                 xcd_swizzle=xcd_swizzle)

    ``c_m`` is the dynamic ``fx.Index`` for runtime ``M``; ``tile_m``,
    ``tile_n``, ``N`` and ``xcd_swizzle`` are compile-time Python ints.
    """
    if xcd_swizzle <= 0:
        return bx, by

    _c1 = fx.arith.constant(1, index=True)
    _c_tm = fx.arith.constant(tile_m, index=True)
    _gx = fx.arith.constant(N // tile_n, index=True)
    _gy = (c_m + _c_tm - _c1) // _c_tm

    _linear_id = bx * _gx + by
    _num_wgs = _gx * _gy

    _c_xcds = fx.arith.constant(num_xcds, index=True)
    _q = _num_wgs // _c_xcds
    _r = _num_wgs % _c_xcds
    _xcd = _linear_id % _c_xcds
    _in_xcd = _linear_id // _c_xcds
    _xcd_lt_r = fx.arith.cmpi(CmpIPredicate.ult, _xcd, _r)
    _clip = fx.arith.select(_xcd_lt_r, _xcd, _r)
    _wgid = _xcd * _q + _clip + _in_xcd

    _c_wgm = fx.arith.constant(xcd_swizzle, index=True)
    _num_wgid_in_group = _c_wgm * _gx
    _group_id = _wgid // _num_wgid_in_group
    _first_pid_m = _group_id * _c_wgm
    _remaining_m = _gy - _first_pid_m
    _cmp_m = fx.arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
    _group_size_m = fx.arith.select(_cmp_m, _remaining_m, _c_wgm)

    _wgid_in_group = _wgid % _num_wgid_in_group
    new_bx = _first_pid_m + (_wgid_in_group % _group_size_m)
    new_by = _wgid_in_group // _group_size_m
    return new_bx, new_by


__all__ = [
    "PreshuffleBLayout",
    "PreshuffleScaleLayout",
    "buffer_copy_gmem16_dwordx4",
    "lds_load_pack_k32",
    "lds_row_major_idx",
    "lds_store_4b_xor16",
    "lds_store_8b_xor16",
    "lds_store_16b_xor16",
    "make_preshuffle_b_layout",
    "make_preshuffle_scale_layout",
    "load_b_pack_k32",
    "load_b_raw_w4a16",
    "unpack_b_w4a16",
    "unpack_b_w4a16_mxfp4",
    "load_b_raw_w4a16_groupwise",
    "unpack_b_w4a16_groupwise",
    "unpack_b_w4a16_mxfp4_groupwise",
    "extract_bf16_scale",
    "e8m0_to_f32_scale",
    "split_row_major_2d",
    "swizzle_xor16",
    "tile_chunk_coord_i32",
    "xcd_remap_bx_by",
]


# ---------------------------------------------------------------------------
# Groupwise scale load helper (shared by W4A16 and W4A8 groupwise paths)
# ---------------------------------------------------------------------------


def _load_groupwise_scale(
    buffer_ops,
    arith,
    *,
    scale_rsrc,
    expert_offset,
    n_blk,
    n_intra,
    k_pos,
    num_groups: int,
    group_size: int,
    n_per_expert: int,
    scale_dtype=None,
):
    """Load one per-group scale value from the scale buffer.

    Computes the linear index into the scale tensor from expert offset,
    N position, and group index derived from ``k_pos``.

    For bf16 scales the tensor uses ``(E, G//2, N, 2)`` layout — two
    adjacent groups for the same N position are packed into one dword.
    We load the raw i32 dword (no extraction) so it can be carried as
    loop state without register copies.  Use :func:`extract_bf16_scale`
    in the compute phase to obtain the f32 value.
    """
    c16 = fx.Index(16)
    n_global = n_blk * c16 + n_intra
    c_group_size = fx.Index(group_size)
    c_npe = fx.Index(n_per_expert)
    group_idx = k_pos // c_group_size
    if scale_dtype is None:
        scale_dtype = T.f32

    if scale_dtype == T.bf16:
        # (E, G//2, N, 2) layout: dword at [e, pair, n] holds bf16 scales
        # for groups 2*pair and 2*pair+1.
        pair_idx = group_idx >> fx.Index(1)  # group_idx // 2
        # Dword index: same flat formula but with G//2 groups
        num_pairs = num_groups // 2
        c_npm1 = fx.Index(num_pairs - 1)
        dword_base = expert_offset * c_npm1 + n_global
        dword_elem = dword_base + pair_idx * c_npe
        dword_idx = arith.index_cast(T.i32, dword_elem)
        # Return raw i32 dword — extraction deferred to compute phase.
        scale_val = buffer_ops.buffer_load(
            scale_rsrc, dword_idx, vec_width=1, dtype=T.i32
        )
    else:
        # (E, G, N) layout with f32 dtype
        c_gm1 = fx.Index(num_groups - 1)
        base_scale = expert_offset * c_gm1 + n_global
        elem_idx = base_scale + group_idx * c_npe
        scale_idx_i32 = arith.index_cast(T.i32, elem_idx)
        scale_val = buffer_ops.buffer_load(
            scale_rsrc, scale_idx_i32, vec_width=1, dtype=T.f32
        )
    return scale_val


def extract_bf16_scale(arith, scale_raw_i32, ku: int):
    """Extract f32 scale from raw i32 dword loaded by bf16 groupwise path.

    In the ``(E, G//2, N, 2)`` layout two adjacent groups share one dword.
    ``ku`` determines which half: even ku → low bf16, odd ku → high bf16.
    """
    if ku % 2 == 0:
        # Low bf16: shift left by 16 to place in upper 16 bits → f32
        return arith.bitcast(T.f32, scale_raw_i32 << fx.Int32(16))
    else:
        # High bf16: mask upper 16 bits → f32
        return arith.bitcast(T.f32, scale_raw_i32 & fx.Int32(0xFFFF0000))


# ---------------------------------------------------------------------------
# W4A16 groupwise load / unpack helpers
# ---------------------------------------------------------------------------


def load_b_raw_w4a16_groupwise(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k,
    ku: int,
    n_blk,
    n_intra,
    lane_div_16,
    elem_type,
    scale_rsrc,
    expert_offset,
    num_groups: int,
    group_size: int,
    n_per_expert: int,
    kpack_bytes: int = 8,
    scale_dtype=None,
):
    """Phase 1 of W4A16 groupwise B load: buffer_loads for weight + scale.

    Reuses :func:`load_b_raw_w4a16` for the weight load, then issues an
    additional ``buffer_load_dword`` for the per-group scale.

    Returns ``(packed32, scale_val)``.
    """
    packed32 = load_b_raw_w4a16(
        buffer_ops,
        arith,
        vector,
        arg_b=arg_b,
        b_rsrc=b_rsrc,
        layout_b=layout_b,
        base_k=base_k,
        ku=ku,
        n_blk=n_blk,
        n_intra=n_intra,
        lane_div_16=lane_div_16,
        elem_type=elem_type,
        kpack_bytes=kpack_bytes,
    )
    k_pos = base_k + fx.Index(ku * 32)
    scale_val = _load_groupwise_scale(
        buffer_ops,
        arith,
        scale_rsrc=scale_rsrc,
        expert_offset=expert_offset,
        n_blk=n_blk,
        n_intra=n_intra,
        k_pos=k_pos,
        num_groups=num_groups,
        group_size=group_size,
        n_per_expert=n_per_expert,
        scale_dtype=scale_dtype,
    )
    return (packed32, scale_val)


def unpack_b_w4a16_groupwise(packed32, scale_val, arith, vector, use_gfx950_cvt=False):
    """Phase 2 of W4A16 groupwise: unpack + scale + convert to bf16."""
    return unpack_b_w4a16(
        packed32, arith, vector, scale_val=scale_val, use_gfx950_cvt=use_gfx950_cvt
    )


def e8m0_to_f32_scale(arith, scale_raw_i32):
    """Decode an E8M0 (uint8 exponent) scale to f32.

    E8M0 stores a biased exponent ``u`` with value ``2^(u-127)``. The f32 with
    that value has bits ``u << 23``. The raw dword may pack multiple E8M0 bytes;
    caller must pass the byte already isolated in the low 8 bits.
    """
    u = fx.Int32(scale_raw_i32) & fx.Int32(0xFF)
    f32_bits = u << fx.Int32(23)
    return fx.Float32(arith.bitcast(T.f32, _arith._to_raw(f32_bits)))


def unpack_b_w4a16_mxfp4_groupwise(packed32, scale_val, arith, vector):
    """Phase 2 of W4A16 mxfp4 groupwise: unpack fp4x2 + e2m1 dequant + scale.

    ``scale_val`` is the per-group f32 scale (already decoded from E8M0).
    """
    return unpack_b_w4a16_mxfp4(packed32, arith, vector, scale_val=scale_val)
