# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run flydsl 0.2.x-style kernels on flydsl 0.1.x.

The MoE kernels ported from PR3987 are written against flydsl 0.2.x.  Everything
performance-relevant they use -- MFMA atoms, tiled_mma, tiled copies, layout
algebra, swizzles, register fragments -- already exists in 0.1.x, so the gap is
not capability but *plumbing*: 0.2.x added argument coercion, typed-value helpers
and a python-side normalisation layer that 0.1.x lacks, and without them the same
source either raises, fails MLIR verification, or segfaults inside the native
layout algebra.

Importing this module installs those pieces.  Every patch mirrors what flydsl
0.2.x does (referenced per patch below) rather than inventing behaviour, and every
patch is a no-op on 0.2.x, so a single source tree works on both releases.

Kept in one module on purpose: the patches are global (they attach to flydsl's own
classes and primitives), so having them in one auditable place beats sprinkling
them through the kernels.
"""

import os

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import primitive as _prim
from flydsl.expr.numeric import Int32 as _I32, Int64 as _I64, Numeric as _Num
from flydsl.expr.primitive import IntTupleType as _IntTupleType
from flydsl.expr.typing import Pointer as _FlyPointer, T
from flydsl.expr.utils.arith import ArithValue as _ArithValue

# ---------------------------------------------------------------------------
# capability flags
# ---------------------------------------------------------------------------

#: 0.2.x exposes the raw-pointer value type as ``fx.Pointer``; 0.1.x does not.
IS_02X = hasattr(fx, "Pointer")

#: 0.2.x declares LDS with ``@fx.union`` + ``fx.SharedAllocator``.
HAS_SHARED_UNION = hasattr(fx, "union")

#: 0.1.x's native ``slice`` aborts on the broadcast coord-tensor layouts the copy
#: helpers build ("Mismatched ranks in slice").  Verified to be a property of the
#: layout rather than the coord: even an all-wildcard coord aborts on
#: ``(16,(12,64)):(1,(16,0))``, while the same shape with strides ``(0,(0,1))``
#: slices fine.  Callers that can derive the indices arithmetically must do so.
SLICE_SUPPORTS_COORD_TENSORS = IS_02X and (
    # escape hatch: force the arithmetic gather on 0.2.x too, so the index math can
    # be validated against a build where everything else is known good
    os.environ.get("AITER_FLYDSL_FORCE_ARITH_GATHER", "0") != "1"
)

#: ``fx.Pointer`` for isinstance tests.  0.1.x *has* pointer values (get_iter
#: returns one), it just does not export the class as ``fx.Pointer``.
PointerType = getattr(fx, "Pointer", _FlyPointer)


def _dbg(tag, msg):
    if os.environ.get("AITER_FLYDSL_012_DEBUG"):
        print(f"[012compat] {tag}: {msg}", flush=True)


# ---------------------------------------------------------------------------
# helpers used by the kernels (available on both releases)
# ---------------------------------------------------------------------------


def recast_iter(dtype, ptr):
    """``recast_iter`` accepting a Numeric subclass on both releases.

    0.2.x wraps the element type into ``PointerType(elem, memspace, alignment)``
    itself; 0.1.x forwards its argument to MLIR and needs the finished pointer
    type ("Result 0 of operation fly.recast_iter must be a Type").
    """
    if IS_02X:
        return fx.recast_iter(dtype, ptr)
    # The alignment must be a multiple of the element size, so widening the element
    # type (e.g. fp8 -> u32 for the 128-bit copies) has to raise it.  Every such
    # recast here sits under a 128-bit copy atom, whose addresses are 16B aligned
    # anyway; claim only the element size so flydsl's add_offset type inference
    # keeps agreeing with the declared pointer type.
    align = max(int(ptr.alignment), dtype.width // 8)
    return fx.recast_iter(
        fx.PointerType.get(dtype.ir_type, ptr.memspace, align), ptr
    )


def as_i64(v):
    """Coerce a python int / fx Integer / ir.Value to a raw i64 ir.Value."""
    iv = getattr(v, "ir_value", None)
    if callable(iv):
        v = iv()
    if isinstance(v, ir.Value) and str(v.type) == "i64":
        return v
    return _I64(v).ir_value()


def signless(v):
    """Return an integer value with a signless MLIR type.

    0.1.x types a torch int32 kernel argument as ``si32``, and ``arith.cmpi``
    rejects a signed operand mixed with a signless one.  Signedness is fixed at
    the *pointer* (see :func:`as_ptr`), so by the time a value is loaded it should
    already be signless; anything else is a bug worth surfacing rather than
    papering over with an ``unrealized_conversion_cast`` (which passes
    verification but then dies in ``reconcile_unrealized_casts``).
    """
    raw = v.ir_value() if callable(getattr(v, "ir_value", None)) else v
    ty = str(getattr(raw, "type", ""))
    if ty.startswith(("si", "ui")):
        raise TypeError(
            f"value is still {ty}; normalise the pointer via as_ptr() instead of "
            "casting the loaded value"
        )
    return raw


def as_ptr(p, dtype=None):
    """Normalise a kernel argument to an iterator suitable for ``make_view``.

    Besides the 0.2.x behaviour (convert memref/pointer, optionally recast the
    element type) this always re-casts through ``p.dtype`` on 0.1.x: a torch int32
    argument arrives as ``si32`` there and *every* value, constant and GEP derived
    from it inherits that signedness, which MLIR arith ops reject and LLVM
    translation cannot represent.  The Numeric classes' ``ir_type`` is always the
    signless form, so recasting through it normalises the whole chain at once.
    """
    try:
        p = fx.get_iter(p)
    except Exception:  # already an iterator / pointer
        pass
    if dtype is None:
        if IS_02X:
            return p
        dtype = p.dtype
    if IS_02X and p.dtype == dtype:
        return p
    return recast_iter(dtype, p)


def make_buffer_tensor(tensor, max_size=True, *, num_records_bytes=None):
    """``rocdl.make_buffer_tensor`` with 0.2.x's bounds semantics on 0.1.x too.

    0.1.x hardcodes the descriptor's numRecords to ``0xFFFFFFFF``, i.e. it always
    behaves like ``max_size=True`` and provides no hardware out-of-bounds check.
    These kernels depend on that check: padded ``sorted_ids`` slots deliberately
    address past the end of the activation buffer and must read 0 rather than
    fault.  Mirrors 0.2.x: explicit byte count, else 0xFFFFFFFF, else
    ``cosize(layout) * elem_bytes``.
    """
    if IS_02X:
        return fx.rocdl.make_buffer_tensor(
            tensor, max_size, num_records_bytes=num_records_bytes
        )

    from flydsl._mlir.dialects import arith as _a, fly, llvm
    from flydsl.expr.meta import _to_raw_value

    raw = _to_raw_value(tensor)
    layout = _prim.get_layout(tensor)
    elem_type = fly.MemRefType(raw.type).element_type
    elem_bits = tensor.dtype.width

    if num_records_bytes is not None:
        nrec = as_i64(num_records_bytes)
    elif max_size:
        nrec = _a.ConstantOp(T.i64, ir.IntegerAttr.get(T.i64, 0xFFFFFFFF)).result
    else:
        cosize = fx.get_scalar(fx.cosize(layout))
        nrec = as_i64(
            cosize * (elem_bits // 8)
            if elem_bits % 8 == 0
            else (cosize * elem_bits + 7) // 8
        )

    # The base must be the tensor's *iterator*, i.e. include any offset already
    # applied to it (`+ e_idx * BLOCK_M * N` for the output, `+ expert_id * N * K`
    # for the weights).  0.2.x passes get_iter(tensor) for exactly this reason;
    # 0.1.x's own make_buffer_tensor uses extract_aligned_pointer_as_index, which
    # returns the aligned *base* of the allocation and silently drops the offset --
    # every block then reads expert 0 and writes block 0.  Go through
    # ptrtoint/inttoptr so the offset survives.
    addr = fx.ptrtoint(fx.get_iter(tensor))
    if str(getattr(addr, "type", "")) != "i64":
        addr = _a.IndexCastOp(T.i64, addr).result
    base = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), addr).result
    stride = _a.ConstantOp(T.i16, ir.IntegerAttr.get(T.i16, 0)).result
    from flydsl.expr.buffer_ops import _get_buffer_flags

    flags = _a.ConstantOp(
        T.i32, ir.IntegerAttr.get(T.i32, _get_buffer_flags())
    ).result
    bd_ptr_type = fly.PointerType.get(
        elem_type,
        address_space=int(fly.AddressSpace.BufferDesc),
        alignment=fx.get_iter(tensor).alignment,
    )
    return _prim.make_view(
        _prim.make_ptr(bd_ptr_type, [base, stride, nrec, flags]), layout
    )


# ---------------------------------------------------------------------------
# LDS: `@fx.union` + fx.SharedAllocator (0.2.x) -> SmemAllocator (0.1.x)
# ---------------------------------------------------------------------------


class _SharedField:
    """One field of a shared-storage union, exposing 0.2.x's ``.peek().view()``."""

    def __init__(self, ptr):
        self._ptr = ptr

    def peek(self):
        return self

    def view(self, layout):
        return fx.make_view(self._ptr, layout)


def union_bytes(fields):
    """Byte size of a union whose fields are ``name -> (dtype, count)``."""
    return max(count * (dtype.width // 8) for dtype, count in fields.values())


def make_lds_allocator(fields):
    """Reserve ``union_bytes(fields)`` in the static LDS global (0.1.x only).

    Returns ``(allocator, byte_offset)``.  Must run outside the kernel: the launch
    function finalizes the global before the kernel body is traced.
    """
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator

    alloc = SmemAllocator(None, arch=get_rocm_arch())
    offset = alloc._align(alloc.ptr, 16)
    alloc.ptr = offset + union_bytes(fields)
    return alloc, offset


def finalize_lds(allocator, ctx):
    """Emit the LDS global into the gpu.module body (0.1.x only)."""
    if allocator is None:
        return
    allocator.finalized = False
    with ir.InsertionPoint(ctx.gpu_module_body):
        allocator.finalize()


def shared_union(allocator, byte_offset, fields):
    """0.1.x replacement for ``fx.SharedAllocator().allocate(<@fx.union ...>)``.

    Uses the *static* LDS global that ``SmemAllocator`` emits, because 0.1.x's
    dynamic-shared path produces a zero-sized ``!llvm.array<0 x i8>`` global.
    ``SmemPtr.get()`` hands back a builtin ``memref``, which ``make_view`` rejects,
    so take the global's address and rebuild it as a *fly* pointer.  A union means
    every field aliases the same bytes, so each field is that one base recast to
    its own element type.
    """
    import types as _types

    from flydsl._mlir.dialects import arith as _a, fly, memref as _memref
    from flydsl.expr.meta import _to_raw_value

    raw = _to_raw_value(allocator.get_base())
    addr = _a.IndexCastOp(
        T.i64, _memref.extract_aligned_pointer_as_index(raw)
    ).result
    if byte_offset:
        addr = _a.AddIOp(addr, as_i64(byte_offset)).result

    def _field(dtype):
        ptr_ty = fly.PointerType.get(
            dtype.ir_type,
            address_space=int(fly.AddressSpace.Shared),
            alignment=16,
        )
        return _SharedField(fx.inttoptr(ptr_ty, addr))

    return _types.SimpleNamespace(
        **{name: _field(dtype) for name, (dtype, _n) in fields.items()}
    )


# ---------------------------------------------------------------------------
# fragments and copy-atom index math
# ---------------------------------------------------------------------------


def retype_fragment(frag, dtype):
    """Re-allocate an MMA fragment with ``dtype`` as its element type, same layout.

    0.1.x types fragments from the MMA atom (f32 for C, the operand dtype for A/B)
    rather than from the source tensor, so the tensor being copied in can have a
    different -- though always same-width -- element type.  Two cases here: the
    i32 sorted-id coord fragment, and fp8 operand fragments once the data path
    moves bytes as u8 (0.1.x cannot represent f8E4M3FNUZ in LLVM IR).

    ``make_fragment_like`` rather than a recast view on purpose: a view would keep
    the original fragment's alloca alive, and an ``alloca ... x f8E4M3FNUZ`` is
    exactly what breaks LLVM translation.  Re-allocating leaves the original dead,
    and callers always fill the fragment with a copy right after, so starting from
    a fresh buffer is correct.
    """
    if IS_02X:
        return frag
    src = getattr(frag, "dtype", None)
    if src is None or src is dtype or src.width != dtype.width:
        return frag
    out = fx.make_fragment_like(frag, dtype)
    _dbg(
        "retype",
        f"{src.__name__} -> {dtype.__name__} | before={fx.get_layout(frag)} "
        f"after={fx.get_layout(out)}",
    )
    return out


def _recast_view_32(t):
    """Reinterpret an 8-bit-lane tensor view as u32 lanes (same bytes).

    Carries the source alignment over rather than asserting 16B: pinning an
    alignment here makes flydsl's ``add_offset`` type inference disagree with the
    declared result type further down the copy machinery.
    """
    from flydsl._mlir.dialects import fly

    bits = t.dtype.width
    it = fx.get_iter(t)
    # exactly the element size: recast_iter rejects an alignment that is not a
    # multiple of it, and over-claiming (e.g. 16) makes add_offset's inferred type
    # disagree with the declared one further down the copy machinery.
    ptr_ty = fly.PointerType.get(fx.Uint32.ir_type, it.memspace, 4)
    return fx.make_view(
        fx.recast_iter(ptr_ty, it),
        fx.recast_layout(fx.get_layout(t), bits, 32),
    )


def _static_int(x):
    for attr in ("to_py_value", "get_static_leaf_int"):
        v = getattr(x, attr, None)
        if v is not None:
            return int(v() if callable(v) else v)
    return int(x)


def tiled_copy_operand_u32(mm, abc, atom_bits, ratio=4):
    """``make_tiled_copy_{A,B}`` for an 8-bit operand moved in u32 lanes.

    ``make_tiled_copy_A`` feeds the MMA's thread-value layout straight to
    ``make_tiled_copy``, and that layout counts *operand elements*, so pairing it
    with a u32 atom hands each thread 4x too many bytes.  The layout's codomain is
    the (MN, K) tile index space with MN stride 1, so converting it means dividing
    every pure-K thread stride by the lane ratio and replacing the value mode with
    the atom's u32 lane count (still contiguous in K).  Same transform the
    hand-written 0.1.x kernels in this tree apply to ``tv_layout_B_tiled``.
    """
    tv = mm.tv_layout_A_tiled if abc == "A" else mm.tv_layout_B_tiled
    mn = _static_int(fx.size(fx.select(mm.tile_size_mnk, [0 if abc == "A" else 1])))
    k = _static_int(fx.size(fx.select(mm.tile_size_mnk, [2])))
    thr_shape, thr_stride = [], []
    for i in range(8):
        try:
            sh, st = tv.shape[0][i], tv.stride[0][i]
        except (IndexError, RuntimeError, ValueError):
            break
        thr_shape.append(_static_int(fx.size(sh)))
        thr_stride.append(_static_int(st))
    assert thr_shape, f"cannot read the thread mode of {tv}"
    strides32 = []
    for s in thr_stride:
        if s and s % mn == 0:
            q = s // mn
            assert q % ratio == 0, f"K stride {s} is not a multiple of {ratio} tiles"
            strides32.append(mn * (q // ratio))
        else:
            strides32.append(s)
    return fx.make_tiled_copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy(atom_bits), fx.Uint32),
        fx.make_layout((tuple(thr_shape), atom_bits // 32), (tuple(strides32), mn)),
        fx.make_tile(fx.make_layout(mn, 1), fx.make_layout(k // ratio, 1)),
    )


def copy_maybe32(copy_atom, copy_atom_bits, src, dst, from_buffer=True):
    """``fx.copy`` that moves 8-bit-lane data in u32 lanes on 0.1.x.

    0.1.x's LLVM cannot legalise a 128-bit vector load/store whose lanes are 8 bit
    ("LLVM ERROR: Do not know how to split the result of this operator" --
    ``v16i8`` is not a legal AMDGPU register type; every hand-written 0.1.x kernel
    in this tree loads ``i32x4`` and bitcasts instead).

    Recasting both sides to u32 is byte-identical: ``recast_layout`` divides the
    extents and strides by 4 while the element type gets 4x wider.  This is the
    same mechanism the PR kernel itself uses for its fp8 -> u32 dequant path, so it
    is a supported operation rather than a trick.
    """
    swizzled = isinstance(
        fx.get_layout(src), fx.ComposedLayout
    ) or isinstance(fx.get_layout(dst), fx.ComposedLayout)
    if (
        IS_02X
        or not from_buffer  # a plain (universal) load of <16 x i8> splits fine
        or src.dtype.width != 8
        or copy_atom_bits < 32
        or swizzled  # recasting a swizzle would change the byte permutation
    ):
        fx.copy(copy_atom, src, dst)
        return
    atom_kind = fx.rocdl.BufferCopy if from_buffer else fx.UniversalCopy
    # Recasting *here* would be wrong for a buffer descriptor: `src` is already
    # partitioned, so its block/thread offset is baked into the descriptor in
    # 8-bit elements and recast_iter does not divide it by 4 -- a 4x address
    # error (the pre-existing 0.1.x kernels in this tree recast the weight at the
    # source, before make_buffer_tensor, for exactly this reason).  A 32-bit atom
    # over the original 8-bit lanes needs no recast at all: v4i8 is a legal i32
    # register, unlike the v16i8 a 128-bit atom would need.
    fx.copy(fx.make_copy_atom(atom_kind(32), src.dtype), src, dst)


def split_atom_index(mode_shape, linear):
    """Spread a flat atom index across ``mode_shape``, which may be nested.

    ``logical_divide`` leaves the atom mode nested (e.g. ``(12, 64)``) when the
    source strides are not contiguous.  0.2.x lets a scalar coord index such a mode
    and de-linearises it column-major; 0.1.x asserts on the rank mismatch.

    The column-major convention is verified rather than assumed: on 0.2.x,
    ``crd2idx(5, make_layout((12, 64), (100, 1)))`` is 500, i.e. exactly
    ``crd2idx((5 % 12, 5 // 12))``.
    """
    rank = getattr(mode_shape, "rank", 1)
    if rank <= 1:
        return linear
    extents = [int(fx.size(mode_shape[i]).get_static_leaf_int) for i in range(rank)]
    coord, rest = [], linear
    for extent in extents[:-1]:  # the last mode takes whatever remains
        coord.append(rest % extent)
        rest = rest // extent
    coord.append(rest)
    return tuple(coord)


# ---------------------------------------------------------------------------
# global patches (0.1.x only)
# ---------------------------------------------------------------------------


def _install():
    from flydsl._mlir.dialects import fly as _fly
    from flydsl.expr import vector as _vec
    from flydsl.expr.numeric import Float8E4M3FNUZ as _F8FNUZ
    from flydsl.expr.utils import arith as _arith_utils

    # -- int-tuple leaf normalisation -------------------------------------
    # Every layout primitive funnels into the native ``infer_int_tuple_type``,
    # which only accepts normalised leaves and *segfaults* on index-typed values
    # such as thread_idx.x.  0.2.x fixed this by normalising in python first
    # (``_expand_int_tuple_leaves``); wrapping the native entry point once covers
    # every primitive.  Note 0.1.x's own helpers coerce dynamic leaves to i32
    # (see its idx2crd, which calls _to_i32) -- widening to i64 like 0.2.x makes
    # later passes emit an invalid `extsi i64 -> i32`.
    _orig_infer = _fly.infer_int_tuple_type

    def _expand(value):
        if isinstance(value, ir.Value) and isinstance(value.type, _IntTupleType):
            return _expand(value.to_py_value())
        if isinstance(value, _Num):
            if isinstance(value.value, ir.Value) and type(value).width < 32:
                return _I32(value).value
            return value.value
        if isinstance(value, ir.Value):
            if isinstance(value.type, ir.IntegerType) and value.type.width != 32:
                # 0.1.x's int-tuple leaves are i32: narrower ones must be widened,
                # and i64 ones (the kernels build byte offsets as Int64) must be
                # narrowed, otherwise a later pass emits `extsi i64 -> i32`.  The
                # element offsets these kernels form stay well inside i32 for any
                # supported shape.
                return _I32(value).value
            if isinstance(value.type, ir.IndexType):
                return _I32(value).value
        if isinstance(value, (list, tuple)):
            return tuple(_expand(v) for v in value)
        return value

    _fly.infer_int_tuple_type = lambda v: _orig_infer(_expand(v))

    # -- fp8-e4m3fnuz reverse mapping --------------------------------------
    # 0.1.x's Numeric.from_ir_type table is missing the entry 0.2.x added ("not in
    # upstream MLIR extras T"), so `.dtype` on any fnuz tensor raises.  Matched by
    # type *name* so it keeps working under the storage-type patch below.
    _orig_from_ir = _Num.from_ir_type
    _Num.from_ir_type = staticmethod(
        lambda t: _F8FNUZ if str(t) == "f8E4M3FNUZ" else _orig_from_ir(t)
    )

    # -- fp8 storage type ---------------------------------------------------
    # 0.1.x's lowering keeps `f8E4M3FNUZ` in the data-movement path (buffer loads,
    # LDS stores, GEPs, register allocas) all the way to LLVM translation, which
    # has no mapping for it: gpu_module_to_binary dies with "unknown LLVM dialect
    # type".  The five 0.1.x kernels in this tree never hit it because they move
    # fp8 as raw bytes.
    #
    # The MFMA operands are already i64-packed -- byte-identical between this
    # kernel and the working ones -- and the atom picks its intrinsic from the
    # dtype *class*, not its MLIR type.  So presenting fp8 as i8 only changes data
    # movement, which is byte-wise anyway.
    #
    # Off by default: the MMA atom validates its operand element type in MLIR
    # ("elemTyA must be f16, bf16, f32, f8E4M3FNUZ, f8E5M2FNUZ, got 'i8'"), so the
    # patch has to be scoped to the data path rather than applied globally.
    if os.environ.get("AITER_FLYDSL_012_FP8_AS_I8", "0") == "1":
        _F8FNUZ._ir_type = staticmethod(lambda: T.i8)
        _dbg("fp8", "Float8E4M3FNUZ presented as i8 for data movement")

    # -- pointer arithmetic -------------------------------------------------
    # 0.1.x has the add_offset primitive but no operator on Pointer; 0.2.x's
    # Pointer.__add__ is exactly ``add_offset(self, offset)``.
    if not hasattr(_FlyPointer, "__add__"):

        def _ptr_add(self, offset):
            # 0.1.x only int-tuple-wraps offsets that are not ir.Values, so a
            # traced i64 reaches AddOffsetOp raw and is rejected.
            if not (
                isinstance(offset, ir.Value)
                and isinstance(offset.type, _IntTupleType)
            ):
                offset = fx.make_int_tuple(offset)
            # A GEP inherits the pointer's element type, and a signed one reaches
            # LLVM translation as `..., si32` and crashes it.
            ty = str(getattr(self, "type", ""))
            if "<si" in ty or "<ui" in ty:
                self = recast_iter(self.dtype, self)
            return fx.add_offset(self, offset)

        _FlyPointer.__add__ = _ptr_add
        _FlyPointer.__radd__ = _ptr_add

    # -- argument coercion on layout primitives -----------------------------
    # 0.2.x wraps plain python tuples into tile / int-tuple values before calling
    # the fly op; 0.1.x forwards them and MLIR rejects them ("Operand N of
    # operation fly.X must be a Value").
    specs = (
        # (name, arg index, arg name, conversion)
        ("flat_divide", 1, "divisor", "tile"),
        ("logical_divide", 1, "divisor", "tile"),
        ("zipped_divide", 1, "divisor", "tile"),
        ("tiled_divide", 1, "divisor", "tile"),
        ("make_tiled_copy", 2, "tile_mn", "tile"),
        ("make_tiled_mma", 2, "permutation", "tile"),
        ("composition", 1, "tiler", "int_tuple"),
        ("logical_product", 1, "tiler", "int_tuple"),
        ("zipped_product", 1, "tiler", "int_tuple"),
        ("slice", 1, "coord", "int_tuple"),
        ("dice", 1, "coord", "int_tuple"),
        ("tiled_copy_partition_src", 2, "thr_int_tuple", "int_tuple"),
        ("tiled_copy_partition_dst", 2, "thr_int_tuple", "int_tuple"),
        ("make_fragment_like", 1, "dtype", "dtype"),
        # a coord tensor's base can be a plain python int (0.2.x coerces `iter`
        # permissively); 0.1.x hands it straight to MLIR
        ("make_view", 0, "iter", "int_tuple"),
    )

    def _convert(val, how):
        if how == "dtype":
            return val.ir_type if hasattr(val, "ir_type") else val
        if isinstance(val, ir.Value):
            return val
        return fx.make_tile(*val) if how == "tile" else fx.make_int_tuple(val)

    for name, pos, argname, how in specs:
        orig = getattr(_prim, name, None)
        if orig is None:
            continue

        def wrapper(*args, _orig=orig, _pos=pos, _name=argname, _how=how, **kwargs):
            if len(args) > _pos and args[_pos] is not None:
                args = list(args)
                args[_pos] = _convert(args[_pos], _how)
                args = tuple(args)
            elif kwargs.get(_name) is not None:
                kwargs[_name] = _convert(kwargs[_name], _how)
            return _orig(*args, **kwargs)

        setattr(_prim, name, wrapper)
        if getattr(fx, name, None) is not None:
            setattr(fx, name, wrapper)

    # recast_iter needs the source iterator to build the pointer type, so it gets
    # a dedicated wrapper rather than a table entry.
    _orig_recast = _prim.recast_iter

    def _recast_wrapper(result_type, src, **kwargs):
        if isinstance(result_type, type) and hasattr(result_type, "ir_type"):
            result_type = fx.PointerType.get(
                result_type.ir_type, src.memspace, src.alignment
            )
        return _orig_recast(result_type, src, **kwargs)

    _prim.recast_iter = _recast_wrapper
    fx.recast_iter = _recast_wrapper

    # -- vector/scalar broadcast -------------------------------------------
    # 0.2.x's Vector wrapper auto-broadcasts a scalar operand against a vector
    # one; without it `frag.load() * scale` emits arith.mulf(vector<16xf32>, f32).
    # ArithValue's binary ops route every operand through `_coerce_other`.
    _orig_coerce = _arith_utils._coerce_other

    def _coerce_broadcast(self, other, **kwargs):
        res = _orig_coerce(self, other, **kwargs)
        if res is NotImplemented:
            return res
        st, rt = str(getattr(self, "type", "")), str(getattr(res, "type", ""))
        if st.startswith("vector<") and not rt.startswith("vector<"):
            return _vec.broadcast(self.type, res)
        return res

    _arith_utils._coerce_other = _coerce_broadcast

    # -- typed-value helpers 0.2.x added -----------------------------------
    def _lanes_type(self, dtype):
        """Element type for scalars, same-length vector type for vector values."""
        ty = str(getattr(self, "type", ""))
        if ty.startswith("vector<"):
            lanes = int(ty[len("vector<") : ty.index("x")])
            return ir.VectorType.get([lanes], dtype.ir_type)
        return dtype.ir_type

    def _elem_width(ty):
        inner = ty[ty.index("x") + 1 : -1] if ty.startswith("vector<") else ty
        return int(inner[1:]) if inner[1:].isdigit() else None

    def _convert_lanes(self, dtype):
        """Elementwise convert a vector value to ``dtype``'s lane type.

        Only the narrowing / widening cases these kernels use are handled;
        anything else raises rather than silently emitting a wrong conversion.
        """
        from flydsl._mlir.dialects import arith as _a

        ty = str(self.type)
        target = _lanes_type(self, dtype)
        sw, dw = _elem_width(ty), _elem_width(str(target))
        src_float = "f" in ty[ty.index("x") + 1 :]
        dst_float = "f" in str(target).split("x")[-1]
        if src_float != dst_float:
            raise TypeError(
                f"vector .to(): int<->float {ty} -> {target} is not backported"
            )
        if sw is None or dw is None or sw == dw:
            return _ArithValue(self)
        if src_float:
            op = _a.TruncFOp if dw < sw else _a.ExtFOp
        else:
            op = _a.TruncIOp if dw < sw else _a.ExtUIOp
        return _ArithValue(op(target, self).result)

    # -- Tensor.fill --------------------------------------------------------
    # 0.1.x's Tensor.fill is a no-op stub (`def fill(self, value): pass`) while
    # 0.2.x stores a splat vector.  The MoE kernels rely on it to zero the MFMA
    # accumulator (`fragC.fill(0)`), so on 0.1.x the accumulator starts as
    # uninitialized registers -- results come out as ~1e38 / NaN.  0.2.x has no
    # `full()` here to reuse, so splat the value directly.
    from flydsl.expr.typing import Tensor as _Tensor

    def _fill(self, value, **_kwargs):
        n = int(fx.size(self.shape).get_static_leaf_int)
        _dbg("fill", f"shape={self.shape} n={n} layout={fx.get_layout(self)}")
        scalar = self.dtype(value)
        raw = (
            scalar.ir_value()
            if callable(getattr(scalar, "ir_value", None))
            else scalar
        )
        return self.store(
            _vec.broadcast(ir.VectorType.get([n], self.dtype.ir_type), raw)
        )

    _Tensor.fill = _fill

    for cls in (_ArithValue, _Num):
        # ``Numeric.to(dtype)`` is ``dtype(self)`` for a Numeric target on 0.2.x;
        # 0.1.x has no ``.to`` on ArithValue, which is what its arithmetic returns.
        if not hasattr(cls, "to"):

            def _to(self, dtype):
                if dtype is type(self):
                    return self
                if isinstance(dtype, type) and issubclass(dtype, _Num):
                    if str(getattr(self, "type", "")).startswith("vector<"):
                        return _convert_lanes(self, dtype)
                    return dtype(self)
                raise TypeError(f"unsupported .to({dtype!r}) on flydsl 0.1.x")

            cls.to = _to

        orig_bitcast = getattr(cls, "bitcast", None)
        if orig_bitcast is None:
            # Numeric has no bitcast at all on 0.1.x; add 0.2.x's version.
            def _bitcast(self, dtype):
                from flydsl.expr import arith as _a

                raw = (
                    self.ir_value()
                    if callable(getattr(self, "ir_value", None))
                    else getattr(self, "value", self)
                )
                return dtype(_a.bitcast(_lanes_type(self, dtype), raw))

        else:
            # ArithValue.bitcast wants a raw ir.Type; accept a Numeric subclass and
            # keep the lane count for vector values.
            def _bitcast(self, dtype, _orig=orig_bitcast, **kwargs):
                if isinstance(dtype, type) and issubclass(dtype, _Num):
                    dtype = _lanes_type(self, dtype)
                return _orig(self, dtype, **kwargs)

        cls.bitcast = _bitcast

    _dbg("install", "flydsl 0.1.x compatibility patches installed")


if not IS_02X:
    _install()
