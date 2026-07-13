"""MoE GEMM stage1/stage2 kernel implementations (FlyDSL MFMA FP8).

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import logging
import os
import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith
from flydsl.expr import gpu, buffer_ops, vector, rocdl
from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

try:
    from flydsl.runtime.device import (
        supports_bf16_global_atomics,
        bf16_global_atomics_arch_description,
    )
except ImportError:
    # Backward compatibility for runtime.device versions that only expose get_rocm_arch.
    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))

    def bf16_global_atomics_arch_description() -> str:
        return "gfx94+/gfx95+/gfx12+"


from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl._mlir.dialects import gpu as _gpu_mlir
from flydsl.expr.typing import T


from .mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_store_4b_xor16,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    load_b_raw_w4a16,
    unpack_b_w4a16,
    tile_chunk_coord_i32,
    swizzle_xor16,
    crd2idx,
)
from .mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog


# ── XCD-locality grid remap ────────────────────────────────────────────────
# The default 2D launch is grid=(gx=n_tiles, gy=expert_blocks) walked X-fast.
# Under multi-XCD chunked dispatch with TG_CHUNK_SIZE==1, consecutive workgroups
# round-robin across the XCDs. With the default layout a single expert-block (one
# gy row) spans the whole X axis, so that expert's n_tiles scatter across all
# XCDs and every XCD pulls the same expert weights into its own L2 (zero reuse).
#
# Remap to a 3D grid (NUM_XCD, gx, ceil(gy/NUM_XCD)):
#   block_id.x -> xcd   (0..NUM_XCD-1, one per XCD via chunk=1)
#   block_id.y -> n_tile  (walks the full original gx)
#   block_id.z -> expert-block group
#   expert_block = z*NUM_XCD + xcd
# X-fast walk then sends (x=0..7,y=0) to XCD0..7, (x=*,y=1) again to XCD0..7, ...
# so each XCD completes ALL n_tiles of one expert before advancing to the next
# expert (z+1) -> that expert's weights stay resident in the XCD's L2.
# gy need not be a multiple of NUM_XCD: trailing blocks (expert_block >= gy) are
# dropped by the existing blk_valid guard (bx_m >= max_token/num_valid).
MOE_XCD_REMAP = os.environ.get("AITER_MOE_XCD_REMAP", "1") not in ("0", "false", "False")
# XCD remap axis selector (only meaningful when MOE_XCD_REMAP is on).
#   default (gy-first): split the sorted-M / expert-block axis (gy) across XCDs ->
#     grid=(NUM_XCD, gx, ceil(gy/NUM_XCD)). Each XCD owns a token-block slice and
#     sweeps all n_tiles per block => input-activation (token) stays L2-resident.
#   AITER_MOE_XCD_GX=1 (gx-first): split the N / n_tile axis (gx) across XCDs ->
#     grid=(NUM_XCD, gy, ceil(gx/NUM_XCD)). Each XCD owns an n_tile slice and sweeps
#     all M blocks per n_tile => expert-weight (B) stays L2-resident.
# In gx-first the rounding overruns the N (by) axis instead of the M (bx) axis, so
# the out-of-range guard is on by < n_tiles (folded into blk_valid) and bx is always
# in range.
MOE_XCD_REMAP_GX = os.environ.get("AITER_MOE_XCD_GX", "0") not in ("0", "false", "False")
MOE_NUM_XCD = int(os.environ.get("AITER_MOE_NUM_XCD", "8"))
# Split-K dispatch axis (experiment): where the k_batch factor is folded into the
# launch grid (only the gy-first remap branch honors this; other modes use "z").
#   "z" (default): grid=(NUM_XCD, gx, ceil(gy/NUM_XCD)*k_batch); blockIdx.z encodes
#       group*k_batch+kz. The kz partials of one output tile dispatch ~8*gx apart
#       (separate rounds) -> the shared atomic output tile is evicted from L2
#       between partial writes.
#   "y": grid=(NUM_XCD, gx*k_batch, ceil(gy/NUM_XCD)); blockIdx.y encodes
#       n_tile*k_batch+kz so a tile's kz partials are adjacent in dispatch order
#       -> the atomic output tile stays L2-resident across its accumulation.
MOE_SPLITK_AXIS = os.environ.get("AITER_MOE_SPLITK_AXIS", "z")
# Debug probe: when set, each workgroup's leader thread prints the
# (block.x, block.y, block.z) -> (expert_block bx, n_tile by, hw XCD id) mapping
# via gpu.printf so the real-kernel (x,y,z)<->XCD correspondence can be inspected.
MOE_XCD_DEBUG = os.environ.get("AITER_MOE_XCD_DEBUG", "0") not in ("0", "false", "False")
# Stage2 depth-D B(weight) prefetch pool (env AITER_MOE_S2_BPOOL_DEPTH=N): keep a
# ring of N in-flight weight tiles. Prologue front-loads N load_b results; each
# consume pops the pool head and issues the tail's next load_b (N tiles ahead), so
# N weight loads stay in flight (emitted early in program order -> in flight across
# the VMEM-non-draining LDS barriers -> higher MLP). Usually passed per-compile via
# the b_pool_depth kwarg (tuner); the env is a manual override. Default off.
try:
    MOE_S2_BPOOL_DEPTH = int(os.environ.get("AITER_MOE_S2_BPOOL_DEPTH", "0") or "0")
except ValueError:
    MOE_S2_BPOOL_DEPTH = 0
# Stage1 depth-D B(weight) prefetch pool (env AITER_MOE_S1_BPOOL_DEPTH=N): same
# ring-pool as stage2, but on the gate(+up) weight tiles. Front-loads N (gate,up)
# tile-pairs so ~N weight loads stay in flight, matching the hand-tuned ASM's deep
# prefetch prologue. Usually passed per-compile via the b_pool_depth kwarg (tuner).
try:
    MOE_S1_BPOOL_DEPTH = int(os.environ.get("AITER_MOE_S1_BPOOL_DEPTH", "0") or "0")
except ValueError:
    MOE_S1_BPOOL_DEPTH = 0
# Stage1 depth-D X(activation) HBM prefetch pool (env AITER_MOE_S1_XPOOL_DEPTH=N):
# same ring as stage2 -- front-load N X-tile HBM->reg loads; ds_write to LDS stays
# 1-ahead (2-buffer ping-pong unchanged). Usually passed via the x_pool_depth kwarg.
try:
    MOE_S1_XPOOL_DEPTH = int(os.environ.get("AITER_MOE_S1_XPOOL_DEPTH", "0") or "0")
except ValueError:
    MOE_S1_XPOOL_DEPTH = 0
# Stage2 depth-D X(activation) HBM prefetch pool (env AITER_MOE_S2_XPOOL_DEPTH=N):
# front-load N X-tile HBM->reg loads into a ring; ds_write to LDS stays 1-ahead
# (2-buffer ping-pong unchanged). Only the HBM load is deepened. X is the larger
# load cost at long token (weights are L2-cached there), so this targets long token.
try:
    MOE_S2_XPOOL_DEPTH = int(os.environ.get("AITER_MOE_S2_XPOOL_DEPTH", "0") or "0")
except ValueError:
    MOE_S2_XPOOL_DEPTH = 0
# Stage2 output bypass-L2 experiment. Stage2 has two output paths -- in-kernel
# atomic accumulation (accumulate=True) and per-(token,slot) store + a separate
# reduction kernel (accumulate=False). The final MoE output has no intra-XCD
# reuse under XCD remap (a token's topk partials land on different XCDs), so
# caching it mostly pollutes L2. When AITER_MOE_OUT_NT is set, mark both output
# paths non-temporal so they stream past L2 instead of allocating lines.
#   _OUT_ATOMIC_AUX : raw buffer-atomic cachepolicy bits. Use SLC=2 only (stream /
#     no-L2-allocate). Do NOT set GLC for the no-return fadd -- GLC flips it to the
#     return-value variant and faults (hipErrorIllegalAddress).
#   _OUT_STORE_CM   : buffer_ops cache_modifier (0 = normal, 2 = non-temporal)
MOE_OUT_NT = os.environ.get("AITER_MOE_OUT_NT", "0") not in ("0", "false", "False")
_OUT_ATOMIC_AUX = 2 if MOE_OUT_NT else 0
_OUT_STORE_CM = 2 if MOE_OUT_NT else 0
# Input-activation (token) bypass-L2 experiment. When AITER_MOE_X_NT is set, the
# X (stage1) / a2 (stage2) activation loads are marked non-temporal so they stream
# past L2. Diagnostic: if remap's L2 gain comes from activation reuse, bypassing X
# should collapse the L2 hit rate.
#   _X_CM     : buffer_ops cache_modifier (0 = normal, 2 = non-temporal)
#   _X_DMA_AUX: raw_ptr_buffer_load_lds aux (SLC=2) for the async-copy path
MOE_X_NT = os.environ.get("AITER_MOE_X_NT", "0") not in ("0", "false", "False")
_X_CM = 2 if MOE_X_NT else 0
_X_DMA_AUX = 2 if MOE_X_NT else 0
# Weight (B) cache-policy override. The per-kernel `b_nt` is parsed from the tuned
# kernel name (b_nt=2 => weights non-temporal/bypass-L2, the default for most
# configs; only `_bnt0` names use b_nt=0 => weights cached). Set AITER_MOE_B_NT to
# force a single value across ALL flydsl moe kernels (e.g. 0 = force weights into
# L2) for clean L2 diagnosis. Empty = keep each kernel's own b_nt.
_MOE_B_NT_OVERRIDE = os.environ.get("AITER_MOE_B_NT", "")
# Scale + index/metadata bypass-L2 experiment. The quant scales (scale_x per token,
# scale_w per expert) and the routing metadata (sorted_token_ids / expert_ids /
# sorted_weights / max_token_id) are small but very hot -- they stay in L2 and
# nearly always hit, so they dominate the residual L2 hit rate even when X/B/OUT
# are all bypassed. When AITER_MOE_SCALE_NT is set, mark these loads non-temporal
# too, to confirm they are the source of the leftover hits.
#   _SCALE_CM: buffer_ops cache_modifier (0 = normal, 2 = non-temporal)
MOE_SCALE_NT = os.environ.get("AITER_MOE_SCALE_NT", "0") not in ("0", "false", "False")
_SCALE_CM = 2 if MOE_SCALE_NT else 0


def _eff_b_nt(b_nt):
    """Effective weight cache_modifier: env override wins when set."""
    return int(_MOE_B_NT_OVERRIDE) if _MOE_B_NT_OVERRIDE != "" else b_nt


# Canonical default knobs (the env-derived module globals above are the defaults).
# The dispatch/remap/cache knobs used to be process-wide env globals; they are now
# per-compile tuning parameters. compile_moe_gemm{1,2} resolve them into LOCALS that
# shadow the module globals of the same name, so every reference inside the kernel
# builders (closures) automatically picks up the per-call value.
def _resolve_moe_knobs(remap, splitk_axis, x_nt, scale_nt, out_nt):
    """Resolve per-compile dispatch/cache knobs.

    Each arg is None => fall back to the env-derived module default.
      remap       : "gy" | "gx" | "off"  (XCD remap axis; gy is the canonical default)
      splitk_axis : "z"  | "y"           (split-K grid fold axis)
      x_nt        : 0 | 2                 (input-activation load cache_modifier)
      scale_nt    : 0 | 2                 (scale + routing-metadata cache_modifier)
      out_nt      : 0 | 2                 (output atomic/store cache_modifier)
    Returns: (xcd_remap, xcd_remap_gx, splitk_axis,
              x_cm, x_dma_aux, scale_cm, out_atomic_aux, out_store_cm)
    """
    if remap is None:
        rmp, rgx = MOE_XCD_REMAP, MOE_XCD_REMAP_GX
    elif remap == "gy":
        rmp, rgx = True, False
    elif remap == "gx":
        rmp, rgx = True, True
    elif remap == "off":
        rmp, rgx = False, False
    else:
        raise ValueError(f"remap must be 'gy'|'gx'|'off'|None, got {remap!r}")
    axis = MOE_SPLITK_AXIS if splitk_axis is None else splitk_axis
    if axis not in ("z", "y"):
        raise ValueError(f"splitk_axis must be 'z'|'y'|None, got {splitk_axis!r}")
    x_cm = _X_CM if x_nt is None else (2 if x_nt else 0)
    x_aux = _X_DMA_AUX if x_nt is None else (2 if x_nt else 0)
    sc_cm = _SCALE_CM if scale_nt is None else (2 if scale_nt else 0)
    o_aux = _OUT_ATOMIC_AUX if out_nt is None else (2 if out_nt else 0)
    o_cm = _OUT_STORE_CM if out_nt is None else (2 if out_nt else 0)
    return rmp, rgx, axis, x_cm, x_aux, sc_cm, o_aux, o_cm


def _moe_knob_tags(xcd_remap, xcd_remap_gx, x_cm, scale_cm, out_aux):
    """Kernel-name suffix for the resolved knobs. Canonical defaults (remap=gy and
    all caches normal) produce an EMPTY tag so existing tuned-kernel names and their
    cached artifacts are byte-identical."""
    if not xcd_remap:
        remap_tag = "_roff"
    elif xcd_remap_gx:
        remap_tag = "_rgx"
    else:
        remap_tag = ""  # gy (canonical default)
    return (
        remap_tag
        + ("_xnt" if x_cm else "")
        + ("_snt" if scale_cm else "")
        + ("_ont" if out_aux else "")
    )


def _barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm.

    Bypasses LLVM SIInsertWaitcnts which would insert a conservative
    s_waitcnt vmcnt(0) lgkmcnt(0) before every S_BARRIER MI.
    """
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    parts.append("s_barrier")
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


def _s_setprio(prio):
    return
    """Emit s_setprio via inline asm to control wave scheduling priority."""
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string=f"s_setprio {prio}",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


def _s_nop(count=1):
    """Emit s_nop via inline asm for transcendental instruction latency."""
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string=f"s_nop {count}",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


def _get_xcc_id():
    """Read the hardware XCC/XCD id via inline asm (HW_REG_XCC_ID==20, field [3:0]).

    Mirrors get_xcc_id() in xcd_chunk_probe3d.hip:
        s_getreg_b32 %0, hwreg(20, 0, 4)
    Returns an i32 holding the XCD index of the executing workgroup.
    """
    return llvm.InlineAsmOp(
        T.i32,
        [],
        "s_getreg_b32 $0, hwreg(20, 0, 4)",
        "=s",
        has_side_effects=True,
    ).result


def _xcd_debug_print(stage, bx, by):
    """Leader-thread gpu.printf of (block.x,y,z) -> (bx, by, hw XCD).

    No-op unless AITER_MOE_XCD_DEBUG is set. `bx` is the expert-block id and
    `by` the n-tile id as computed by the kernel (remap or default layout), so
    the printout shows the real launch->XCD correspondence on hardware.
    """
    if not MOE_XCD_DEBUG:
        return
    tx = gpu.thread_id("x")
    is_leader = arith.cmpi(arith.CmpIPredicate.eq, tx, fx.Index(0))
    _dbg_if = scf.IfOp(is_leader)
    with _if_then(_dbg_if):
        raw = arith._to_raw
        bidx = arith.index_cast(T.i32, gpu.block_id("x"))
        bidy = arith.index_cast(T.i32, gpu.block_id("y"))
        bidz = arith.index_cast(T.i32, gpu.block_id("z"))
        bx_i = arith.index_cast(T.i32, bx)
        by_i = arith.index_cast(T.i32, by)
        xcd = _get_xcc_id()
        # flydsl gpu.printf is variadic: printf(format, v0, v1, ...) -- spread args.
        _gpu_mlir.printf(
            f"XCDDBG s{stage} blk=(%d,%d,%d) bx=%d by=%d -> xcd=%d\n",
            raw(bidx), raw(bidy), raw(bidz), raw(bx_i), raw(by_i), raw(xcd),
        )


def _mfma_i32_16x16x64_i8(a_v4i32, b_v4i32, acc_v4i32):
    """v_mfma_i32_16x16x64_i8 via inline asm (gfx950 only).

    Each source (A, B) is 128 bits (4 VGPRs) and the accumulator is 4 x i32.
    Doubles K-per-MFMA to 64, halving instruction count vs 2x K=32 chain.
    """
    return llvm.InlineAsmOp(
        T.i32x4,
        [a_v4i32, b_v4i32, acc_v4i32],
        "v_mfma_i32_16x16x64_i8 $0, $1, $2, $3",
        "=v,v,v,0",
        has_side_effects=True,
    ).result


def _pack_i64x4_to_i32x8(x0, x1, x2, x3):
    """Pack 4 x i64 into vector<8 x i32> (256-bit) for K=128 MFMA operands."""
    v4 = vector.from_elements(T.vec(4, T.i64), [x0, x1, x2, x3])
    return vector.bitcast(T.vec(8, T.i32), v4)


@contextmanager
def _if_then(if_op):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    """Compat helper for SCF IfOp else-region across old/new Python APIs."""
    if getattr(if_op, "else_block", None) is None:
        raise RuntimeError("IfOp has no else block")
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def _persist_anti_licm_tx(tx, mi):
    """Defeat LICM register bloat inside the persist_m loop.

    Without this, every tid-derived LDS address is loop-invariant, so LLVM
    hoists the whole address book above the persist loop and pins ~19 VGPRs live
    across the loop body (occupancy loss, e.g. stage2 VGPR 124->162). We fold an
    induction-variable-dependent opaque zero into the thread id: ``opaque(mi)``
    is an inline-asm identity the optimizer cannot see through, so
    ``opaque(mi) - mi`` is provably 0 at runtime (results unchanged) yet appears
    loop-variant to the compiler. The address chain is then recomputed (and
    recycled) each iteration instead of being held live. Returns the perturbed
    ``tx``; call once right after entering the persist loop body.
    """
    mi_i32 = arith.index_cast(T.i32, mi)
    opaque_iv = llvm.InlineAsmOp(
        res=ir.IntegerType.get_signless(32),
        operands_=[mi_i32],
        asm_string="",
        constraints="=v,0",
        has_side_effects=False,
        is_align_stack=False,
    ).result
    return tx + arith.index_cast(T.index, arith.subi(opaque_iv, mi_i32))


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    # NOTE: aiter swap passes these for API symmetry; stage1 uses dynamic memrefs so they are ignored.
    doweight_stage1: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    act: str = "silu",
    use_g1u1: bool = True,
    use_cshuffle_epilog: bool | None = None,
    use_async_copy: bool = False,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    k_batch: int = 1,
    persist_m: int = 1,
    remap: str | None = None,
    splitk_axis: str | None = None,
    x_nt: int | None = None,
    scale_nt: int | None = None,
    out_nt: int | None = None,
    b_pool_depth: int = 0,
    x_pool_depth: int = 0,
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    in_dtype:
      - "fp8": X/W are fp8
      - "fp16": X/W are fp16
      - "bf16": X/W are bf16
      - "int8": X/W are int8 (X is [tokens, K])
      - "int8smooth": X/W are int8, but X is pre-expanded to [tokens*topk, K] with per-(token,slot)
        quant scales (used to emulate MoE smoothquant behavior where each (token,slot)->expert route can
        have a distinct input scaling before quantization).
      - "int4": W4A8 path: X is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
      - "int4_bf16": W4A16 path: X is bf16, W is packed int4 unpacked to bf16 in-kernel

    waves_per_eu:
      Controls LDS-based occupancy. When >= 1, the allocator is padded so that
      each workgroup claims at least ``160KB // (waves_per_eu + 1) + 1`` bytes
      of LDS, limiting the number of concurrent workgroups per CU.
      0 means no padding (default).

    b_nt:
      Non-temporal cache modifier for B (weight) buffer loads.
      0 = normal caching, 2 = non-temporal (GLC+SLC).

    remap / splitk_axis / x_nt / scale_nt / out_nt:
      Per-compile dispatch & cache-policy knobs (None => env default). They shadow
      the module globals of the same name so the kernel builder picks up per-call
      values; see _resolve_moe_knobs / _moe_knob_tags.
    """
    b_nt = _eff_b_nt(b_nt)
    # Resolve dispatch/cache knobs into LOCALS that shadow the module globals; all
    # references inside the nested moe_gemm1 builder resolve to these.
    (
        MOE_XCD_REMAP,
        MOE_XCD_REMAP_GX,
        MOE_SPLITK_AXIS,
        _X_CM,
        _X_DMA_AUX,
        _SCALE_CM,
        _OUT_ATOMIC_AUX,
        _OUT_STORE_CM,
    ) = _resolve_moe_knobs(remap, splitk_axis, x_nt, scale_nt, out_nt)
    # persist_m (workgroup merge along M) is the inverse of split-K; mutually exclusive.
    persist_m = int(persist_m)
    if persist_m > 1 and k_batch > 1:
        raise ValueError(
            f"persist_m={persist_m} and k_batch={k_batch} are mutually exclusive"
        )

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)

    if in_dtype not in (
        "fp8",
        "fp16",
        "bf16",
        "int8",
        "int8smooth",
        "int4",
        "int4_bf16",
    ):
        raise ValueError(
            f"in_dtype must be one of ('fp8','fp16','bf16','int8','int8smooth','int4','int4_bf16'), got {in_dtype!r}"
        )
    is_int4_bf16 = in_dtype == "int4_bf16"
    is_f16 = in_dtype == "fp16"
    is_bf16 = is_int4_bf16 or in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    needs_scale_w = (not is_f16_or_bf16) or is_int4_bf16
    elem_bytes = 2 if is_f16_or_bf16 else 1
    #if out_dtype not in ("f16", "bf16"):
    #    raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")

    # NOTE: don't materialize MLIR types outside an active MLIR Context.
    def out_mlir():
        return (lambda ty: ty() if callable(ty) else ty)(
            T.f16 if out_dtype == "f16" else T.bf16
        )

    tile_k_bytes = int(tile_k) * int(elem_bytes)
    # K64-byte micro-step: always 64 bytes per `ku`. For fp16 this is 32 elements.
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: X is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype == "int8") or is_int4
    x_is_token_slot = in_dtype == "int8smooth"
    # "int8smooth" still uses int8 MFMA, but X/scale_x are provided per (token,slot).
    is_int8 = is_int8 or x_is_token_slot

    # ── Split-K (partition the GEMM K=model_dim across `k_batch` workgroups) ──
    # When k_batch>1 each CTA computes a K-slice [kz*K_per_batch, (kz+1)*K_per_batch)
    # and the silu/mul activation is deferred to a post-kernel reduction step
    # (see moe_kernels.py: tmp_out -> silu_and_mul). The kernel therefore writes
    # *raw* gate/up partials into a 2*inter_dim buffer via atomic-add.
    # k_batch==1 keeps the original (activation-fused, direct-store) path untouched.
    _is_splitk1 = int(k_batch) > 1
    if _is_splitk1:
        if int(model_dim) % int(k_batch) != 0:
            raise ValueError(
                f"split-K: model_dim={model_dim} not divisible by k_batch={k_batch}"
            )
        _k_per_batch1 = int(model_dim) // int(k_batch)
        if _k_per_batch1 % int(tile_k) != 0:
            raise ValueError(
                f"split-K: K_per_batch={_k_per_batch1} not divisible by tile_k={tile_k}"
            )
        if (_k_per_batch1 // int(tile_k)) < 2:
            raise ValueError(
                "split-K: K_per_batch must be >= 2*tile_k (pipeline needs >=2 tail tiles)"
            )
        # The stage1 main loop is a ping-pong pipeline that processes tiles in
        # pairs and then a fixed 2-tile tail (pair_iters = (total_tiles-2)//2).
        # That requires an EVEN number of K-tiles per slice; an odd count drops
        # the unpaired middle tile (silently wrong output). Split-K shrinks the
        # per-slice K, so reject k_batch values that make it odd.
        if (_k_per_batch1 // int(tile_k)) % 2 != 0:
            raise ValueError(
                f"split-K: K_per_batch/tile_k={_k_per_batch1 // int(tile_k)} must be "
                f"even (model_dim={model_dim}, k_batch={k_batch}, tile_k={tile_k}); "
                "the stage1 ping-pong loop has a fixed 2-tile tail."
            )
    else:
        _k_per_batch1 = int(model_dim)
    # Encode the split-K dispatch axis in the kernel tag so the y-axis variant does
    # not collide with the z-axis artifact in the (name-keyed) flydsl/disk cache.
    _sk_axis_tag1 = (
        "y"
        if (_is_splitk1 and MOE_SPLITK_AXIS == "y" and MOE_XCD_REMAP and not MOE_XCD_REMAP_GX)
        else ""
    )
    _sk_tag1 = f"_sk{k_batch}{_sk_axis_tag1}" if _is_splitk1 else ""

    _is_gfx950 = str(gpu_arch).startswith("gfx950")
    _use_k64_mfma = _is_gfx950 and is_int8
    _use_k128_mfma_fp8 = (
        _is_gfx950 and not is_int8 and not is_f16_or_bf16
        and (tile_k_bytes % 128) == 0
    )

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    def _out_elem_type():
        return T.bf16 if out_dtype == "bf16" else T.f16

    def _out_vec_type():
        return T.vec(1, T.bf16) if out_dtype == "bf16" else T.vec(1, T.f16)

    mfma_f32_bf16_k16 = None
    if is_bf16:
        mfma_f32_bf16_k16 = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
            rocdl, "mfma_f32_16x16x16_bf16_1k", None
        )
        if mfma_f32_bf16_k16 is None:
            raise AttributeError(
                "BF16 K16 MFMA op not found: expected `rocdl.mfma_f32_16x16x16bf16_1k` "
                "(or `rocdl.mfma_f32_16x16x16_bf16_1k`)."
            )

    num_waves = tile_n // 32
    total_threads = num_waves * 64
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    # Keep MoE stage1 X gmem->LDS pipeline consistent with the optimized GEMM kernel:
    # split into <=16B pieces and use direct buffer_load for smaller widths.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    # LDS128 mode (same idea as test_preshuffle_gemm.py):
    # - LDS stride == tile_k (no extra padding) + XOR16 swizzle
    # - Use ds_{read,write}_b128 (16B) and extract 8B halves for MFMA steps
    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLYDSL_MOE_STAGE1_CSHUFFLE", "1") in (
            "1",
            "true",
            "True",
            "YES",
            "yes",
        )
    use_cshuffle_epilog = bool(use_cshuffle_epilog)
    #if out_dtype != "f16" and use_cshuffle_epilog:
    #    raise ValueError(
    #        "stage1 cshuffle epilog currently supports only f16 output (out_dtype='f16')"
    #    )

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Keep an explicit ABI tag so signature changes can't accidentally reuse an old binary.
    g1u_tag = "g1u0" if not use_g1u1 else "g1u1"
    _async_tag = "_async" if use_async_copy else ""
    _wpe_tag = f"_wpe{waves_per_eu}" if waves_per_eu >= 1 else ""
    _bnt_tag = f"_bnt{b_nt}" if b_nt != 2 else ""

    _w_rows_per_expert_static = int(inter_dim) if not use_g1u1 else int(2 * inter_dim)
    _w_storage_elem_bytes = 2 if is_f16_or_bf16 else 1
    _w_physical_k_bytes_static = int(model_dim) * _w_storage_elem_bytes
    _w_nbytes_static = (
        int(experts) * _w_rows_per_expert_static * _w_physical_k_bytes_static
    )
    _use_wptr64 = _w_nbytes_static >= (1 << 31)
    _wptr64_tag = "_wptr64" if _use_wptr64 else ""

    _knob_tag = _moe_knob_tags(
        MOE_XCD_REMAP, MOE_XCD_REMAP_GX, _X_CM, _SCALE_CM, _OUT_ATOMIC_AUX
    )
    _pm_tag = f"_pm{persist_m}" if persist_m != 1 else ""
    module_name = (
        f"mfma_moe1_{g1u_tag}_{in_dtype}_{out_dtype}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}{_async_tag}{_wpe_tag}{_bnt_tag}{_wptr64_tag}{_sk_tag1}{_knob_tag}{_pm_tag}"
        f"_abi6_wptr64gate"  # ABI bumped: optional 64-bit W load path gated by static size check
    ).replace("-", "_")

    # ── LDS sizing (pure Python; no MLIR Context needed) ─────────────────────
    # Reuse the same LDS bytes for both:
    # - ping-pong X tiles (2 * tile_m * lds_stride bytes)
    # - optional epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
    # lds_tid (tile_m i32 values) lives after the main buffer and is used
    # concurrently with the epilogue cshuffle region.
    _use_cshuffle_epilog = bool(use_cshuffle_epilog)
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_tid_bytes = int(tile_m) * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    _lds_tid_byte_off = max(lds_x_bytes, lds_out_bytes)

    if waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _cur_lds = allocator._align(allocator.ptr, 128)
        if _cur_lds < _min_lds:
            allocator.ptr += _min_lds - _cur_lds

    if True:

        @flyc.kernel(known_block_size=[total_threads, 1, 1])
        def moe_gemm1(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_max_token_ids: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            # Unwrap tensor handles to memrefs for ext dialect helpers (e.g. fly.extract_aligned_pointer_as_index).
            #arg_out = arg_out.value
            #arg_x = arg_x.value
            #arg_w = arg_w.value
            #arg_scale_x = arg_scale_x.value
            #arg_scale_w = arg_scale_w.value
            #arg_sorted_token_ids = arg_sorted_token_ids.value
            #arg_expert_ids = arg_expert_ids.value
            #arg_sorted_weights = arg_sorted_weights.value
            #arg_max_token_ids = arg_max_token_ids.value

            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            inter_in = arith.index_cast(T.index, i32_inter_in)
            k_in = arith.index_cast(T.index, i32_k_in)
            size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
            x_elem = (
                T.bf16
                if is_bf16
                else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
            )
            # For int4/int4_bf16, weights are stored as packed bytes (i8) and unpacked in-kernel.
            w_elem = (
                T.i8
                if (is_int4 or is_int4_bf16)
                else (
                    T.bf16
                    if is_bf16
                    else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
                )
            )
            vec1_bf16 = T.vec(1, T.bf16)
            vec4_bf16 = T.vec(4, T.bf16)
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)

            def silu(x):
                # device fast path:
                #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
                #   sig = rcp(1 + emu)                   -> v_rcp_f32
                #   y = x * sig
                #
                # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
                # sequences that introduce extra compares/cndmasks.
                t = x * (-1.4426950408889634)  # -log2(e)
                emu = rocdl.exp2(T.f32, t)
                _s_nop(1)
                den = 1.0 + emu
                sig = rocdl.rcp(T.f32, den)
                return x * sig

            def gelu(x):
                # e^(x*(c1*x*x+c2))
                #x3 = x2 * x
                #t = x3 * (-0.102942064487430885) - x * (2.302209467435655)
                x2 = x * x
                t = x * (x2 * (-0.102942064487430885) - (2.302209467435655))
                #t = (
                #    (x * x * (-0.07135400176048279) - 1.595770001411438)
                #    * x
                #    * (1.4426950408889634)
                #)
                emu = rocdl.exp2(T.f32, t)
                _s_nop(1)
                den = 1.0 + emu
                sig = rocdl.rcp(T.f32, den)
                return x * sig
                '''# Tanh-form GELU matching reference assembly:
                #   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
                # Uses sign-extraction so exp2 argument is always <= 0.
                _c1 = 0.044714998453855515    # 0x3d372713
                _c2 = 0.7978845238685608      # 0x3f4c4229  sqrt(2/pi)
                _c3 = -2.885390043258667      # 0xc038aa3b  -2*log2(e)

                x2 = x * x
                z = _c2 * (x * (_c1 * x2) + x)
                z_i32 = arith.bitcast(T.i32, z)
                sign_i32 = arith.andi(z_i32, arith.constant(-2147483648, type=T.i32))
                one_i32 = arith.constant(0x3F800000, type=T.i32)
                sign_one = arith.bitcast(T.f32, arith.ori(one_i32, sign_i32))
                t = (_c3 * sign_one) * z
                e = rocdl.exp2(T.f32, t)
                _s_nop(1)
                den = 1.0 + e
                r = rocdl.rcp(T.f32, den)
                se = sign_one * e
                tanh_z = (sign_one - se) * r
                half_x = 0.5 * x
                return half_x * (1.0 + tanh_z)'''

            acc_init = (
                arith.constant_vector(0, T.i32x4)
                if is_int8
                else arith.constant_vector(0.0, T.f32x4)
            )

            # B preshuffle layout: match GEMM test helper exactly.
            _w_rows_per_expert = inter_dim if not use_g1u1 else (2 * inter_dim)
            c_n_total = arith.index(experts * _w_rows_per_expert)
            # For packed int4 (W4A8/W4A16), kpack_bytes=8.
            kpack_bytes = 8 if (is_int4 or is_int4_bf16) else 16
            w_elem_bytes = 1 if (is_int4 or is_int4_bf16) else elem_bytes
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in,
                kpack_bytes=kpack_bytes,
                elem_bytes=w_elem_bytes,
            )
            layout_b = b_layout.layout_b

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Split-K dispatch axis (see MOE_SPLITK_AXIS). "y" only applies to the
            # gy-first remap branch; kz/k_start are then derived from blockIdx.y
            # inside that branch. Otherwise blockIdx.z encodes group*k_batch+kz.
            _sk_axis_y1 = (
                _is_splitk1
                and MOE_SPLITK_AXIS == "y"
                and MOE_XCD_REMAP
                and not MOE_XCD_REMAP_GX
            )
            if _is_splitk1 and not _sk_axis_y1:
                _bidz_sk = gpu.block_id("z")
                kz_sk = _bidz_sk % fx.Index(k_batch)
                _sk_group_z = _bidz_sk // fx.Index(k_batch)
                k_start = kz_sk * fx.Index(_k_per_batch1)
            else:
                _sk_group_z = None
                k_start = None
            # persist_m loop: each WG serially sweeps persist_m M-blocks (inverse of
            # split-K). persist follows M (see launch grid): remap=gy folds into the
            # group (z) index, remap=gx / no-remap into the M index (block_id.y).
            _persist1 = persist_m > 1
            if _persist1:
                _c_pm1 = arith.constant(persist_m, index=True)
                _for_persist1 = scf.ForOp(
                    arith.constant(0, index=True),
                    _c_pm1,
                    arith.constant(1, index=True),
                )
                _for_ip1 = ir.InsertionPoint(_for_persist1.body)
                _for_ip1.__enter__()
                _mi1 = _for_persist1.induction_variable
                # Anti-LICM: keep tid-derived LDS addresses loop-variant so they
                # are recomputed (not hoisted + pinned) each persist iteration.
                tx = _persist_anti_licm_tx(tx, _mi1)

            def _pm_fold1(v):
                return (v * _c_pm1 + _mi1) if _persist1 else v

            # Align with Aiter launch mapping (NSwizzle==false):
            # - blockIdx.x -> N dimension (tile along inter_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            if MOE_XCD_REMAP and MOE_XCD_REMAP_GX:
                # gx-first: grid=(NUM_XCD, gy, ceil(gx/NUM_XCD)). N (n_tile) is split
                # across XCDs; bx (sorted M) walks the full gy on block_id.y so it is
                # always in range. The rounding now overruns by (n_tile), so guard
                # by < n_tiles and fold it into blk_valid.
                _xcd = gpu.block_id("x")  # 0..NUM_XCD-1 (one per XCD, chunk=1)
                bx = _pm_fold1(gpu.block_id("y"))  # tile along sorted M (full range)
                _bz = _sk_group_z if _is_splitk1 else gpu.block_id("z")  # n_tile group
                by = _bz * fx.Index(MOE_NUM_XCD) + _xcd  # tile along inter_dim (n-tile)
                bx_m = bx * fx.Index(tile_m)
                bx_m_i32 = arith.index_cast(T.i32, bx_m)
                # n_tiles must match the launcher's gx = inter_in // tile_n.
                n_tiles_i32 = arith.index_cast(T.i32, inter_in // fx.Index(tile_n))
                by_in_range = arith.cmpi(
                    arith.CmpIPredicate.ult, arith.index_cast(T.i32, by), n_tiles_i32
                )
                # Only in-range (by) blocks load max_token_id and run the token check;
                # padding blocks (by >= n_tiles) skip the load and yield invalid directly.
                _rng_if = scf.IfOp(by_in_range, [ir.IntegerType.get_signless(1)], has_else=True)
                with _if_then(_rng_if):
                    maxids_rsrc = buffer_ops.create_buffer_resource(
                        arg_max_token_ids,
                        max_size=False,
                        num_records_bytes=fx.Index(4),
                    )
                    max_token_id_i32 = buffer_ops.buffer_load(
                        maxids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
                    )
                    tok_ok = arith.cmpi(
                        arith.CmpIPredicate.ult, bx_m_i32, max_token_id_i32
                    )
                    scf.YieldOp([tok_ok])
                with _if_else(_rng_if):
                    scf.YieldOp([arith.constant(0, type=ir.IntegerType.get_signless(1))])
                blk_valid = _rng_if.results[0]
            elif MOE_XCD_REMAP:
                # grid=(NUM_XCD, gx, ceil(gy/NUM_XCD)); see MOE_XCD_REMAP note.
                _xcd = gpu.block_id("x")  # 0..NUM_XCD-1 (one per XCD, chunk=1)
                if _sk_axis_y1:
                    # Split-K on the -2 axis: blockIdx.y = n_tile*k_batch + kz, so a
                    # tile's kz partials are adjacent in dispatch order. blockIdx.z is
                    # the pure expert-block group.
                    _bycomb = gpu.block_id("y")
                    by = _bycomb // fx.Index(k_batch)  # tile along inter_dim (n-tile)
                    kz_sk = _bycomb % fx.Index(k_batch)
                    k_start = kz_sk * fx.Index(_k_per_batch1)
                    _bz = gpu.block_id("z")  # expert-block group
                else:
                    by = gpu.block_id("y")  # tile along inter_dim (n-tile)
                    _bz = (
                        _sk_group_z
                        if _is_splitk1
                        else _pm_fold1(gpu.block_id("z"))
                    )  # expert-block group (persist folds into the group dim)
                bx = _bz * fx.Index(MOE_NUM_XCD) + _xcd  # tile along sorted M
                # The XCD remap rounds the grid up, so bx can run past the real
                # number of expert blocks (size_expert_ids). Flag those pure-padding
                # blocks here and fold into blk_valid below so they exit via the
                # whole-kernel gate (no OOB, no wasted buffer/gmem work).
                bx_in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, bx),
                    i32_size_expert_ids_in,
                )
                # Keep bx_m / bx_m_i32 in the enclosing scope so the gated body below can
                # use bx_m (sorted_row, _tid_row, etc.) -- they must dominate that region.
                bx_m = bx * fx.Index(tile_m)
                bx_m_i32 = arith.index_cast(T.i32, bx_m)
                # Only in-range blocks load max_token_id and run the token check; padding
                # blocks (bx >= size_expert_ids) skip the load and yield invalid directly.
                _rng_if = scf.IfOp(bx_in_range, [ir.IntegerType.get_signless(1)], has_else=True)
                with _if_then(_rng_if):
                    maxids_rsrc = buffer_ops.create_buffer_resource(
                        arg_max_token_ids,
                        max_size=False,
                        num_records_bytes=fx.Index(4),
                    )
                    max_token_id_i32 = buffer_ops.buffer_load(
                        maxids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
                    )
                    tok_ok = arith.cmpi(
                        arith.CmpIPredicate.ult, bx_m_i32, max_token_id_i32
                    )
                    scf.YieldOp([tok_ok])
                with _if_else(_rng_if):
                    scf.YieldOp([arith.constant(0, type=ir.IntegerType.get_signless(1))])
                blk_valid = _rng_if.results[0]
            else:
                by = gpu.block_id("x")  # tile along inter_dim
                bx = _pm_fold1(gpu.block_id("y"))  # tile along sorted M
                # Block validity: compute as early as possible so invalid blocks skip all buffer-resource
                # setup, LDS pointer math, and gmem prefetch work.
                bx_m = bx * fx.Index(tile_m)
                maxids_rsrc = buffer_ops.create_buffer_resource(
                    arg_max_token_ids,
                    max_size=False,
                    num_records_bytes=fx.Index(4),
                )
                max_token_id_i32 = buffer_ops.buffer_load(
                    maxids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
                )
                bx_m_i32 = arith.index_cast(T.i32, bx_m)
                blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, max_token_id_i32)
            _xcd_debug_print(1, bx, by)
            # Common constants/atoms (hoisted): keep IR small like GEMM.
            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            # Everything below is gated by `blk_valid` to avoid doing buffer-resource setup and
            # gmem work for padding blocks.
            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                base_ptr = allocator.get_base()
                lds_x_ptr = SmemPtr(
                    base_ptr,
                    lds_alloc_offset,
                    (
                        T.bf16
                        if is_bf16
                        else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
                    ),
                    shape=(lds_total_elems,),
                )
                lds_x = lds_x_ptr.get()
                # Alias LDS bytes as fp16 for optional CShuffle epilogue.
                lds_out = (
                    SmemPtr(
                        base_ptr, lds_x_ptr.byte_offset, _out_elem_type(), shape=(tile_m * tile_n,)
                    ).get()
                    if _use_cshuffle_epilog
                    else None
                )

                # lds_tid: sorted_token_ids preloaded into LDS for epilogue
                lds_tid = SmemPtr(
                    base_ptr, lds_x_ptr.byte_offset + _lds_tid_byte_off, T.i32, shape=(tile_m,)
                ).get()

                # Buffer resources: for dynamic memrefs, provide `num_records_bytes` explicitly so
                # hardware OOB behavior is stable (otherwise it falls back to a large max size).
                c_topk = fx.Index(topk)

                # X: [tokens, k] bytes = tokens*k*elem_bytes
                x_rows = tokens_in * (c_topk if x_is_token_slot else fx.Index(1))
                x_nbytes_idx = x_rows * k_in * arith.index(int(elem_bytes))
                x_rsrc = buffer_ops.create_buffer_resource(
                    arg_x, max_size=False, num_records_bytes=x_nbytes_idx
                )

                w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

                if _use_wptr64:
                    from flydsl._mlir.dialects import fly as _fly

                    _llvm_ptr_ty_as1 = ir.Type.parse("!llvm.ptr<1>")
                    w_base_ptr = _fly.extract_aligned_pointer_as_index(
                        _llvm_ptr_ty_as1, arg_w
                    )
                    _kpack_elems_b = int(kpack_bytes) // int(w_elem_bytes)
                    _stride_nlane_b = arith.constant(_kpack_elems_b, index=True)
                    _stride_klane_b = arith.constant(
                        16 * _kpack_elems_b, index=True
                    )
                    _stride_k0_b = arith.constant(
                        64 * _kpack_elems_b, index=True
                    )
                    _c_k_bytes_b = k_in * arith.constant(
                        int(w_elem_bytes), index=True
                    )
                    _c_k0_b = _c_k_bytes_b // arith.constant(64, index=True)
                    _stride_n0_b = _c_k0_b * _stride_k0_b
                else:
                    w_base_ptr = None
                    _llvm_ptr_ty_as1 = None
                    _stride_n0_b = _stride_k0_b = None
                    _stride_klane_b = _stride_nlane_b = None

                # OUT: [tokens, topk, inter] f16/bf16 -> bytes = tokens*topk*inter*out_elem_bytes
                out_elem_bytes = 2  # f16/bf16
                # Split-K (g1u1) writes 2*inter_dim raw gate/up partial columns, so the
                # output buffer-resource must cover the wider stride or the up-partial
                # atomics (cols >= inter_dim) would be dropped as OOB.
                _out_cols_in = (
                    (inter_in * fx.Index(2))
                    if (_is_splitk1 and use_g1u1)
                    else inter_in
                )
                out_nbytes_idx = (
                    tokens_in * c_topk * _out_cols_in * fx.Index(out_elem_bytes)
                )
                out_rsrc = buffer_ops.create_buffer_resource(
                    arg_out, max_size=False, num_records_bytes=out_nbytes_idx
                )

                # scale_x: fp16/bf16 path ignores (implicit scale=1.0); int4_bf16 also uses 1.0.
                if is_f16_or_bf16:
                    sx_rsrc = None
                else:
                    sx_rows = tokens_in * (c_topk if x_is_token_slot else fx.Index(1))
                    sx_nbytes_idx = sx_rows * fx.Index(4)
                    sx_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_idx
                    )
                # scale_w: fp16/bf16 (non-int4) path ignores; int4_bf16 needs dequant scale.
                if not needs_scale_w:
                    sw_rsrc = None
                else:
                    sw_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_w, max_size=False
                    )

                sorted_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_token_ids, max_size=False
                )
                sorted_w_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_weights, max_size=False
                )

                # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
                expert_rsrc = buffer_ops.create_buffer_resource(
                    arg_expert_ids,
                    max_size=False,
                    num_records_bytes=(size_expert_ids_in * fx.Index(4)),
                )

                # Expert id for this M tile (keep address math in `index`)
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=T.i32
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                inter2_idx = arith.index(_w_rows_per_expert)
                expert_off_idx = expert_idx * inter2_idx  # index

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16/bf16 we require 16B.
                if is_f16_or_bf16:
                    if bytes_per_thread_x % 16 != 0:
                        raise ValueError(
                            f"[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                        )
                    x_load_bytes = 16
                else:
                    if bytes_per_thread_x % 16 == 0:
                        x_load_bytes = 16
                    elif bytes_per_thread_x % 8 == 0:
                        x_load_bytes = 8
                    elif bytes_per_thread_x % 4 == 0:
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                        )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // arith.index(4)
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = fx.Index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32
                mask24 = fx.Int32(0xFFFFFF)
                tokens_i32 = arith.index_cast(T.i32, tokens_in)
                topk_i32 = fx.Int32(topk)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                # decode token once (per thread's M-slice) and build a base row offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    # NOTE: rows beyond `num_valid_ids` can contain garbage (within the allocated
                    # buffer). That's OK as long as we never use an out-of-range token id to index X.
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                    )
                    t_raw = fused_i & mask24
                    # NOTE: aiter moe_sorting uses sentinel token_id == tokens for padding.
                    # Do NOT rely on buffer OOB semantics for X loads; explicitly mask to a safe row.
                    t_valid_i32 = arith.cmpi(arith.CmpIPredicate.ult, t_raw, tokens_i32)
                    if x_is_token_slot:
                        s_raw = fused_i >> 24
                        # X is indexed by token-slot in **slot-major** order:
                        #   row_ts = slot * tokens + token
                        # This matches CK's moe_smoothquant output layout.
                        row_ts_i32 = s_raw * tokens_i32 + t_raw
                        row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                        # Apply bounds check to token-slot index
                        row_ts_safe = t_valid_i32.select(row_ts_idx, fx.Index(0))
                        x_row_base_div4.append(row_ts_safe * c_k_div4)
                    else:
                        t_idx = arith.index_cast(T.index, t_raw)
                        t_safe = t_valid_i32.select(t_idx, fx.Index(0))
                        x_row_base_div4.append(t_safe * c_k_div4)

                vec4_x = T.vec(4, x_elem)

                def load_x(idx_i32):
                    """Load `x_load_bytes` bytes from X (gmem) into regs.

                    For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                    idx_i32 is in dword units; convert to element index for _buffer_load_vec.
                    """
                    if x_load_bytes == 16:
                        idx_elem = (
                            idx_i32 if elem_bytes == 1 else (idx_i32 * fx.Index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                            elem_bytes=elem_bytes,
                            cache_modifier=_X_CM,
                        )
                    # For 8B/4B, load raw i32 dwords directly.
                    if x_load_bytes == 8:
                        return buffer_ops.buffer_load(
                            x_rsrc, idx_i32, vec_width=2, dtype=T.i32,
                            cache_modifier=_X_CM,
                        )
                    return buffer_ops.buffer_load(
                        x_rsrc, idx_i32, vec_width=1, dtype=T.i32,
                        cache_modifier=_X_CM,
                    )

                def load_x_tile(base_k):
                    """Prefetch the per-thread X tile portion (gmem -> regs) for a given K base (in elements)."""
                    base_k_div4 = (base_k * arith.index(int(elem_bytes))) // fx.Index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(T.i32x4, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(x_vec)
                        else:
                            parts.append(x_vec)
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(lane_id, layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)

                # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16 * a_kpack_elems.
                # A-side kpack is always 16 bytes (activation elements); B-side kpack_bytes
                # may differ (e.g. 8 for int4 weights), but that only affects B preshuffle.
                row_a_lds = lane_mod_16
                a_kpack_elems = 16 // elem_bytes
                col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
                col_offset_base_bytes = (
                    col_offset_base
                    if elem_bytes == 1
                    else (col_offset_base * arith.index(int(elem_bytes)))
                )

                # Dynamic N tiling within block (same as existing kernels)
                by_n = by * fx.Index(tile_n)
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = fx.Index(n_per_wave)
                wave_n_id = wave_id % fx.Index(num_waves)
                n_tile_base = wave_n_id * c_n_per_wave

                # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
                n_intra_gate = []
                n_blk_gate = []
                n_intra_up = []
                n_blk_up = []
                col_g_list = []
                inter_idx = arith.index(inter_dim)
                c_n0_static = experts * _w_rows_per_expert // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.index(ni * 16)
                    col_g = by_n + n_tile_base
                    col_g = col_g + offset
                    col_g = col_g + lane_mod_16
                    col_g_list.append(col_g)

                    row_gate = expert_off_idx + col_g

                    coord_gate = fx.idx2crd(row_gate, layout_n_blk_intra)
                    n_blk_gate.append(fx.get(coord_gate, 0))
                    n_intra_gate.append(fx.get(coord_gate, 1))
                    if use_g1u1:
                        row_up = row_gate + inter_idx
                        coord_up = fx.idx2crd(row_up, layout_n_blk_intra)
                        n_blk_up.append(fx.get(coord_up, 0))
                        n_intra_up.append(fx.get(coord_up, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)
                _num_b_loads_gate = k_unroll * 2 * num_acc_n
                _num_b_loads = _num_b_loads_gate * (2 if use_g1u1 else 1)
                # NOTE: the async-copy barriers below use vmcnt=0 (full drain).
                # The async global->LDS DMA for X is slower than the B VGPR loads,
                # so a partial vmcnt count lets MFMA read the X tile from LDS
                # before the DMA lands (read-before-write race). Full drain has
                # negligible measured e2e cost and fixes the corruption.

                # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
                def _load_b_gep_vec_i32(*, n_blk, k0, k1, n_intra, load_bytes):
                    elem_idx_i = (
                        n_blk * _stride_n0_b
                        + k0 * _stride_k0_b
                        + k1 * _stride_klane_b
                        + n_intra * _stride_nlane_b
                    )
                    byte_idx = elem_idx_i * arith.constant(
                        int(w_elem_bytes), index=True
                    )
                    ptr = llvm.GEPOp(
                        _llvm_ptr_ty_as1,
                        w_base_ptr,
                        [arith.index_cast(T.i64, byte_idx)],
                        [-2147483648],
                        T.i8,
                        llvm.GEPNoWrapFlags.none,
                    ).result
                    vec_width = load_bytes // 4
                    return llvm.LoadOp(
                        T.vec(vec_width, T.i32), ptr, alignment=load_bytes
                    ).result

                def _load_b_pack_k32_via_gep(*, base_k, ki_step, n_blk, n_intra):
                    c64_idx = arith.constant(64, index=True)
                    base_k_bytes = base_k * arith.constant(
                        int(w_elem_bytes), index=True
                    )
                    k0_base = base_k_bytes // c64_idx
                    k0 = k0_base + arith.constant(ki_step // 2, index=True)
                    k1 = lane_div_16

                    raw_vec = _load_b_gep_vec_i32(
                        n_blk=n_blk, k0=k0, k1=k1, n_intra=n_intra,
                        load_bytes=int(kpack_bytes),
                    )
                    half = ki_step % 2
                    if half == 0:
                        d0 = vector.extract(
                            raw_vec, static_position=[0], dynamic_position=[]
                        )
                        d1 = vector.extract(
                            raw_vec, static_position=[1], dynamic_position=[]
                        )
                    else:
                        d0 = vector.extract(
                            raw_vec, static_position=[2], dynamic_position=[]
                        )
                        d1 = vector.extract(
                            raw_vec, static_position=[3], dynamic_position=[]
                        )
                    v2 = vector.from_elements(T.vec(2, T.i32), [d0, d1])
                    v64 = vector.bitcast(T.vec(1, T.i64), v2)
                    return vector.extract(
                        v64, static_position=[0], dynamic_position=[]
                    )

                def load_b_pack(base_k, ki_step, ni, blk_list, intra_list):
                    if _use_wptr64:
                        return _load_b_pack_k32_via_gep(
                            base_k=base_k,
                            ki_step=ki_step,
                            n_blk=blk_list[ni],
                            n_intra=intra_list[ni],
                        )
                    return load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ki_step,
                        n_blk=blk_list[ni],
                        n_intra=intra_list[ni],
                        lane_div_16=lane_div_16,  # 0..3
                        elem_type=w_elem,
                        kpack_bytes=kpack_bytes,
                        elem_bytes=w_elem_bytes,
                        unpack_int4=is_int4,
                        cache_modifier=b_nt,
                    )

                def load_b_tile(base_k, blk_list, intra_list):
                    """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base.

                    Returns a list of length `k_unroll`, where each entry is a tuple:
                      (packs_half0[ni], packs_half1[ni])  for the K64 micro-step.
                    """
                    if is_int4_bf16:
                        # W4A16: 2-phase load+unpack for VMEM latency hiding
                        # Phase 1: Issue ALL buffer_loads first.
                        raw_data = []
                        for ku in range_constexpr(k_unroll):
                            raw_ku = []
                            for ni in range_constexpr(num_acc_n):
                                raw = load_b_raw_w4a16(
                                    buffer_ops,
                                    arith,
                                    vector,
                                    arg_b=arg_w,
                                    b_rsrc=w_rsrc,
                                    layout_b=layout_b,
                                    base_k=base_k,
                                    ku=ku,
                                    n_blk=blk_list[ni],
                                    n_intra=intra_list[ni],
                                    lane_div_16=lane_div_16,
                                    elem_type=w_elem,
                                    kpack_bytes=kpack_bytes,
                                    cache_modifier=b_nt,
                                )
                                raw_ku.append(raw)
                            raw_data.append(raw_ku)
                        # Phase 2: Unpack ALL (by now early loads have completed).
                        b_tile = []
                        for ku in range_constexpr(k_unroll):
                            packs0 = []
                            packs1 = []
                            for ni in range_constexpr(num_acc_n):
                                b0, b1 = unpack_b_w4a16(raw_data[ku][ni], arith, vector)
                                packs0.append(b0)
                                packs1.append(b1)
                            b_tile.append((packs0, packs1))
                        return b_tile
                    b_tile = []
                    for ku in range_constexpr(k_unroll):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            b0 = load_b_pack(base_k, ki0, ni, blk_list, intra_list)
                            b1 = load_b_pack(base_k, ki1, ni, blk_list, intra_list)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                acc_gate = [acc_init] * (num_acc_n * m_repeat)
                acc_up = [acc_init] * (num_acc_n * m_repeat) if use_g1u1 else None

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif x_load_bytes == 8:
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                            )
                        else:
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                            )

                # ---- Async DMA path: global -> LDS directly (bypass registers) ----
                if use_async_copy:
                    if bytes_per_thread_x % 16 != 0:
                        raise ValueError(
                            f"use_async_copy requires bytes_per_thread_x divisible by 16, "
                            f"got {bytes_per_thread_x} (tile_m={tile_m}, tile_k={tile_k}, "
                            f"elem_bytes={elem_bytes}). Try larger tile_m or tile_k."
                        )
                    _dma_bytes = 16
                    _wave_size = 64
                    _num_dma_loads = bytes_per_thread_x // 16

                    def dma_x_tile_to_lds(base_k, lds_base):
                        c4_idx = fx.Index(4)
                        base_k_div4 = (base_k * arith.index(int(elem_bytes))) // c4_idx

                        lds_ptr_i64 = None
                        for i in range_constexpr(_num_dma_loads):
                            row_local_i = x_row_local[i]
                            col_local_i32_i = x_col_local_i32[i]
                            col_local_sw = swizzle_xor16(
                                row_local_i, col_local_i32_i * c4_idx, k_blocks16
                            )
                            row_k_dw = x_row_base_div4[i] + base_k_div4
                            global_byte_idx = row_k_dw * c4_idx + col_local_sw
                            global_offset = arith.index_cast(T.i32, global_byte_idx)

                            if i == 0:
                                lds_byte_off = lds_base * arith.index(int(elem_bytes))
                                lds_addr = (
                                    memref.extract_aligned_pointer_as_index(lds_x)
                                    + lds_byte_off
                                    + wave_id * arith.index(_wave_size * _dma_bytes)
                                )
                                lds_ptr_i64 = rocdl.readfirstlane(
                                    T.i64, arith.index_cast(T.i64, lds_addr)
                                )
                            else:
                                lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                    total_threads * _dma_bytes, type=T.i64
                                )

                            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                            lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                            rocdl.raw_ptr_buffer_load_lds(
                                x_rsrc,
                                lds_ptr,
                                arith.constant(_dma_bytes, type=T.i32),
                                global_offset,
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                                arith.constant(_X_DMA_AUX, type=T.i32),
                            )

                    def prefetch_x_to_lds(base_k, lds_base):
                        dma_x_tile_to_lds(base_k, lds_base)

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base_bytes, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(int(elem_bytes)))
                    )
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def compute_tile(
                    acc_gate_in,
                    acc_up_in,
                    b_gate_tile_in,
                    b_up_tile_in,
                    lds_base,
                    *,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                    a1_prefetch=None,
                ):
                    gate_list = list(acc_gate_in)
                    up_list = list(acc_up_in) if use_g1u1 else None
                    mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
                    mfma_fn = (
                        mfma_i32_k32
                        if is_int8
                        else (
                            mfma_f32_bf16_k16
                            if is_bf16
                            else (
                                rocdl.mfma_f32_16x16x16f16
                                if is_f16
                                else rocdl.mfma_f32_16x16x32_fp8_fp8
                            )
                        )
                    )

                    # Optional: prefetch epilogue scales while we are about to run the last MFMA tile,
                    # matching the preshuffle GEMM pattern of overlapping scale loads with MFMA.
                    epilogue_pf = None
                    if prefetch_epilogue:
                        expert_off_pf = expert_off_idx
                        sw_gate_pf = []
                        sw_up_pf = []
                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            row_gate_idx = expert_off_pf + col_g
                            sw_gate_pf.append(
                                fx.Float32(1.0)
                                if not needs_scale_w
                                else buffer_ops.buffer_load(
                                    sw_rsrc, row_gate_idx, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                )
                            )
                            if use_g1u1:
                                row_up_idx = row_gate_idx + inter_idx
                                sw_up_pf.append(
                                    fx.Float32(1.0)
                                    if not needs_scale_w
                                    else buffer_ops.buffer_load(
                                        sw_rsrc, row_up_idx, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                    )
                                )
                        epilogue_pf = (sw_gate_pf, sw_up_pf)

                    def _i64_to_v4f16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.f16x4, v1)

                    def _i64_to_v4i16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.i16x4, v1)

                    def _combine_i64_to_v4i32(lo, hi):
                        v2 = vector.from_elements(T.i64x2, [lo, hi])
                        return vector.bitcast(T.i32x4, v2)

                    def mfma_k64(acc_in, a0, a1, b0, b1):
                        if _use_k64_mfma:
                            a_v4 = _combine_i64_to_v4i32(a0, a1)
                            b_v4 = _combine_i64_to_v4i32(b0, b1)
                            return _mfma_i32_16x16x64_i8(a_v4, b_v4, acc_in)
                        if is_f16:
                            a0v = _i64_to_v4f16(a0)
                            a1v = _i64_to_v4f16(a1)
                            b0v = _i64_to_v4f16(b0)
                            b1v = _i64_to_v4f16(b1)
                            acc_mid = mfma_fn(mfma_res_ty, [a0v, b0v, acc_in, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc_mid, 0, 0, 0])
                        if is_bf16:
                            a0v = _i64_to_v4i16(a0)
                            a1v = _i64_to_v4i16(a1)
                            b0v = _i64_to_v4i16(b0)
                            b1v = _i64_to_v4i16(b1)
                            acc_mid = mfma_fn(mfma_res_ty, [a0v, b0v, acc_in, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc_mid, 0, 0, 0])
                        acc_mid = mfma_fn(mfma_res_ty, [a0, b0, acc_in, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1, b1, acc_mid, 0, 0, 0])

                    if _use_k128_mfma_fp8:
                        for ku128 in range_constexpr(k_unroll // 2):
                            ku0 = ku128 * 2
                            ku1 = ku0 + 1
                            bg0_p0, bg0_p1 = b_gate_tile_in[ku0]
                            bg1_p0, bg1_p1 = b_gate_tile_in[ku1]
                            bu0_p0, bu0_p1 = b_up_tile_in[ku0] if use_g1u1 else (None, None)
                            bu1_p0, bu1_p1 = b_up_tile_in[ku1] if use_g1u1 else (None, None)
                            ki64_0 = arith.index(ku0 * 64)
                            ki64_1 = arith.index(ku1 * 64)
                            col_base0 = col_offset_base_bytes + ki64_0
                            col_base1 = col_offset_base_bytes + ki64_1

                            _s_setprio(1)
                            for mi in range_constexpr(m_repeat):
                                mi_val = arith.index(mi * 16)
                                curr_row_a_lds = row_a_lds + mi_val
                                if (a0_prefetch is not None) and (ku0 == 0) and (mi == 0):
                                    a0, a1 = a0_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_base)
                                if (a1_prefetch is not None) and (ku1 == 1) and (mi == 0):
                                    a2, a3 = a1_prefetch
                                else:
                                    a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_base)
                                a_128 = _pack_i64x4_to_i32x8(a0, a1, a2, a3)

                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    b_128_g = _pack_i64x4_to_i32x8(
                                        bg0_p0[ni], bg0_p1[ni], bg1_p0[ni], bg1_p1[ni],
                                    )
                                    gate_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        T.f32x4,
                                        [a_128, b_128_g, gate_list[acc_idx],
                                         0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
                                    )
                                    if use_g1u1:
                                        b_128_u = _pack_i64x4_to_i32x8(
                                            bu0_p0[ni], bu0_p1[ni], bu1_p0[ni], bu1_p1[ni],
                                        )
                                        up_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                            T.f32x4,
                                            [a_128, b_128_u, up_list[acc_idx],
                                             0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
                                        )
                            _s_setprio(0)
                    else:
                        for ku in range_constexpr(k_unroll):
                            b_gate_packs0, b_gate_packs1 = b_gate_tile_in[ku]
                            b_up_packs0, b_up_packs1 = b_up_tile_in[ku] if use_g1u1 else (None, None)
                            ki64 = arith.index(ku * 64)
                            col_base = col_offset_base_bytes + ki64

                            _s_setprio(1)
                            for mi in range_constexpr(m_repeat):
                                mi_val = arith.index(mi * 16)
                                curr_row_a_lds = row_a_lds + mi_val

                                if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                                    a0, a1 = a0_prefetch
                                elif (a1_prefetch is not None) and (ku == 1) and (mi == 0):
                                    a0, a1 = a1_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(
                                        curr_row_a_lds, col_base, lds_base
                                    )

                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    gate_list[acc_idx] = mfma_k64(
                                        gate_list[acc_idx],
                                        a0,
                                        a1,
                                        b_gate_packs0[ni],
                                        b_gate_packs1[ni],
                                    )
                                    if use_g1u1:
                                        up_list[acc_idx] = mfma_k64(
                                            up_list[acc_idx],
                                            a0,
                                            a1,
                                            b_up_packs0[ni],
                                            b_up_packs1[ni],
                                        )
                            _s_setprio(0)
                    return gate_list, up_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.index(tile_m * lds_stride)
                lds_base_cur = fx.Index(0)
                lds_base_nxt = lds_tile_elems

                # Optional scheduler hints (copied from tuned GEMM); can be disabled via env.
                rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    mfma_group = num_acc_n if not use_g1u1 else (num_acc_n * 2)

                    # Use equivalent K=32 MFMA count for pipeline time slots,
                    # regardless of actual MFMA variant (K=64/K=128 have proportionally
                    # higher latency, so the scheduling window is the same).
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group

                    if use_async_copy:
                        a_vmem_load = max(1, tile_m // 32)
                        b_vmem_total = _num_b_loads
                        vmem_count = b_vmem_total + 2 + a_vmem_load

                        rocdl.sched_vmem(a_vmem_load)
                        rocdl.sched_mfma(a_vmem_load)

                        if tile_m == 16:
                            for i in range_constexpr(2):
                                rocdl.sched_dsrd(1)
                                rocdl.sched_mfma(1)
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(1)
                            _tail_vmem = max(0, vmem_count - a_vmem_load - 2)
                            for i in range_constexpr(_tail_vmem):
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(1)
                        else:
                            _dsrd_vmem_iters = a_vmem_load * 4
                            for i in range_constexpr(_dsrd_vmem_iters):
                                rocdl.sched_dsrd(1)
                                rocdl.sched_mfma(1)
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(mfma_group)
                            _tail_vmem = max(0, vmem_count - _dsrd_vmem_iters)
                            for i in range_constexpr(_tail_vmem):
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(mfma_group)
                    else:
                        mfma_per_iter = 2 * mfma_group
                        sche_iters = (
                            0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                        )

                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(2)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)

                        dswr_tail = num_x_loads
                        if dswr_tail > sche_iters:
                            dswr_tail = sche_iters
                        dswr_start = sche_iters - dswr_tail
                        for sche_i in range_constexpr(sche_iters):
                            rocdl.sched_vmem(1)
                            rocdl.sched_mfma(mfma_group)
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(mfma_group)
                            if sche_i >= dswr_start - 1:
                                rocdl.sched_dswr(1)

                    rocdl.sched_barrier(0)

                # Preload sorted_token_ids into lds_tid for epilogue
                _c_tile_m_idx = arith.constant(tile_m, index=True)
                _tid_in_range = arith.cmpi(arith.CmpIPredicate.ult, tx, _c_tile_m_idx)
                _if_tid = scf.IfOp(_tid_in_range)
                with _if_then(_if_tid):
                    _tid_row = bx_m + tx
                    _tid_val = buffer_ops.buffer_load(
                        sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                    )
                    _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                    vector.store(_tid_vec1, lds_tid, [tx])

                # Prologue: prefetch tile0, store to LDS(cur), sync.
                # Split-K: start at this CTA's K-slice base (k_start); == 0 otherwise.
                k0 = k_start if _is_splitk1 else fx.Index(0)
                if use_async_copy:
                    prefetch_x_to_lds(k0, lds_base_cur)
                    b_gate_cur = load_b_tile(k0, n_blk_gate, n_intra_gate)
                    b_up_cur = load_b_tile(k0, n_blk_up, n_intra_up) if use_g1u1 else []
                else:
                    x_regs0 = load_x_tile(k0)
                    b_gate_cur = load_b_tile(k0, n_blk_gate, n_intra_gate)
                    b_up_cur = load_b_tile(k0, n_blk_up, n_intra_up) if use_g1u1 else []
                    store_x_tile_to_lds(x_regs0, lds_base_cur)
                if use_async_copy:
                    _barrier(vmcnt=0, lgkmcnt=0)
                else:
                    _barrier(lgkmcnt=0)

                # Loop-carried ping/pong state.
                lds_base_pong = lds_base_cur  # current/compute
                lds_base_ping = lds_base_nxt  # next/load+store

                # Cross-tile A0+A1 LDS prefetch: issue ds_reads back-to-back so LDS
                # bandwidth is fully utilized and the second read completes during
                # MFMA execution.
                _a1_col_bytes = col_offset_base_bytes + arith.index(64)
                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base_bytes, lds_base_pong
                )
                a1_prefetch_pong = (
                    lds_load_packs_k64(row_a_lds, _a1_col_bytes, lds_base_pong)
                    if k_unroll >= 2
                    else None
                )

                # Unrolled ping-pong main loop (2 tiles per iteration), leaving 2 tail tiles.
                # Keep this as constexpr expansion to avoid SCF child-region dominance issues
                # when carrying MFMA accumulators/prefetch values into the tail section.
                c2_tile_k = arith.index(tile_k * 2)
                # Split-K: each CTA sweeps only its K-slice [k_start, k_start+K_per_batch).
                total_tiles = int(_k_per_batch1) // int(tile_k)
                pair_iters = max((total_tiles - 2) // 2, 0)
                # End of this CTA's K-slice (== k_in when not split-K, so IR is identical).
                _k_slice_end = (
                    (k_start + fx.Index(_k_per_batch1)) if _is_splitk1 else k_in
                )

                # Deep B(weight)-prefetch pool (opt-in via b_pool_depth): front-load N
                # (gate,up) tile-pairs so ~N weight loads stay in flight (match hand-tuned
                # ASM's deep-prefetch prologue). Gated by _use_pool1 so the default
                # (no-pool) path below is byte-identical.
                _bp_depth1 = b_pool_depth if b_pool_depth else MOE_S1_BPOOL_DEPTH
                _use_pool1 = (_bp_depth1 >= 2) and not use_async_copy

                def _load_b_pair1(kk):
                    return (
                        load_b_tile(kk, n_blk_gate, n_intra_gate),
                        load_b_tile(kk, n_blk_up, n_intra_up) if use_g1u1 else [],
                    )

                def _k_of1(t):
                    _kk = arith.index(t * tile_k)
                    return (k_start + _kk) if _is_splitk1 else _kk

                _bpool1 = []
                if _use_pool1:
                    _pooln1 = min(_bp_depth1, total_tiles)
                    _bpool1 = [(b_gate_cur, b_up_cur)] + [
                        _load_b_pair1(_k_of1(t)) for t in range(1, _pooln1)
                    ]
                    if _pooln1 > 1:
                        rocdl.sched_vmem(
                            (_pooln1 - 1) * k_unroll * num_acc_n * 2 * (2 if use_g1u1 else 1)
                        )

                # X(activation) HBM prefetch pool: front-load tiles 1..depth into regs;
                # ds_write to LDS stays 1-ahead (tile 0's X already stored in prologue).
                _xp_depth1 = x_pool_depth if x_pool_depth else MOE_S1_XPOOL_DEPTH
                _use_xpool1 = (_xp_depth1 >= 2) and not use_async_copy
                _xpool1 = []
                if _use_xpool1:
                    _xpool1 = [
                        load_x_tile(_k_of1(t))
                        for t in range(1, min(_xp_depth1 + 1, total_tiles))
                    ]
                    if len(_xpool1) > 0:
                        rocdl.sched_vmem(len(_xpool1) * num_x_loads)

                for pair_i in range_constexpr(pair_iters):
                    k_iv = arith.index(pair_i * (tile_k * 2))
                    if _is_splitk1:
                        k_iv = k_start + k_iv
                    # ---- stage 0: prefetch+store ping, compute pong ----
                    next_k1 = k_iv + tile_k
                    if use_async_copy:
                        prefetch_x_to_lds(next_k1, lds_base_ping)
                    elif _use_xpool1:
                        x_regs_ping = _xpool1.pop(0)  # tile 2i+1
                        _xtl0 = (2 * pair_i + 1) + _xp_depth1
                        if _xtl0 < total_tiles:
                            _xpool1.append(load_x_tile(_k_of1(_xtl0)))
                    else:
                        x_regs_ping = load_x_tile(next_k1)
                    if _use_pool1:
                        _bg0, _bu0 = _bpool1.pop(0)
                        _tl0 = 2 * pair_i + _bp_depth1
                        if _tl0 < total_tiles:
                            _bpool1.append(_load_b_pair1(_k_of1(_tl0)))
                    else:
                        b_gate_ping = load_b_tile(next_k1, n_blk_gate, n_intra_gate)
                        b_up_ping = load_b_tile(next_k1, n_blk_up, n_intra_up) if use_g1u1 else []
                        _bg0, _bu0 = b_gate_cur, b_up_cur

                    acc_gate, acc_up, _ = compute_tile(
                        acc_gate,
                        acc_up,
                        _bg0,
                        _bu0,
                        lds_base_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a1_prefetch=a1_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    a1_prefetch_pong = None
                    if not use_async_copy:
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    if use_async_copy:
                        _barrier(vmcnt=0, lgkmcnt=0)
                    else:
                        _barrier(lgkmcnt=0)

                    # Cross-tile prefetch for the ping tile we are about to compute.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_ping
                    )
                    a1_prefetch_ping = (
                        lds_load_packs_k64(row_a_lds, _a1_col_bytes, lds_base_ping)
                        if k_unroll >= 2
                        else None
                    )

                    # ---- stage 1: prefetch+store pong, compute ping ----
                    next_k2 = k_iv + c2_tile_k
                    if use_async_copy:
                        prefetch_x_to_lds(next_k2, lds_base_pong)
                    elif _use_xpool1:
                        x_regs_pong = _xpool1.pop(0)  # tile 2i+2
                        _xtl1 = (2 * pair_i + 2) + _xp_depth1
                        if _xtl1 < total_tiles:
                            _xpool1.append(load_x_tile(_k_of1(_xtl1)))
                    else:
                        x_regs_pong = load_x_tile(next_k2)
                    if _use_pool1:
                        _bg1, _bu1 = _bpool1.pop(0)
                        _tl1 = 2 * pair_i + 1 + _bp_depth1
                        if _tl1 < total_tiles:
                            _bpool1.append(_load_b_pair1(_k_of1(_tl1)))
                    else:
                        b_gate_next = load_b_tile(next_k2, n_blk_gate, n_intra_gate)
                        b_up_next = load_b_tile(next_k2, n_blk_up, n_intra_up) if use_g1u1 else []
                        _bg1, _bu1 = b_gate_ping, b_up_ping

                    acc_gate, acc_up, _ = compute_tile(
                        acc_gate,
                        acc_up,
                        _bg1,
                        _bu1,
                        lds_base_ping,
                        a0_prefetch=a0_prefetch_ping,
                        a1_prefetch=a1_prefetch_ping,
                    )
                    a0_prefetch_ping = None
                    a1_prefetch_ping = None
                    if not use_async_copy:
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                    hot_loop_scheduler()
                    if use_async_copy:
                        _barrier(vmcnt=0, lgkmcnt=0)
                    else:
                        _barrier(lgkmcnt=0)

                    # Cross-tile prefetch for the next pong tile.
                    a0_prefetch_pong = lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_pong
                    )
                    a1_prefetch_pong = (
                        lds_load_packs_k64(row_a_lds, _a1_col_bytes, lds_base_pong)
                        if k_unroll >= 2
                        else None
                    )

                    # Advance pong state to next_k2 for next iteration.
                    if not _use_pool1:
                        b_gate_cur = b_gate_next
                        b_up_cur = b_up_next

                # Tail: 2 remaining tiles at (k_end - 2*tile_k) and (k_end - tile_k),
                # where k_end == k_in for the non-split-K path and the K-slice end otherwise.
                # Rebuild prefetch in the current block: values produced inside the `range(...)`
                # loop body may live in a child region and cannot be used here.
                k_tail0 = _k_slice_end - c2_tile_k
                if _use_pool1:
                    b_gate_cur, b_up_cur = _bpool1.pop(0)  # tile total_tiles-2
                else:
                    b_gate_cur = load_b_tile(k_tail0, n_blk_gate, n_intra_gate)
                    b_up_cur = load_b_tile(k_tail0, n_blk_up, n_intra_up) if use_g1u1 else []
                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base_bytes, lds_base_pong
                )
                a1_prefetch_pong = (
                    lds_load_packs_k64(row_a_lds, _a1_col_bytes, lds_base_pong)
                    if k_unroll >= 2
                    else None
                )
                k_tail1 = _k_slice_end - tile_k
                if use_async_copy:
                    prefetch_x_to_lds(k_tail1, lds_base_ping)
                elif _use_xpool1:
                    x_regs_ping = _xpool1.pop(0)  # last tile's X (held in pool)
                else:
                    x_regs_ping = load_x_tile(k_tail1)
                if _use_pool1:
                    b_gate_ping, b_up_ping = _bpool1.pop(0)  # tile total_tiles-1 (last)
                else:
                    b_gate_ping = load_b_tile(k_tail1, n_blk_gate, n_intra_gate)
                    b_up_ping = load_b_tile(k_tail1, n_blk_up, n_intra_up) if use_g1u1 else []

                acc_gate, acc_up, _ = compute_tile(
                    acc_gate,
                    acc_up,
                    b_gate_cur,
                    b_up_cur,
                    lds_base_pong,
                    a0_prefetch=a0_prefetch_pong,
                    a1_prefetch=a1_prefetch_pong,
                )
                a0_prefetch_pong = None
                a1_prefetch_pong = None
                if not use_async_copy:
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                hot_loop_scheduler()
                if use_async_copy:
                    _barrier(vmcnt=0, lgkmcnt=0)
                else:
                    _barrier(lgkmcnt=0)

                # Cross-tile prefetch for the final ping tile.
                a0_prefetch_ping = lds_load_packs_k64(
                    row_a_lds, col_offset_base_bytes, lds_base_ping
                )
                a1_prefetch_ping = (
                    lds_load_packs_k64(row_a_lds, _a1_col_bytes, lds_base_ping)
                    if k_unroll >= 2
                    else None
                )

                # Epilogue: compute last tile with epilogue scale prefetch to overlap loads with MFMA.
                acc_gate, acc_up, epilogue_pf = compute_tile(
                    acc_gate,
                    acc_up,
                    b_gate_ping,
                    b_up_ping,
                    lds_base_ping,
                    prefetch_epilogue=True,
                    a0_prefetch=a0_prefetch_ping,
                    a1_prefetch=a1_prefetch_ping,
                )

                # Store epilogue to out[t, slot, inter]
                expert_off = expert_off_idx
                tokens_i32_v = tokens_i32
                topk_i32_v = topk_i32
                inter_i32_v = fx.Int32(inter_dim)
                mask24_i32 = fx.Int32(0xFFFFFF)

                if epilogue_pf is not None:
                    sw_gate_vals, sw_up_vals = epilogue_pf
                else:
                    sw_gate_vals = []
                    sw_up_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_gate_idx = expert_off + col_g
                        row_up_idx = row_gate_idx + inter_idx
                        sw_gate_vals.append(
                            fx.Float32(1.0)
                            if not needs_scale_w
                            else buffer_ops.buffer_load(
                                sw_rsrc, row_gate_idx, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                            )
                        )
                        if use_g1u1:
                            sw_up_vals.append(
                                fx.Float32(1.0)
                                if not needs_scale_w
                                else buffer_ops.buffer_load(
                                    sw_rsrc, row_up_idx, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                )
                            )
                        else:
                            sw_up_vals = None

                # Epilogue hoists to keep IR + Python build time small:
                col_i32_list = []
                for ni in range_constexpr(num_acc_n):
                    col_i32_list.append(arith.index_cast(T.i32, col_g_list[ni]))

                inter_i32_local = inter_i32_v

                if _is_splitk1:
                    # ---- Split-K epilogue: atomic-accumulate RAW gate/up partials ----
                    # Each kz CTA owns a K-slice and contributes a partial sum. The
                    # silu/mul activation (and routing weight) is deferred to the
                    # post-kernel reduction (moe_kernels.py -> silu_and_mul), so here
                    # we write the *pre-activation* gate/up values (scaled by sx/sw)
                    # into a [tokens*topk, 2*inter_dim] buffer with atomic-add.
                    # Two passes share the CShuffle LDS scratch (gate, then up),
                    # each producing contiguous EVec=2 fragments for packed atomics.
                    if lds_out is None:
                        raise RuntimeError(
                            "split-K stage1 requires the CShuffle LDS buffer (lds_out); "
                            "do not disable use_cshuffle_epilog for k_batch>1."
                        )
                    _sk_out_cols = (2 * inter_dim) if use_g1u1 else inter_dim
                    _sk_out_cols_i32 = fx.Int32(_sk_out_cols)
                    _sk_evec = 2
                    _sk_nlane = min(32, tile_n // _sk_evec)
                    _sk_zero_i32 = fx.Int32(0)
                    _sk_aux_i32 = fx.Int32(_OUT_ATOMIC_AUX)
                    _sk_c2_i32 = fx.Int32(2)  # 2 bytes per f16/bf16 element
                    _sk_mask_even = fx.Int32(0xFFFFFFFE)
                    _sk_frag_ty = T.bf16 if out_dtype == "bf16" else T.f16
                    _sk_state = {"acc": None, "sw": None, "noff": 0}

                    def _sk_atomic_add_x2(val_x2, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_x2,
                            out_rsrc,
                            byte_off_i32,
                            _sk_zero_i32,
                            _sk_aux_i32,
                        )

                    def _sk_write_row(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        fused2 = memref.load(lds_tid, [row_in_tile])
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        t_valid = arith.cmpi(
                            arith.CmpIPredicate.ult, t2, tokens_i32_v
                        )
                        if x_is_token_slot:
                            ts2 = s2 * tokens_i32_v + t2
                            sx = (
                                fx.Float32(1.0)
                                if is_f16_or_bf16
                                else arith.select(
                                    t_valid,
                                    buffer_ops.buffer_load(
                                        sx_rsrc, ts2, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                    ),
                                    fx.Float32(0.0),
                                )
                            )
                        else:
                            sx = (
                                fx.Float32(1.0)
                                if is_f16_or_bf16
                                else arith.select(
                                    t_valid,
                                    buffer_ops.buffer_load(
                                        sx_rsrc, t2, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                    ),
                                    fx.Float32(0.0),
                                )
                            )
                        _acc = _sk_state["acc"]
                        _sw = _sk_state["sw"]
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            sw = _sw[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                _acc[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            if is_int8:
                                v = arith.sitofp(T.f32, v)
                            v = v * sx * sw
                            v_out = arith.trunc_f(_out_elem_type(), v)
                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(_out_vec_type(), [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def _sk_precompute_row(*, row_local, row):
                        fused2 = memref.load(lds_tid, [row_local])
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        t_valid = arith.cmpi(
                            arith.CmpIPredicate.ult, t2, tokens_i32_v
                        )
                        ts = t2 * topk_i32_v + s2
                        row_base_elem = ts * _sk_out_cols_i32
                        return (row_base_elem, t_valid)

                    def _sk_store_pair(
                        *, row_local, row, row_ctx, col_pair0, col_g0, frag
                    ):
                        row_base_elem = row_ctx
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_elem = row_base_elem + col_i32 + fx.Int32(_sk_state["noff"])
                        idx_elem_even = idx_elem & _sk_mask_even
                        byte_off = idx_elem_even * _sk_c2_i32
                        _sk_atomic_add_x2(frag, byte_off)

                    def _sk_run_pass(acc, sw_vals, noff):
                        _sk_state["acc"] = acc
                        _sk_state["sw"] = sw_vals
                        _sk_state["noff"] = noff
                        c_shuffle_epilog(
                            arith=arith,
                            vector=vector,
                            gpu=gpu,
                            scf=scf,
                            range_constexpr=range_constexpr,
                            tile_m=tile_m,
                            tile_n=tile_n,
                            e_vec=_sk_evec,
                            cshuffle_nlane=_sk_nlane,
                            block_size=total_threads,
                            m_repeat=m_repeat,
                            num_acc_n=num_acc_n,
                            tx=tx,
                            lane_div_16=lane_div_16,
                            lane_mod_16=lane_mod_16,
                            bx_m=bx_m,
                            by_n=by_n,
                            n_tile_base=n_tile_base,
                            lds_out=lds_out,
                            frag_elem_type=_sk_frag_ty,
                            write_row_to_lds=_sk_write_row,
                            precompute_row=_sk_precompute_row,
                            store_pair=_sk_store_pair,
                        )

                    # Pass 1: gate partials -> columns [0, inter_dim)
                    _sk_run_pass(acc_gate, sw_gate_vals, 0)
                    # Pass 2: up partials -> columns [inter_dim, 2*inter_dim)
                    if use_g1u1:
                        _sk_run_pass(acc_up, sw_up_vals, inter_dim)
                    return

                # Uses EVec=4 (buffer store "x4" of fp16 elements).
                use_cshuffle_epilog_flag = _use_cshuffle_epilog

                if use_cshuffle_epilog_flag:
                    if lds_out is None:
                        raise RuntimeError(
                            "CShuffle epilogue enabled but lds_out is not allocated/aliased."
                        )

                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        fused2 = memref.load(lds_tid, [row_in_tile])
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        # aiter moe_sorting uses sentinel token_id == tokens for padding.
                        # Do NOT rely on buffer OOB semantics for scale loads; explicitly mask.
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        if x_is_token_slot:
                            # slot-major: slot*tokens + token
                            ts2 = s2 * tokens_i32_v + t2
                            sx = (
                                fx.Float32(1.0)
                                if is_f16_or_bf16
                                else arith.select(
                                    t_valid,
                                    buffer_ops.buffer_load(
                                        sx_rsrc, ts2, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                    ),
                                    fx.Float32(0.0),
                                )
                            )
                        else:
                            sx = (
                                fx.Float32(1.0)
                                if is_f16_or_bf16
                                else arith.select(
                                    t_valid,
                                    buffer_ops.buffer_load(
                                        sx_rsrc, t2, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                    ),
                                    fx.Float32(0.0),
                                )
                            )

                        # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                        if doweight_stage1:
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                            )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            sw_gate = sw_gate_vals[ni]
                            sw_up = sw_up_vals[ni] if use_g1u1 else None

                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            vu = vector.extract(
                                acc_up[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            ) if use_g1u1 else None

                            if is_int8:
                                vg = arith.sitofp(T.f32, vg)
                                vu = arith.sitofp(T.f32, vu) if use_g1u1 else None
                            vg = vg * sx * sw_gate
                            vu = (vu * sx * sw_up) if use_g1u1 else None

                            if use_g1u1:
                                if act == "silu":
                                    y = silu(vg) * vu
                                elif act == "gelu":
                                    y = gelu(vg) * vu
                                else:
                                    y = silu(vg) * vu
                            else:
                                if act == "silu":
                                    y = silu(vg)
                                elif act == "gelu":
                                    y = gelu(vg)
                                else:
                                    y = silu(vg)
                            if doweight_stage1:
                                y = y * tw
                            y16 = arith.trunc_f(_out_elem_type(), y)

                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(_out_vec_type(), [y16])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        fused2 = memref.load(lds_tid, [row_local])
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        return (t2 * topk_i32_v + s2) * inter_i32_local

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        fused2 = memref.load(lds_tid, [row_local])
                        t2 = fused2 & mask24_i32
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        _if_valid = scf.IfOp(t_valid)
                        with _if_then(_if_valid):
                            idx0 = row_ctx
                            col_i32 = arith.index_cast(T.i32, col_g0)
                            idx_out = idx0 + col_i32
                            # Vectorized fp16 store (EVec=4).
                            buffer_ops.buffer_store(frag, out_rsrc, idx_out)

                    mfma_epilog(
                        use_cshuffle=True,
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        block_size=total_threads,
                        e_vec=4,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        frag_elem_type=_out_elem_type(),
                    )
                    # NOTE: no early `return` here. An early return would skip the
                    # persist_m loop close (scf.YieldOp + InsertionPoint exit) emitted
                    # at the end of the kernel body, leaving the scf.for region open.
                    # The default-epilog path below is gated on use_cshuffle instead.

                def _stage1_store_row(*, mi: int, ii: int, row_in_tile, row):
                    fused2 = memref.load(lds_tid, [row_in_tile])
                    t2_raw = fused2 & mask24_i32
                    s2_raw = fused2 >> 24
                    t2 = t2_raw
                    s2 = s2_raw
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)

                    # Do NOT rely on buffer OOB semantics for scale loads; explicitly mask.
                    if x_is_token_slot:
                        # slot-major: slot*tokens + token
                        ts2 = s2 * tokens_i32_v + t2
                        sx0 = (
                            fx.Float32(1.0)
                            if is_f16_or_bf16
                            else arith.select(
                                t_valid,
                                buffer_ops.buffer_load(
                                    sx_rsrc, ts2, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                ),
                                fx.Float32(0.0),
                            )
                        )
                    else:
                        sx0 = (
                            fx.Float32(1.0)
                            if is_f16_or_bf16
                            else arith.select(
                                t_valid,
                                buffer_ops.buffer_load(
                                    sx_rsrc, t2, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                ),
                                fx.Float32(0.0),
                            )
                        )
                    sx = sx0

                    # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                    idx0 = (t2 * topk_i32_v + s2) * inter_i32_local

                    # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                    if doweight_stage1:
                        tw = buffer_ops.buffer_load(
                            sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                        )

                    _if_valid = scf.IfOp(t_valid)
                    with _if_then(_if_valid):
                        for ni in range_constexpr(num_acc_n):
                            col_i32 = col_i32_list[ni]
                            sw_gate = sw_gate_vals[ni]
                            sw_up = sw_up_vals[ni] if use_g1u1 else None

                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            if use_g1u1:
                                vu = vector.extract(
                                    acc_up[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )

                            if is_int8:
                                vg = arith.sitofp(T.f32, vg)
                                vu = arith.sitofp(T.f32, vu) if use_g1u1 else None
                            vg = vg * sx * sw_gate
                            vu = (vu * sx * sw_up) if use_g1u1 else None

                            if use_g1u1:
                                if act == "silu":
                                    y = silu(vg) * vu
                                elif act == "gelu":
                                    y = gelu(vg) * vu
                                else:
                                    y = silu(vg) * vu
                            else:
                                if act == "silu":
                                    y = silu(vg)
                                elif act == "gelu":
                                    y = gelu(vg)
                                else:
                                    y = silu(vg)
                            if doweight_stage1:
                                y = y * tw
                            y = arith.trunc_f(out_mlir(), y)
                            idx_out0 = idx0 + col_i32
                            buffer_ops.buffer_store(y, out_rsrc, idx_out0)

                if not use_cshuffle_epilog_flag:
                    mfma_epilog(
                        use_cshuffle=False,
                        arith=arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=_stage1_store_row,
                    )

            if _persist1:
                # barrier so LDS from this M-block is fully consumed before the next
                # persist iteration reuses it; then close the persist loop.
                gpu.barrier()
                scf.YieldOp([])
                _for_ip1.__exit__(None, None, None)

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    @flyc.jit
    def launch_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = inter_in // fx.Index(tile_n)
        gy = size_expert_ids_in
        # Split-K multiplies the launch z-dim by k_batch. The kernel decodes
        # blockIdx.z as (group * k_batch + kz); k_batch==1 leaves the grid intact.
        _sk_kb = fx.Index(k_batch)

        # persist_m (>1) divides the M-carrying grid dim so each WG serially sweeps
        # persist_m M-blocks. persist follows M: remap=gy -> group (z) dim; remap=gx /
        # no-remap -> M dim (block_id.y).
        def _pm_ceil(dim):
            if persist_m == 1:
                return dim
            return (dim + fx.Index(persist_m - 1)) // fx.Index(persist_m)

        if MOE_XCD_REMAP:
            if MOE_XCD_REMAP_GX:
                # (NUM_XCD, expert_blocks, ceil(n_tiles/NUM_XCD)): split N across XCDs
                # so each XCD keeps its weight n_tile slice L2-resident.
                gz = (gx + fx.Index(MOE_NUM_XCD - 1)) // fx.Index(MOE_NUM_XCD)
                gz = gz * _sk_kb if _is_splitk1 else gz
                grid_dims = (fx.Index(MOE_NUM_XCD), _pm_ceil(gy), gz)
            else:
                # (NUM_XCD, n_tiles, ceil(expert_blocks/NUM_XCD)) for XCD L2 locality
                gz = (gy + fx.Index(MOE_NUM_XCD - 1)) // fx.Index(MOE_NUM_XCD)
                if _is_splitk1 and MOE_SPLITK_AXIS == "y":
                    # Split-K on the -2 axis: fold k_batch into the n_tile dim so a
                    # tile's kz partials dispatch adjacently (output stays L2-hot).
                    grid_dims = (fx.Index(MOE_NUM_XCD), gx * _sk_kb, gz)
                else:
                    gz = gz * _sk_kb if _is_splitk1 else gz
                    grid_dims = (fx.Index(MOE_NUM_XCD), gx, _pm_ceil(gz))
        else:
            grid_dims = (
                (gx, gy, _sk_kb) if _is_splitk1 else (gx, _pm_ceil(gy), 1)
            )

        moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=grid_dims,
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm1


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    accumulate: bool = True,
    use_async_copy: bool = False,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    k_batch: int = 1,
    persist_m: int = 1,
    remap: str | None = None,
    splitk_axis: str | None = None,
    x_nt: int | None = None,
    scale_nt: int | None = None,
    out_nt: int | None = None,
    b_pool_depth: int = 0,
    x_pool_depth: int = 0,
    persist_n: int = 0,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    in_dtype:
      - "fp8": A2/W are fp8
      - "fp16": A2/W are fp16
      - "bf16": A2/W are bf16
      - "int8": A2/W are int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 unpacked to int8 in-kernel
      - "int4_bf16": W4A16 path: A2 is bf16, W is packed int4 unpacked to bf16 in-kernel

    Stage2 output supports:
      - out_dtype="f16": fp16 half2 atomics (fast, can overflow to +/-inf for bf16 workloads)
      - out_dtype="f32": fp32 scalar atomics (slower, but avoids fp16 atomic overflow)

    `use_cshuffle_epilog` controls whether we use the LDS CShuffle epilogue before
    global atomics (recommended for performance).

    waves_per_eu:
      Controls LDS-based occupancy (see compile_moe_gemm1 docstring). 0 = no padding.

    b_nt:
      Non-temporal cache modifier for B (weight) buffer loads.
      0 = normal caching, 2 = non-temporal (GLC+SLC).

    remap / splitk_axis / x_nt / scale_nt / out_nt:
      Per-compile dispatch & cache-policy knobs (None => env default); see
      _resolve_moe_knobs / _moe_knob_tags.
    """
    b_nt = _eff_b_nt(b_nt)
    (
        MOE_XCD_REMAP,
        MOE_XCD_REMAP_GX,
        MOE_SPLITK_AXIS,
        _X_CM,
        _X_DMA_AUX,
        _SCALE_CM,
        _OUT_ATOMIC_AUX,
        _OUT_STORE_CM,
    ) = _resolve_moe_knobs(remap, splitk_axis, x_nt, scale_nt, out_nt)
    # persist_m (workgroup merge along M) is the inverse of split-K; the two are
    # mutually exclusive. Each WG serially sweeps persist_m M-blocks.
    persist_m = int(persist_m)
    if persist_m > 1 and k_batch > 1:
        raise ValueError(
            f"persist_m={persist_m} and k_batch={k_batch} are mutually exclusive"
        )

    # ── persist_n: N-merge factor (analogous to persist_m along M) ───────────
    # persist_n = how many consecutive N-tiles (output/model_dim tiles) each
    # workgroup serially sweeps. The N tiles of a given M-block share the SAME
    # stage2 activation X (X depends only on M/inter_dim, not on the output
    # dim), so folding several N-tiles into one WG lets X stay L2-resident and
    # reused across them, and shrinks the launch N-grid dim by persist_n.
    #   persist_n <= 1 (default) -> each WG covers 1 N-tile (grid N = gx_total,
    #                               IR byte-identical to before).
    #   persist_n = k > 1        -> grid N = gx_total/k; each WG loops over k
    #                               consecutive N-tiles (base = block_id*k + j).
    # Source precedence: the per-compile `persist_n` arg (from the `_pn{n}`
    # kernel-name token) takes precedence, else env FLYDSL_MOE_STAGE2_PERSIST_N,
    # else full-N default. A requested value that does not divide gx_total (or an
    # unsupported combination) silently falls back to the default (persist_n=1).
    # NOTE: kept intentionally independent of / mutually simple with the other
    # workgroup-shaping knobs: persist_n>1 is only honored for the plain path
    # (k_batch==1, persist_m==1) and when N stays on a single grid axis (i.e.
    # not the gx-first XCD remap, which already splits N across XCDs).
    _gx_total = int(model_dim) // int(tile_n)
    _pn_env = int(os.environ.get("FLYDSL_MOE_STAGE2_PERSIST_N", "0") or "0")
    _pn_req = int(persist_n) if int(persist_n) > 0 else _pn_env
    _persist_n_ok = (
        _pn_req > 1
        and _pn_req <= _gx_total
        and (_gx_total % _pn_req == 0)
        and int(k_batch) == 1
        and int(persist_m) == 1
        and not MOE_XCD_REMAP_GX
    )
    _persist_n = _pn_req if _persist_n_ok else 1

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)

    if in_dtype not in (
        "fp8",
        "fp16",
        "bf16",
        "int8",
        "int8smooth",
        "int4",
        "int4_bf16",
    ):
        raise ValueError(
            f"in_dtype must be one of ('fp8','fp16','bf16','int8','int8smooth','int4','int4_bf16'), got {in_dtype!r}"
        )
    is_int4_bf16 = in_dtype == "int4_bf16"
    is_f16 = in_dtype == "fp16"
    is_bf16 = is_int4_bf16 or in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    needs_scale_w = (not is_f16_or_bf16) or is_int4_bf16
    elem_bytes = 2 if is_f16_or_bf16 else 1
    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(
            f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}"
        )
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype in ("int8", "int8smooth")) or is_int4

    # ── Split-K (partition the GEMM K=inter_dim across `k_batch` workgroups) ──
    # Stage2 is a plain linear down-projection that already atomic-accumulates
    # its result into the final output. Split-K therefore only needs to slice
    # the K-loop per CTA and let the existing atomics sum the partials, so it
    # requires accumulate=True. k_batch==1 keeps the original path untouched.
    _is_splitk2 = int(k_batch) > 1
    if _is_splitk2:
        if not bool(accumulate):
            raise ValueError("split-K (k_batch>1) stage2 requires accumulate=True")
        if int(inter_dim) % int(k_batch) != 0:
            raise ValueError(
                f"split-K: inter_dim={inter_dim} not divisible by k_batch={k_batch}"
            )
        _k_per_batch2 = int(inter_dim) // int(k_batch)
        if _k_per_batch2 % int(tile_k) != 0:
            raise ValueError(
                f"split-K: K_per_batch={_k_per_batch2} not divisible by tile_k={tile_k}"
            )
        if (_k_per_batch2 // int(tile_k)) < 2:
            raise ValueError(
                "split-K: K_per_batch must be >= 2*tile_k (pipeline needs >=2 tail tiles)"
            )
    else:
        _k_per_batch2 = int(inter_dim)
    _sk_axis_tag2 = (
        "y"
        if (_is_splitk2 and MOE_SPLITK_AXIS == "y" and MOE_XCD_REMAP and not MOE_XCD_REMAP_GX)
        else ""
    )
    _sk_tag2 = f"_sk{k_batch}{_sk_axis_tag2}" if _is_splitk2 else ""

    _is_gfx950 = str(gpu_arch).startswith("gfx950")
    _use_k64_mfma = _is_gfx950 and is_int8

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    mfma_f32_bf16_k16 = None
    if is_bf16:
        mfma_f32_bf16_k16 = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
            rocdl, "mfma_f32_16x16x16_bf16_1k", None
        )
        if mfma_f32_bf16_k16 is None:
            raise AttributeError(
                "BF16 K16 MFMA op not found: expected `rocdl.mfma_f32_16x16x16bf16_1k` "
                "(or `rocdl.mfma_f32_16x16x16_bf16_1k`)."
            )

    num_waves = tile_n // 32
    total_threads = num_waves * 64
    tile_k_bytes = int(tile_k) * int(elem_bytes)
    _use_k128_mfma_fp8 = (
        _is_gfx950 and not is_int8 and not is_f16_or_bf16
        and (tile_k_bytes % 128) == 0
    )
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    # gfx950+ has buffer_atomic_pk_add_bf16 → bf16 can use buffer atomics (same as f16).
    # gfx942 only has global_atomic_pk_add_bf16 → must use global atomics with raw pointer.
    _has_buffer_atomic_bf16 = str(gpu_arch).startswith(("gfx95", "gfx12"))
    _needs_global_atomic_bf16 = out_is_bf16 and not _has_buffer_atomic_bf16
    if out_is_bf16:
        if not supports_bf16_global_atomics(gpu_arch):
            raise ValueError(
                f"out_dtype='bf16' requires bf16 global atomics ({bf16_global_atomics_arch_description()}), got arch={gpu_arch!r}"
            )

    if out_is_f32:
        # Match origin/dev_a16w4: f32 output uses scalar atomics and does NOT use the CShuffle epilogue.
        _use_cshuffle_epilog = (
            False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        )
        if _use_cshuffle_epilog:
            raise ValueError(
                "out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False)."
            )
    else:
        if use_cshuffle_epilog is None:
            _use_cshuffle_epilog = os.environ.get(
                "FLYDSL_MOE_STAGE2_CSHUFFLE", "1"
            ) in (
                "1",
                "true",
                "True",
                "YES",
                "yes",
            )
        else:
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
        if not _use_cshuffle_epilog:
            raise ValueError(
                "stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1)."
            )

    # NOTE: Keep this as a callable so we don't require an MLIR Context at Python-time.
    def out_elem():
        ty = T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)
        return ty() if callable(ty) else ty

    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    _async_tag2 = "_async" if use_async_copy else ""
    _wpe_tag2 = f"_wpe{waves_per_eu}" if waves_per_eu >= 1 else ""
    _bnt_tag2 = f"_bnt{b_nt}" if b_nt != 2 else ""

    _w_storage_elem_bytes_s2 = 2 if is_f16_or_bf16 else 1
    _w_physical_k_bytes_static_s2 = int(inter_dim) * _w_storage_elem_bytes_s2
    _w_nbytes_static_s2 = (
        int(experts) * int(model_dim) * _w_physical_k_bytes_static_s2
    )
    _use_wptr64 = _w_nbytes_static_s2 >= (1 << 31)
    _wptr64_tag = "_wptr64" if _use_wptr64 else ""

    _knob_tag = _moe_knob_tags(
        MOE_XCD_REMAP, MOE_XCD_REMAP_GX, _X_CM, _SCALE_CM, _OUT_ATOMIC_AUX
    )
    _pm_tag = f"_pm{persist_m}" if persist_m != 1 else ""
    # Effective B/X prefetch-pool depths: per-kernel param (for tuning) overrides the
    # module-level env default (for standalone experiments). Baked into the name so
    # each depth is a distinct compiled kernel.
    _bp_depth = b_pool_depth if b_pool_depth else MOE_S2_BPOOL_DEPTH
    _xp_depth = x_pool_depth if x_pool_depth else MOE_S2_XPOOL_DEPTH
    _pool_tag = (f"_bp{_bp_depth}" if _bp_depth >= 2 else "") + (
        f"_xp{_xp_depth}" if _xp_depth >= 2 else "")
    # Encode persist_n so distinct per-WG N-loop lengths get separate compile
    # caches. Only emitted when partial (>1) so the default stays byte-identical.
    _pn_tag = f"_pn{_persist_n}" if _persist_n > 1 else ""
    module_name = (
        f"mfma_moe2_{in_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}{_async_tag2}{_wpe_tag2}{_bnt_tag2}{_wptr64_tag}{_sk_tag2}{_knob_tag}{_pm_tag}{_pool_tag}{_pn_tag}"
        f"_abi5_wptr64gate"  # ABI bumped: optional 64-bit W load path gated by static size check
    ).replace("-", "_")

    # ── CShuffle epilogue e_vec (pure Python; must be computed before @flyc.kernel
    # because the AST rewriter intercepts `if` statements inside kernel bodies and
    # turns them into closure dispatches, which breaks variable reassignment) ────
    _cshuffle_nlane = 32
    if bool(accumulate):
        _e_vec = 2
    else:
        _e_vec = 8 if int(tile_n) % (_cshuffle_nlane * 8) == 0 else 2
        _cshuffle_stride = _cshuffle_nlane * _e_vec
        if int(tile_n) % _cshuffle_stride != 0:
            raise ValueError(
                f"tile_n={tile_n} must be divisible by {_cshuffle_stride} when accumulate=False"
            )

    # ── LDS sizing (pure Python; no MLIR Context needed) ─────────────────────
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
    lds_out_bytes = (
        2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )  # f16 bytes
    lds_tid_bytes = int(tile_m) * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    _lds_tid_byte_off2 = max(lds_x_bytes, lds_out_bytes)

    if waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _cur_lds = allocator._align(allocator.ptr, 128)
        if _cur_lds < _min_lds:
            allocator.ptr += _min_lds - _cur_lds

    _store_nt = 2 if not bool(accumulate) else 0

    if True:

        @flyc.kernel(known_block_size=[total_threads, 1, 1])
        def moe_gemm2(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            # Unwrap tensor handles to memrefs for ext dialect helpers (e.g. fly.extract_aligned_pointer_as_index).
            #arg_out = arg_out.value
            #arg_x = arg_x.value
            #arg_w = arg_w.value
            #arg_scale_x = arg_scale_x.value
            #arg_scale_w = arg_scale_w.value
            #arg_sorted_token_ids = arg_sorted_token_ids.value
            #arg_expert_ids = arg_expert_ids.value
            #arg_sorted_weights = arg_sorted_weights.value
            #arg_num_valid_ids = arg_num_valid_ids.value
            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            n_in = arith.index_cast(T.index, i32_n_in)
            k_in = arith.index_cast(T.index, i32_k_in)
            size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
            x_elem = (
                T.bf16
                if is_bf16
                else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
            )
            # For int4/int4_bf16, weights are stored as packed bytes (i8) and unpacked in-kernel.
            w_elem = (
                T.i8
                if (is_int4 or is_int4_bf16)
                else (
                    T.bf16
                    if is_bf16
                    else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
                )
            )
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)

            acc_init = (
                arith.constant_vector(0, T.i32x4)
                if is_int8
                else arith.constant_vector(0.0, T.f32x4)
            )

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.index(experts * model_dim)
            kpack_bytes = 8 if (is_int4 or is_int4_bf16) else 16
            w_elem_bytes = 1 if (is_int4 or is_int4_bf16) else elem_bytes
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in,
                kpack_bytes=kpack_bytes,
                elem_bytes=w_elem_bytes,
            )
            layout_b = b_layout.layout_b

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Split-K dispatch axis (see MOE_SPLITK_AXIS). "y" only applies to the
            # gy-first remap branch; kz/k_start are then derived from blockIdx.y
            # inside that branch. Otherwise blockIdx.z encodes group*k_batch+kz.
            _sk_axis_y2 = (
                _is_splitk2
                and MOE_SPLITK_AXIS == "y"
                and MOE_XCD_REMAP
                and not MOE_XCD_REMAP_GX
            )
            if _is_splitk2 and not _sk_axis_y2:
                _bidz_sk = gpu.block_id("z")
                kz_sk = _bidz_sk % fx.Index(k_batch)
                _sk_group_z = _bidz_sk // fx.Index(k_batch)
                k_start = kz_sk * fx.Index(_k_per_batch2)
            else:
                _sk_group_z = None
                k_start = None
            # persist_m loop: each WG serially sweeps persist_m M-blocks (the inverse
            # of split-K). persist follows the M dim (see launch grid): under remap=gy
            # it folds into the group (z) index, under remap=gx / no-remap into the M
            # index (block_id.y). persist_m==1 leaves IR/behavior unchanged.
            _persist2 = persist_m > 1
            if _persist2:
                _c_pm2 = arith.constant(persist_m, index=True)
                _for_persist2 = scf.ForOp(
                    arith.constant(0, index=True),
                    _c_pm2,
                    arith.constant(1, index=True),
                )
                _for_ip2 = ir.InsertionPoint(_for_persist2.body)
                _for_ip2.__enter__()
                _mi2 = _for_persist2.induction_variable
                # Anti-LICM: keep tid-derived LDS addresses loop-variant so they
                # are recomputed (not hoisted + pinned) each persist iteration.
                tx = _persist_anti_licm_tx(tx, _mi2)

            def _pm_fold2(v):
                return (v * _c_pm2 + _mi2) if _persist2 else v

            # Align with Aiter launch mapping:
            # - blockIdx.x -> N dimension (tile along model_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            if MOE_XCD_REMAP and MOE_XCD_REMAP_GX:
                # gx-first: grid=(NUM_XCD, gy, ceil(gx/NUM_XCD)). N (n_tile) split across
                # XCDs; bx (sorted M) walks the full gy on block_id.y => always in range.
                # The rounding overruns by (n_tile), so guard by < n_tiles (folded into
                # blk_valid below).
                _xcd = gpu.block_id("x")  # 0..NUM_XCD-1 (one per XCD, chunk=1)
                bx = _pm_fold2(gpu.block_id("y"))  # tile along sorted M (full range)
                _bz = _sk_group_z if _is_splitk2 else gpu.block_id("z")  # n_tile group
                by = _bz * fx.Index(MOE_NUM_XCD) + _xcd  # tile along model_dim (n-tile)
                # n_tiles must match the launcher's gx = n_in // tile_n.
                _n_tiles_i32 = arith.index_cast(T.i32, n_in // fx.Index(tile_n))
                by_in_range = arith.cmpi(
                    arith.CmpIPredicate.ult, arith.index_cast(T.i32, by), _n_tiles_i32
                )
            elif MOE_XCD_REMAP:
                # grid=(NUM_XCD, gx, ceil(gy/NUM_XCD)); see MOE_XCD_REMAP note.
                _xcd = gpu.block_id("x")  # 0..NUM_XCD-1 (one per XCD, chunk=1)
                if _sk_axis_y2:
                    # Split-K on the -2 axis: blockIdx.y = n_tile*k_batch + kz, so a
                    # tile's kz partials are adjacent in dispatch order (output stays
                    # L2-hot). blockIdx.z is the pure expert-block group.
                    _bycomb = gpu.block_id("y")
                    by = _bycomb // fx.Index(k_batch)  # tile along model_dim (n-tile)
                    kz_sk = _bycomb % fx.Index(k_batch)
                    k_start = kz_sk * fx.Index(_k_per_batch2)
                    _bz = gpu.block_id("z")  # expert-block group
                else:
                    by = gpu.block_id("y")  # tile along model_dim (n-tile)
                    _bz = (
                        _sk_group_z
                        if _is_splitk2
                        else _pm_fold2(gpu.block_id("z"))
                    )  # expert-block group (persist folds into the group dim)
                bx = _bz * fx.Index(MOE_NUM_XCD) + _xcd  # tile along sorted M
                # The XCD remap rounds the grid up, so bx can run past the real
                # number of expert blocks (size_expert_ids). Flag those pure-padding
                # blocks here and fold into blk_valid below so they exit via the
                # whole-kernel gate (no OOB, no wasted buffer/gmem work).
                bx_in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, bx),
                    i32_size_expert_ids_in,
                )
            else:
                by = gpu.block_id("x")  # tile along model_dim
                bx = _pm_fold2(gpu.block_id("y"))  # tile along sorted M

            _xcd_debug_print(2, bx, by)

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                (
                    T.bf16
                    if is_bf16
                    else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
                ),
                shape=(lds_total_elems,),
            )
            lds_x = lds_x_ptr.get()
            # Alias the same underlying LDS bytes as f16/bf16 for epilogue shuffle.
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    (T.bf16 if out_is_bf16 else T.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # lds_tid: sorted_token_ids preloaded into LDS for epilogue
            lds_tid = SmemPtr(
                base_ptr, lds_x_ptr.byte_offset + _lds_tid_byte_off2, T.i32, shape=(tile_m,)
            ).get()

            # Buffer resources.
            # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
            # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
            c_topk = fx.Index(topk)

            # X(A2): [tokens*topk, inter_dim] bytes = tokens*topk*k*elem_bytes
            x_nbytes_idx = (tokens_in * c_topk) * k_in * arith.index(int(elem_bytes))
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_idx
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            if _use_wptr64:
                from flydsl._mlir.dialects import fly as _fly

                _llvm_ptr_ty_as1 = ir.Type.parse("!llvm.ptr<1>")
                w_base_ptr = _fly.extract_aligned_pointer_as_index(
                    _llvm_ptr_ty_as1, arg_w
                )
                _kpack_elems_b = int(kpack_bytes) // int(w_elem_bytes)
                _stride_nlane_b = arith.constant(_kpack_elems_b, index=True)
                _stride_klane_b = arith.constant(
                    16 * _kpack_elems_b, index=True
                )
                _stride_k0_b = arith.constant(
                    64 * _kpack_elems_b, index=True
                )
                _c_k_bytes_b = k_in * arith.constant(
                    int(w_elem_bytes), index=True
                )
                _c_k0_b = _c_k_bytes_b // arith.constant(64, index=True)
                _stride_n0_b = _c_k0_b * _stride_k0_b
            else:
                w_base_ptr = None
                _llvm_ptr_ty_as1 = None
                _stride_n0_b = _stride_k0_b = None
                _stride_klane_b = _stride_nlane_b = None

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = tokens_in * n_in * fx.Index(out_elem_bytes)
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in * fx.Index(topk) * n_in * fx.Index(out_elem_bytes)
                )
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_idx
            )
            # scale_x: fp16/bf16 path ignores (implicit scale=1.0); int4_bf16 also uses 1.0.
            if is_f16_or_bf16:
                sx_rsrc = None
            else:
                # scale_x (A2 scale): [tokens*topk] f32 -> bytes = tokens*topk*4
                sx_nbytes_idx = (tokens_in * c_topk) * fx.Index(4)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_idx
                )
            # scale_w: fp16/bf16 (non-int4) path ignores; int4_bf16 needs dequant scale.
            if not needs_scale_w:
                sw_rsrc = None
            else:
                # scale_w: [experts*model_dim] f32 (static shape in practice)
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (CK-style padded length)
            sorted_nbytes_idx = size_expert_ids_in * fx.Index(tile_m) * fx.Index(4)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_idx,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_idx
            )

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            eid_nbytes_idx = size_expert_ids_in * fx.Index(4)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_idx
            )
            bx_m = bx * fx.Index(tile_m)

            # Early-exit guard (as in 2ce65fb): some routing paths can produce extra/garbage
            # expert blocks beyond `num_valid_ids`. Skip those blocks entirely to avoid OOB.
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            if MOE_XCD_REMAP and not MOE_XCD_REMAP_GX:
                # Only in-range blocks load num_valid_ids and run the token check; padding
                # blocks (bx >= size_expert_ids) skip the load and yield invalid directly.
                # num_valid_i32 is also yielded out (the gated body uses it in precompute_row);
                # padding blocks yield a dummy 0 that is never consumed (blk_valid is False).
                _i1_ty = ir.IntegerType.get_signless(1)
                _rng_if = scf.IfOp(bx_in_range, [_i1_ty, T.i32], has_else=True)
                with _if_then(_rng_if):
                    numids_rsrc = buffer_ops.create_buffer_resource(
                        arg_num_valid_ids,
                        max_size=False,
                        num_records_bytes=fx.Index(4),
                    )
                    _nv = buffer_ops.buffer_load(
                        numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
                    )
                    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, _nv)
                    scf.YieldOp([tok_ok, _nv])
                with _if_else(_rng_if):
                    scf.YieldOp(
                        [arith.constant(0, type=_i1_ty), arith.constant(0, type=T.i32)]
                    )
                blk_valid = _rng_if.results[0]
                num_valid_i32 = _rng_if.results[1]
            elif MOE_XCD_REMAP and MOE_XCD_REMAP_GX:
                # gx-first: bx (sorted M) is always in range; the rounding overruns by
                # (n_tile). Only in-range (by) blocks load num_valid_ids and run the token
                # check; padding blocks (by >= n_tiles) skip the load and yield invalid.
                _i1_ty = ir.IntegerType.get_signless(1)
                _rng_if = scf.IfOp(by_in_range, [_i1_ty, T.i32], has_else=True)
                with _if_then(_rng_if):
                    numids_rsrc = buffer_ops.create_buffer_resource(
                        arg_num_valid_ids,
                        max_size=False,
                        num_records_bytes=fx.Index(4),
                    )
                    _nv = buffer_ops.buffer_load(
                        numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
                    )
                    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, _nv)
                    scf.YieldOp([tok_ok, _nv])
                with _if_else(_rng_if):
                    scf.YieldOp(
                        [arith.constant(0, type=_i1_ty), arith.constant(0, type=T.i32)]
                    )
                blk_valid = _rng_if.results[0]
                num_valid_i32 = _rng_if.results[1]
            else:
                # non-remap: bx (sorted M) is always in range, so load num_valid directly.
                numids_rsrc = buffer_ops.create_buffer_resource(
                    arg_num_valid_ids,
                    max_size=False,
                    num_records_bytes=fx.Index(4),
                )
                num_valid_i32 = buffer_ops.buffer_load(
                    numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
                )
                blk_valid = arith.cmpi(
                    arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32
                )

            def _moe_gemm2_then_body(_by_tile):
                # `_by_tile` is the N-tile (model_dim tile) index this invocation
                # computes. With persist_n<=1 it is simply `by`; with persist_n>1
                # the caller sweeps `persist_n` consecutive tiles per WG.
                # Expert id for this M tile.
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=T.i32
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                n_idx = fx.Index(model_dim)
                expert_off_idx = expert_idx * n_idx  # index

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16/bf16 we require 16B.
                if is_f16_or_bf16:
                    if bytes_per_thread_x % 16 != 0:
                        raise ValueError(
                            f"[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                        )
                    x_load_bytes = 16
                else:
                    if bytes_per_thread_x % 16 == 0:
                        x_load_bytes = 16
                    elif bytes_per_thread_x % 8 == 0:
                        x_load_bytes = 8
                    elif bytes_per_thread_x % 4 == 0:
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                        )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // arith.index(4)
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = fx.Index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = fx.Int32(topk)
                mask24 = fx.Int32(0xFFFFFF)
                # Sentinel clamp uses `tokens` as the upper bound: t_valid = (t < tokens).
                tokens_i32 = arith.index_cast(T.i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                vec4_x = T.vec(4, x_elem)

                def load_x(idx_i32):
                    if x_load_bytes == 16:
                        idx_elem = (
                            idx_i32 if elem_bytes == 1 else (idx_i32 * fx.Index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                            elem_bytes=elem_bytes,
                            cache_modifier=_X_CM,
                        )
                    if x_load_bytes == 8:
                        return buffer_ops.buffer_load(
                            x_rsrc, idx_i32, vec_width=2, dtype=T.i32,
                            cache_modifier=_X_CM,
                        )
                    return buffer_ops.buffer_load(
                        x_rsrc, idx_i32, vec_width=1, dtype=T.i32,
                        cache_modifier=_X_CM,
                    )

                # decode routed token once (per thread's M-slice) and build a base offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                    )
                    t_i32 = fused_i & mask24
                    s_i32 = fused_i >> 24
                    # aiter moe_sorting uses sentinel token_id == tokens for padding.
                    # Do NOT rely on buffer OOB semantics for A2/scale loads; explicitly mask.
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = t_valid & s_valid
                    t_safe = ts_valid.select(t_i32, fx.Int32(0))
                    s_safe = ts_valid.select(s_i32, fx.Int32(0))
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                    # Base row offset in dword units: row_ts_idx * (k_in/4)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (base_k * arith.index(int(elem_bytes))) // fx.Index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(T.i32x4, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(x_vec)
                        else:
                            parts.append(x_vec)
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(lane_id, layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)

                row_a_lds = lane_mod_16
                # A-side kpack is always 16 bytes; kpack_bytes is B-side (may be 8 for int4).
                a_kpack_elems = 16 // elem_bytes
                col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
                col_offset_base_bytes = (
                    col_offset_base
                    if elem_bytes == 1
                    else (col_offset_base * arith.index(int(elem_bytes)))
                )

                # Dynamic N tiling within block.
                by_n = _by_tile * fx.Index(tile_n)
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = fx.Index(n_per_wave)
                wave_n_id = wave_id % fx.Index(num_waves)
                n_tile_base = wave_n_id * c_n_per_wave

                # Precompute (n_blk, n_intra) for B, and col indices for output.
                n_intra_list = []
                n_blk_list = []
                col_g_list = []
                c_n0_static = experts * model_dim // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.index(ni * 16)
                    col_g = by_n + n_tile_base + offset + lane_mod_16
                    col_g_list.append(col_g)

                    row_w = expert_off_idx + col_g
                    coord_w = fx.idx2crd(row_w, layout_n_blk_intra)
                    n_blk_list.append(fx.get(coord_w, 0))
                    n_intra_list.append(fx.get(coord_w, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)
                _num_b_loads2 = k_unroll * 2 * num_acc_n
                # NOTE: the async-copy barriers below use vmcnt=0 (full drain) for
                # the same X DMA read-before-write reason as stage1.

                # --- B Load Logic (K64) ---
                def _load_b_gep_vec_i32(*, n_blk, k0, k1, n_intra, load_bytes):
                    elem_idx_i = (
                        n_blk * _stride_n0_b
                        + k0 * _stride_k0_b
                        + k1 * _stride_klane_b
                        + n_intra * _stride_nlane_b
                    )
                    byte_idx = elem_idx_i * arith.constant(
                        int(w_elem_bytes), index=True
                    )
                    ptr = llvm.GEPOp(
                        _llvm_ptr_ty_as1,
                        w_base_ptr,
                        [arith.index_cast(T.i64, byte_idx)],
                        [-2147483648],
                        T.i8,
                        llvm.GEPNoWrapFlags.none,
                    ).result
                    vec_width = load_bytes // 4
                    return llvm.LoadOp(
                        T.vec(vec_width, T.i32), ptr, alignment=load_bytes
                    ).result

                def _load_b_pack_k32_via_gep(*, base_k, ki_step, n_blk, n_intra):
                    c64_idx = arith.constant(64, index=True)
                    base_k_bytes = base_k * arith.constant(
                        int(w_elem_bytes), index=True
                    )
                    k0_base = base_k_bytes // c64_idx
                    k0 = k0_base + arith.constant(ki_step // 2, index=True)
                    k1 = lane_div_16

                    raw_vec = _load_b_gep_vec_i32(
                        n_blk=n_blk, k0=k0, k1=k1, n_intra=n_intra,
                        load_bytes=int(kpack_bytes),
                    )
                    half = ki_step % 2
                    if half == 0:
                        d0 = vector.extract(
                            raw_vec, static_position=[0], dynamic_position=[]
                        )
                        d1 = vector.extract(
                            raw_vec, static_position=[1], dynamic_position=[]
                        )
                    else:
                        d0 = vector.extract(
                            raw_vec, static_position=[2], dynamic_position=[]
                        )
                        d1 = vector.extract(
                            raw_vec, static_position=[3], dynamic_position=[]
                        )
                    v2 = vector.from_elements(T.vec(2, T.i32), [d0, d1])
                    v64 = vector.bitcast(T.vec(1, T.i64), v2)
                    return vector.extract(
                        v64, static_position=[0], dynamic_position=[]
                    )

                def load_b_pack(base_k, ki_step, ni):
                    if _use_wptr64:
                        return _load_b_pack_k32_via_gep(
                            base_k=base_k,
                            ki_step=ki_step,
                            n_blk=n_blk_list[ni],
                            n_intra=n_intra_list[ni],
                        )
                    return load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ki_step,
                        n_blk=n_blk_list[ni],
                        n_intra=n_intra_list[ni],
                        lane_div_16=lane_div_16,  # 0..3
                        elem_type=w_elem,
                        kpack_bytes=kpack_bytes,
                        elem_bytes=w_elem_bytes,
                        unpack_int4=is_int4,
                        cache_modifier=b_nt,
                    )

                def load_b_tile(base_k):
                    """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base.

                    Returns a list of length `k_unroll`, where each entry is a tuple:
                      (packs_half0[ni], packs_half1[ni])  for the K64 micro-step.
                    """
                    if is_int4_bf16:
                        # W4A16: 2-phase load+unpack for VMEM latency hiding
                        raw_data = []
                        for ku in range_constexpr(k_unroll):
                            raw_ku = []
                            for ni in range_constexpr(num_acc_n):
                                raw = load_b_raw_w4a16(
                                    buffer_ops,
                                    arith,
                                    vector,
                                    arg_b=arg_w,
                                    b_rsrc=w_rsrc,
                                    layout_b=layout_b,
                                    base_k=base_k,
                                    ku=ku,
                                    n_blk=n_blk_list[ni],
                                    n_intra=n_intra_list[ni],
                                    lane_div_16=lane_div_16,
                                    elem_type=w_elem,
                                    kpack_bytes=kpack_bytes,
                                    cache_modifier=b_nt,
                                )
                                raw_ku.append(raw)
                            raw_data.append(raw_ku)
                        b_tile = []
                        for ku in range_constexpr(k_unroll):
                            packs0 = []
                            packs1 = []
                            for ni in range_constexpr(num_acc_n):
                                b0, b1 = unpack_b_w4a16(raw_data[ku][ni], arith, vector)
                                packs0.append(b0)
                                packs1.append(b1)
                            b_tile.append((packs0, packs1))
                        return b_tile
                    b_tile = []
                    for ku in range_constexpr(k_unroll):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            b0 = load_b_pack(base_k, ki0, ni)
                            b1 = load_b_pack(base_k, ki1, ni)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif x_load_bytes == 8:
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                            )
                        else:
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                            )

                # ---- Async DMA path for stage2: global -> LDS directly ----
                if use_async_copy:
                    if bytes_per_thread_x % 16 != 0:
                        raise ValueError(
                            f"use_async_copy requires bytes_per_thread_x divisible by 16, "
                            f"got {bytes_per_thread_x} (tile_m={tile_m}, tile_k={tile_k}, "
                            f"elem_bytes={elem_bytes}). Try larger tile_m or tile_k."
                        )
                    _dma_bytes2 = 16
                    _wave_size2 = 64
                    _num_dma_loads2 = bytes_per_thread_x // 16

                    def dma_x_tile_to_lds(base_k, lds_base):
                        c4_idx = fx.Index(4)
                        base_k_div4 = (base_k * arith.index(int(elem_bytes))) // c4_idx

                        lds_ptr_i64 = None
                        for i in range_constexpr(_num_dma_loads2):
                            row_local_i = x_row_local[i]
                            col_local_i32_i = x_col_local_i32[i]
                            col_local_sw = swizzle_xor16(
                                row_local_i, col_local_i32_i * c4_idx, k_blocks16
                            )
                            row_k_dw = x_row_base_div4[i] + base_k_div4
                            global_byte_idx = row_k_dw * c4_idx + col_local_sw
                            global_offset = arith.index_cast(T.i32, global_byte_idx)

                            if i == 0:
                                lds_byte_off = lds_base * arith.index(int(elem_bytes))
                                lds_addr = (
                                    memref.extract_aligned_pointer_as_index(lds_x)
                                    + lds_byte_off
                                    + wave_id * arith.index(_wave_size2 * _dma_bytes2)
                                )
                                lds_ptr_i64 = rocdl.readfirstlane(
                                    T.i64, arith.index_cast(T.i64, lds_addr)
                                )
                            else:
                                lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                    total_threads * _dma_bytes2, type=T.i64
                                )

                            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                            lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                            rocdl.raw_ptr_buffer_load_lds(
                                x_rsrc,
                                lds_ptr,
                                arith.constant(_dma_bytes2, type=T.i32),
                                global_offset,
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                                arith.constant(_X_DMA_AUX, type=T.i32),
                            )

                    def prefetch_x_to_lds(base_k, lds_base):
                        dma_x_tile_to_lds(base_k, lds_base)

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base_bytes, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(int(elem_bytes)))
                    )
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def compute_tile(
                    acc_in,
                    b_tile_in,
                    lds_base,
                    *,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                    a1_prefetch=None,
                ):
                    acc_list = list(acc_in)
                    mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
                    mfma_fn = (
                        mfma_i32_k32
                        if is_int8
                        else (
                            mfma_f32_bf16_k16
                            if is_bf16
                            else (
                                rocdl.mfma_f32_16x16x16f16
                                if is_f16
                                else rocdl.mfma_f32_16x16x32_fp8_fp8
                            )
                        )
                    )

                    epilogue_pf = None
                    if prefetch_epilogue:
                        expert_off_pf = expert_off_idx
                        sw_pf = []
                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            row_w_idx = expert_off_pf + col_g
                            sw_pf.append(
                                fx.Float32(1.0)
                                if not needs_scale_w
                                else buffer_ops.buffer_load(
                                    sw_rsrc, row_w_idx, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                )
                            )
                        # Also prefetch per-row routed/topk weights (sorted_weights) when enabled.
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * fx.Index(4)
                            ii_idx_list_pf = [fx.Index(ii) for ii in range(4)]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.index(mi * 16)
                                for ii in range_constexpr(4):
                                    row_off_pf = (
                                        lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    )
                                    row_in_tile_pf = mi_base_pf + row_off_pf
                                    sorted_row_pf = bx_m + row_in_tile_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc,
                                            sorted_row_pf,
                                            vec_width=1,
                                            dtype=T.f32,
                                        )
                                    )
                        epilogue_pf = (sw_pf, tw_pf)

                    def _i64_to_v4f16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.f16x4, v1)

                    def _i64_to_v4i16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.i16x4, v1)

                    def _combine_i64_to_v4i32(lo, hi):
                        v2 = vector.from_elements(T.i64x2, [lo, hi])
                        return vector.bitcast(T.i32x4, v2)

                    def mfma_k64(acc0, a0, a1, b0, b1):
                        if _use_k64_mfma:
                            a_v4 = _combine_i64_to_v4i32(a0, a1)
                            b_v4 = _combine_i64_to_v4i32(b0, b1)
                            return _mfma_i32_16x16x64_i8(a_v4, b_v4, acc0)
                        if is_f16:
                            a0v = _i64_to_v4f16(a0)
                            a1v = _i64_to_v4f16(a1)
                            b0v = _i64_to_v4f16(b0)
                            b1v = _i64_to_v4f16(b1)
                            acc1 = mfma_fn(mfma_res_ty, [a0v, b0v, acc0, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc1, 0, 0, 0])
                        if is_bf16:
                            a0v = _i64_to_v4i16(a0)
                            a1v = _i64_to_v4i16(a1)
                            b0v = _i64_to_v4i16(b0)
                            b1v = _i64_to_v4i16(b1)
                            acc1 = mfma_fn(mfma_res_ty, [a0v, b0v, acc0, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc1, 0, 0, 0])
                        acc1 = mfma_fn(mfma_res_ty, [a0, b0, acc0, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1, b1, acc1, 0, 0, 0])

                    if _use_k128_mfma_fp8:
                        for ku128 in range_constexpr(k_unroll // 2):
                            ku0 = ku128 * 2
                            ku1 = ku0 + 1
                            b0_p0, b0_p1 = b_tile_in[ku0]
                            b1_p0, b1_p1 = b_tile_in[ku1]
                            ki64_0 = arith.index(ku0 * 64)
                            ki64_1 = arith.index(ku1 * 64)
                            col_base0 = col_offset_base_bytes + ki64_0
                            col_base1 = col_offset_base_bytes + ki64_1

                            _s_setprio(1)
                            for mi in range_constexpr(m_repeat):
                                mi_val = arith.index(mi * 16)
                                curr_row_a_lds = row_a_lds + mi_val
                                if (a0_prefetch is not None) and (ku0 == 0) and (mi == 0):
                                    a0, a1 = a0_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_base)
                                if (a1_prefetch is not None) and (ku1 == 1) and (mi == 0):
                                    a2, a3 = a1_prefetch
                                else:
                                    a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_base)
                                a_128 = _pack_i64x4_to_i32x8(a0, a1, a2, a3)

                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    b_128 = _pack_i64x4_to_i32x8(
                                        b0_p0[ni], b0_p1[ni], b1_p0[ni], b1_p1[ni],
                                    )
                                    acc_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        T.f32x4,
                                        [a_128, b_128, acc_list[acc_idx],
                                         0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
                                    )
                            _s_setprio(0)
                    else:
                        for ku in range_constexpr(k_unroll):
                            b_packs0, b_packs1 = b_tile_in[ku]
                            ki64 = arith.index(ku * 64)
                            col_base = col_offset_base_bytes + ki64

                            _s_setprio(1)
                            for mi in range_constexpr(m_repeat):
                                mi_val = arith.index(mi * 16)
                                curr_row_a_lds = row_a_lds + mi_val

                                if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                                    a0, a1 = a0_prefetch
                                elif (a1_prefetch is not None) and (ku == 1) and (mi == 0):
                                    a0, a1 = a1_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(
                                        curr_row_a_lds, col_base, lds_base
                                    )

                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    acc_list[acc_idx] = mfma_k64(
                                        acc_list[acc_idx],
                                        a0,
                                        a1,
                                        b_packs0[ni],
                                        b_packs1[ni],
                                    )
                            _s_setprio(0)
                    return acc_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.index(tile_m * lds_stride)
                lds_base_cur = fx.Index(0)
                lds_base_nxt = lds_tile_elems

                rocdl.sched_barrier(0)

                # def hot_loop_scheduler():
                #     mfma_group = num_acc_n
                #     # K64 micro-step: 2x K32 MFMA per accumulator update.
                #     mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                #     mfma_per_iter = 2 * mfma_group
                #     sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                #     rocdl.sched_dsrd(2)
                #     rocdl.sched_mfma(1)
                #     rocdl.sched_mfma(1)
                #     if num_acc_n < 4:
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(2)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(2)
                #         rocdl.sched_vmem(1)

                #     dswr_tail = num_x_loads
                #     if dswr_tail > sche_iters:
                #         dswr_tail = sche_iters
                #     dswr_start = sche_iters - dswr_tail
                #     for sche_i in range_constexpr(sche_iters):
                #         rocdl.sched_mfma(mfma_group // 2)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(mfma_group // 2)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(mfma_group)
                #         if sche_i >= dswr_start - 1:
                #             rocdl.sched_dswr(1)
                #     rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    mfma_group = num_acc_n

                    # Use equivalent K=32 MFMA count for pipeline time slots,
                    # regardless of actual MFMA variant (K=64/K=128 have proportionally
                    # higher latency, so the scheduling window is the same).
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group

                    if use_async_copy:
                        a_vmem_load = max(1, tile_m // 32)
                        b_vmem_total = _num_b_loads2
                        vmem_count = b_vmem_total + 2 + a_vmem_load

                        rocdl.sched_vmem(a_vmem_load)
                        rocdl.sched_mfma(a_vmem_load)

                        if tile_m == 16:
                            for i in range_constexpr(2):
                                rocdl.sched_dsrd(1)
                                rocdl.sched_mfma(1)
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(1)
                            _tail_vmem = max(0, vmem_count - a_vmem_load - 2)
                            for i in range_constexpr(_tail_vmem):
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(1)
                        else:
                            _dsrd_vmem_iters = a_vmem_load * 4
                            for i in range_constexpr(_dsrd_vmem_iters):
                                rocdl.sched_dsrd(1)
                                rocdl.sched_mfma(1)
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(mfma_group)
                            _tail_vmem = max(0, vmem_count - _dsrd_vmem_iters)
                            for i in range_constexpr(_tail_vmem):
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(mfma_group)
                    else:
                        mfma_per_iter = 2 * mfma_group
                        sche_iters = (
                            0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                        )

                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(1)
                        if tile_m == 16:
                            rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)
                        if tile_m == 16:
                            rocdl.sched_vmem(1)
                        if num_acc_n < 4:
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(1)
                            if tile_m == 16:
                                rocdl.sched_vmem(1)
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(1)
                            if tile_m == 16:
                                rocdl.sched_vmem(1)
                            rocdl.sched_mfma(1)

                        dswr_tail = num_x_loads
                        if dswr_tail > sche_iters:
                            dswr_tail = sche_iters
                        dswr_start = sche_iters - dswr_tail
                        for sche_i in range_constexpr(sche_iters):
                            rocdl.sched_vmem(1)
                            rocdl.sched_mfma(mfma_group)
                            rocdl.sched_dsrd(1)
                            rocdl.sched_mfma(mfma_group)
                            if sche_i >= dswr_start - 1:
                                rocdl.sched_dswr(1)

                    rocdl.sched_barrier(0)

                # Preload sorted_token_ids into lds_tid for epilogue
                _c_tile_m_idx = arith.constant(tile_m, index=True)
                _tid_in_range = arith.cmpi(arith.CmpIPredicate.ult, tx, _c_tile_m_idx)
                _if_tid = scf.IfOp(_tid_in_range)
                with _if_then(_if_tid):
                    _tid_row = bx_m + tx
                    _tid_val = buffer_ops.buffer_load(
                        sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                    )
                    _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                    vector.store(_tid_vec1, lds_tid, [tx])

                # Prologue.
                # Split-K: start at this CTA's K-slice base (k_start); == 0 otherwise.
                k0 = k_start if _is_splitk2 else fx.Index(0)
                if use_async_copy:
                    prefetch_x_to_lds(k0, lds_base_cur)
                    b_cur = load_b_tile(k0)
                else:
                    x_regs0 = load_x_tile(k0)
                    b_cur = load_b_tile(k0)
                    store_x_tile_to_lds(x_regs0, lds_base_cur)
                if use_async_copy:
                    _barrier(vmcnt=0, lgkmcnt=0)
                else:
                    _barrier(lgkmcnt=0)

                acc = [acc_init] * (num_acc_n * m_repeat)
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt

                # Cross-tile A0+A1 LDS prefetch: issue ds_reads back-to-back.
                _a1_col_bytes2 = col_offset_base_bytes + arith.index(64)
                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base_bytes, lds_base_pong
                )
                a1_prefetch_pong = (
                    lds_load_packs_k64(row_a_lds, _a1_col_bytes2, lds_base_pong)
                    if k_unroll >= 2
                    else None
                )

                # Main loop: process K tiles in 2-tile ping-pong steps.
                #
                # IMPORTANT: for odd number of K tiles, leave **1** tail tile; for even, leave **2**.
                # Otherwise the 2-tile tail below would double-count the last tile when num_tiles is odd
                # (e.g. inter_dim=192, tile_k=64 -> 3 tiles).
                # Split-K: each CTA sweeps only its K-slice [k_start, k_start+K_per_batch).
                num_k_tiles_py = int(_k_per_batch2) // int(tile_k)
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if k_main2_py < 0:
                    k_main2_py = 0
                # End of this CTA's K-slice (== k_in when not split-K, so IR is identical).
                _k_slice_end = (
                    (k_start + fx.Index(_k_per_batch2)) if _is_splitk2 else k_in
                )

                c2_tile_k = arith.index(tile_k * 2)
                pair_iters = k_main2_py // (int(tile_k) * 2)
                _use_pool = (_bp_depth >= 2) and not use_async_copy

                def _k_of(t):
                    return k0 + arith.index(t * tile_k)

                _bpool = []
                if _use_pool:
                    # Front-load N weight tiles into the ring pool. Reuse the
                    # prologue's b_cur as tile 0, then load tiles 1..N-1.
                    _pooln = min(_bp_depth, num_k_tiles_py)
                    _bpool = [b_cur] + [load_b_tile(_k_of(t)) for t in range(1, _pooln)]
                    # Hint the scheduler to issue the front-loaded pool weight loads
                    # up front (each load_b_tile = k_unroll*num_acc_n*2 buffer_loads).
                    if _pooln > 1:
                        rocdl.sched_vmem((_pooln - 1) * k_unroll * num_acc_n * 2)
                # X HBM prefetch pool: tile 0's X already loaded+stored in prologue;
                # front-load tiles 1..depth into registers (ds_write stays 1-ahead).
                _use_xpool = (_xp_depth >= 2) and not use_async_copy
                _xpool = []
                if _use_xpool:
                    _xpool = [load_x_tile(_k_of(t))
                              for t in range(1, min(_xp_depth + 1, num_k_tiles_py))]
                    if len(_xpool) > 0:
                        rocdl.sched_vmem(len(_xpool) * num_x_loads)
                for pair_i in range_constexpr(pair_iters):
                    k_iv = arith.index(pair_i * (tile_k * 2))
                    if _is_splitk2:
                        k_iv = k_start + k_iv
                    next_k1 = k_iv + tile_k
                    if use_async_copy:
                        prefetch_x_to_lds(next_k1, lds_base_ping)
                    elif _use_xpool:
                        x_regs_ping = _xpool.pop(0)  # tile 2i+1
                        _xtl0 = (2 * pair_i + 1) + _xp_depth
                        if _xtl0 < num_k_tiles_py:
                            _xpool.append(load_x_tile(_k_of(_xtl0)))
                    else:
                        x_regs_ping = load_x_tile(next_k1)
                    if _use_pool:
                        # take head (tile 2i), issue tail load (tile 2i+depth)
                        _b0 = _bpool.pop(0)
                        _tl0 = 2 * pair_i + _bp_depth
                        if _tl0 < num_k_tiles_py:
                            _bpool.append(load_b_tile(_k_of(_tl0)))
                    else:
                        b_ping = load_b_tile(next_k1)
                        _b0 = b_cur

                    acc, _ = compute_tile(
                        acc, _b0, lds_base_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a1_prefetch=a1_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    a1_prefetch_pong = None
                    if not use_async_copy:
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    if use_async_copy:
                        _barrier(vmcnt=0, lgkmcnt=0)
                    else:
                        _barrier(lgkmcnt=0)

                    # Cross-tile prefetch for the ping tile we are about to compute.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_ping
                    )
                    a1_prefetch_ping = (
                        lds_load_packs_k64(row_a_lds, _a1_col_bytes2, lds_base_ping)
                        if k_unroll >= 2
                        else None
                    )

                    next_k2 = k_iv + c2_tile_k
                    if use_async_copy:
                        prefetch_x_to_lds(next_k2, lds_base_pong)
                    elif _use_xpool:
                        x_regs_pong = _xpool.pop(0)  # tile 2i+2
                        _xtl1 = (2 * pair_i + 2) + _xp_depth
                        if _xtl1 < num_k_tiles_py:
                            _xpool.append(load_x_tile(_k_of(_xtl1)))
                    else:
                        x_regs_pong = load_x_tile(next_k2)
                    if _use_pool:
                        _b1 = _bpool.pop(0)
                        _tl1 = 2 * pair_i + 1 + _bp_depth
                        if _tl1 < num_k_tiles_py:
                            _bpool.append(load_b_tile(_k_of(_tl1)))
                    else:
                        b_next = load_b_tile(next_k2)
                        _b1 = b_ping

                    acc, _ = compute_tile(
                        acc, _b1, lds_base_ping,
                        a0_prefetch=a0_prefetch_ping,
                        a1_prefetch=a1_prefetch_ping,
                    )
                    a0_prefetch_ping = None
                    a1_prefetch_ping = None
                    if not use_async_copy:
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                    hot_loop_scheduler()
                    if use_async_copy:
                        _barrier(vmcnt=0, lgkmcnt=0)
                    else:
                        _barrier(lgkmcnt=0)

                    # Cross-tile prefetch for the next pong tile.
                    a0_prefetch_pong = lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_pong
                    )
                    a1_prefetch_pong = (
                        lds_load_packs_k64(row_a_lds, _a1_col_bytes2, lds_base_pong)
                        if k_unroll >= 2
                        else None
                    )

                    if _use_pool:
                        pass  # pool self-manages b state
                    else:
                        b_cur = b_next

                if odd_k_tiles:
                    # Tail: single remaining tile (already in `b_cur`/pool / `lds_base_pong`).
                    _bt = _bpool.pop(0) if _use_pool else b_cur
                    acc, epilogue_pf = compute_tile(
                        acc,
                        _bt,
                        lds_base_pong,
                        prefetch_epilogue=True,
                        a0_prefetch=a0_prefetch_pong,
                        a1_prefetch=a1_prefetch_pong,
                    )
                else:
                    # Tail: 2 remaining tiles (k_end == k_in for non-split-K).
                    k_tail1 = _k_slice_end - tile_k
                    if use_async_copy:
                        prefetch_x_to_lds(k_tail1, lds_base_ping)
                    elif _use_xpool:
                        x_regs_ping = _xpool.pop(0)  # last tile's X (held in pool)
                    else:
                        x_regs_ping = load_x_tile(k_tail1)
                    if _use_pool:
                        _bt0 = _bpool.pop(0)   # tile N-2 (held in pool)
                        b_ping = _bpool.pop(0)  # tile N-1 (held in pool)
                    else:
                        _bt0 = b_cur
                        b_ping = load_b_tile(k_tail1)

                    acc, _ = compute_tile(
                        acc, _bt0, lds_base_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a1_prefetch=a1_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    a1_prefetch_pong = None
                    if not use_async_copy:
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    if use_async_copy:
                        _barrier(vmcnt=0, lgkmcnt=0)
                    else:
                        _barrier(lgkmcnt=0)

                    # Epilogue tile with sw prefetch.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_ping
                    )
                    a1_prefetch_ping = (
                        lds_load_packs_k64(row_a_lds, _a1_col_bytes2, lds_base_ping)
                        if k_unroll >= 2
                        else None
                    )
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_ping,
                        lds_base_ping,
                        prefetch_epilogue=True,
                        a0_prefetch=a0_prefetch_ping,
                        a1_prefetch=a1_prefetch_ping,
                    )

                # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
                # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.
                expert_off = expert_off_idx
                mask24_i32 = fx.Int32(0xFFFFFF)
                model_i32 = fx.Int32(model_dim)
                topk_i32_v = topk_i32

                zero_i32 = fx.Int32(0)
                out_aux_i32 = fx.Int32(_OUT_ATOMIC_AUX)  # buffer-atomic cachepolicy (bypass L2 when set)
                c2_i32 = fx.Int32(2)  # 2B element size for f16/bf16
                mask_even_i32 = fx.Int32(
                    0xFFFFFFFE
                )  # align element index to even for half2 atomics

                e_vec = _e_vec

                def atomic_add_f16x2(val_f16x2, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f16x2,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        out_aux_i32,
                    )

                sw_pf = None
                tw_pf = None
                if epilogue_pf is not None:
                    sw_pf, tw_pf = epilogue_pf

                # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
                if sw_pf is not None:
                    sw_vals = sw_pf
                else:
                    sw_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_w_idx = expert_off + col_g
                        sw_vals.append(
                            fx.Float32(1.0)
                            if not needs_scale_w
                            else buffer_ops.buffer_load(
                                sw_rsrc, row_w_idx, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                            )
                        )

                if out_is_f32:
                    # origin/dev_a16w4: f32 output uses scalar f32 atomics and skips CShuffle/LDS.
                    c4_i32 = fx.Int32(4)

                    def atomic_add_f32(val_f32, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f32,
                            out_rsrc,
                            byte_off_i32,
                            zero_i32,
                            out_aux_i32,
                        )

                    def _stage2_row_atomic(*, mi: int, ii: int, row_in_tile, row):
                        fused2 = memref.load(lds_tid, [row_in_tile])
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24

                        # Mask sentinel (token_id==tokens, slot==topk) to avoid OOB scale_x loads.
                        # For invalid rows, force sx=0 so they contribute exactly 0 to output.
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        ts_ok = t_ok & s_ok
                        t2_safe = ts_ok.select(t2, fx.Int32(0))
                        s2_safe = ts_ok.select(s2, fx.Int32(0))
                        ts2 = t2_safe * topk_i32_v + s2_safe
                        sx = (
                            arith.select(ts_ok, fx.Float32(1.0), fx.Float32(0.0))
                            if is_f16_or_bf16
                            else arith.select(
                                ts_ok,
                                buffer_ops.buffer_load(
                                    sx_rsrc, ts2, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                                ),
                                fx.Float32(0.0),
                            )
                        )

                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = ts_ok.select(tw_pf[tw_idx], fx.Float32(0.0))
                            else:
                                tw = arith.select(
                                    ts_ok,
                                    buffer_ops.buffer_load(
                                        sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                                    ),
                                    fx.Float32(0.0),
                                )
                            # sx and tw are both per-row; fold into one product here
                            # (still 0 for invalid rows since both carry the ts_ok mask).
                            sxtw = sx * tw
                        else:
                            sxtw = sx

                        idx0 = (
                            t2_safe * model_i32
                        )  # i32 element index base (safe for sentinel rows)

                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            csw = sxtw * sw_vals[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            if is_int8:
                                v = arith.sitofp(T.f32, v)
                            v = v * csw
                            col_i32 = arith.index_cast(T.i32, col_g)
                            idx_elem = idx0 + col_i32
                            byte_off = idx_elem * c4_i32
                            atomic_add_f32(v, byte_off)

                    default_epilog(
                        arith=arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=_stage2_row_atomic,
                    )
                else:
                    if lds_out is None:
                        raise RuntimeError(
                            "FLYDSL_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                        )

                    # For bf16 global atomics (gfx942 only), precompute the output base address.
                    # gfx950+ has buffer_atomic_pk_add_bf16, so bf16 uses buffer atomics there.
                    out_base_idx = None
                    if _needs_global_atomic_bf16:
                        from flydsl._mlir.dialects import fly as _fly
                        _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                        #out_base_idx = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty, arg_out)
                        out_base_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty, arg_out)
                        out_base_i64 = llvm.ptrtoint(T.i64, out_base_ptr)
                        out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                        lds_col_remap=None,
                    ):
                        # sx (per-token activation scale): decode+load inline. No
                        # safe-clamp/select mask -- invalid/padding rows are dropped by
                        # store_pair's row-validity guard (sentinel row_id==tokens makes
                        # ts2 OOB -> buffer returns 0), so masking here is redundant.
                        fused2 = memref.load(lds_tid, [row_in_tile])
                        ts2 = (fused2 & mask24_i32) * topk_i32_v + (fused2 >> 24)
                        sx = (
                            fx.Float32(1.0)
                            if is_f16_or_bf16
                            else buffer_ops.buffer_load(
                                sx_rsrc, ts2, vec_width=1, dtype=T.f32, cache_modifier=_SCALE_CM
                            )
                        )

                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = tw_pf[tw_idx]
                            else:
                                tw = buffer_ops.buffer_load(
                                    sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                                )
                            # Both sx (per-token) and tw (routed weight) are per-row,
                            # i.e. invariant across ni; fold them into ONE product here
                            # ("first time") so the ni loop drops a multiply. sw stays
                            # per-column. FP mul is non-associative so the compiler
                            # cannot hoist this itself.
                            sxtw = sx * tw
                        else:
                            sxtw = sx

                        # Depth-1 software pipeline over ni: pre-extract v for ni=0,
                        # then each step scales/stores the current element while
                        # pre-extracting the NEXT one. Overlaps the AccVGPR-read +
                        # scale latency across ni, with only ONE extra value live
                        # (unlike a full pool -> no VGPR pressure / occupancy loss).
                        _v_cur = vector.extract(
                            acc[mi * num_acc_n], static_position=[ii], dynamic_position=[]
                        )
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            # Per-(row,ni) scale, independent of v -> computed off the
                            # AccVGPR-read critical path (overlaps the extract latency).
                            csw = sxtw * sw_vals[ni]
                            v = _v_cur
                            if ni + 1 < num_acc_n:
                                _v_cur = vector.extract(
                                    acc[mi * num_acc_n + ni + 1],
                                    static_position=[ii], dynamic_position=[],
                                )
                            if is_int8:
                                v = arith.sitofp(T.f32, v)
                            v = v * csw
                            v_out = arith.trunc_f(out_elem(), v)

                            # Interleaved LDS layout (when enabled): remap the column so
                            # that a reader thread's n_reps fragments land adjacent -> the
                            # CShuffle read side can use one wide ds_read instead of n_reps
                            # narrow reads (halves the read-address VALU + ds_read count).
                            lds_col = (
                                lds_col_remap(col_local)
                                if lds_col_remap is not None
                                else col_local
                            )
                            lds_idx = row_base_lds + lds_col
                            vec1_out = T.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        fused2 = memref.load(lds_tid, [row_local])
                        row_i32 = arith.index_cast(T.i32, row)
                        # Guard ONLY row < num_valid_ids (rejects the uninitialized garbage
                        # tail). Sentinel padding rows (token==tokens) within num_valid are
                        # neutralized elsewhere (sx OOB->0 => value 0; out idx OOB->dropped),
                        # so t<tokens & s<topk are redundant.
                        row_valid = arith.cmpi(
                            arith.CmpIPredicate.ult, row_i32, num_valid_i32
                        )
                        t = fused2 & mask24_i32
                        # Hoist the per-row output base index (idx0) up-front (invariant
                        # across ni/columns); store_pair then only adds the column. Keeping
                        # it here (not deferred into the store guard) lets the scheduler
                        # overlap the index math across rows.
                        if bool(accumulate):
                            idx0 = t * model_i32
                        else:
                            s = fused2 >> 24
                            idx0 = (t * topk_i32_v + s) * model_i32
                        return (idx0, row_valid)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        idx0 = row_ctx
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_elem = idx0 + col_i32
                        idx_elem_even = idx_elem & mask_even_i32
                        if _needs_global_atomic_bf16:
                            # gfx942: no buffer_atomic_pk_add_bf16, use global atomicrmw fadd
                            if bool(accumulate):
                                byte_off = idx_elem_even * c2_i32
                                byte_off_idx = arith.index_cast(T.index, byte_off)
                                ptr_addr_idx = out_base_idx + byte_off_idx
                                out_ptr = buffer_ops.create_llvm_ptr(
                                    ptr_addr_idx, address_space=1
                                )
                                out_ptr_v = (
                                    out_ptr._value
                                    if hasattr(out_ptr, "_value")
                                    else out_ptr
                                )
                                frag_v = (
                                    frag._value if hasattr(frag, "_value") else frag
                                )
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd,
                                    out_ptr_v,
                                    frag_v,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent",
                                    alignment=4,
                                )
                            else:
                                buffer_ops.buffer_store(
                                    frag, out_rsrc, idx_elem_even,
                                    cache_modifier=_store_nt,
                                )
                        else:
                            # f16, or bf16 on gfx950+ (has buffer_atomic_pk_add_bf16)
                            byte_off = idx_elem_even * c2_i32
                            if bool(accumulate):
                                atomic_add_f16x2(frag, byte_off)
                            else:
                                buffer_ops.buffer_store(
                                    frag, out_rsrc, idx_elem_even,
                                    cache_modifier=_store_nt,
                                )

                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        block_size=total_threads,
                        e_vec=e_vec,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=(T.bf16 if out_is_bf16 else T.f16),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        interleave_n_reps=True,
                    )

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                if _persist_n > 1:
                    # persist_n N-loop: this WG serially sweeps `_persist_n`
                    # consecutive N-tiles for its M-block. `by` is the per-WG
                    # N-tile counter (launch N-grid was divided by _persist_n),
                    # so the base tile is by*_persist_n. X for this M-block is
                    # identical across the tiles, so re-streaming it keeps the
                    # activation L2-resident (reused) across the sweep. A barrier
                    # separates iterations so tile j's LDS is fully consumed
                    # before tile j+1 reuses the same static LDS.
                    _pn_base = by * fx.Index(_persist_n)
                    for _pn_j in range_constexpr(_persist_n):
                        #if _pn_j > 0:
                        #    gpu.barrier()
                        _moe_gemm2_then_body(_pn_base + fx.Index(_pn_j))
                else:
                    _moe_gemm2_then_body(by)

            if _persist2:
                # barrier so LDS from this M-block is fully consumed before the next
                # persist iteration reuses it; then close the persist loop.
                gpu.barrier()
                scf.YieldOp([])
                _for_ip2.__exit__(None, None, None)

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    @flyc.jit
    def launch_moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_in = arith.index_cast(T.index, i32_n_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        # persist_n (>1) folds `_persist_n` consecutive N-tiles into each WG, so
        # the N-tile grid dim shrinks by that factor (each WG loops the merged
        # tiles internally). _persist_n<=1 keeps the original expression / IR.
        gx = (
            n_in // fx.Index(tile_n)
            if _persist_n <= 1
            else n_in // fx.Index(int(tile_n) * int(_persist_n))
        )
        gy = size_expert_ids_in
        # Split-K multiplies the launch z-dim by k_batch. The kernel decodes
        # blockIdx.z as (group * k_batch + kz); k_batch==1 leaves the grid intact.
        _sk_kb = fx.Index(k_batch)

        # persist_m (>1) divides the M-carrying grid dim so each WG serially sweeps
        # persist_m M-blocks. persist follows M: under remap=gy that is the group
        # (z) dim; under remap=gx / no-remap it is the M dim (gy on block_id.y).
        def _pm_ceil(dim):
            if persist_m == 1:
                return dim
            return (dim + fx.Index(persist_m - 1)) // fx.Index(persist_m)

        if MOE_XCD_REMAP:
            if MOE_XCD_REMAP_GX:
                # (NUM_XCD, expert_blocks, ceil(n_tiles/NUM_XCD)): split N across XCDs
                # so each XCD keeps its weight n_tile slice L2-resident.
                gz = (gx + fx.Index(MOE_NUM_XCD - 1)) // fx.Index(MOE_NUM_XCD)
                gz = gz * _sk_kb if _is_splitk2 else gz
                grid_dims = (fx.Index(MOE_NUM_XCD), _pm_ceil(gy), gz)
            else:
                # (NUM_XCD, n_tiles, ceil(expert_blocks/NUM_XCD)) for XCD L2 locality
                gz = (gy + fx.Index(MOE_NUM_XCD - 1)) // fx.Index(MOE_NUM_XCD)
                if _is_splitk2 and MOE_SPLITK_AXIS == "y":
                    # Split-K on the -2 axis: fold k_batch into the n_tile dim so a
                    # tile's kz partials dispatch adjacently (output stays L2-hot).
                    grid_dims = (fx.Index(MOE_NUM_XCD), gx * _sk_kb, gz)
                else:
                    gz = gz * _sk_kb if _is_splitk2 else gz
                    grid_dims = (fx.Index(MOE_NUM_XCD), gx, _pm_ceil(gz))
        else:
            grid_dims = (
                (gx, gy, _sk_kb) if _is_splitk2 else (gx, _pm_ceil(gy), 1)
            )

        moe_gemm2(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=grid_dims,
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm2


# MoE Reduction Kernel (reduce sum over topk dimension)
@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
    use_mask: bool = False,
):
    """Compile a reduction kernel that sums over the topk dimension.

    Input:  X [tokens, topk, model_dim]
            valid_mask [tokens, topk] (optional, if use_mask=True)
    Output: Y [tokens, model_dim]

    This kernel performs: Y[t, d] = sum(X[t, :, d]) for all t, d.
    When use_mask=True, only sums slots where valid_mask[t,k]=1.
    Used in conjunction with compile_moe_gemm2(accumulate=False) to avoid atomic contention.
    """
    # Kernel Config
    BLOCK_SIZE = 256
    VEC_WIDTH = 8
    if dtype_str == "f32":
        elem_type_tag = "f32"
    elif dtype_str == "f16":
        elem_type_tag = "f16"
    elif dtype_str == "bf16":
        elem_type_tag = "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    def compute_type():
        return T.f32

    def i32_type():
        return T.i32

    def i8_type():
        return T.i8

    def elem_type():
        ty = (
            T.f32
            if elem_type_tag == "f32"
            else (T.f16 if elem_type_tag == "f16" else T.bf16)
        )
        return ty() if callable(ty) else ty

    if True:

        @flyc.kernel
        def moe_reduction_kernel(
            X: fx.Tensor,
            Y: fx.Tensor,
            valid_mask: fx.Tensor,
            i32_m_tokens: fx.Int32,
        ):
            m_tokens = arith.index_cast(T.index, i32_m_tokens)
            c_topk = fx.Index(topk)
            c_model_dim = fx.Index(model_dim)
            c_elem_bytes = arith.index(4 if dtype_str == "f32" else 2)
            x_nbytes_idx = m_tokens * c_topk * c_model_dim * c_elem_bytes
            y_nbytes_idx = m_tokens * c_model_dim * c_elem_bytes
            mask_nbytes_idx = m_tokens * c_topk
            x_rsrc = buffer_ops.create_buffer_resource(
                X, max_size=False, num_records_bytes=x_nbytes_idx
            )
            y_rsrc = buffer_ops.create_buffer_resource(
                Y, max_size=False, num_records_bytes=y_nbytes_idx
            )
            mask_rsrc = buffer_ops.create_buffer_resource(
                valid_mask, max_size=False, num_records_bytes=mask_nbytes_idx
            )

            token_idx = gpu.block_id("x")
            tile_idx = gpu.block_id("y")
            tid = gpu.thread_id("x")

            # Guard: token in range
            token_i32 = arith.index_cast(i32_type(), token_idx)
            m_tokens_i32 = arith.index_cast(i32_type(), m_tokens)
            tok_ok = arith.cmpi(arith.CmpIPredicate.ult, token_i32, m_tokens_i32)
            _if_tok = scf.IfOp(tok_ok)
            with _if_then(_if_tok):
                tile_cols = BLOCK_SIZE * VEC_WIDTH
                c_tile_cols = fx.Index(tile_cols)
                c_vecw = fx.Index(VEC_WIDTH)

                col_base = tile_idx * c_tile_cols + tid * c_vecw

                # Guard: any work in bounds
                col_ok = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(i32_type(), col_base),
                    arith.index_cast(i32_type(), c_model_dim),
                )
                _if_col = scf.IfOp(col_ok)
                with _if_then(_if_col):
                    # Fast path: full vector in-bounds -> vector load/store.
                    end_ok = arith.cmpi(
                        arith.CmpIPredicate.ule,
                        arith.index_cast(i32_type(), col_base + c_vecw),
                        arith.index_cast(i32_type(), c_model_dim),
                    )
                    _if_full = scf.IfOp(end_ok, has_else=True)
                    with _if_then(_if_full):
                        c0_i8 = arith.constant(0, type=i8_type())
                        token_base = token_idx * c_topk
                        for lane in range_constexpr(VEC_WIDTH):
                            col = col_base + fx.Index(lane)
                            a = arith.constant(0.0, type=compute_type())
                            for k in range_constexpr(topk):
                                k_idx = fx.Index(k)
                                x_idx = (token_base + k_idx) * c_model_dim + col
                                x_idx_i32 = arith.index_cast(i32_type(), x_idx)
                                if use_mask:
                                    m_idx = token_base + k_idx
                                    m_idx_i32 = arith.index_cast(i32_type(), m_idx)
                                    mv = buffer_ops.buffer_load(
                                        mask_rsrc,
                                        m_idx_i32,
                                        vec_width=1,
                                        dtype=i8_type(),
                                    )
                                    mv_ok = arith.cmpi(
                                        arith.CmpIPredicate.ne, mv, c0_i8
                                    )
                                    v = arith.select(
                                        mv_ok,
                                        buffer_ops.buffer_load(
                                            x_rsrc,
                                            x_idx_i32,
                                            vec_width=1,
                                            dtype=elem_type(),
                                        ),
                                        arith.constant(0.0, type=elem_type()),
                                    )
                                else:
                                    v = buffer_ops.buffer_load(
                                        x_rsrc,
                                        x_idx_i32,
                                        vec_width=1,
                                        dtype=elem_type(),
                                    )
                                if dtype_str in ("f16", "bf16"):
                                    v = arith.extf(compute_type(), v)
                                a = a + v
                            v = a
                            if dtype_str in ("f16", "bf16"):
                                v = arith.trunc_f(elem_type(), v)
                            y_idx = token_idx * c_model_dim + col
                            y_idx_i32 = arith.index_cast(i32_type(), y_idx)
                            buffer_ops.buffer_store(
                                v, y_rsrc, y_idx_i32, cache_modifier=_OUT_STORE_CM
                            )

                    with _if_else(_if_full):
                        # Tail path: scalar load/store per lane.
                        for lane in range_constexpr(VEC_WIDTH):
                            col = col_base + fx.Index(lane)
                            lane_ok = arith.cmpi(
                                arith.CmpIPredicate.ult,
                                arith.index_cast(i32_type(), col),
                                arith.index_cast(i32_type(), c_model_dim),
                            )
                            _if_lane = scf.IfOp(lane_ok)
                            with _if_then(_if_lane):
                                a = arith.constant(0.0, type=compute_type())
                                token_base = token_idx * c_topk
                                c0_i8 = arith.constant(0, type=i8_type())
                                for k in range_constexpr(topk):
                                    k_idx = fx.Index(k)
                                    x_idx = (token_base + k_idx) * c_model_dim + col
                                    x_idx_i32 = arith.index_cast(i32_type(), x_idx)
                                    if use_mask:
                                        m_idx = token_base + k_idx
                                        m_idx_i32 = arith.index_cast(i32_type(), m_idx)
                                        mv = buffer_ops.buffer_load(
                                            mask_rsrc,
                                            m_idx_i32,
                                            vec_width=1,
                                            dtype=i8_type(),
                                        )
                                        mv_ok = arith.cmpi(
                                            arith.CmpIPredicate.ne, mv, c0_i8
                                        )
                                        v = arith.select(
                                            mv_ok,
                                            buffer_ops.buffer_load(
                                                x_rsrc,
                                                x_idx_i32,
                                                vec_width=1,
                                                dtype=elem_type(),
                                            ),
                                            arith.constant(0.0, type=elem_type()),
                                        )
                                    else:
                                        v = buffer_ops.buffer_load(
                                            x_rsrc,
                                            x_idx_i32,
                                            vec_width=1,
                                            dtype=elem_type(),
                                        )
                                    if dtype_str in ("f16", "bf16"):
                                        v = arith.extf(compute_type(), v)
                                    a = a + v

                                out = a
                                if dtype_str in ("f16", "bf16"):
                                    out = arith.trunc_f(elem_type(), out)
                                y_idx = token_idx * c_model_dim + col
                                y_idx_i32 = arith.index_cast(i32_type(), y_idx)
                                buffer_ops.buffer_store(
                                    out, y_rsrc, y_idx_i32, cache_modifier=_OUT_STORE_CM
                                )

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    tile_size = BLOCK_SIZE * VEC_WIDTH
    gy_static = (model_dim + tile_size - 1) // tile_size

    @flyc.jit
    def launch_moe_reduction(
        X: fx.Tensor,
        Y: fx.Tensor,
        valid_mask: fx.Tensor,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, i32_m_tokens)
        moe_reduction_kernel(X, Y, valid_mask, i32_m_tokens).launch(
            grid=(gx, gy_static, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction


# MoE GEMM2 Execution Modes
class MoeGemm2Mode:
    """Execution mode for MoE GEMM2."""

    ATOMIC = "atomic"  # Use atomic accumulation (default)
    REDUCE = "reduce"  # Use non-atomic write + reduce kernel


class _MoeGemm2ReduceWrapper:
    """Wrapper combining GEMM2 (no atomics) with reduction kernel.

    This wrapper handles the intermediate buffer allocation and orchestrates
    the two-phase computation:
    1. GEMM2 outputs to [tokens*topk, model_dim] without atomics
    2. Reduce sums over topk to produce [tokens, model_dim]
    """

    def __init__(
        self,
        gemm2_exe,
        reduce_exe,
        topk: int,
        model_dim: int,
        out_dtype_str: str = "f16",
        use_mask: bool = False,
        zero_intermediate: bool = True,
    ):
        self._gemm2_exe = gemm2_exe
        self._reduce_exe = reduce_exe
        self._topk = topk
        self._model_dim = model_dim
        self._out_dtype_str = out_dtype_str
        self._use_mask = use_mask
        self._zero_intermediate = zero_intermediate

    def _get_torch_dtype(self):
        """Convert dtype string to torch dtype."""
        import torch

        dtype_map = {
            "f16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "f32": torch.float32,
        }
        return dtype_map.get(self._out_dtype_str, torch.float16)

    def __call__(
        self,
        arg_out,
        arg_x,
        arg_w,
        arg_scale_x,
        arg_scale_w,
        arg_sorted_token_ids,
        arg_expert_ids,
        arg_sorted_weights,
        arg_num_valid_ids,
        tokens_in,
        n_in,
        k_in,
        size_expert_ids_in,
        valid_mask=None,
        stream=None,
    ):
        """Execute GEMM2 + reduce.

        Args match moe_gemm2 kernel signature (see compile_moe_gemm2).
        """
        import torch

        if stream is None:
            stream = torch.cuda.current_stream()
        intermediate = torch.empty(
            tokens_in * self._topk,
            self._model_dim,
            device=arg_out.device,
            dtype=self._get_torch_dtype(),
        )
        if self._zero_intermediate and not self._use_mask:
            intermediate.zero_()
        # Phase 1: GEMM2 (no atomics) -> [tokens*topk, model_dim]
        self._gemm2_exe(
            intermediate.view(-1),
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            tokens_in,
            n_in,
            k_in,
            size_expert_ids_in,
            stream,
        )
        # Phase 2: Reduce over topk -> [tokens, model_dim]
        X = intermediate.view(tokens_in, self._topk, self._model_dim)
        Y = arg_out.view(tokens_in, self._model_dim)
        if not self._use_mask:
            if valid_mask is not None:
                logging.warning(
                    "valid_mask provided but use_mask=False; ignoring valid_mask"
                )
            valid_mask = torch.empty(
                (0, self._topk), device=arg_out.device, dtype=torch.uint8
            )
        self._reduce_exe(X, Y, valid_mask, tokens_in, stream)

    @property
    def mode(self) -> str:
        """Return the execution mode."""
        return MoeGemm2Mode.REDUCE


def compile_moe_gemm2_ex(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    # Extended parameters for mode control
    mode: str = MoeGemm2Mode.ATOMIC,
    valid_mask=None,
    zero_intermediate: bool = True,
):
    """Compile MoE GEMM2 kernel with optional reduction.

    This is the extended interface that supports explicit mode control.

    Args:
        mode: Execution mode selection:
            - "atomic": Use atomic accumulation (original behavior)
            - "reduce": Use non-atomic write + reduce kernel

        zero_intermediate: If all output slots are valid,
            set False to increase performance

    Returns:
        Compiled executable (either wrapped or raw depending on mode).
    """
    # Compile based on mode
    if mode == MoeGemm2Mode.REDUCE:
        # Determine if we need masked reduction
        use_mask = valid_mask is not None

        # Compile GEMM2 with accumulate=False
        gemm2_exe = compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            group_size=group_size,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            accumulate=False,
        )
        # Compile reduction kernel with masking support
        out_s = str(out_dtype).strip().lower()
        if out_s in ("f16", "fp16", "half"):
            dtype_str = "f16"
        elif out_s in ("bf16", "bfloat16"):
            dtype_str = "bf16"
        else:
            dtype_str = "f32"
        reduce_exe = compile_moe_reduction(
            topk=topk,
            model_dim=model_dim,
            dtype_str=dtype_str,
            use_mask=use_mask,
        )
        return _MoeGemm2ReduceWrapper(
            gemm2_exe=gemm2_exe,
            reduce_exe=reduce_exe,
            topk=topk,
            model_dim=model_dim,
            out_dtype_str=dtype_str,
            use_mask=use_mask,
            zero_intermediate=zero_intermediate,
        )
    else:
        # Compile GEMM2 with accumulate=True (atomic mode)
        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            group_size=group_size,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            accumulate=True,
        )
