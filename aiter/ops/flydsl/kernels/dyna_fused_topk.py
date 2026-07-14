# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL ``dyna_fused_topk`` -- per-token *dynamic* top-k softmax router.

A per-token *dynamic* variant of the MoE router top-k softmax: on top of the
usual ``softmax -> top-k`` it takes a per-token ``dyna_k`` tensor, keeps each
token's first ``dyna_k`` experts, optionally renormalizes the kept weights to
sum to 1, and pads the unused tail of the ``max_topk``-wide output row.

Reference semantics (per token ``t``)::

    probs          = softmax(gating_output, dim=-1)        # [T, E]
    w, idx         = topk(probs, max_topk, dim=-1)         # [T, max_topk] desc
    k              = clamp(dyna_k[t], 1, max_topk)
    valid[t, j]    = j < k
    kept_sum       = (w * valid).sum(-1)
    out_w[t, j]    = (w[t, j] / kept_sum if renormalize else w[t, j])
                     if valid else 0.0
    out_id[t, j]   = idx[t, j] if valid else pad_id

Two layouts are emitted (chosen by :func:`_resolve_layout`): a sub-warp
multi-token fast path when ``E`` factors into ``VPT*TPT``, and a one-wave (64
lanes) per-token fallback otherwise (e.g. E=192). Both pick the top-``max_topk``
with ``max_topk`` constexpr-unrolled arg-max passes, each guarded by a
wave-uniform ``if j < dyna_k`` so the dropped tail's work is skipped.

Public entrypoint: :func:`build_dyna_fused_topk_module` returns a
``@flyc.jit`` launcher ``launch(gating, dyna_k, out_w, out_id, pad_id, rows,
num_blocks, stream)``; wrap it from ``flydsl_dyna_fused_topk``.

The id tail (``j >= dyna_k``) is set to ``pad_id`` (the wrapper defaults it to
``num_experts``, the sentinel ``moe_sorting`` skips), so dropped experts are not
routed and stage-1/stage-2 compute is saved.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, range_constexpr
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import T, Int32

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf

__all__ = [
    "build_dyna_fused_topk_module",
    "dyna_topk_tokens_per_block",
    "BLOCK_THREADS",
    "WAVES_PER_BLOCK",
]

WARP_SIZE = 64
# A block runs WAVES_PER_BLOCK wavefronts; tokens/block depends on the layout.
BLOCK_THREADS = 256
WAVES_PER_BLOCK = BLOCK_THREADS // WARP_SIZE  # 4

# log2(e): exp(x) == exp2(x * LOG2E), so the softmax can use the hardware exp2.
_LOG2E = 1.4426950408889634

# A finite stand-in for -inf used to mask already-selected / invalid experts.
_NEG_BIG = -3.0e38
# A finite stand-in for +inf used as the "no candidate" index in the min-reduce.
_BIG_IDX = 0x7FFFFFFF


_SUPPORTED_IN_DTYPES = ("f32", "bf16", "fp16", "f16")


# Token count at/above which the "large batch" (throughput) layout is used.
# Below it the small/mid-batch layout wins (see _resolve_layout). Tuned on
# gfx942 device kernel time; the crossover sits between ~2k and ~16k tokens.
LARGE_BATCH_TOKENS = 16384


def _resolve_layout(E, large_batch=False):
    """Pick the kernel layout for ``E`` experts and batch regime ``large_batch``.

    Sub-warp fast path when ``E == VPT * THREADS_PER_TOKEN`` factors with both
    powers of two and ``TPT <= 64``; pick the largest ``VPT <= VPT_CAP``. The cap
    is tuned on gfx942 device time: 16 for ``E <= 128``, and for ``E >= 256``
    16 for small/mid batches but 32 for large batches (``T >= LARGE_BATCH_TOKENS``,
    a large-T throughput win that regresses small/mid T). Otherwise fall back to
    one wave (64 lanes) per token (e.g. E=192).

    Returns ``(use_subwarp, VPT, THREADS_PER_TOKEN, TOKENS_PER_WARP,
    TOKENS_PER_BLOCK, LOG2_TPT)``.
    """
    vpt_cap = 32 if (E >= 256 and large_batch) else 16
    for vpt in (32, 16, 8, 4, 2, 1):
        if vpt > vpt_cap or E % vpt != 0:
            continue
        tpt = E // vpt
        if tpt > WARP_SIZE or (tpt & (tpt - 1)) != 0:
            continue
        tpw = WARP_SIZE // tpt
        return True, vpt, tpt, tpw, WAVES_PER_BLOCK * tpw, tpt.bit_length() - 1
    cn = (E + WARP_SIZE - 1) // WARP_SIZE
    return False, cn, WARP_SIZE, 1, WAVES_PER_BLOCK, 6


def dyna_topk_tokens_per_block(num_experts, large_batch=False):
    """Tokens processed per thread block for ``num_experts`` / ``large_batch``
    (the layout is chosen by :func:`_resolve_layout`). The host wrapper uses this
    to size the launch grid (``num_blocks = ceil(T / TOKENS_PER_BLOCK)``)."""
    return _resolve_layout(int(num_experts), large_batch)[4]


def _build_subwarp_dyna(
    E, K, in_dtype, renormalize, VPT, TPT, TPW, TPB, LOG2_TPT,
    scoring_func="softmax",
):
    """Multi-token-per-block dynamic top-k router: each token is served by a
    ``TPT``-lane sub-warp group, lane ``expert_lane`` owning experts
    ``v*TPT + expert_lane`` for ``v in [0, VPT)``; the group leader writes.

    ``scoring_func`` selects the per-expert score: ``"softmax"`` (row-normalized
    probabilities, needs the row max/sum reductions) or ``"sigmoid"`` (per-expert
    independent ``1/(1+e^-x)``, no reductions)."""
    _OFFS = tuple(TPT >> (s + 1) for s in range(LOG2_TPT))

    @flyc.kernel
    def dyna_fused_topk_kernel(
        gating: fx.Tensor,
        dyna_k: fx.Tensor,
        out_w: fx.Tensor,
        out_id: fx.Tensor,
        pad_id: Int32,
        rows: Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        if in_dtype == "bf16":
            in_elem_ty = T.bf16
        elif in_dtype == "fp16":
            in_elem_ty = T.f16
        else:
            in_elem_ty = f32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c6_i32 = arith.constant(6, type=i32)
        c63_i32 = arith.constant(WARP_SIZE - 1, type=i32)
        c64_i32 = arith.constant(WARP_SIZE, type=i32)
        cE_i32 = arith.constant(E, type=i32)
        cK_i32 = arith.constant(K, type=i32)
        cKstride_b = arith.constant(K * 4, type=i32)
        cBIG_idx = arith.constant(_BIG_IDX, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c1_f32 = arith.constant(1.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        c_neg_big = arith.constant(_NEG_BIG, type=f32)
        c_vpt = arith.constant(VPT, type=i32)
        c_tpw = arith.constant(TPW, type=i32)
        c_tpb = arith.constant(TPB, type=i32)
        c_log2tpt = arith.constant(LOG2_TPT, type=i32)
        c_tptmask = arith.constant(TPT - 1, type=i32)
        c_false = arith.cmpi(CmpIPredicate.ne, c0_i32, c0_i32)

        g_rsrc = buffer_ops.create_buffer_resource(gating, max_size=True)
        dk_rsrc = buffer_ops.create_buffer_resource(dyna_k, max_size=True)
        ow_rsrc = buffer_ops.create_buffer_resource(out_w, max_size=True)
        oid_rsrc = buffer_ops.create_buffer_resource(out_id, max_size=True)

        bid = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        rows_i32 = ArithValue(rows)
        pad_id_i32 = ArithValue(pad_id)

        lane = tid & c63_i32
        warp_id = tid >> c6_i32
        token_in_warp = lane >> c_log2tpt
        expert_lane = lane & c_tptmask
        local_token = warp_id * c_tpw + token_in_warp
        token = bid * c_tpb + local_token
        tok_ok = arith.cmpi(CmpIPredicate.slt, token, rows_i32)
        token_safe = arith.select(tok_ok, token, c0_i32)

        def _av(x):
            return x if isinstance(x, ArithValue) else ArithValue(x)

        def grp_reduce_max(v):
            v = _av(v)
            for off in _OFFS:
                peer = v.shuffle_xor(arith.constant(off, type=i32), c64_i32)
                v = arith.maximumf(v, peer)
            return v

        def grp_reduce_add(v):
            v = _av(v)
            for off in _OFFS:
                peer = v.shuffle_xor(arith.constant(off, type=i32), c64_i32)
                v = _av(v) + _av(peer)
            return v

        e_ids = []

        def _argmax_pass(wv):
            """Fused (val, idx) arg-max over the group; ties -> smaller id."""
            lv = c_neg_big
            li = cBIG_idx
            for v in range_constexpr(VPT):
                gt = arith.cmpf(arith.CmpFPredicate.OGT, wv[v], lv)
                lv = arith.select(gt, wv[v], lv)
                li = arith.select(gt, e_ids[v], li)
            val = _av(lv)
            idx = _av(li)
            for off in _OFFS:
                ko = arith.constant(off, type=i32)
                pv = _av(val.shuffle_xor(ko, c64_i32))
                pi = _av(idx.shuffle_xor(ko, c64_i32))
                gt = arith.cmpf(arith.CmpFPredicate.OGT, pv, val)
                eq = arith.cmpf(arith.CmpFPredicate.OEQ, pv, val)
                lower = arith.cmpi(CmpIPredicate.slt, pi, idx)
                take = arith.select(gt, gt, arith.select(eq, lower, gt))
                val = arith.select(take, pv, val)
                idx = arith.select(take, pi, idx)
            new_wv = []
            for v in range_constexpr(VPT):
                is_chosen = arith.cmpi(CmpIPredicate.eq, e_ids[v], idx)
                new_wv.append(arith.select(is_chosen, c_neg_big, wv[v]))
            return val, idx, new_wv

        row_elem_base = _av(token_safe) * cE_i32

        # Strided load: lane holds experts v*TPT + expert_lane (all valid).
        x_vals = []
        for v in range_constexpr(VPT):
            e_v = arith.constant(v * TPT, type=i32) + expert_lane
            g_off = row_elem_base + _av(e_v)
            xv = buffer_ops.buffer_load(g_rsrc, g_off, vec_width=1, dtype=in_elem_ty)
            if in_dtype != "f32":
                xv = arith.extf(f32, xv)
            e_ids.append(e_v)
            x_vals.append(_av(xv))

        if scoring_func == "sigmoid":
            # sigmoid(x) is monotonic in the logit, so the top-k *selection* can
            # run directly on the raw logits and only the K winners need a
            # sigmoid (see _weight_of). This skips the row reductions *and* the
            # VPT-K sigmoids the dropped experts would otherwise cost. All lanes
            # are valid in this layout, so the logits feed selection unmasked.
            sel_v = [_av(x_vals[v]) for v in range_constexpr(VPT)]
        else:
            # Numerically-stable softmax over the TPT-lane group. softmax needs
            # the row sum regardless, so build the full prob array and select on
            # it (the prob is also the emitted weight, so _weight_of is identity).
            local_max = c_neg_big
            for v in range_constexpr(VPT):
                local_max = arith.maximumf(local_max, x_vals[v])
            row_max = grp_reduce_max(local_max)

            exps = []
            local_sum = _av(c0_f32)
            for v in range_constexpr(VPT):
                t = (x_vals[v] - row_max) * c_log2e
                ev = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                exps.append(_av(ev))
                local_sum = local_sum + _av(ev)
            sum_exp = grp_reduce_add(local_sum)
            inv_sum = _av(
                llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [sum_exp], [], [])
            )
            sel_v = [(_av(exps[v]) * inv_sum) for v in range_constexpr(VPT)]

        def _weight_of(gv):
            """Map a selected winner value to its emitted weight: identity for
            softmax (already a prob), sigmoid for the sigmoid path (the winner is
            a raw logit -- only the K winners pay a transcendental)."""
            if scoring_func == "sigmoid":
                t = (c0_f32 - _av(gv)) * c_log2e
                ev = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                return _av(
                    llvm.call_intrinsic(
                        f32, "llvm.amdgcn.rcp.f32", [c1_f32 + _av(ev)], [], []
                    )
                )
            return _av(gv)

        row_byte_base = _av(token) * cKstride_b
        k_raw = buffer_ops.buffer_load(dk_rsrc, token_safe, vec_width=1, dtype=i32)
        k_dyn = arith.minui(arith.maxsi(k_raw, c1_i32), cK_i32)
        is_leader = arith.cmpi(CmpIPredicate.eq, expert_lane, c0_i32)
        do_write = arith.select(is_leader, tok_ok, c_false)

        sel_w = []
        sel_id = []
        kept_sum = _av(c0_f32)
        for k in range_constexpr(K):
            cj = arith.constant(k, type=i32)
            cond = arith.cmpi(CmpIPredicate.slt, cj, k_dyn)
            ifk = scf.IfOp(
                cond, results_=[f32] * VPT + [f32, f32, i32], has_else=True
            )
            with ir.InsertionPoint(ifk.then_block):
                gv, gi, new_sel = _argmax_pass(sel_v)
                wj = _weight_of(gv)
                new_kept = kept_sum + _av(wj)
                scf.YieldOp(new_sel + [new_kept, wj, gi])
            with ir.InsertionPoint(ifk.else_block):
                scf.YieldOp(
                    [sel_v[v] for v in range_constexpr(VPT)]
                    + [kept_sum, c0_f32, pad_id_i32]
                )
            sel_v = [_av(ifk.results[v]) for v in range_constexpr(VPT)]
            kept_sum = _av(ifk.results[VPT])
            sel_w.append(_av(ifk.results[VPT + 1]))
            sel_id.append(_av(ifk.results[VPT + 2]))
        inv_kept = _av(
            llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [kept_sum], [], [])
        )
        _if0 = scf.IfOp(do_write)
        with ir.InsertionPoint(_if0.then_block):
            for j in range_constexpr(K):
                cj = arith.constant(j, type=i32)
                vj = arith.cmpi(CmpIPredicate.slt, cj, k_dyn)
                w_val = sel_w[j] * inv_kept if renormalize else sel_w[j]
                w_out = arith.select(vj, w_val, c0_f32)
                id_out = arith.select(vj, sel_id[j], pad_id_i32)
                byte_off = row_byte_base + arith.constant(j * 4, type=i32)
                buffer_ops.buffer_store(
                    w_out, ow_rsrc, byte_off, offset_is_bytes=True
                )
                buffer_ops.buffer_store(
                    id_out, oid_rsrc, byte_off, offset_is_bytes=True
                )
            scf.YieldOp([])

    @flyc.jit
    def launch_dyna_fused_topk(
        gating: fx.Tensor,
        dyna_k: fx.Tensor,
        out_w: fx.Tensor,
        out_id: fx.Tensor,
        pad_id: Int32,
        rows: Int32,
        num_blocks: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        grid_x = arith.index_cast(T.index, num_blocks)
        launcher = dyna_fused_topk_kernel(
            gating, dyna_k, out_w, out_id, pad_id, rows
        )
        launcher.launch(
            grid=(grid_x, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch_dyna_fused_topk


@functools.lru_cache(maxsize=None)
def build_dyna_fused_topk_module(
    num_experts: int,
    max_topk: int,
    renormalize: bool = True,
    in_dtype: str = "f32",
    large_batch: bool = False,
    scoring_func: str = "softmax",
):
    """Build (and cache) a launcher for the dynamic top-k softmax router.

    The dropped tail (``j >= dyna_k[t]``) gets ``topk_weights == 0`` and its id
    slot is written with the host scalar ``pad_id`` (the wrapper defaults it to
    ``num_experts``, the value ``moe_sorting`` skips), so the dropped experts are
    not routed and stage-1/stage-2 compute is saved.

    Parameters
    ----------
    num_experts : int
        Number of experts ``E`` (last dim of ``gating_output``). Compile-time.
        Any positive ``E`` is supported (the wavefront handles
        ``CN = ceil(E / 64)`` experts per lane).
    max_topk : int
        Maximum / padded top-k width. Must satisfy ``1 <= max_topk <= E``.
    renormalize : bool, default True
        If True, the kept weights are normalized to sum to 1 (``w_j / kept_sum``).
        If False, the kept weights are emitted as raw scores (plain truncation,
        kept sum < 1).
    scoring_func : {"softmax", "sigmoid"}, default "softmax"
        Per-expert scoring. ``"softmax"`` uses row-normalized probabilities;
        ``"sigmoid"`` uses per-expert independent ``1/(1+e^-x)`` (no row
        reductions). The top-k selection is identical either way (both are
        monotonic in the logit); only the emitted weights / the renormalize
        denominator differ.
    in_dtype : {"f32", "bf16", "fp16"}, default "f32"
        Element type of ``gating`` *as loaded by the kernel*. For "bf16"/"fp16"
        the kernel loads the 16-bit value and widens it to f32 with ``extf``
        (a lossless widening) before the softmax -- the "native low-precision"
        path that avoids a separate host-side up-cast pass + temp f32 buffer.
        The softmax/top-k math and the f32 ``out_w`` / i32 ``out_id`` outputs
        are identical regardless of ``in_dtype``.
    large_batch : bool, default False
        Select the throughput-tuned sub-warp layout (larger VPT) for ``E >= 256``
        large-batch (``T >= LARGE_BATCH_TOKENS``) launches. The host wrapper sets
        this from the token count; it only affects layout selection (see
        :func:`_resolve_layout`), never the result.
    """
    E = int(num_experts)
    K = int(max_topk)
    if E <= 0:
        raise ValueError(f"num_experts must be positive, got {E}")
    if not (1 <= K <= E):
        raise ValueError(f"max_topk must be in [1, num_experts={E}], got {K}")
    if in_dtype not in _SUPPORTED_IN_DTYPES:
        raise ValueError(
            f"in_dtype must be one of {_SUPPORTED_IN_DTYPES}, got {in_dtype!r}"
        )
    if scoring_func not in ("softmax", "sigmoid"):
        raise ValueError(
            f"scoring_func must be 'softmax' or 'sigmoid', got {scoring_func!r}"
        )
    in_dtype = "fp16" if in_dtype == "f16" else in_dtype

    use_sub, VPT, TPT, TPW, TPB, LOG2_TPT = _resolve_layout(E, large_batch)
    if use_sub:
        return _build_subwarp_dyna(
            E, K, in_dtype, renormalize, VPT, TPT, TPW, TPB, LOG2_TPT, scoring_func
        )

    CN = (E + WARP_SIZE - 1) // WARP_SIZE  # experts handled per lane

    @flyc.kernel
    def dyna_fused_topk_kernel(
        gating: fx.Tensor,   # (rows, E)        f32
        dyna_k: fx.Tensor,   # (rows,)          i32  (per-token k)
        out_w: fx.Tensor,    # (rows, max_topk) f32
        out_id: fx.Tensor,   # (rows, max_topk) i32
        pad_id: Int32,       # padding sentinel for the id tail (host scalar)
        rows: Int32,         # token count (for the tail-wave guard)
    ):
        f32 = T.f32
        i32 = T.i32
        if in_dtype == "bf16":
            in_elem_ty = T.bf16
        elif in_dtype == "fp16":
            in_elem_ty = T.f16
        else:
            in_elem_ty = f32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c6_i32 = arith.constant(6, type=i32)
        c63_i32 = arith.constant(WARP_SIZE - 1, type=i32)
        c64_i32 = arith.constant(WARP_SIZE, type=i32)
        cE_i32 = arith.constant(E, type=i32)
        cK_i32 = arith.constant(K, type=i32)
        cKstride_b = arith.constant(K * 4, type=i32)
        cBIG_idx = arith.constant(_BIG_IDX, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c1_f32 = arith.constant(1.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        c_neg_big = arith.constant(_NEG_BIG, type=f32)
        c_waves = arith.constant(WAVES_PER_BLOCK, type=i32)

        g_rsrc = buffer_ops.create_buffer_resource(gating, max_size=True)
        dk_rsrc = buffer_ops.create_buffer_resource(dyna_k, max_size=True)
        ow_rsrc = buffer_ops.create_buffer_resource(out_w, max_size=True)
        oid_rsrc = buffer_ops.create_buffer_resource(out_id, max_size=True)

        bid = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        rows_i32 = ArithValue(rows)
        pad_id_i32 = ArithValue(pad_id)

        lane = tid & c63_i32
        wave = tid >> c6_i32
        token = bid * c_waves + wave   # one wave -> one token

        def _av(x):
            return x if isinstance(x, ArithValue) else ArithValue(x)

        def wave_reduce_max(v):
            v = _av(v)
            for sh in (32, 16, 8, 4, 2, 1):
                peer = v.shuffle_xor(arith.constant(sh, type=i32), c64_i32)
                v = arith.maximumf(v, peer)
            return v

        def wave_reduce_add(v):
            v = _av(v)
            for sh in (32, 16, 8, 4, 2, 1):
                peer = v.shuffle_xor(arith.constant(sh, type=i32), c64_i32)
                v = _av(v) + _av(peer)
            return v

        in_range = arith.cmpi(CmpIPredicate.slt, token, rows_i32)

        # Tail waves (token >= rows) skip uniformly.
        _if = scf.IfOp(in_range)
        with ir.InsertionPoint(_if.then_block):
            row_elem_base = token * cE_i32

            # Lane c owns experts e_c = lane + 64*c (tail lanes invalid).
            e_ids = []
            valids = []
            x_vals = []
            for c in range_constexpr(CN):
                e_c = lane + arith.constant(c * WARP_SIZE, type=i32)
                valid_c = arith.cmpi(CmpIPredicate.slt, e_c, cE_i32)
                e_safe = arith.select(valid_c, e_c, c0_i32)
                g_off = row_elem_base + _av(e_safe)
                xc = buffer_ops.buffer_load(
                    g_rsrc, g_off, vec_width=1, dtype=in_elem_ty
                )
                if in_dtype != "f32":
                    xc = arith.extf(f32, xc)
                e_ids.append(e_c)
                valids.append(valid_c)
                x_vals.append(_av(xc))

            if scoring_func == "sigmoid":
                # Per-expert sigmoid (no reductions). Invalid tail lanes are set
                # sigmoid is monotonic in the logit, so selection runs on the
                # raw logits (invalid tail lanes masked to a negative sentinel so
                # they are never picked) and only the K winners get a sigmoid
                # (see _weight_of) -- no row reductions, no dropped-expert
                # sigmoids.
                sel_c = []
                for c in range_constexpr(CN):
                    sel_c.append(arith.select(valids[c], x_vals[c], c_neg_big))
            else:
                # Numerically-stable softmax over the 64-lane wave. softmax needs
                # the row sum, so build the full prob array and select on it
                # (the prob is also the weight, so _weight_of is identity).
                local_max = c_neg_big
                for c in range_constexpr(CN):
                    xm = arith.select(valids[c], x_vals[c], c_neg_big)
                    local_max = arith.maximumf(local_max, xm)
                row_max = wave_reduce_max(local_max)

                exps = []
                local_sum = _av(c0_f32)
                for c in range_constexpr(CN):
                    t = (x_vals[c] - row_max) * c_log2e
                    ev = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.exp2.f32", [t], [], []
                    )
                    ev = arith.select(valids[c], ev, c0_f32)
                    exps.append(ev)
                    local_sum = local_sum + _av(ev)
                sum_exp = wave_reduce_add(local_sum)
                inv_sum = _av(
                    llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [sum_exp], [], [])
                )

                sel_c = []
                for c in range_constexpr(CN):
                    wv = _av(exps[c]) * inv_sum
                    sel_c.append(arith.select(valids[c], wv, c_neg_big))

            def _weight_of(gv):
                """Selected winner -> emitted weight: identity for softmax (a
                prob), sigmoid for the sigmoid path (winner is a raw logit -- only
                the K winners pay a transcendental)."""
                if scoring_func == "sigmoid":
                    t = (c0_f32 - _av(gv)) * c_log2e
                    ev = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.exp2.f32", [t], [], []
                    )
                    return _av(
                        llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [c1_f32 + _av(ev)], [], []
                        )
                    )
                return _av(gv)

            row_byte_base = token * cKstride_b

            def _argmax_pass(wc):
                """Fused (val, idx) arg-max over the 64-lane wave; ties -> smaller
                id. Returns (gv, gi, new_wc) with the winner masked to -inf."""
                lv = c_neg_big
                li = cBIG_idx
                for c in range_constexpr(CN):
                    gt = arith.cmpf(arith.CmpFPredicate.OGT, wc[c], lv)
                    lv = arith.select(gt, wc[c], lv)
                    li = arith.select(gt, e_ids[c], li)
                v = _av(lv)
                idx = _av(li)
                for sh in (32, 16, 8, 4, 2, 1):
                    ksh = arith.constant(sh, type=i32)
                    pv = _av(v.shuffle_xor(ksh, c64_i32))
                    pi = _av(idx.shuffle_xor(ksh, c64_i32))
                    gt = arith.cmpf(arith.CmpFPredicate.OGT, pv, v)
                    eq = arith.cmpf(arith.CmpFPredicate.OEQ, pv, v)
                    lower = arith.cmpi(CmpIPredicate.slt, pi, idx)
                    take = arith.select(gt, gt, arith.select(eq, lower, gt))
                    v = arith.select(take, pv, v)
                    idx = arith.select(take, pi, idx)
                gv = v
                gi = idx
                new_wc = []
                for c in range_constexpr(CN):
                    is_chosen = arith.cmpi(CmpIPredicate.eq, e_ids[c], gi)
                    new_wc.append(arith.select(is_chosen, c_neg_big, wc[c]))
                return gv, gi, new_wc

            k_raw = buffer_ops.buffer_load(dk_rsrc, token, vec_width=1, dtype=i32)
            k_dyn = arith.minui(arith.maxsi(k_raw, c1_i32), cK_i32)

            sel_w = []
            sel_id = []
            kept_sum = _av(c0_f32)
            for k in range_constexpr(K):
                cj = arith.constant(k, type=i32)
                cond = arith.cmpi(CmpIPredicate.slt, cj, k_dyn)
                ifk = scf.IfOp(
                    cond,
                    results_=[f32] * CN + [f32, f32, i32],
                    has_else=True,
                )
                with ir.InsertionPoint(ifk.then_block):
                    gv, gi, new_sel = _argmax_pass(sel_c)
                    wj = _weight_of(gv)
                    new_kept = kept_sum + _av(wj)
                    scf.YieldOp(new_sel + [new_kept, wj, gi])
                with ir.InsertionPoint(ifk.else_block):
                    scf.YieldOp(
                        [sel_c[c] for c in range_constexpr(CN)]
                        + [kept_sum, c0_f32, pad_id_i32]
                    )
                sel_c = [_av(ifk.results[c]) for c in range_constexpr(CN)]
                kept_sum = _av(ifk.results[CN])
                sel_w.append(_av(ifk.results[CN + 1]))
                sel_id.append(_av(ifk.results[CN + 2]))

            inv_kept = _av(
                llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [kept_sum], [], [])
            )
            is_lane0 = arith.cmpi(CmpIPredicate.eq, lane, c0_i32)
            _if0 = scf.IfOp(is_lane0)
            with ir.InsertionPoint(_if0.then_block):
                for j in range_constexpr(K):
                    cj = arith.constant(j, type=i32)
                    vj = arith.cmpi(CmpIPredicate.slt, cj, k_dyn)
                    w_val = sel_w[j] * inv_kept if renormalize else sel_w[j]
                    w_out = arith.select(vj, w_val, c0_f32)
                    id_out = arith.select(vj, sel_id[j], pad_id_i32)
                    byte_off = row_byte_base + arith.constant(j * 4, type=i32)
                    buffer_ops.buffer_store(
                        w_out, ow_rsrc, byte_off, offset_is_bytes=True
                    )
                    buffer_ops.buffer_store(
                        id_out, oid_rsrc, byte_off, offset_is_bytes=True
                    )
                scf.YieldOp([])

            scf.YieldOp([])

    @flyc.jit
    def launch_dyna_fused_topk(
        gating: fx.Tensor,
        dyna_k: fx.Tensor,
        out_w: fx.Tensor,
        out_id: fx.Tensor,
        pad_id: Int32,
        rows: Int32,
        num_blocks: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        grid_x = arith.index_cast(T.index, num_blocks)
        launcher = dyna_fused_topk_kernel(
            gating, dyna_k, out_w, out_id, pad_id, rows
        )
        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_dyna_fused_topk
