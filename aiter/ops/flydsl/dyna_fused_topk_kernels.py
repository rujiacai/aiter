# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL MoE router (dynamic top-k softmax) APIs.

Exposes :func:`flydsl_dyna_fused_topk`, the per-token *dynamic* top-k softmax
router. It produces the same ``(topk_weights, topk_ids)`` output contract as the
static top-k softmax router but additionally takes a per-token ``dyna_k`` tensor
and emits a fixed ``max_topk``-wide row.

For each token the dropped tail (``j >= dyna_k[t]``) gets weight ``0`` and its
id slot written with ``pad_id`` (default ``num_experts``). ``moe_sorting``
already drops the ``num_experts`` sentinel (its histogram has a
``num_experts + 1`` padding column that is never emitted), so the dropped
experts are not routed and stage-1/stage-2 compute is saved -- a drop-in for
``moe_sorting`` with no sort-kernel change. When ``renormalize=True`` the kept
weights are normalized to sum to 1.
"""

from __future__ import annotations

import torch

from .kernels.dyna_fused_topk import (
    LARGE_BATCH_TOKENS,
    build_dyna_fused_topk_module,
    dyna_topk_tokens_per_block,
)
from .moe_kernels import _run_compiled

__all__ = ["flydsl_dyna_fused_topk"]


_SUPPORTED_IN_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_TORCH_TO_IN_DTYPE = {
    torch.float32: "f32",
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
}


def flydsl_dyna_fused_topk(
    gating_output: torch.Tensor,
    dyna_k: torch.Tensor,
    max_topk: int,
    *,
    topk_weights: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    renormalize: bool = True,
    pad_id: int = None,
    native: bool = True,
    scoring_func: str = "softmax",
    stream: torch.cuda.Stream = None,
):
    """Per-token dynamic top-k softmax router.

    For each token ``t`` this computes ``softmax(gating_output[t])``, selects the
    top ``max_topk`` experts (descending weight, ties broken by smaller expert
    id), keeps only the first ``dyna_k[t]`` of them, optionally renormalizes the
    kept weights to sum to 1, and writes a fixed ``max_topk``-wide row. The
    dropped tail (``j >= dyna_k[t]``) gets ``topk_weights == 0`` and its id slot
    set to ``pad_id`` (the ``moe_sorting``-skipped sentinel).

    Parameters
    ----------
    gating_output : torch.Tensor
        ``(T, E)`` router logits, ``float32`` / ``bfloat16`` / ``float16``,
        contiguous. With ``native=True`` (default) a bf16/fp16 input is fed to
        the kernel as-is and widened to f32 *inside* the kernel (``extf``, a
        lossless widening); with ``native=False`` the host up-casts to f32
        first (an extra full pass + temp f32 buffer). Both produce identical
        results because bf16/fp16 -> f32 widening is exact.
    dyna_k : torch.Tensor
        ``(T,)`` ``int32`` per-token number of experts to keep. Values are
        clamped to ``[1, max_topk]`` inside the kernel (matching the torch
        reference), so out-of-range values keep at least the top-1 expert and
        cap at ``max_topk``.
    max_topk : int
        Padded top-k width. Must satisfy ``1 <= max_topk <= E``.
    topk_weights : torch.Tensor, optional
        ``(T, max_topk)`` ``float32`` output buffer; allocated if ``None``.
    topk_ids : torch.Tensor, optional
        ``(T, max_topk)`` ``int32`` output buffer; allocated if ``None``.
    renormalize : bool, default True
        If True, normalize the kept weights to sum to 1. If False, emit raw
        scores (kept sum < 1).
    scoring_func : {"softmax", "sigmoid"}, default "softmax"
        Per-expert scoring. ``"softmax"`` uses row-normalized probabilities;
        ``"sigmoid"`` uses per-expert ``1/(1+e^-x)``. Selection is identical
        (both monotonic in the logit); only the weights / renormalize
        denominator differ.
    native : bool, default True
        If ``True``, run the native low-precision kernel for bf16/fp16 inputs
        (load 16-bit, widen to f32 in-register). If ``False``, up-cast to f32
        on the host first (legacy path). Numerically identical; ``native``
        avoids the extra host cast + temp f32 buffer.
    pad_id : int, optional
        Sentinel written to the dropped ``topk_ids`` tail. Defaults to
        ``num_experts`` (``E``) -- the convention ``moe_sorting`` already skips
        (its histogram has a ``num_experts + 1`` padding column that is never
        emitted), so the output is a drop-in for ``moe_sorting`` with no kernel
        change.
    stream : torch.cuda.Stream, optional
        Defaults to the current CUDA stream.

    Returns
    -------
    (topk_weights, topk_ids) : Tuple[torch.Tensor, torch.Tensor]
    """
    assert gating_output.dim() == 2, (
        f"gating_output must be 2-D (T, E), got {tuple(gating_output.shape)}"
    )
    if gating_output.dtype not in _SUPPORTED_IN_DTYPES:
        raise ValueError(
            f"gating_output dtype {gating_output.dtype} unsupported; "
            f"expected one of {_SUPPORTED_IN_DTYPES}"
        )
    # The kernel ingests tensors through DLPack, which rejects tensors that
    # require grad ("Can't export tensors that require gradient"). Detach so an
    # autograd router output (e.g. a gating linear in training) works directly;
    # detach is a cheap storage-sharing view.
    gating_output = gating_output.detach()
    # native=False (legacy): up-cast bf16/fp16 to f32 on the host (extra pass +
    # temp f32 buffer). native=True: keep the 16-bit input and let the kernel
    # widen with extf -- numerically identical, no host cast.
    if not native and gating_output.dtype != torch.float32:
        gating_output = gating_output.float()
    if not gating_output.is_contiguous():
        gating_output = gating_output.contiguous()
    in_dtype = _TORCH_TO_IN_DTYPE[gating_output.dtype]

    T_tokens, E = int(gating_output.shape[0]), int(gating_output.shape[1])
    K = int(max_topk)
    if not (1 <= K <= E):
        raise ValueError(f"max_topk must be in [1, num_experts={E}], got {K}")

    if pad_id is None:
        pad_id = E  # moe_sorting-skipped sentinel (== num_experts)

    if dyna_k.numel() != T_tokens:
        raise ValueError(
            f"dyna_k must have {T_tokens} elements, got {dyna_k.numel()}"
        )
    dyna_k_i32 = dyna_k.to(dtype=torch.int32).contiguous()

    if topk_weights is None:
        topk_weights = torch.empty(
            T_tokens, K, dtype=torch.float32, device=gating_output.device
        )
    if topk_ids is None:
        topk_ids = torch.empty(
            T_tokens, K, dtype=torch.int32, device=gating_output.device
        )

    assert topk_weights.shape == (T_tokens, K) and topk_weights.is_contiguous()
    assert topk_ids.shape == (T_tokens, K) and topk_ids.is_contiguous()
    assert topk_weights.dtype == torch.float32
    assert topk_ids.dtype == torch.int32

    if T_tokens == 0:
        return topk_weights, topk_ids

    # Layout is T-dependent for E >= 256: small/mid batches use a smaller VPT
    # (broad latency win), large batches (T >= LARGE_BATCH_TOKENS) use the
    # throughput-tuned larger VPT. The choice only affects the layout, not the
    # result; it must be consistent between the grid size and the built kernel.
    large_batch = T_tokens >= LARGE_BATCH_TOKENS
    tpb = dyna_topk_tokens_per_block(E, large_batch)
    num_blocks = (T_tokens + tpb - 1) // tpb

    exe = build_dyna_fused_topk_module(
        num_experts=E,
        max_topk=K,
        renormalize=bool(renormalize),
        in_dtype=in_dtype,
        large_batch=large_batch,
        scoring_func=scoring_func,
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    # Launch through the shared helper so the compiled function is cached on the
    # (lru-cached) ``exe`` and re-dispatched cheaply on subsequent calls. Tensors
    # are passed flattened; the kernel addresses them with linear element/byte
    # offsets, so 1-D views are equivalent and avoid shape-specialisation churn.
    _run_compiled(
        exe,
        (
            gating_output.view(-1),
            dyna_k_i32.view(-1),
            topk_weights.view(-1),
            topk_ids.view(-1),
            int(pad_id),
            T_tokens,
            num_blocks,
            stream,
        ),
    )

    return topk_weights, topk_ids
