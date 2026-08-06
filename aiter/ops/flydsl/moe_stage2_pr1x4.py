# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""PR3987's ``down_prefill_1x4`` as a fused_moe stage2.

Unlike the other FlyDSL stage2 kernels this one stores one row per *sorted slot*
into a padded buffer instead of atomic-accumulating into the token-major output,
so it needs that intermediate plus the two kernels that scatter it back
(``invert_sorted_ids`` + ``sorted_sum``).  Selected by the ``_pr1x4`` tag in the
stage2 kernel name; see ``moe_kernels._parse_flydsl_kernel_name``.
"""

import functools
import os

import torch

_FP8 = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)


def _use_triton_reduce() -> bool:
    return os.environ.get("AITER_PR1X4_TRITON_REDUCE", "0") in ("1", "true", "True", "yes", "YES")


@functools.cache
def _down_kernel(N, K, weight_dtype, quant_type, topk, block_m, block_n, E):
    """compile_gemm has no cache of its own, and recompiling per call is fatal."""
    from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import compile_gemm

    return compile_gemm(
        N=N,
        K=K,
        weight_dtype=weight_dtype,
        weight_quant_type=quant_type,
        TOPK=topk,
        BLOCK_TILE_SIZE_M=block_m,
        BLOCK_TILE_SIZE_N=block_n,
        stage="down",
        alg="prefill_1x4",
        E=E,
    )


@functools.cache
def _per_expert_scale_buf(E, device_index):
    return torch.empty(E, dtype=torch.float32, device=f"cuda:{device_index}")


def _launch(kernel_fn, *args):
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    _run_compiled(kernel_fn, *args, torch.cuda.current_stream())


def flydsl_moe_stage2_pr1x4(
    inter_states,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    block_m=64,
    tile_n=128,
    w2_scale=None,
    a2_scale=None,
    sorted_weights=None,
):
    if sorted_weights is None:
        raise ValueError(
            "flydsl_moe_stage2_pr1x4 always applies the routing weight, so it "
            "cannot be paired with doweight_stage1"
        )
    if w2.dtype not in _FP8 and w2.dtype != torch.bfloat16:
        raise ValueError(f"unsupported stage2 weight dtype {w2.dtype}")

    E = w2.shape[0]
    B, N2 = out.shape[0], out.shape[-1]
    K2 = inter_states.shape[-1]
    task_num = sorted_expert_ids.shape[0]
    weight_dtype = "fp8" if w2.dtype in _FP8 else "bf16"

    # The kernel indexes the weight scale per expert (`p_w_scale + expert_id`), so
    # a single per-tensor value has to be materialised for every expert -- a
    # 1-element tensor would read out of bounds for expert > 0.
    if w2_scale is None:
        raise ValueError("flydsl_moe_stage2_pr1x4 needs a per-tensor weight scale")
    if w2_scale.numel() == 1:
        ws = _per_expert_scale_buf(E, out.device.index)
        ws.copy_(w2_scale.reshape(1).expand(E))
    elif w2_scale.numel() == E:
        ws = w2_scale.reshape(E).to(torch.float32)
    else:
        raise ValueError(
            f"expected a per-tensor or per-expert weight scale, got {tuple(w2_scale.shape)}"
        )
    if a2_scale is None:
        a2_scale = torch.ones(1, dtype=torch.float32, device=out.device)

    gemm2_out = torch.empty(
        [task_num * block_m, N2], dtype=out.dtype, device=out.device
    )
    _launch(
        _down_kernel(N2, K2, weight_dtype, "per_tensor", topk, block_m, tile_n, E),
        inter_states,
        w2,
        gemm2_out,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        ws,
        a2_scale,
        B,
        task_num,
    )

    from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import (
        invert_sorted_ids,
        sorted_sum,
    )

    loc_ids = torch.empty([B, topk], dtype=torch.int32, device=out.device)
    # Scan only [0, num_valid): the tail of sorted_token_ids is uninitialised, and
    # its garbage can map real tokens onto rows the down kernel never wrote.
    invert_sorted_ids(topk)(
        sorted_token_ids, loc_ids, num_valid_ids, sorted_token_ids.shape[0], B
    )
    if _use_triton_reduce():
        # Attribution probe: the reduce-mode stage2 already ships a Triton kernel
        # for exactly this gather-and-sum over a sorted-row partial buffer, and it
        # is measurably faster than sorted_sum on the same bytes.
        from aiter.ops.flydsl._fused_post import fused_topk_sum_gather

        fused_topk_sum_gather(
            out, gemm2_out.view(-1), loc_ids,
            token_num=B, topk=topk, model_dim=N2,
        )
        return out
    sorted_sum(topk, N2)(loc_ids, gemm2_out, out, B)
    return out
