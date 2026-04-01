import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    torch_fp4_moe_2stage_with_smooth,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.utility import fp4_utils


def _prepare_smoothed_fp4_weights(w1_q, w2_q, w1_scale, w2_scale, s1, s2):
    e, n, model_dim_half = w1_q.shape
    model_dim = model_dim_half * 2
    inter_dim = w2_q.shape[2] * 2

    w1_f32 = fp4_utils.mxfp4_to_f32(w1_q)
    w1_s_f32 = fp4_utils.e8m0_to_f32(w1_scale)
    w1_f32 = (
        w1_f32.view(e, n, model_dim // 32, 32) * w1_s_f32.view(e, n, model_dim // 32, 1)
    ).view(e, n, model_dim)
    w1_f32 = w1_f32 * s1.view(e, 1, model_dim)

    w2_f32 = fp4_utils.mxfp4_to_f32(w2_q)
    w2_s_f32 = fp4_utils.e8m0_to_f32(w2_scale)
    w2_f32 = (
        w2_f32.view(e, model_dim, inter_dim // 32, 32)
        * w2_s_f32.view(e, model_dim, inter_dim // 32, 1)
    ).view(e, model_dim, inter_dim)
    w2_f32 = w2_f32 * s2.view(e, 1, inter_dim)

    quant_fp4 = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q2, w1_s2 = quant_fp4(w1_f32, quant_dtype=dtypes.fp4x2)
    w2_q2, w2_s2 = quant_fp4(w2_f32, quant_dtype=dtypes.fp4x2)
    w1_q2 = w1_q2.view(e, n, model_dim // 2)
    w2_q2 = w2_q2.view(e, model_dim, inter_dim // 2)
    return w1_q2, w2_q2, w1_s2, w2_s2


def test_torch_fp4_moe_2stage_with_smooth_matches_manual_fp4_2stage():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    device = "cuda"

    token_num = 32
    experts = 8
    model_dim = 256
    inter_dim = 128
    topk = 2
    dtype = dtypes.bf16

    x = torch.randn(token_num, model_dim, dtype=dtype, device=device)
    w1_bf16 = torch.randn(experts, inter_dim, model_dim, dtype=dtype, device=device) / 10
    w2_bf16 = torch.randn(experts, model_dim, inter_dim, dtype=dtype, device=device) / 10

    quant_fp4 = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_scale = quant_fp4(w1_bf16, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant_fp4(w2_bf16, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(experts, inter_dim, model_dim // 2)
    w2_q = w2_q.view(experts, model_dim, inter_dim // 2)

    topk_ids = torch.randint(0, experts, (token_num, topk), dtype=torch.int32, device=device)
    topk_weight = torch.rand(token_num, topk, dtype=torch.float32, device=device)
    topk_weight = topk_weight / topk_weight.sum(dim=1, keepdim=True)

    fc1_smooth_scale = (
        1.0 + 0.05 * torch.randn(experts, 1, model_dim, dtype=torch.float32, device=device)
    )
    fc2_smooth_scale = (
        1.0 + 0.05 * torch.randn(experts, 1, inter_dim, dtype=torch.float32, device=device)
    )

    out_test = torch_fp4_moe_2stage_with_smooth(
        x,
        w1_q,
        w2_q,
        topk_weight,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Gelu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        fc1_smooth_scale=fc1_smooth_scale,
        fc2_smooth_scale=fc2_smooth_scale,
        doweight_stage1=False,
    )

    w1_q_ref, w2_q_ref, w1_s_ref, w2_s_ref = _prepare_smoothed_fp4_weights(
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        fc1_smooth_scale.to(torch.float32).view(experts, -1),
        fc2_smooth_scale.to(torch.float32).view(experts, -1),
    )
    x_q, x_s = quant_fp4(x.to(torch.float32), quant_dtype=dtypes.fp4x2)
    x_q = x_q.view(token_num, model_dim // 2)

    out1_ref = torch_moe_stage1(
        x_q,
        w1_q_ref,
        w2_q_ref,
        topk_weight,
        topk_ids,
        dtype=torch.float32,
        activation=ActivationType.Gelu,
        quant_type=QuantType.per_1x32,
        a1_scale=x_s,
        w1_scale=w1_s_ref,
        doweight=False,
    )
    a2_q, a2_s = quant_fp4(
        out1_ref.view(token_num * topk, inter_dim), quant_dtype=dtypes.fp4x2
    )
    a2_q = a2_q.view(token_num, topk, inter_dim // 2)
    out_ref = torch_moe_stage2(
        a2_q,
        w1_q_ref,
        w2_q_ref,
        topk_weight,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_s_ref,
        a2_scale=a2_s,
        doweight=True,
    )

    assert torch.allclose(out_test.float(), out_ref.float(), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out_test).all()