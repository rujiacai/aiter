import argparse
import os
from statistics import mean

import torch
import torch.profiler as tpf

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, get_2stage_cfgs, moe_sorting
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2, get_flydsl_kernel_params
from aiter.ops.quant import get_torch_quant


def _build_inputs(
    token: int,
    model_dim: int,
    inter_dim: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
    block_m: int,
):
    torch.manual_seed(0)
    device = "cuda"

    x = torch.randn(token, model_dim, dtype=dtype, device=device)
    score = torch.randn(token, num_experts, dtype=dtype, device=device)
    topk_weight, topk_ids = fused_topk(x, score, topk, True)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids=topk_ids,
        topk_weights=topk_weight,
        num_experts=num_experts,
        model_dim=model_dim,
        moebuf_dtype=dtype,
        block_size=block_m,
    )

    inter_bf16 = torch.randn(token, topk, inter_dim, dtype=dtype, device=device) / 10
    w2_bf16 = torch.randn(num_experts, model_dim, inter_dim, dtype=dtype, device=device) / 10

    fp4_quant = get_torch_quant(QuantType.per_1x32)
    inter_fp4, a2_scale = fp4_quant(inter_bf16)
    inter_fp4 = inter_fp4.view(token, topk, inter_dim // 2)
    w2_fp4, w2_scale = fp4_quant(w2_bf16)
    w2_fp4 = w2_fp4.view(num_experts, model_dim, inter_dim // 2)

    return (
        inter_fp4,
        w2_fp4,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        sorted_weights,
        a2_scale,
        w2_scale,
    )


def _run_stage2(
    *,
    inter_states,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    sorted_weights,
    a2_scale,
    w2_scale,
    topk: int,
    params: dict,
    out: torch.Tensor,
):
    # print(f"params: {params}")
    return flydsl_moe_stage2(
        inter_states=inter_states,
        w2=w2,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=params["tile_m"],
        tile_n=params["tile_n"],
        tile_k=params["tile_k"],
        a_dtype=params["a_dtype"],
        b_dtype=params["b_dtype"],
        out_dtype=params["out_dtype"],
        mode=params.get("mode", "atomic"),
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        sorted_weights=sorted_weights,
        sort_block_m=params.get("sort_block_m", 0),
        persist=params.get("persist", False),
    )


def main():
    parser = argparse.ArgumentParser(description="Stage2-only FP4 benchmark.")
    parser.add_argument("--token", type=int, default=20480)
    parser.add_argument("--model-dim", type=int, default=4096)
    parser.add_argument("--inter-dim", type=int, default=1536)
    parser.add_argument("--experts", type=int, default=400)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--kernel-name2",
        type=str,
        default="",
        help="Override stage2 FlyDSL kernel name.",
    )
    parser.add_argument(
        "--hip-visible-devices",
        type=str,
        default="",
        help="Set HIP_VISIBLE_DEVICES before running.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=0,
        help="If >0, run torch profiler for N iterations and print kernel table.",
    )
    args = parser.parse_args()

    if args.hip_visible_devices:
        os.environ["HIP_VISIBLE_DEVICES"] = args.hip_visible_devices

    dtype = torch.bfloat16
    q_type = QuantType.per_1x32
    use_g1u1 = False
    activation = ActivationType.Gelu

    meta = get_2stage_cfgs(
        args.token,
        args.model_dim,
        args.inter_dim,
        args.experts,
        args.topk,
        dtype,
        torch.float4_e2m1fn_x2,
        torch.float4_e2m1fn_x2,
        q_type,
        use_g1u1,
        activation,
        False,
        0,
        0,
        True,
    )
    kernel_name2 = args.kernel_name2 or meta.stage2.keywords.get("kernelName", "")
    params = get_flydsl_kernel_params(kernel_name2)
    if params is None:
        raise ValueError(f"Invalid stage2 kernel name: {kernel_name2}")

    print(f"[stage2-only] kernelName2={kernel_name2}")
    print(
        f"[stage2-only] tile=({params['tile_m']},{params['tile_n']},{params['tile_k']}), "
        f"mode={params.get('mode','atomic')}, persist={params.get('persist', False)}"
    )

    (
        inter_states,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        sorted_weights,
        a2_scale,
        w2_scale,
    ) = _build_inputs(
        token=args.token,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        num_experts=args.experts,
        topk=args.topk,
        dtype=dtype,
        block_m=meta.block_m,
    )

    out = torch.empty((args.token, args.model_dim), dtype=dtype, device="cuda")

    for _ in range(args.warmup):
        out.zero_()
        _run_stage2(
            inter_states=inter_states,
            w2=w2,
            sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            sorted_weights=sorted_weights,
            a2_scale=a2_scale,
            w2_scale=w2_scale,
            topk=args.topk,
            params=params,
            out=out,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lat_ms = []
    for _ in range(args.iters):
        out.zero_()
        start.record()
        _run_stage2(
            inter_states=inter_states,
            w2=w2,
            sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            sorted_weights=sorted_weights,
            a2_scale=a2_scale,
            w2_scale=w2_scale,
            topk=args.topk,
            params=params,
            out=out,
        )
        end.record()
        end.synchronize()
        lat_ms.append(start.elapsed_time(end))
    print(f"[stage2-only] avg_total={mean(lat_ms) * 1000.0:.3f} us/iter")

    if args.profile_iters > 0:
        with tpf.profile(
            activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ) as prof:
            for _ in range(args.profile_iters):
                out.zero_()
                _run_stage2(
                    inter_states=inter_states,
                    w2=w2,
                    sorted_token_ids=sorted_token_ids,
                    sorted_expert_ids=sorted_expert_ids,
                    num_valid_ids=num_valid_ids,
                    sorted_weights=sorted_weights,
                    a2_scale=a2_scale,
                    w2_scale=w2_scale,
                    topk=args.topk,
                    params=params,
                    out=out,
                )
            torch.cuda.synchronize()

        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=20,
            )
        )
        for evt in prof.key_averages():
            if "moe_gemm2_0" in evt.key:
                self_cuda_total = getattr(
                    evt,
                    "self_cuda_time_total",
                    getattr(evt, "self_device_time_total", 0.0),
                )
                avg_us = self_cuda_total / max(evt.count, 1)
                print(
                    f"[stage2-only] profiler {evt.key}: "
                    f"avg_self_cuda={avg_us:.3f} us over {evt.count} calls"
                )


if __name__ == "__main__":
    main()
