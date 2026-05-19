# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Benchmark: (AR+RMSNorm) + static_per_tensor_quant  vs  fused kernel
#            non-graph and graph modes, latency = min over TIMED_ITERS runs
#
# Usage:
#   python bench_fused_ar_rms_per_tensor_quant.py
#   python bench_fused_ar_rms_per_tensor_quant.py -t 8 -d bf16

import os
from typing import Optional, List, Tuple
import aiter
import torch
import torch.distributed as dist
import argparse
import itertools
import pandas as pd
from aiter import dtypes
from aiter.ops.quant import static_per_tensor_quant
from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    graph_capture,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from aiter.dist.communication_op import (
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_fused_allreduce_rmsnorm_per_tensor_quant,
)
from multiprocessing import set_start_method, Pool, freeze_support
import logging

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

WARMUP_ITERS = 20
TIMED_ITERS  = 100   # number of individual timings; min is taken across all


def time_fn_min(fn, iters, group):
    """Per-iteration timing: barrier once, then record each call individually, return min us."""
    dist.barrier(group=group)
    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True))
              for _ in range(iters)]
    for s, e in events:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return min(s.elapsed_time(e) * 1e3 for s, e in events)   # ms -> us


def build_graphs(run_unfused, run_fused):
    """Capture both kernels into CUDA graphs (after warmup)."""
    for _ in range(WARMUP_ITERS):
        run_unfused()
        run_fused()
    torch.cuda.synchronize()

    g_unfused = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        with torch.cuda.graph(g_unfused, stream=gc.stream):
            run_unfused()

    g_fused = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        with torch.cuda.graph(g_fused, stream=gc.stream):
            run_fused()

    return g_unfused, g_fused


def bench_worker(
    tp_size: int,
    pp_size: int,
    rankID: int,
    shapes: List[Tuple[int, int]],
    dtype: torch.dtype,
    scale_factor: float,
    distributed_init_method: Optional[str],
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size, rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)

    group        = get_tp_group().device_group
    fp8_t        = dtypes.fp8
    eps          = 1e-6
    scale_tensor = torch.tensor(scale_factor, dtype=torch.float32, device=device)

    dist.barrier(group=group)
    torch.cuda.synchronize()

    results = []
    for shape in shapes:
        m, n = shape
        x       = torch.randn(shape, dtype=dtype, device=device)
        weight  = torch.randn((n,),  dtype=dtype, device=device)
        out_fp8 = torch.empty(shape, dtype=fp8_t,  device=device)

        def run_unfused():
            rms_out, _ = tensor_model_parallel_fused_allreduce_rmsnorm(
                x, x, weight, eps
            )
            static_per_tensor_quant(out_fp8, rms_out, scale_tensor)

        def run_fused():
            tensor_model_parallel_fused_allreduce_rmsnorm_per_tensor_quant(
                x, x, weight, eps, scale_tensor
            )

        # eager warmup
        for _ in range(WARMUP_ITERS):
            run_unfused()
            run_fused()
        torch.cuda.synchronize()

        # eager min timing
        us_unfused_eager = time_fn_min(run_unfused, TIMED_ITERS, group)
        us_fused_eager   = time_fn_min(run_fused,   TIMED_ITERS, group)

        # graph capture + min timing
        g_unfused, g_fused = build_graphs(run_unfused, run_fused)

        us_unfused_graph = time_fn_min(g_unfused.replay, TIMED_ITERS, group)
        us_fused_graph   = time_fn_min(g_fused.replay,   TIMED_ITERS, group)

        if rankID == 0:
            results.append((m, n,
                            us_unfused_eager, us_fused_eager,
                            us_unfused_graph, us_fused_graph))

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return results


# Configuration
l_dtype  = ["bf16"]
l_m      = [2, 4, 8, 16, 32, 64, 128, 512, 1024]
l_n      = [4096]
l_tp     = [8]
l_pp     = [1]
l_scale  = [0.5]

parser = argparse.ArgumentParser(
    description="Benchmark: (AR+RMSNorm + static_per_tensor_quant) vs fused kernel"
)
parser.add_argument("-d", "--dtype",  type=str, choices=["fp16", "bf16"], default=None)
parser.add_argument("-t", "--tp",     type=int, default=None)
parser.add_argument("-n", "--hidden", type=int, default=None)
parser.add_argument("--scale",        type=float, default=None)

if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype  is not None: l_dtype = [dtypes.d_dtypes[args.dtype]]
    else:                        l_dtype = [dtypes.d_dtypes[k] for k in l_dtype]
    if args.tp     is not None: l_tp    = [args.tp]
    if args.hidden is not None: l_n     = [args.hidden]
    if args.scale  is not None: l_scale = [args.scale]

    from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip

    all_rows = []
    for dtype, tp, pp, n, scale in itertools.product(l_dtype, l_tp, l_pp, l_n, l_scale):
        shapes = [(m, n) for m in l_m]
        init_method = get_distributed_init_method(get_ip(), get_open_port())

        pool = Pool(processes=tp)
        try:
            rets = [
                pool.apply_async(
                    bench_worker,
                    args=(tp, pp, i, shapes, dtype, scale, init_method),
                )
                for i in range(tp)
            ]
            pool.close()
            all_results = [r.get(timeout=600) for r in rets]
        except Exception:
            pool.terminate()
            raise
        finally:
            pool.join()

        rank0_results = next(r for r in all_results if r)
        for m, n, ue, fe, ug, fg in rank0_results:
            all_rows.append({
                "dtype":             str(dtype).replace("torch.", ""),
                "tp":                tp,
                "m":                 m,
                "n":                 n,
                "unfused_eager(us)": round(ue, 2),
                "fused_eager(us)":   round(fe, 2),
                "speedup_eager":     round(ue / fe, 4) if fe > 0 else float("inf"),
                "unfused_graph(us)": round(ug, 2),
                "fused_graph(us)":   round(fg, 2),
                "speedup_graph":     round(ug / fg, 4) if fg > 0 else float("inf"),
            })
            logger.info(
                f"dtype={str(dtype).replace('torch.','')} tp={tp} m={m:>5} n={n}: "
                f"eager unfused={ue:>8.2f}us fused={fe:>8.2f}us spd={ue/fe:.4f}x | "
                f"graph unfused={ug:>8.2f}us fused={fg:>8.2f}us spd={ug/fg:.4f}x"
            )

    df = pd.DataFrame(all_rows)
    logger.info(
        "\nBenchmark summary (min latency, baseline=static_per_tensor_quant HIP kernel):\n%s",
        df.to_markdown(index=False),
    )
