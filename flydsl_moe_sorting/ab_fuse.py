#!/usr/bin/env python3
# A/B: -memset sort-only time, AITER_SORT_FUSE_CUMSUM on vs off.
import os
import torch
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.ops.flydsl.moe_sorting_api import moe_sorting_atomic

E, topk, block, model_dim = 193, 9, 16, 4096
TOKENS = [1, 8, 64, 256, 1024, 4096, 16384, 32768]
torch.cuda.manual_seed(0)
os.environ["AITER_SORT_SKIP_ZERO"] = "1"  # -memset scenario (the focus)


def bench(fn, *args):
    _, us = run_perftest(fn, *args, num_warmup=15, num_iters=50)
    return us


print(f"{'token':>7} | {'fuse=off':>9} | {'fuse=on':>9} | {'speedup':>7}")
print("-" * 42)
for t in TOKENS:
    ti = torch.randint(0, E, (t, topk), dtype=torch.int32, device="cuda")
    tw = torch.rand((t, topk), dtype=torch.float32, device="cuda")

    os.environ["AITER_SORT_FUSE_CUMSUM"] = "0"
    off = bench(moe_sorting_atomic, ti, tw, E, model_dim, dtypes.bf16, block)
    os.environ["AITER_SORT_FUSE_CUMSUM"] = "1"
    on = bench(moe_sorting_atomic, ti, tw, E, model_dim, dtypes.bf16, block)

    print(f"{t:>7} | {off:9.2f} | {on:9.2f} | {off/on:6.2f}x")
