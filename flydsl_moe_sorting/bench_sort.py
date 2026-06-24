#!/usr/bin/env python3
# Correctness + pure-sort benchmark: CK moe_sorting vs FlyDSL atomicAdd sort.
#
#   python flydsl_moe_sorting/bench_sort.py
#
# - Verifies each (token, topk) is placed in the correct expert block, fully
#   covered, weights preserved (order-insensitive, since atomicAdd is unordered).
# - Benchmarks CK vs FlyDSL across a token sweep (E=193, topk=9, block=16).
import sys
import torch
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.fused_moe import moe_sorting as ck_moe_sorting
from aiter.ops.flydsl.moe_sorting_api import moe_sorting_atomic


def gen(M, E, topk, device="cuda"):
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
    topk_weights = torch.rand((M, topk), dtype=torch.float32, device=device)
    return topk_ids, topk_weights


def verify(topk_ids, topk_weights, E, block, out):
    sorted_ids, sorted_w, sorted_e, num_valid, moe_buf = out
    M, topk = topk_ids.shape
    total = int(num_valid[0].item())
    assert total % block == 0, f"total {total} not multiple of block {block}"
    nblocks = total // block
    ids = sorted_ids[:total].cpu().tolist()
    ws = sorted_w[:total].cpu().tolist()
    se = sorted_e[:nblocks].cpu().tolist()
    ti = topk_ids.cpu()
    tw = topk_weights.cpu()
    sentinel = (topk << 24) | M
    seen = {}
    for b in range(nblocks):
        e = se[b]
        for s in range(b * block, (b + 1) * block):
            v = ids[s]
            if v == sentinel:
                continue
            k = (v >> 24) & 0xFF
            tok = v & 0xFFFFFF
            if ti[tok, k].item() != e:
                return False, f"block {b} expert {e} but (tok={tok},k={k})->{ti[tok,k].item()}"
            seen[(tok, k)] = ws[s]
    if len(seen) != M * topk:
        return False, f"covered {len(seen)} != {M*topk}"
    for (tok, k), w in seen.items():
        if abs(w - tw[tok, k].item()) > 1e-5:
            return False, f"weight mismatch (tok={tok},k={k})"
    return True, "ok"


def run_case(M, E, topk, block, model_dim=4096, do_perf=False):
    topk_ids, topk_weights = gen(M, E, topk)
    out = moe_sorting_atomic(topk_ids, topk_weights, E, model_dim, dtypes.bf16, block_size=block)
    ok, msg = verify(topk_ids, topk_weights, E, block, out)
    print(f"[verify] M={M} E={E} topk={topk} block={block}: {'PASS' if ok else 'FAIL'} ({msg})")
    if not ok:
        return ok
    if do_perf:
        _, ck_us = run_perftest(
            ck_moe_sorting, topk_ids, topk_weights, E, model_dim, dtypes.bf16, block,
            None, None, 0, num_warmup=15, num_iters=50,
        )
        _, fly_us = run_perftest(
            moe_sorting_atomic, topk_ids, topk_weights, E, model_dim, dtypes.bf16, block,
            num_warmup=15, num_iters=50,
        )
        print(f"[perf]   M={M}: CK={ck_us:.2f}us  FlyDSL={fly_us:.2f}us  speedup={ck_us/fly_us:.2f}x")
    return ok


if __name__ == "__main__":
    torch.cuda.manual_seed(0)
    for M, E, topk, block in [(8, 8, 2, 16), (64, 32, 5, 16), (256, 193, 9, 16)]:
        if not run_case(M, E, topk, block):
            sys.exit(1)
    print("--- token sweep (E=193, topk=9, block=16): CK vs FlyDSL ---")
    for M in [1, 2, 8, 64, 256, 1024, 4096, 16384, 32768]:
        run_case(M, 193, 9, 16, do_perf=True)
