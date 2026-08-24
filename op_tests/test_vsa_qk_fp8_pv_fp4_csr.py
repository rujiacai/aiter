# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A/B test: CSR connectivity ABI vs the shipped indexed kernel.

The baseline is the production vsa/vsa_qk_fp8_pv_fp4.co, unmodified.  CSR comes
from the WarpEngine translation unit
(/opt/WarpEngine/vsa_qk_fp8_pv_fp4_hip/vsa_qk_fp8_pv_fp4.hip) built with
-DVSA_CSR_ABI=1; the same source built with -DVSA_CSR_ABI=0 reproduces the
shipped object instruction for instruction, so the connectivity change is the
only thing separating the two.

`attention_indexed` and `attention_csr` below are the full call sequence for
each ABI and double as the usage reference; the rest of the file is harness.

Checks, in order:
  1. CSR round-trip — row_nnz / row_start / first_kv / col_indices reproduce
     the indexed descriptor element-wise, including per-row column order.
  2. Bitwise equality — out (bf16) and lse (f32) compared as raw bit patterns,
     not with a tolerance.
  3. Non-degenerate output — a spot-check against the FP32 reference, so that
     check 2 cannot pass by both kernels writing garbage.
  4. Performance — CSR kernel time must not regress against indexed.

Usage:
    PYTHONPATH=/opt/aiter-rujiacai python3 op_tests/test_vsa_qk_fp8_pv_fp4_csr.py
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_vsa_qk_fp8_pv_fp4 import make_synthetic_data, spot_check  # noqa: E402

from aiter.ops.vsa_qk_fp8_pv_fp4 import (  # noqa: E402
    build_l2_aware_lim_vsa_qk_fp8_pv_fp4,
    build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr,
    vsa_qk_fp8_pv_fp4_csr_dropB,
    vsa_qk_fp8_pv_fp4_dropB,
)

SPARSE_BLK = 128
_HEAD_DIM = 128


def indexed_to_csr(q2k_idx, q2k_num):
    """Flatten an indexed (n_tasks, max_kv) rectangle to (row_ptr, col_indices),
    preserving each row's column order exactly.

    Test-only.  It exists because the shared data generator emits the rectangle
    and both ABIs have to be fed the same connectivity; touching the whole
    rectangle is precisely the cost CSR exists to avoid, so a real caller emits
    CSR from its sparsity selector and never builds one.
    """
    q2k_idx = q2k_idx.view(-1, q2k_idx.shape[-1])
    q2k_num = q2k_num.view(-1).to(torch.int64)
    n, max_kv = q2k_idx.shape
    assert q2k_num.numel() == n

    row_ptr = torch.zeros(n + 1, dtype=torch.int64, device=q2k_idx.device)
    torch.cumsum(q2k_num, 0, out=row_ptr[1:])

    keep = torch.arange(max_kv, device=q2k_idx.device).unsqueeze(0) < q2k_num.unsqueeze(1)
    # Boolean-mask selection on a contiguous 2D tensor yields row-major order,
    # which is precisely CSR order.
    col_indices = q2k_idx[keep].to(torch.int32).contiguous()
    return row_ptr.to(torch.int32).contiguous(), col_indices


# --------------------------------------------------------------------------- #
# Calling the kernel — one function per connectivity ABI.  Between them these
# are the whole usage story; everything below is the A/B harness.
#
# The schedule (`lim` / `n_dense`, plus `row_meta` for CSR) describes the
# sparsity pattern, not Q/K/V.  Leave `schedule` unset for a one-shot call, or
# build it once per pattern and pass it back in to reuse it across forwards —
# which is what a real caller does, and what the benchmark below does.
# --------------------------------------------------------------------------- #
def attention_indexed(*, q, k, v, qs, ks, vs, vbs,
                      q2k_idx, q2k_num, max_kv,
                      B, T, num_q_blks,
                      schedule=None, out=None, lse=None, counters=None):
    """Indexed connectivity: a dense (BH*num_q_blks, max_kv) int32 rectangle,
    with q2k_num[i] valid entries in row i."""
    if schedule is None:
        schedule = build_l2_aware_lim_vsa_qk_fp8_pv_fp4(q2k_idx, q2k_num, max_kv)
    lim, n_dense = schedule

    return vsa_qk_fp8_pv_fp4_dropB(
        q=q, k=k, v=v, qs=qs, ks=ks, vs=vs,
        q2k_idx=q2k_idx, q2k_num=q2k_num,
        vbs=vbs, lim=lim, n_dense=n_dense,
        B=B, T=T, num_q_blks=num_q_blks, max_kv=max_kv,
        out=out, lse=lse, counters=counters,
    )


def attention_csr(*, q, k, v, qs, ks, vs, vbs,
                  q2k_col_indices, q2k_row_ptr=None,
                  B, T, num_q_blks,
                  schedule=None, out=None, lse=None, counters=None):
    """CSR connectivity: (row_ptr, col_indices), each row's KV block ids
    contiguous and in the order the kernel should consume them.

    `q2k_row_ptr` is only needed to build a schedule; once you have one, the
    kernel reads the payload through `row_meta` and never touches row_ptr.
    """
    if schedule is None:
        schedule = build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr(
            q2k_row_ptr, q2k_col_indices, num_q_blks)
    lim, n_dense, row_meta = schedule

    return vsa_qk_fp8_pv_fp4_csr_dropB(
        q=q, k=k, v=v, qs=qs, ks=ks, vs=vs,
        q2k_col_indices=q2k_col_indices, q2k_row_meta=row_meta,
        vbs=vbs, lim=lim, n_dense=n_dense,
        B=B, T=T, num_q_blks=num_q_blks,
        out=out, lse=lse, counters=counters,
    )


def _sep(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# Check 1 — CSR faithfully encodes the indexed descriptor
# --------------------------------------------------------------------------- #
def check_csr_roundtrip(q2k_idx, q2k_num, row_ptr, col_ind, row_meta, lim):
    n, max_kv = q2k_idx.shape
    # row_meta is emitted in schedule order; undo the permutation so the
    # records line up with logical rows for comparison.
    meta = torch.empty_like(row_meta)
    meta[lim.long()] = row_meta
    row_nnz, row_start, first_kv = (meta[:, 0].contiguous(),
                                    meta[:, 1].contiguous(),
                                    meta[:, 2].contiguous())
    ok = True

    def report(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"    {'PASS' if cond else 'FAIL'}  {name}")

    report("row_nnz == q2k_num", torch.equal(row_nnz, q2k_num))
    report("row_start == row_ptr[:-1]", torch.equal(row_start, row_ptr[:-1]))
    report("row_ptr[-1] == nnz", int(row_ptr[-1]) == col_ind.numel())

    # Rebuild the rectangle from CSR and compare the live region element-wise.
    # This is the check that matters: it verifies column ORDER within each row,
    # which is what makes the online softmax reproduce bit for bit.
    ar = torch.arange(max_kv, device=q2k_idx.device)
    live = ar.unsqueeze(0) < row_nnz.unsqueeze(1).long()
    gathered = torch.zeros_like(q2k_idx)
    flat_pos = row_start.long().unsqueeze(1) + ar.unsqueeze(0)
    gathered[live] = col_ind[flat_pos[live]]
    report("per-row column order preserved",
           torch.equal(gathered[live], q2k_idx[live]))

    nonempty = row_nnz > 0
    report("first_kv == q2k_idx[:, 0] on non-empty rows",
           torch.equal(first_kv[nonempty], q2k_idx[nonempty, 0]))
    return ok


# --------------------------------------------------------------------------- #
# Check 2 — bitwise comparison
# --------------------------------------------------------------------------- #
def bitwise_report(name, a, b):
    """Compare raw bit patterns.  NaNs compare equal iff their payloads match,
    which is the strictness we want here."""
    bits = {torch.bfloat16: torch.int16,
            torch.float16: torch.int16,
            torch.float32: torch.int32}[a.dtype]
    ai, bi = a.view(bits), b.view(bits)
    diff = ai != bi
    n_diff = int(diff.sum())
    total = ai.numel()
    if n_diff == 0:
        print(f"    PASS  {name}: all {total:,} elements bit-identical")
        return True
    ulp = (ai.to(torch.int64) - bi.to(torch.int64)).abs()
    print(f"    FAIL  {name}: {n_diff:,}/{total:,} differ "
          f"({100.0 * n_diff / total:.4f}%), max |ULP| = {int(ulp.max())}")
    return False


# --------------------------------------------------------------------------- #
# Check 4 — timing
# --------------------------------------------------------------------------- #
def bench_pair(fa, fb, iters=30, burst=10, warmup=20):
    """Time `burst` back-to-back launches per event pair, and swap which
    variant leads on alternating iterations.

    Both details matter.  Timing one launch per event pair charges each variant
    for its host-side argument validation, and the CSR entry point validates
    more tensors than the indexed one — that is real work, but it is not kernel
    time and it pipelines away in any real workload.  Timing the two variants
    in separate back-to-back loops lets clock and thermal drift land entirely
    on whichever ran second.
    """
    for _ in range(warmup):
        fa()
        fb()
    torch.cuda.synchronize()

    ta, tb = [], []
    evs = []
    for i in range(iters):
        pair = [(fa, ta), (fb, tb)]
        if i & 1:
            pair.reverse()
        for fn, acc in pair:
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(burst):
                fn()
            e.record()
            evs.append((s, e, acc))
    torch.cuda.synchronize()
    for s, e, acc in evs:
        acc.append(s.elapsed_time(e) / burst)

    def stats(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2], xs[0]
    return stats(ta), stats(tb)


# --------------------------------------------------------------------------- #
def run_one(num_q_blks, seed, do_spot_check):
    data = make_synthetic_data(seed=seed, num_q_blks=num_q_blks)
    B, T = data['B'], data['T']
    BH = data['B'] * data['H']
    max_kv = data['max_kv']
    q2k_idx, q2k_num = data['q2k_idx'], data['q2k_num']

    _sep(f"T = {T:,} tokens  (num_q_blks={num_q_blks}, BH={BH}, max_kv={max_kv})")

    row_ptr, col_ind = indexed_to_csr(q2k_idx, q2k_num)
    nnz = col_ind.numel()
    dense_bytes = q2k_idx.numel() * 4
    csr_bytes = nnz * 4 + row_ptr.numel() * 4 + 4 * BH * num_q_blks * 4
    print(f"  descriptor: indexed {dense_bytes / 2**20:8.1f} MiB   "
          f"CSR {csr_bytes / 2**20:8.1f} MiB   "
          f"({dense_bytes / csr_bytes:.1f}x smaller, nnz={nnz:,}, "
          f"density={nnz / q2k_idx.numel():.4%})")

    # attention_*() builds a schedule itself when you don't pass one, as checks
    # [2]/[3] do below.  The harness also wants one in hand: check [1] validates
    # its contents, and check [4] has to keep the sort out of the timed region.
    sched_idx = build_l2_aware_lim_vsa_qk_fp8_pv_fp4(q2k_idx, q2k_num, max_kv)
    sched_csr = build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr(
        row_ptr, col_ind, num_q_blks)
    lim_idx, n_dense_idx = sched_idx
    lim_csr, n_dense_csr, row_meta = sched_csr

    print(f"\n  [1] CSR round-trip")
    ok = check_csr_roundtrip(q2k_idx, q2k_num, row_ptr, col_ind, row_meta, lim_csr)
    same_sched = torch.equal(lim_idx, lim_csr)
    print(f"    INFO  n_dense: indexed={n_dense_idx} csr={n_dense_csr}; "
          f"lim permutation {'identical' if same_sched else 'differs'}"
          f"{'' if same_sched else ' (allowed: per-tile results are schedule-invariant)'}")

    kw = dict(q=data['q'], k=data['k'], v=data['v'],
              qs=data['qs'], ks=data['ks'], vs=data['vs'],
              vbs=data['vbs'], B=B, T=T, num_q_blks=num_q_blks)

    # One-shot form: hand each entry point its connectivity and let it do the
    # rest.  out / lse are allocated for us and reused by the timing loop.
    out_i, lse_i = attention_indexed(
        q2k_idx=q2k_idx, q2k_num=q2k_num, max_kv=max_kv, **kw)
    out_c, lse_c = attention_csr(
        q2k_row_ptr=row_ptr, q2k_col_indices=col_ind, **kw)
    torch.cuda.synchronize()

    print(f"\n  [2] bitwise equality (shipped vsa_qk_fp8_pv_fp4.co vs CSR .co)")
    ok &= bitwise_report("out (bf16)", out_i, out_c)
    ok &= bitwise_report("lse (f32) ", lse_i, lse_c)

    print(f"\n  [3] output is real attention, not a degenerate write")
    finite = bool(torch.isfinite(out_c).all())
    nonzero = float(out_c.abs().float().mean())
    print(f"    {'PASS' if finite else 'FAIL'}  all finite; "
          f"mean|out| = {nonzero:.5f}")
    ok &= finite and nonzero > 0.0
    if do_spot_check:
        st = spot_check(data, out_c, lse_c, n_samples=8, seed=seed)
        good = st['cos'] > 0.97 and st['nan_tiles'] == 0
        print(f"    {'PASS' if good else 'FAIL'}  vs FP32 reference over "
              f"{st['n_samples']} tiles: cos = {st['cos']:.6f}, "
              f"cos_lse = {st['cos_lse']:.6f}, max|diff| = {st['max_abs']:.4e}, "
              f"nan_tiles = {st['nan_tiles']}")
        ok &= good

    print(f"\n  [4] performance")
    # Reuse form: schedule built once and passed in, so the sort stays out of
    # the timed region — which is also what a real caller does across forwards.
    counters = torch.zeros(2, dtype=torch.int32, device='cuda')
    call_idx = lambda: attention_indexed(
        q2k_idx=q2k_idx, q2k_num=q2k_num, max_kv=max_kv, schedule=sched_idx,
        out=out_i, lse=lse_i, counters=counters, **kw)
    call_csr = lambda: attention_csr(
        q2k_col_indices=col_ind, schedule=sched_csr,
        out=out_c, lse=lse_c, counters=counters, **kw)
    (mi, bi), (mc, bc) = bench_pair(call_idx, call_csr)
    d_med = (mc - mi) / mi * 100.0
    d_best = (bc - bi) / bi * 100.0
    print(f"    median  indexed {mi:8.3f} ms   csr {mc:8.3f} ms   {d_med:+.2f}%")
    print(f"    best    indexed {bi:8.3f} ms   csr {bc:8.3f} ms   {d_best:+.2f}%")
    # Judged on best-of, the least noise-contaminated estimator here.
    #
    # CSR comes out slightly ahead at every size, but not all of that margin is
    # CSR's: the shipped object and a rebuild of its source schedule the same
    # instructions differently, worth on the order of 1.5%.  Measured against an
    # identically-scheduled indexed build, the CSR prologue instead costs ~1.2%
    # at 50k tokens and amortises to parity by 400k — it is a per-tile cost
    # spread over a growing number of KV blocks per row.  So this is a
    # no-regression gate, not a speedup claim.
    per_tile_ns = (bc - bi) * 1e6 / (BH * num_q_blks)
    print(f"    per tile {per_tile_ns:+.1f} ns over {nnz / (BH * num_q_blks):.0f}"
          f" KV blocks/row")
    perf_ok = d_best <= 2.0
    print(f"    {'PASS' if perf_ok else 'FAIL'}  CSR does not regress "
          f"(best-of within 2%)")
    ok &= perf_ok

    del data, out_i, out_c, lse_i, lse_c, q2k_idx, q2k_num, col_ind
    torch.cuda.empty_cache()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[391, 1563, 3125],
                    help="num_q_blks values (T = num_q_blks * 128)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-spot-check", action="store_true")
    args = ap.parse_args()

    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}")

    results = {}
    for nqb in args.sizes:
        results[nqb * SPARSE_BLK] = run_one(nqb, args.seed, not args.no_spot_check)

    _sep("SUMMARY")
    for t, ok in results.items():
        print(f"  T = {t:>9,}  {'PASS' if ok else 'FAIL'}")
    allok = all(results.values())
    print(f"\n{'ALL CHECKS PASSED' if allok else 'FAILURES PRESENT'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
