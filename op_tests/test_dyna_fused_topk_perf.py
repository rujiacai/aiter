# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Device-side performance comparison for the FlyDSL ``dyna_fused_topk`` router.

For every ``(dtype, token, E, topk)`` it times (GPU ``self_device_time_total``
via :func:`aiter.test_common.run_perftest`):

* ``dyna_full`` -- FlyDSL dynamic router with ``dyna_k == topk`` for every token
  (the *max-k* mode; same selection workload as a static top-k router).
* ``dyna_var``  -- FlyDSL dynamic router with a per-token ``dyna_k`` drawn from
  ``[1, topk]`` (the *dynamic-k* mode; skips the dropped-tail arg-max work).
* ``CK``        -- :func:`aiter.topk_softmax` (the deployed HIP/CK static router).
* ``ASM``       -- :func:`aiter.topk_softmax_asm` (only the few ``(E, topk)`` the
  assembly kernel supports; skipped otherwise).

The reported ratios ``CK/dyna`` and ``ASM/dyna`` are relative to ``dyna_full``
(``> 1`` means the dynamic router's kernel is faster). ``dyna_full`` is also
checked against an eager-torch reference for correctness.

Usage:
    python op_tests/test_dyna_fused_topk_perf.py
    python op_tests/test_dyna_fused_topk_perf.py -d bf16 -e 256 -t 4096 -k 8
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import _dyna_fused_topk_torch
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_dyna_fused_topk
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# (E, topk) combinations the ASM topk_softmax kernel ships specializations for.
_ASM_COMBOS = [(128, 4), (128, 6), (128, 8), (256, 6), (256, 8), (384, 8)]


def _asm_supported(E, topk, dtype):
    return (
        (E, topk) in _ASM_COMBOS
        and dtype in (dtypes.bf16, dtypes.fp32)
        and get_gfx() in ("gfx942", "gfx950")
    )


def _run_dyna(gating, dyna_k, topk, scoring_func="softmax"):
    w, i = flydsl_dyna_fused_topk(gating, dyna_k, topk, scoring_func=scoring_func)
    return w, i


def _run_ck(gating, topk, renormalize):
    M = gating.shape[0]
    w = torch.empty((M, topk), dtype=dtypes.fp32, device=gating.device)
    i = torch.empty((M, topk), dtype=dtypes.i32, device=gating.device)
    tei = torch.empty((M, topk), dtype=dtypes.i32, device=gating.device)
    aiter.topk_softmax(w, i, tei, gating, renormalize)
    return w, i


def _run_asm(gating, topk, renormalize):
    M = gating.shape[0]
    # ASM writes M padded up to a multiple of 4 rows.
    Mpad = (M + 3) // 4 * 4
    w = torch.empty((Mpad, topk), dtype=dtypes.fp32, device=gating.device)
    i = torch.empty((Mpad, topk), dtype=dtypes.i32, device=gating.device)
    tei = torch.empty((M, topk), dtype=dtypes.i32, device=gating.device)
    aiter.topk_softmax_asm(w, i, tei, gating, renormalize)
    return w, i


@benchmark()
def test_dyna_fused_topk(dtype, token, E, topk, renormalize=True, scoring_func="softmax"):
    gating = torch.randn((token, E), dtype=dtype, device="cuda").contiguous()
    dyna_k_full = torch.full((token,), topk, dtype=dtypes.i32, device="cuda")
    dyna_k_var = torch.randint(1, topk + 1, (token,), dtype=dtypes.i32, device="cuda")

    ret = {"scoring_func": scoring_func}

    # max-k mode (dyna_k == topk): same selection workload as a static router.
    (w_full, id_full), us_full = run_perftest(
        _run_dyna, gating, dyna_k_full, topk, scoring_func
    )
    ret["dyna_full us"] = us_full

    # dynamic-k mode: per-token k in [1, topk]; tail arg-max work is skipped.
    (_w, _i), us_var = run_perftest(_run_dyna, gating, dyna_k_var, topk, scoring_func)
    ret["dyna_var us"] = us_var

    # Correctness of the max-k path against the eager-torch reference.
    w_ref, id_ref = _dyna_fused_topk_torch(gating, dyna_k_full, topk, scoring_func=scoring_func)
    id_r, perm_r = torch.sort(id_ref)
    id_g, perm_g = torch.sort(id_full)
    ret["err"] = checkAllclose(
        w_ref.gather(1, perm_r), w_full.gather(1, perm_g),
        atol=2e-3, msg="dyna_full vs torch weights", printLog=False,
    )

    # CK / ASM are softmax-only static routers, so they are only a meaningful
    # baseline for the softmax scoring sweep.
    if scoring_func == "softmax":
        # CK static router (the deployed baseline).
        (_w, _i), us_ck = run_perftest(_run_ck, gating, topk, renormalize)
        ret["CK us"] = us_ck
        ret["CK/dyna"] = us_ck / us_full if us_full else float("nan")

        # ASM router, only where a specialization exists.
        if _asm_supported(E, topk, dtype):
            (_w, _i), us_asm = run_perftest(_run_asm, gating, topk, renormalize)
            ret["ASM us"] = us_asm
            ret["ASM/dyna"] = us_asm / us_full if us_full else float("nan")

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="dyna_fused_topk vs CK vs ASM device-time benchmark",
)
parser.add_argument(
    "-d", "--dtype", type=dtypes.str2Dtype, nargs="*",
    choices=[dtypes.d_dtypes["fp32"], dtypes.d_dtypes["bf16"]],
    metavar="{fp32, bf16}", default=[dtypes.d_dtypes["bf16"]],
    help="Data type(s). e.g.: -d bf16 fp32",
)
parser.add_argument(
    "-e", "--expert", type=int, nargs="*", default=[64, 128, 192, 256],
    help="Number of experts E. e.g.: -e 64 128 192 256",
)
parser.add_argument(
    "-t", "--token", type=int, nargs="*",
    default=[1, 16, 32, 64, 256, 1024, 2048, 16384, 65536],
    help="Number of tokens T. e.g.: -t 1 256 65536",
)
parser.add_argument(
    "-k", "--topk", type=int, nargs="*", default=[4, 8, 12, 16, 20],
    help="Number of top-k. e.g.: -k 4 8 12 16 20",
)
parser.add_argument(
    "-s", "--scoring_func", type=str, nargs="*", choices=["softmax", "sigmoid"],
    default=["softmax", "sigmoid"],
    help="Scoring function(s). e.g.: -s softmax sigmoid",
)
args = parser.parse_args()

df = []
for scoring_func in args.scoring_func:
    for dtype in args.dtype:
        for e in args.expert:
            for k in args.topk:
                if k > e:
                    continue
                for m in args.token:
                    df.append(test_dyna_fused_topk(dtype, m, e, k, scoring_func=scoring_func))
df = pd.DataFrame(df)
try:
    summary = df.to_markdown(index=False)
except ImportError:
    # ``to_markdown`` needs the optional ``tabulate`` package; fall back.
    summary = df.to_string(index=False)
aiter.logger.info("dyna_fused_topk perf summary:\n%s", summary)
