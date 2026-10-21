# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import functools

import torch

from aiter.ops.flydsl import moe_kernels as _moe_kernels

_SUPPORTED = {
    (16, False, "atomic"),
    (16, True, "atomic"),
    (32, False, "atomic"),
    (32, True, "atomic"),
    (64, False, "atomic"),
    (64, True, "atomic"),
    (128, False, "nonatomic"),
    (128, False, "nonatomic_mxfp4"),
    (32, False, "nonatomic_cshuffle"),
    (64, False, "nonatomic_cshuffle"),
    (128, False, "nonatomic_cshuffle"),
    (128, False, "nonatomic_lds"),
}


def _epilog_of(atomic, mxfp4out, cshuffle=False, ldsepi=False):
    if mxfp4out:
        return "nonatomic_mxfp4"
    if cshuffle:
        return "nonatomic_cshuffle"
    if ldsepi and not atomic:
        return "nonatomic_lds"
    return "atomic" if atomic else "nonatomic"


@functools.cache
def _get_compiled_mxfp4_gemm2_port(
    BM,
    use_nt,
    NE,
    N_OUT,
    epilog,
    D_INTER,
    D_INTER_REAL=None,
    BN=256,
    BK=256,
    xcd_swizzle=0,
    skippad=False,
    occ2=False,
):
    from .kernels.mxfp4_gemm2 import compile_gemm2_a4w4_port

    return compile_gemm2_a4w4_port(
        BM=BM,
        use_nt=use_nt,
        NE=NE,
        N_OUT=N_OUT,
        epilog=epilog,
        D_INTER=D_INTER,
        D_INTER_REAL=D_INTER_REAL,
        BN=BN,
        BK=BK,
        xcd_swizzle=xcd_swizzle,
        skippad=skippad,
        occ2=occ2,
    )


@functools.cache
def _dummy_out_scale(device):
    return torch.empty(1, dtype=torch.uint8, device=device)


def _assert_supported(
    *,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    BM,
    use_nt,
    atomic,
    mxfp4out,
    cshuffle=False,
    ldsepi=False,
    skippad=False,
    occ2=False,
    BN=256,
    BK=256,
):
    if skippad and not ldsepi:
        raise NotImplementedError("flydsl mxfp4 gemm2 skippad requires ldsepi")
    if occ2 and not ldsepi:
        raise NotImplementedError("flydsl mxfp4 gemm2 occ2 requires ldsepi")
    if occ2 and D_INTER != 2 * BK:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 occ2 supports D_INTER == {2 * BK} only, "
            f"got {D_INTER}"
        )
    if D_INTER % BK != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 contraction D_INTER (=inter_dim) must be a "
            f"multiple of {BK}; D_INTER not divisible by {BK} (e.g. "
            f"384/192) is not supported by this BK={BK} kernel "
            f"(got D_INTER={D_INTER})"
        )
    if D_HIDDEN % BN != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 requires D_HIDDEN (=N_OUT=model_dim) % {BN} == 0, "
            f"got H={D_HIDDEN}"
        )
    epilog = _epilog_of(atomic, mxfp4out, cshuffle, ldsepi)
    if (BM, use_nt, epilog) not in _SUPPORTED:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 unsupported variant "
            f"(BM={BM}, use_nt={use_nt}, epilog={epilog})"
        )


def flydsl_mxfp4_gemm2(
    *,
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    flat_out,
    M_logical,
    max_sorted,
    BM,
    use_nt,
    atomic,
    mxfp4out,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    flat_out_scale=None,
    cshuffle=False,
    ldsepi=False,
    skippad=False,
    occ2=False,
    D_INTER_REAL=None,
    BN=256,
    BK=256,
    xcd_swizzle=0,
    stream=None,
):
    _assert_supported(
        NE=NE,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        topk=topk,
        BM=BM,
        use_nt=use_nt,
        atomic=atomic,
        mxfp4out=mxfp4out,
        cshuffle=cshuffle,
        ldsepi=ldsepi,
        skippad=skippad,
        occ2=occ2,
        BN=BN,
        BK=BK,
    )
    if occ2 and D_INTER_REAL not in (None, D_INTER):
        raise NotImplementedError("flydsl mxfp4 gemm2 occ2 needs an unpadded D_INTER")
    epilog = _epilog_of(atomic, mxfp4out, cshuffle, ldsepi)
    launch = _get_compiled_mxfp4_gemm2_port(
        BM,
        use_nt,
        NE,
        D_HIDDEN,
        epilog,
        D_INTER,
        D_INTER_REAL,
        BN,
        BK,
        xcd_swizzle,
        skippad,
        occ2,
    )

    max_m_blocks = (max_sorted + BM - 1) // BM
    # skippad stores through a buffer descriptor with 32-bit byte offsets.
    if skippad and max_m_blocks * BM * D_HIDDEN * 2 > 0x7FFFFFF0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 skippad: output of {max_m_blocks * BM} x "
            f"{D_HIDDEN} bf16 exceeds the 2 GiB buffer-offset range"
        )

    out_scale = flat_out_scale if mxfp4out else _dummy_out_scale(flat_out.device)
    # occ2: tile counters, one per XCD, that must be zero at launch.
    counter = (
        torch.zeros(8, dtype=torch.int32, device=flat_out.device)
        if occ2
        else _dummy_out_scale(flat_out.device)
    )

    _moe_kernels._run_compiled(
        launch,
        (
            inter_sorted_quant.data_ptr(),
            inter_sorted_shuffled_scale.data_ptr(),
            w2_u8.data_ptr(),
            w2_scale_u8.data_ptr(),
            sorted_expert_ids.data_ptr(),
            cumsum_tensor.data_ptr(),
            sorted_token_ids.data_ptr(),
            sorted_weights.data_ptr(),
            M_logical,
            max_m_blocks,
            flat_out.data_ptr(),
            out_scale.data_ptr(),
            counter.data_ptr(),
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
