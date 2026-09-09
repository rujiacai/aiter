// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// PISA CSR fused exact/approx attention (loads
// AITER_ASM_DIR/gfx950/vsa/vsa_qk_fp8_pv_fp4_pisa_csr.co).

#pragma once

#include <torch/extension.h>

void vsa_qk_fp8_pv_fp4_pisa_stats(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& kscale,
    const torch::Tensor& vscale,
    const torch::Tensor& vbs,
    const torch::Tensor& k_center,
    const torch::Tensor& value_center,
    const torch::Tensor& used_ids,
    int64_t n_used,
    int64_t BH,
    int64_t T);

void vsa_qk_fp8_pv_fp4_pisa_csr(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& qscale,
    const torch::Tensor& kscale,
    const torch::Tensor& vscale,
    const torch::Tensor& exact_col_indices,
    const torch::Tensor& exact_row_meta,
    const torch::Tensor& approx_col_indices,
    const torch::Tensor& approx_row_meta,
    const torch::Tensor& vbs,
    const torch::Tensor& lim,
    const torch::Tensor& k_center,
    const torch::Tensor& value_center,
    const torch::Tensor& out,
    const torch::Tensor& lse,
    const torch::Tensor& counters,
    int64_t B,
    int64_t T,
    int64_t num_q_blks,
    int64_t n_dense,
    int64_t centroids_ready);
