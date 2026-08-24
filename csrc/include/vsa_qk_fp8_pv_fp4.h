// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Public C++ entry point for the QK=FP8 / PV=FP4 mixed-precision VSA
// block-sparse attention kernel (loads
// /opt/aiter/hsa/gfx950/vsa/vsa_qk_fp8_pv_fp4.co at first call).
//
// Q/K  are FP8 E4M3 with one E8M0 scale byte per 32-element group.
// V    is  FP4 MXFP4 E2M1 with one E8M0 scale byte per (32 K-row, D-channel)
//      block, packed into HBM-coalesced (BH, T/128, D, 4) uint8 layout so the
//      kernel reads one dword per thread (4-byte aggregation of 4 ubyte
//      scales) per VS prefetch.
//
// Numerical contract vs an FP32 software reference using the same E8M0
// quantisation (sparsity = 0.0846, n_attended_kv = 0.0846 * num_q_blks):
//   - cosine similarity stable in [0.9826, 0.9833] across T = 50k..1M
//   - cos(LSE)         = 1.000000 (FP32 tree reduction)
//   - max |diff|       decreases with T (~2.4e-2 at 50k -> ~6e-3 at 1M)
//   - no NaN / Inf at any size up to T = 1M tokens
//
// Layout convention: inputs are flattened to (BH, T, D) where BH = B * H.
//
// Caller responsibilities:
//   - `lim` and `n_dense` are the L2-aware tile-ordering hints — build them
//     ONCE per (q2k_idx, q2k_num) shape via the Python helper
//     `aiter.build_l2_aware_lim_vsa_qk_fp8_pv_fp4(...)`.
//   - `counters` must be a contiguous int32 tensor with at least 2 elements
//     (used by the kernel as two atomic dispatch counters: counters[0] for
//     the dense-tile partition, counters[1] for the sparse-tile partition);
//     the launcher zeros it on every call.

#pragma once

#include <torch/extension.h>

void vsa_qk_fp8_pv_fp4(const torch::Tensor& q,        // (BH, T, 128)  float8_e4m3fn
                       const torch::Tensor& k,        // (BH, T, 128)  float8_e4m3fn
                       const torch::Tensor& v,        // (BH, T, 64)   uint8 (FP4 packed)
                       const torch::Tensor& qscale,   // (BH, T, 4)    uint8 (E8M0)
                       const torch::Tensor& kscale,   // (BH, T, 4)    uint8 (E8M0)
                       const torch::Tensor& vscale,   // (BH, T/128, 128, 4) uint8 (E8M0)
                       const torch::Tensor& q2k_idx,  // (BH*num_q_blks, max_kv) int32
                       const torch::Tensor& q2k_num,  // (BH*num_q_blks,)        int32
                       const torch::Tensor& vbs,      // (num_q_blks,)           int32
                       const torch::Tensor& lim,      // (BH*num_q_blks,)        int32
                       const torch::Tensor& out,      // (BH, T, 128)            bfloat16
                       const torch::Tensor& lse,      // (BH, T)                 float32
                       const torch::Tensor& counters, // (>=2,)                  int32
                       int64_t B,
                       int64_t T,
                       int64_t num_q_blks,
                       int64_t max_kv,
                       int64_t n_dense);

// CSR connectivity ABI.  Identical maths to vsa_qk_fp8_pv_fp4 — the two are
// built from one translation unit with only -DVSA_CSR_ABI flipped — but the
// (q2k_idx, q2k_num, max_kv) rectangle gives way to a flat CSR payload plus one
// packed record per tile.  The rectangle costs O(BH * num_q_blks * max_kv),
// which reaches tens of GiB per head at long contexts; CSR is O(nnz + rows).
//
//   q2k_col_indices : (nnz,) int32, each row's KV block ids contiguous and in
//                     the order the kernel should consume them.
//   q2k_row_meta    : (BH*num_q_blks, 4) int32 {nnz, row_start, first_kv, _},
//                     16-byte aligned, in SCHEDULE order — record t describes
//                     the tile lim[t], not logical row t.
//
// Both come from `aiter.build_l2_aware_lim_vsa_qk_fp8_pv_fp4_csr(...)`.
// row_start is int32, so a single launch is capped at 2^31 CSR entries.
void vsa_qk_fp8_pv_fp4_csr(const torch::Tensor& q,
                           const torch::Tensor& k,
                           const torch::Tensor& v,
                           const torch::Tensor& qscale,
                           const torch::Tensor& kscale,
                           const torch::Tensor& vscale,
                           const torch::Tensor& q2k_col_indices,
                           const torch::Tensor& q2k_row_meta,
                           const torch::Tensor& vbs,
                           const torch::Tensor& lim,
                           const torch::Tensor& out,
                           const torch::Tensor& lse,
                           const torch::Tensor& counters,
                           int64_t B,
                           int64_t T,
                           int64_t num_q_blks,
                           int64_t n_dense);
