// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

union MlaWorkInfo
{
    struct
    {
        int32_t batch_idx;
        int32_t partial_qo_loc;
        int32_t qo_start;
        int32_t qo_end;
        int32_t kv_start;
        int32_t kv_end;
        int32_t kv_offset;
        int32_t padding[1];
    };
    uint32_t u32All[8];
};
constexpr size_t kSizeMlaWorkInfoInDw = sizeof(MlaWorkInfo) / sizeof(uint32_t);
static_assert(kSizeMlaWorkInfoInDw == 8);

union MlaPartialTileInfo
{
    struct
    {
        int32_t q_start;
        int32_t q_end;
    };
    uint32_t u32All[2];
};
constexpr size_t kSizeMlaPartialTileInfoInDw = sizeof(MlaPartialTileInfo) / sizeof(uint32_t);
static_assert(kSizeMlaPartialTileInfoInDw == 2);

void get_mla_metadata_v1(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                         const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
                         const torch::Tensor& kv_last_page_lens, // [batch size]
                         const int32_t num_heads_per_head_k,
                         const int32_t num_heads_k,
                         const bool is_causal,
                         torch::Tensor& work_metadata_ptrs,
                         torch::Tensor& work_indptr,
                         torch::Tensor& work_info,
                         torch::Tensor& reduce_indptr,
                         torch::Tensor& reduce_final_map,
                         torch::Tensor& reduce_partial_map,
                         const int32_t page_size,
                         const int32_t kv_granularity,
                         const int32_t max_seqlen_qo,
                         const int32_t uni_seqlen_qo,
                         const bool fast_mode,
                         const int32_t topk,
                         const int32_t max_split_per_batch,
                         const bool intra_batch_mode,
                         const std::optional<at::ScalarType> dtype_q,
                         const std::optional<at::ScalarType> dtype_kv,
                         const bool is_cp_round_robin = false);

std::vector<torch::Tensor>
get_mla_metadata_v1_no_redundant(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                                 const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
                                 const int32_t num_heads_per_head_k,
                                 const int32_t num_heads_k,
                                 const bool is_causal,
                                 const int32_t kv_granularity);

// bf16 MLA paged decode over a unified KV pool (page_size=1, K and V share the
// 512-wide row) indexed by per-token CSR slot lists. Query token t attends over
// kv_indices[kv_indptr[t] : kv_indptr[t+1]] and nothing else, so any causality
// the caller wants -- MTP draft masking, sliding windows, per-token compressed
// slot selection -- is expressed by which slots it puts in each row.
// gfx942 only. kv_splits=0 picks the split count from the CU count.
torch::Tensor mla_decode_v4_bf16(torch::Tensor q,           // [T, H, 512] bf16
                                 torch::Tensor unified_kv,  // [num_slots, 512] bf16
                                 torch::Tensor kv_indices,  // [nnz] int32
                                 torch::Tensor kv_indptr,   // [T+1] int32
                                 torch::Tensor attn_sink,   // [H] fp32
                                 double softmax_scale,
                                 int64_t kv_splits);

// bf16 MLA sparse paged prefill over two KV sources: a paged `unified_kv` pool
// holding the prefix (page_size=1, K and V share the 512-wide row) and a flat
// `kv` holding this forward's own K. Each is indexed by its own per-token CSR
// slot list, and query token t attends over the concatenation of
//   unified_kv[kv_indices_prefix[kv_indptr_prefix[t] : kv_indptr_prefix[t+1]]]
//   kv        [kv_indices_extend[kv_indptr_extend[t] : kv_indptr_extend[t+1]]]
// and nothing else, so any causality the caller wants is expressed by which
// slots it puts in each row. `-1` entries are skipped when check_sentinel.
//
// The gfx942 counterpart of pa_sparse_prefill_opus, which is gfx950-only.
// H must be a positive multiple of 16; D is fixed at 512.
torch::Tensor mla_prefill_v4_bf16(torch::Tensor q,                  // [N, H, 512] bf16
                                  torch::Tensor unified_kv,         // [total_pages, 512] bf16
                                  torch::Tensor kv_indices_prefix,  // [nnz_p] int32
                                  torch::Tensor kv_indptr_prefix,   // [N+1] int32
                                  torch::Tensor kv,                 // [total_tokens, 512] bf16
                                  torch::Tensor kv_indices_extend,  // [nnz_e] int32
                                  torch::Tensor kv_indptr_extend,   // [N+1] int32
                                  torch::Tensor attn_sink,          // [H] fp32
                                  double softmax_scale,
                                  bool check_sentinel);

void mla_reduce_v1(const torch::Tensor& partial_output,
                   const torch::Tensor& partial_lse,
                   const torch::Tensor& reduce_indptr,
                   const std::optional<torch::Tensor>& reduce_final_map,
                   const torch::Tensor& reduce_partial_map,
                   const int max_seqlen_q,
                   const int num_kv_splits,
                   torch::Tensor& final_output,
                   std::optional<torch::Tensor>& final_lse);

void get_pa_metadata_v1(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                        const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
                        const int32_t num_heads_per_head_k,
                        const int32_t num_heads_k,
                        const bool is_causal,
                        torch::Tensor& work_metadata_ptrs,
                        torch::Tensor& work_info_set,
                        torch::Tensor& work_indptr,
                        torch::Tensor& reduce_indptr,
                        torch::Tensor& reduce_final_map,
                        torch::Tensor& reduce_partial_map,
                        const int32_t kv_granularity,
                        const int32_t max_seqlen_qo,
                        const int32_t uni_seqlen_qo,
                        const bool fast_mode,
                        const int32_t topk,
                        const int32_t max_split_per_batch);

void hk_mla_decode_fwd(
    torch::Tensor& query,                   // [num_seqs, num_heads, head_size]
    torch::Tensor& kv_buffer,               // [num_page, page_size, num_kv_heads, head_size]
    const torch::Tensor& qo_indptr,         // [batch_size+1]
    const torch::Tensor& kv_indptr,         // [batch_size+1]
    const torch::Tensor& kv_page_indices,   // [num_page_used]
    const torch::Tensor& kv_last_page_lens, // [batch_size]
    const torch::Tensor& work_indptr,       // metadata
    const torch::Tensor& work_info_set,
    const int max_seqlen_q,
    const float softmax_scale,
    torch::Tensor& split_output,  // Output: [batch_size, num_kv_splits, num_heads, v_head_dim]
    torch::Tensor& split_lse,     // Output: [batch_size, num_kv_splits, num_heads,  1]
    torch::Tensor& final_output); // Output: [batch_size, num_heads, v_head_dim]
