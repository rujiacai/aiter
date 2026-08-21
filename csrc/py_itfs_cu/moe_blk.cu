// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host launcher for the prebuilt blockwise-fp8 MoE code objects under
// hsa/{arch}/moe_blk/. They are ordinary AMDGPU code objects, so AiterAsmKernel
// loads them exactly like the hand-written asm kernels.
//
// Unlike the asm families there is no CSV lookup: the code object name is fully
// determined by the shape/tile/flags the caller already knows, so Python builds
// it and passes it down.
#include <hip/hip_runtime.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "aiter_hip_common.h"

namespace {

// Natural (compiler) layout, not the padded asm convention. The two stages have
// genuinely different signatures and must not share a struct: stage1 carries the
// smooth_scale pointer and the clamp, stage2 has neither, so reusing stage1's
// layout would shift every scalar by 8 bytes. The totals below have to match
// kernarg_segment_size in manifest.csv (100 and 88).
struct __attribute__((packed)) MoeBlkStage1Args
{
    void* ptr_out;
    void* ptr_x;
    void* ptr_w;
    void* ptr_scale_x;
    void* ptr_scale_w;
    void* ptr_sorted_token_ids;
    void* ptr_expert_ids;
    void* ptr_sorted_weights;
    void* ptr_num_valid_ids;
    void* ptr_smooth_scale;
    unsigned int tokens;
    unsigned int n_in;
    unsigned int k_in;
    unsigned int size_expert_ids;
    float swiglu_limit;
};
static_assert(sizeof(MoeBlkStage1Args) == 100, "stage1 kernarg layout drifted");

struct __attribute__((packed)) MoeBlkStage2Args
{
    void* ptr_out;
    void* ptr_x;
    void* ptr_w;
    void* ptr_scale_x;
    void* ptr_scale_w;
    void* ptr_sorted_token_ids;
    void* ptr_expert_ids;
    void* ptr_sorted_weights;
    void* ptr_num_valid_ids;
    unsigned int tokens;
    unsigned int n_in;
    unsigned int k_in;
    unsigned int size_expert_ids;
};
static_assert(sizeof(MoeBlkStage2Args) == 88, "stage2 kernarg layout drifted");

// One workgroup per (N tile, sorted M block); 256 threads is baked into every
// code object in this family (max_flat_workgroup_size in the manifest).
constexpr int MOE_BLK_BLOCK = 256;

template <typename Args>
void launch_moe_blk(const char* kernel_name,
                    const char* co_name,
                    Args& args,
                    int grid_x,
                    int grid_y,
                    const hipStream_t stream)
{
    size_t arg_size = sizeof(Args);
    // One kernel object per .co, kept alive for the process: hipModuleLoadData
    // is far too slow to redo per call.
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> cache;
    static std::mutex cache_mutex;

    AiterAsmKernel* impl = nullptr;
    {
        std::lock_guard<std::mutex> guard(cache_mutex);
        auto it = cache.find(co_name);
        if(it == cache.end())
        {
            std::string path = std::string("moe_blk/") + co_name;
            it = cache.emplace(co_name,
                               std::make_unique<AiterAsmKernel>(kernel_name, path.c_str()))
                     .first;
        }
        impl = it->second.get();
    }

    impl->launch_kernel({&args,
                         &arg_size,
                         grid_x,
                         grid_y,
                         1,
                         MOE_BLK_BLOCK,
                         1,
                         1,
                         stream});
}

void* data_or_null(aiter_tensor_t* t) { return t == nullptr ? nullptr : t->ptr; }

} // namespace

AITER_CTYPES_ERROR_DEF

// stage1: gate/up GEMM + activation. `n_in` is inter_dim (gate and up are two
// separate B tiles), so grid.x = inter_dim / tile_n.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    moe_blk_stage1,
    (aiter_tensor_t * out,
     aiter_tensor_t* input,
     aiter_tensor_t* w1,
     aiter_tensor_t* a1_scale,
     aiter_tensor_t* w1_scale,
     aiter_tensor_t* sorted_token_ids,
     aiter_tensor_t* sorted_expert_ids,
     aiter_tensor_t* sorted_weights,
     aiter_tensor_t* num_valid_ids,
     aiter_tensor_t* smooth_scale,
     int tokens,
     int inter_dim,
     int model_dim,
     int tile_n,
     float swiglu_limit,
     const char* co_name,
     hipStream_t stream),
    (out,
     input,
     w1,
     a1_scale,
     w1_scale,
     sorted_token_ids,
     sorted_expert_ids,
     sorted_weights,
     num_valid_ids,
     smooth_scale,
     tokens,
     inter_dim,
     model_dim,
     tile_n,
     swiglu_limit,
     co_name,
     stream))
{
    const int size_expert_ids = sorted_expert_ids->size(0);
    MoeBlkStage1Args args{out->ptr,
                          input->ptr,
                          w1->ptr,
                          data_or_null(a1_scale),
                          data_or_null(w1_scale),
                          sorted_token_ids->ptr,
                          sorted_expert_ids->ptr,
                          data_or_null(sorted_weights),
                          num_valid_ids->ptr,
                          data_or_null(smooth_scale),
                          static_cast<unsigned int>(tokens),
                          static_cast<unsigned int>(inter_dim),
                          static_cast<unsigned int>(model_dim),
                          static_cast<unsigned int>(size_expert_ids),
                          swiglu_limit};

    AITER_CHECK(inter_dim % tile_n == 0,
                __func__,
                "inter_dim must be a multiple of tile_n");
    launch_moe_blk("moe_gemm1_0", co_name, args, inter_dim / tile_n, size_expert_ids, stream);
}

// stage2: down GEMM with atomic topk reduction. `n_in` is model_dim.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    moe_blk_stage2,
    (aiter_tensor_t * out,
     aiter_tensor_t* inter_states,
     aiter_tensor_t* w2,
     aiter_tensor_t* a2_scale,
     aiter_tensor_t* w2_scale,
     aiter_tensor_t* sorted_token_ids,
     aiter_tensor_t* sorted_expert_ids,
     aiter_tensor_t* sorted_weights,
     aiter_tensor_t* num_valid_ids,
     int tokens,
     int model_dim,
     int inter_dim,
     int tile_n,
     const char* co_name,
     hipStream_t stream),
    (out,
     inter_states,
     w2,
     a2_scale,
     w2_scale,
     sorted_token_ids,
     sorted_expert_ids,
     sorted_weights,
     num_valid_ids,
     tokens,
     model_dim,
     inter_dim,
     tile_n,
     co_name,
     stream))
{
    const int size_expert_ids = sorted_expert_ids->size(0);
    MoeBlkStage2Args args{out->ptr,
                          inter_states->ptr,
                          w2->ptr,
                          data_or_null(a2_scale),
                          data_or_null(w2_scale),
                          sorted_token_ids->ptr,
                          sorted_expert_ids->ptr,
                          data_or_null(sorted_weights),
                          num_valid_ids->ptr,
                          static_cast<unsigned int>(tokens),
                          static_cast<unsigned int>(model_dim),
                          static_cast<unsigned int>(inter_dim),
                          static_cast<unsigned int>(size_expert_ids)};

    AITER_CHECK(model_dim % tile_n == 0,
                __func__,
                "model_dim must be a multiple of tile_n");
    launch_moe_blk("moe_gemm2_0", co_name, args, model_dim / tile_n, size_expert_ids, stream);
}
