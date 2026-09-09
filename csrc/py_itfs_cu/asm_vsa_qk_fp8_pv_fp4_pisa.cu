// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// HIP launcher for fused PISA CSR attention (exact + approximate routes).
// Loads AITER_ASM_DIR/gfx950/vsa/vsa_qk_fp8_pv_fp4_pisa_csr.co.

#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include "vsa_qk_fp8_pv_fp4_pisa.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kBlockX   = 256;
constexpr int kLdsBytes = 36864;
constexpr int kNumCus   = 256;
constexpr int kOcc      = 2;
constexpr int kGridCap  = kNumCus * kOcc;

// Fused kernarg — must match vsa_qk_fp8_pv_fp4_pisa.hip VSA_PARAMS:
//   int B, T, num_q_blks, max_kv_blks (centroids_ready),
//   lim, Q, K, V,
//   exact_col, exact_meta, approx_col, approx_meta,
//   vbs, qscale, kscale, vscale, k_center, value_center,
//   Out, Lse, d_counter, s_counter, n_dense
struct __attribute__((packed)) KernelArgsFused {
    int32_t B;
    int32_t T;
    int32_t num_q_blks;
    int32_t max_kv_blks;
    void*   logical_idx_mapping;
    void*   Q;
    void*   K;
    void*   V;
    void*   exact_col_indices;
    void*   exact_row_meta;
    void*   approx_col_indices;
    void*   approx_row_meta;
    void*   variable_block_sizes;
    void*   qscale;
    void*   kscale;
    void*   vscale;
    void*   k_center;
    void*   value_center;
    void*   Out;
    void*   Lse;
    void*   d_counter;
    void*   s_counter;
    int32_t n_dense;
    char    _pad[400 - (4 * 4 + 8 * 18 + 4)];
};
static_assert(sizeof(KernelArgsFused) == 400,
              "PISA fused KernelArgsFused must be 400 bytes");

struct __attribute__((packed)) KernelArgsStats {
    void*   K;
    void*   V;
    void*   kscale;
    void*   vscale;
    void*   variable_block_sizes;
    void*   k_center;
    void*   value_center;
    void*   used_ids;
    int32_t n_used;
    int32_t BH;
    int32_t T;
};

class PisaQkFp8PvFp4AsmKernel {
   public:
    PisaQkFp8PvFp4AsmKernel(const char* fused_base, const char* stats_name,
                            const char* hsaco) {
        const char* AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        AITER_CHECK(AITER_ASM_DIR != nullptr,
                    "AITER_ASM_DIR not set (needed to locate ", hsaco, ")");
        const std::string full_path =
            std::string(AITER_ASM_DIR) + "/gfx950/" + hsaco;
        std::cout << "[aiter] hipModuleLoad: " << full_path
                  << " GetFunction: " << fused_base << "_h{1,2,4} + "
                  << stats_name;
        HIP_CALL(hipModuleLoad(&module_, full_path.c_str()));

        const std::string b = fused_base;
        HIP_CALL(hipModuleGetFunction(&func_h1_, module_, (b + "_h1").c_str()));
        HIP_CALL(hipModuleGetFunction(&func_h2_, module_, (b + "_h2").c_str()));
        HIP_CALL(hipModuleGetFunction(&func_h4_, module_, (b + "_h4").c_str()));
        HIP_CALL(hipModuleGetFunction(&func_stats_, module_, stats_name));
        std::cout << " Success" << std::endl;
    }

    ~PisaQkFp8PvFp4AsmKernel() { HIP_CALL(hipModuleUnload(module_)); }

    struct Selected {
        hipFunction_t func;
        int num_heads;
    };
    Selected select_for_bh(int64_t BH) const {
        if (BH % 4 == 0) return {func_h4_, 4};
        if (BH % 2 == 0) return {func_h2_, 2};
        return {func_h1_, 1};
    }

    hipFunction_t stats() const { return func_stats_; }

    static void launch_one(hipFunction_t func,
                           void*         args,
                           size_t        arg_size,
                           int           gdx,
                           int           bdx,
                           int           lds_bytes,
                           hipStream_t   stream) {
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          args,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size,
                          HIP_LAUNCH_PARAM_END};
        HIP_CALL(hipModuleLaunchKernel(func,
                                       gdx, 1, 1,
                                       bdx, 1, 1,
                                       lds_bytes,
                                       stream,
                                       nullptr,
                                       config));
    }

   private:
    hipModule_t   module_     = nullptr;
    hipFunction_t func_h1_    = nullptr;
    hipFunction_t func_h2_    = nullptr;
    hipFunction_t func_h4_    = nullptr;
    hipFunction_t func_stats_ = nullptr;
};

PisaQkFp8PvFp4AsmKernel& impl() {
    static PisaQkFp8PvFp4AsmKernel kernel(
        "vsa_qk_fp8_pv_fp4_pisa_csr_kernel",
        "vsa_qk_fp8_pv_fp4_pisa_stats_kernel",
        "vsa/vsa_qk_fp8_pv_fp4_pisa_csr.co");
    return kernel;
}

}  // namespace

void vsa_qk_fp8_pv_fp4_pisa_stats(const torch::Tensor& k,
                                  const torch::Tensor& v,
                                  const torch::Tensor& kscale,
                                  const torch::Tensor& vscale,
                                  const torch::Tensor& vbs,
                                  const torch::Tensor& k_center,
                                  const torch::Tensor& value_center,
                                  const torch::Tensor& used_ids,
                                  int64_t n_used,
                                  int64_t BH,
                                  int64_t T) {
    TORCH_CHECK(T > 0 && (T % 128) == 0,
                "vsa_qk_fp8_pv_fp4_pisa_stats: T must be a positive multiple of 128");
    TORCH_CHECK(BH > 0, "vsa_qk_fp8_pv_fp4_pisa_stats: BH must be > 0");
    TORCH_CHECK(used_ids.dtype() == torch::kInt32 && used_ids.is_contiguous(),
                "vsa_qk_fp8_pv_fp4_pisa_stats: used_ids must be contiguous int32");
    if (n_used > 0) {
        TORCH_CHECK(used_ids.numel() >= n_used,
                    "vsa_qk_fp8_pv_fp4_pisa_stats: used_ids.numel() < n_used");
    }

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(k));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    KernelArgsStats a{};
    a.K                    = k.data_ptr();
    a.V                    = v.data_ptr();
    a.kscale               = kscale.data_ptr();
    a.vscale               = vscale.data_ptr();
    a.variable_block_sizes = vbs.data_ptr();
    a.k_center             = k_center.data_ptr();
    a.value_center         = value_center.data_ptr();
    a.used_ids             = n_used > 0 ? used_ids.data_ptr() : nullptr;
    a.n_used               = static_cast<int32_t>(n_used);
    a.BH                   = static_cast<int32_t>(BH);
    a.T                    = static_cast<int32_t>(T);

    const int64_t blocks = T / 128;
    const int64_t grid64 = (n_used > 0) ? n_used : (BH * blocks);
    const int grid_x = static_cast<int>(grid64 > 0 ? grid64 : 1);
    PisaQkFp8PvFp4AsmKernel::launch_one(
        impl().stats(), &a, sizeof(a), grid_x, kBlockX, 0, stream);
}

void vsa_qk_fp8_pv_fp4_pisa_csr(const torch::Tensor& q,
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
                                int64_t centroids_ready) {
    TORCH_CHECK(counters.dtype() == torch::kInt32 && counters.numel() >= 2,
                "vsa_qk_fp8_pv_fp4_pisa_csr: counters must be int32 with >= 2 "
                "elements");
    TORCH_CHECK(counters.is_contiguous(),
                "vsa_qk_fp8_pv_fp4_pisa_csr: counters must be contiguous");

    const int64_t BH = q.size(0);
    TORCH_CHECK(BH % B == 0,
                "vsa_qk_fp8_pv_fp4_pisa_csr: q.size(0)=", BH,
                " must be divisible by B=", B);
    const int64_t total_tiles = BH * num_q_blks;
    TORCH_CHECK(total_tiles > 0,
                "vsa_qk_fp8_pv_fp4_pisa_csr: empty workload (BH*num_q_blks == 0)");

    TORCH_CHECK(exact_col_indices.dtype() == torch::kInt32 &&
                    exact_col_indices.is_contiguous(),
                "vsa_qk_fp8_pv_fp4_pisa_csr: exact_col_indices must be contiguous int32");
    TORCH_CHECK(approx_col_indices.dtype() == torch::kInt32 &&
                    approx_col_indices.is_contiguous(),
                "vsa_qk_fp8_pv_fp4_pisa_csr: approx_col_indices must be contiguous int32");
    TORCH_CHECK(exact_row_meta.dtype() == torch::kInt32 &&
                    exact_row_meta.is_contiguous(),
                "vsa_qk_fp8_pv_fp4_pisa_csr: exact_row_meta must be contiguous int32");
    TORCH_CHECK(approx_row_meta.dtype() == torch::kInt32 &&
                    approx_row_meta.is_contiguous(),
                "vsa_qk_fp8_pv_fp4_pisa_csr: approx_row_meta must be contiguous int32");
    TORCH_CHECK(lim.dtype() == torch::kInt32 && lim.is_contiguous(),
                "vsa_qk_fp8_pv_fp4_pisa_csr: lim must be contiguous int32");
    TORCH_CHECK(lim.numel() == total_tiles,
                "vsa_qk_fp8_pv_fp4_pisa_csr: lim must hold BH*num_q_blks=",
                total_tiles, " entries, got ", lim.numel());
    TORCH_CHECK(exact_row_meta.dim() == 2 &&
                    exact_row_meta.size(0) == total_tiles &&
                    exact_row_meta.size(1) == 4,
                "vsa_qk_fp8_pv_fp4_pisa_csr: exact_row_meta must be (",
                total_tiles, ", 4)");
    TORCH_CHECK(approx_row_meta.dim() == 2 &&
                    approx_row_meta.size(0) == total_tiles &&
                    approx_row_meta.size(1) == 4,
                "vsa_qk_fp8_pv_fp4_pisa_csr: approx_row_meta must be (",
                total_tiles, ", 4)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(exact_row_meta.data_ptr()) % 16 == 0,
                "vsa_qk_fp8_pv_fp4_pisa_csr: exact_row_meta must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(approx_row_meta.data_ptr()) % 16 == 0,
                "vsa_qk_fp8_pv_fp4_pisa_csr: approx_row_meta must be 16-byte aligned");
    TORCH_CHECK(exact_col_indices.numel() <= INT32_MAX &&
                    approx_col_indices.numel() <= INT32_MAX,
                "vsa_qk_fp8_pv_fp4_pisa_csr: CSR nnz exceeds int32 row_start");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    HIP_CALL(hipMemsetAsync(counters.data_ptr(), 0,
                            counters.numel() * sizeof(int32_t),
                            stream));

    const auto picked   = impl().select_for_bh(BH);
    const int32_t B_eff = static_cast<int32_t>(BH / picked.num_heads);

    KernelArgsFused a{};
    a.B                    = B_eff;
    a.T                    = static_cast<int32_t>(T);
    a.num_q_blks           = static_cast<int32_t>(num_q_blks);
    a.max_kv_blks          = static_cast<int32_t>(centroids_ready);
    a.logical_idx_mapping  = lim.data_ptr();
    a.Q                    = q.data_ptr();
    a.K                    = k.data_ptr();
    a.V                    = v.data_ptr();
    a.exact_col_indices    = exact_col_indices.data_ptr();
    a.exact_row_meta       = exact_row_meta.data_ptr();
    a.approx_col_indices   = approx_col_indices.data_ptr();
    a.approx_row_meta      = approx_row_meta.data_ptr();
    a.variable_block_sizes = vbs.data_ptr();
    a.qscale               = qscale.data_ptr();
    a.kscale               = kscale.data_ptr();
    a.vscale               = vscale.data_ptr();
    a.k_center             = k_center.data_ptr();
    a.value_center         = value_center.data_ptr();
    a.Out                  = out.data_ptr();
    a.Lse                  = lse.data_ptr();
    a.d_counter            = counters.data_ptr();
    a.s_counter            = reinterpret_cast<int32_t*>(counters.data_ptr()) + 1;
    a.n_dense              = static_cast<int32_t>(n_dense);

    const int grid_x = (total_tiles < kGridCap)
                           ? static_cast<int>(total_tiles)
                           : kGridCap;
    PisaQkFp8PvFp4AsmKernel::launch_one(
        picked.func, &a, sizeof(a), grid_x, kBlockX, kLdsBytes, stream);
}
