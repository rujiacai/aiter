// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

#include "common/arithmetic.hpp"

namespace aiter::mxfp4_moe::moe_sort_scales {

constexpr int MASK_TOKEN_ID = 0x00FFFFFF;

template <int BM, int NUM_EXPERTS, int D_HIDDEN, int BK,
          int N_CTAS, int THREADS_PER_CTA>
__global__ void sort_scales_kernel_impl(
    const int M, const int MAX_SORTED,
    const uint8_t *a_scale, const int32_t *sorted_token_ids, const int32_t *cumsum_tensor,
    uint8_t *a_scale_sorted_shuffled) {

    static_assert(BM == 32 || BM == 64 || BM == 128 || BM == 256,
                  "BM ∈ {32, 64, 128, 256}");
    static_assert(BK == 128 || BK == 256, "BK ∈ {128, 256}");

    constexpr int A_SCALE_COLS = exact_div<D_HIDDEN, 32>();
    constexpr int MN_PACK = 2;
    constexpr int K_PACK  = BK / 128;
    constexpr int C_M1    = BM / (16 * MN_PACK);
    constexpr int C_K1    = (D_HIDDEN / 32) / (4 * K_PACK);
    constexpr int K_LANE  = 4;
    constexpr int N_LANE  = 16;
    constexpr int DWORDS_PER_CHUNK = C_M1 * C_K1 * K_LANE * N_LANE;

    static_assert(C_M1 >= 1, "BM too small for mn_pack=2 (need BM ≥ 32)");

    const int n_chunks       = MAX_SORTED / BM;
    const int actual_sorted  = cumsum_tensor[0];
    const int actual_n_chunks = (actual_sorted + BM - 1) / BM;

    // One thread per (chunk, mi, k_lane, n_lane) walks all C_K1 k-units: the
    // two sorted rows' token ids are loaded once instead of once per output
    // dword, and each k-unit reads its K_PACK*4 scale bytes of a row in one
    // load. Consecutive lanes still write consecutive output dwords.
    static_assert(A_SCALE_COLS % (K_PACK * 4) == 0, "k-unit must tile a scale row");
    using KUnit = std::conditional_t<K_PACK == 2, uint64_t, uint32_t>;
    constexpr int KU_STRIDE = K_LANE * N_LANE;  // output dwords between k-units
    const int total_work = n_chunks * C_M1 * KU_STRIDE;
    constexpr int total_threads = N_CTAS * THREADS_PER_CTA;
    const int global_tid = blockIdx.x * THREADS_PER_CTA + threadIdx.x;
    uint32_t* out_dw = reinterpret_cast<uint32_t*>(a_scale_sorted_shuffled);

    for (int work_id = global_tid; work_id < total_work; work_id += total_threads) {
        int r = work_id;
        const int n_lane = r % N_LANE;  r /= N_LANE;
        const int k_lane = r % K_LANE;  r /= K_LANE;
        const int mi     = r % C_M1;    r /= C_M1;
        const int chunk  = r;
        uint32_t* dst = out_dw + ((long long)(chunk * C_M1 + mi) * C_K1) * KU_STRIDE
                        + k_lane * N_LANE + n_lane;

        if (chunk >= actual_n_chunks) {
            #pragma unroll 4
            for (int ku = 0; ku < C_K1; ku++)
                dst[ku * KU_STRIDE] = 0;
            continue;
        }

        const KUnit* rows[MN_PACK];
        #pragma unroll
        for (int im_a = 0; im_a < MN_PACK; im_a++) {
            const int sorted_row = chunk * BM + (mi * MN_PACK + im_a) * 16 + n_lane;
            int tid = 0;
            if (sorted_row < actual_sorted) {
                const int sti_val = sorted_token_ids[sorted_row] & MASK_TOKEN_ID;
                tid = (sti_val < M) ? sti_val : 0;
            }
            rows[im_a] = reinterpret_cast<const KUnit*>(a_scale + (long long)tid * A_SCALE_COLS);
        }

        #pragma unroll 4
        for (int ku = 0; ku < C_K1; ku++) {
            uint32_t packed = 0;
            #pragma unroll
            for (int im_a = 0; im_a < MN_PACK; im_a++) {
                const KUnit unit = rows[im_a][ku];
                #pragma unroll
                for (int ikxdl = 0; ikxdl < K_PACK; ikxdl++) {
                    const uint32_t b = (unit >> (8 * (ikxdl * 4 + k_lane))) & 0xFFu;
                    packed |= b << (8 * (ikxdl * MN_PACK + im_a));
                }
            }
            dst[ku * KU_STRIDE] = packed;
        }
    }
}

template <int BM, int NE, int D_HIDDEN, int BK,
          int N_CTAS, int THREADS_PER_CTA>
inline void launch(
    hipStream_t stream, int M, int max_sorted,
    const uint8_t *a_scale, const int32_t *sorted_token_ids, const int32_t *cumsum,
    uint8_t *a_scale_sorted_shuffled)
{
    sort_scales_kernel_impl<BM, NE, D_HIDDEN, BK, N_CTAS, THREADS_PER_CTA>
        <<<N_CTAS, THREADS_PER_CTA, 0, stream>>>(
            M, max_sorted, a_scale, sorted_token_ids, cumsum, a_scale_sorted_shuffled);
}

} // namespace aiter::mxfp4_moe::moe_sort_scales
