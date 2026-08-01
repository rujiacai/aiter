// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// bf16 MLA sparse paged-prefill for gfx942 (MI300/MI308), two-source unified_kv
// CSR contract. The gfx942 counterpart of pa_sparse_prefill_opus, which is
// gfx950-only, and a drop-in for the same call site.
//
// Build note: the optCompilerConfig entry for module_mla_prefill_v4_bf16 passes
// `-mllvm -enable-post-misched=1`, cancelling the `=0` aiter applies globally,
// for the same reason decode_v4_bf16.cu does -- this kernel leans on the
// post-RA scheduler to hide MFMA and ds_read latency behind the
// software-pipelined tile loop. The cancellation works because core.py sorts
// the flag list, so `=1` lands after `=0` and LLVM takes the later one; verify
// it survived if that sort changes.
//
// Structurally this is the decode kernel from ../MLAAttetion with a second KV
// source spliced in. That is not a coincidence: in MLA every query token picks
// its own sparse slot list, so no two tokens share a KV tile and the GEMM's M
// dimension is the head count, not the token count. Prefill is therefore a
// batched decode, and the Triton reference
// (sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_prefill.py) is
// literally its decode kernel with a second region appended to the tile loop.
//
// Contract: query token t attends over the concatenation of
//   unified_kv[kv_indices_prefix[kv_indptr_prefix[t] : kv_indptr_prefix[t+1]]]
//   kv        [kv_indices_extend[kv_indptr_extend[t] : kv_indptr_extend[t+1]]]
// and nothing else. `-1` entries are skipped. Online softmax is order-invariant
// so the region order does not affect the result, and with an empty extend
// region this is bit-identical to the decode kernel on the same prefix row.
//
// The two regions are walked as one virtual row rather than as two loops. A row
// is loaded cooperatively by a whole wave (64 lanes x 8 bf16 = one 512-wide
// row), so the row index is wave-uniform and so is the choice of source buffer;
// splicing costs one wave-uniform select per prefetched row and, unlike two
// back-to-back loops, never drains the prefetch pipeline at the boundary.
//
// Sentinel handling is the one place the two-source form needs state the decode
// kernel did not: validity is decided by the wave that loaded the row but
// consumed by the wave that owns the softmax row, so it crosses waves. It goes
// through LDS as one qword per wave (lds_vm), OR-reduced by the reader once per
// tile -- a broadcast read, no bank conflict, and no atomics or extra barrier,
// which a shared bitmask would have needed to reset.
//
// Everything else -- the AGPR-pinned Q, the QSPLIT partial chains, the branched
// accumulator rescale, the PV gather, the LDS padding -- is carried over from
// the decode kernel unchanged, including the reasoning recorded there.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

#include "mla.h"

#ifndef MLA_BK
#define MLA_BK 48
#endif

// Upper bound on head tiles per workgroup; heads past it spill over into more
// blocks along grid.y.
//
// This is a register knob, not a tiling preference. A wave's accumulator is
// HT*16 heads by DH/NW columns, so at NW=8 it costs 16*HT VGPRs, and Q costs
// another 64 that live across the whole kv loop. HT=7 is the first tile count
// where that no longer fits: the pair tops out at the 256 VGPR + 128 AGPR
// ceiling and starts spilling to scratch (100 B/lane at HT=7, 132 at HT=8),
// and the accumulator lands in AGPRs, which turns the once-per-tile softmax
// rescale into 128 v_accvgpr_read + 128 v_accvgpr_write around its 64
// multiplies. That was 48% of all VALU in the main loop at HT=8.
//
// 6 is the largest tile count that stays clear of both. H=128 does not divide
// by 6 and settles on HT=4 with two blocks of heads, which measured 73.0
// against 68.2 TFLOP/s for the spilling HT=8; H=96 keeps HT=6 and one block.
#ifndef MLA_MAXHT
#define MLA_MAXHT 6
#endif

namespace {

typedef __bf16 bf16_t;
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

constexpr int WARP = 64;
constexpr int DH = 512;       // kv_lora_rank
constexpr int NKC = DH / 16;  // QK contraction steps
// The two kv tile widths the host picks between. Wide keeps a workgroup's whole
// CU to itself; narrow halves lds_k so a second one can share it.
constexpr int BK_WIDE = MLA_BK;
constexpr int BK_NARROW = 16;

#ifndef MLA_KPAD
#define MLA_KPAD 4
#endif
constexpr int KPAD = MLA_KPAD;
constexpr int KD = DH + KPAD;

#ifndef MLA_PPAD
#define MLA_PPAD 4
#endif
constexpr int PPAD = MLA_PPAD;

#ifndef MLA_QSPLIT
#define MLA_QSPLIT 2
#endif
constexpr int QSPLIT = MLA_QSPLIT;

// Everything the wave count and the kv tile width decide together. Both are
// template parameters rather than build constants because the best pair depends
// on H, and the host has to be able to pick per call -- see the launch site.
template <int NW_, int BK>
struct Cfg {
    static constexpr int NW = NW_;
    static constexpr int NTHREADS = NW * WARP;
    static constexpr int DVW = DH / NW;  // dv slice per wave in PV
    static constexpr int DVT = DVW / 16;
    static constexpr int MAX_HT = NW;  // a QK wave must own a whole head tile
    static constexpr int KVT = BK / 16;
    static constexpr int PD = BK + PPAD;
    static constexpr int COOP = BK * DH / (NTHREADS * 8);
    static_assert(DVW % 16 == 0, "dv slice must be a whole number of MFMA tiles");
    static_assert(MAX_HT <= 8, "the host dispatch enumerates HT up to 8");
    static_assert(BK % 16 == 0, "kv tile must be a whole number of MFMA tiles");
    static_assert(BK <= 64, "the validity mask is one qword per wave");
    static_assert(COOP >= 1, "kv tile too small for the workgroup");
};

constexpr float NEG_INF = -3.0e38f;
constexpr float LOG2E_F = 1.4426950408889634f;

template <int CTRL>
__device__ __forceinline__ float dpp_ror(float v) {
    return __builtin_bit_cast(
        float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xf, 0xf, false));
}

__device__ __forceinline__ float row_max16(float v) {
    v = fmaxf(v, dpp_ror<0x121>(v));
    v = fmaxf(v, dpp_ror<0x122>(v));
    v = fmaxf(v, dpp_ror<0x124>(v));
    return fmaxf(v, dpp_ror<0x128>(v));
}

__device__ __forceinline__ float row_sum16(float v) {
    v += dpp_ror<0x121>(v);
    v += dpp_ror<0x122>(v);
    v += dpp_ror<0x124>(v);
    return v + dpp_ror<0x128>(v);
}

__device__ __forceinline__ f32x4 mfma16(bf16x4 a, bf16x4 b, f32x4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// Park a read-only MFMA operand in the AGPR file. See the decode kernel for why
// this is an empty asm rather than a hand-written MFMA.
__device__ __forceinline__ void pin_agpr(bf16x4& x) { asm("" : "+a"(x)); }

#ifndef MLA_XCD
#define MLA_XCD 4
#endif
// MI308X is 4 XCDs of 20 CUs, each with its own 4 MB L2, and the dispatcher
// hands workgroup `L` to XCD `L % XCD`. The decode kernel remaps that so the
// blocks that stream byte-identical KV -- the hblocks of one token -- land on
// one die instead of being spread over every die by `hblk % 4`.
//
// Prefill must not do that unconditionally, and the reason is load balance
// rather than locality. The remap gives XCD `c` the contiguous logical range
// [c*per, (c+1)*per), and prefill's work per token need not be uniform: where
// it is not, a contiguous range hands XCD 0 all the short tokens and XCD 3 all
// the long ones, and the kernel waits for XCD 3.
//
// So it only applies when there is a sharing group to keep together at all.
// There is not for any head count this kernel ships: HT = min(H/16, NW), so
// hblocks is 1 for every H <= 128 and the remap would only ever be permuting
// tokens. It is left in for H > 128, where the groups do exist -- and where the
// same skew argument means it should be re-measured before being trusted.
__device__ __forceinline__ int xcd_swizzle(int L, int N, int nhb) {
    if (MLA_XCD <= 1 || nhb <= 1) return L;
    const int per = N / MLA_XCD;
    if (per == 0 || L >= per * MLA_XCD) return L;
    return (L % MLA_XCD) * per + (L / MLA_XCD);
}

// At HT=1 -- a TP=8 rank holding 16 of 128 heads -- one head tile is all there
// is to hand out, so the wave that owns it does all NKC*KVT of QK's MFMAs while
// every other wave waits: 64 of the workgroup's 128 MFMAs land on one wave, and
// the critical path is 72 against an ideal 16.
//
// Cutting QK along KV instead, which is what OPUS's 16mx1_16nx4 does, does not
// fit here. The MFMA needs 16 KV rows per wave, so NW=4 waves want BK=64, and
// lds_k alone would be 64*516*2 = 66 KiB against a 64 KiB budget -- at D=512 a
// KV row is a full KiB and LDS holds 64 of them, full stop.
//
// Cutting along the contraction instead does fit -- wave w takes NKC/NW of the
// 32 steps, all waves share the one K tile, and the partials are summed through
// a 17 KiB LDS buffer before the softmax -- and it does what it says: the
// critical path drops from 72 MFMAs to 16. It is still off by default because
// it measured *slower* (CSA N=512 18.7 against 19.5 TFLOP/s), which is the
// useful result here.
//
// The reason is that H=16 is not MFMA-bound at all. Per tile a workgroup issues
// 368 LDS ops against 128 MFMAs, and 256 of those 368 are PV's B-operand
// gather: that operand wants one dv column across four kv rows, gfx942 has no
// ds_read_tr16, so it costs four 2-byte reads where an 8-byte read would do.
// At HT=8 the same 256 ops are amortised over eight times the MFMAs; at HT=1
// there is nothing to amortise them against. Splitting the contraction spreads
// QK's reads over every wave but raises the workgroup's total LDS count by the
// reduction traffic, and the CU's LDS issue port is shared, so the total is
// what binds. Left buildable for when a transposed-V layout makes PV cheap
// enough that QK becomes the constraint.
#ifndef MLA_DSPLIT
#define MLA_DSPLIT 0
#endif
// Blocks per CU the compiler must leave room for. At HT=1 the workgroup has one
// head tile of work and little to overlap with, so the latency hiding has to
// come from a second resident workgroup rather than from within this one.
#ifndef MLA_MINBLK
#define MLA_MINBLK 1
#endif
constexpr int DSPLIT_SPAD = 17;  // pad the kv row so the i-stride walks banks

// SENT compiles in the `-1` skip. The Triton reference turns the same check off
// when the caller guarantees dense rows, and it is not free here either: it
// costs the LDS mask round trip below and a select in the softmax.
template <int NW, int HT, int BK, int SENT, int DS>
__global__ __launch_bounds__(NW * WARP, MLA_MINBLK) void mla_prefill_kernel(
    const bf16_t* __restrict__ q,                  // [N, H, DH]
    const bf16_t* __restrict__ unified_kv,         // [total_pages, DH]
    const int* __restrict__ kv_indices_prefix,     // [nnz_p]
    const int* __restrict__ kv_indptr_prefix,      // [N+1]
    const bf16_t* __restrict__ kv_extend,          // [total_tokens, DH]
    const int* __restrict__ kv_indices_extend,     // [nnz_e]
    const int* __restrict__ kv_indptr_extend,      // [N+1]
    const float* __restrict__ attn_sink,           // [H]
    bf16_t* __restrict__ out,                      // [N, H, DH]
    const int H, const float qk_scale) {
    using C = Cfg<NW, BK>;
    static_assert(HT >= 1 && HT <= C::MAX_HT, "one QK wave per head tile");
    static_assert(!DS || HT == 1, "the contraction split only applies to a lone head tile");
    static_assert(!DS || NKC % NW == 0, "the contraction must divide over the waves");
    constexpr int HB = HT * 16;
    constexpr int NTHREADS = C::NTHREADS;
    constexpr int DVW = C::DVW;
    constexpr int DVT = C::DVT;
    constexpr int KVT = C::KVT;
    constexpr int PD = C::PD;
    constexpr int COOP = C::COOP;
    // Contraction steps this wave owns. Without the split a QK wave walks the
    // whole of D for its head tile; with it, NKC/NW of D for the only one.
    constexpr int DW = DS ? NKC / NW : NKC;

    const int ntok = gridDim.x, nhb = gridDim.y;
    int lin = xcd_swizzle(blockIdx.x + ntok * blockIdx.y, ntok * nhb, nhb);
    const int hblk = lin % nhb;
    const int tok = lin / nhb;

    const int tid = threadIdx.x;
    const int wid = tid >> 6;
    const int lane = tid & (WARP - 1);
    const int lm = lane & 15;
    const int lg = lane >> 4;

    const int p_start = kv_indptr_prefix[tok];
    const int p_len = kv_indptr_prefix[tok + 1] - p_start;
    const int e_start = kv_indptr_extend[tok];
    const int e_len = kv_indptr_extend[tok + 1] - e_start;
    const int total = p_len + e_len;

    const int hb_base = hblk * HB;
    if (total <= 0) {
        // Both regions empty: the reference still runs the sink finalization,
        // which drives alpha to zero and leaves a zero row.
        for (int i = tid; i < HB * DH; i += NTHREADS)
            out[((size_t)tok * H + hb_base + i / DH) * DH + i % DH] = (bf16_t)0.f;
        return;
    }

    // With the contraction split every wave contributes to QK and the head tile
    // is always 0; the softmax that follows it still belongs to one wave.
    const int qk_ht = DS ? 0 : wid;
    const bool qk_wave = DS ? true : (wid < HT);
    const bool softmax_wave = DS ? (wid == 0) : (wid < HT);
    const int dc_base = DS ? wid * DW : 0;  // this wave's first contraction step
    const int dv_base = wid * DVW;

    __shared__ bf16_t lds_k[BK * KD];
    __shared__ bf16_t lds_p[HT][16][PD];
    __shared__ float lds_alpha[HB];
    __shared__ unsigned long long lds_vm[NW];
    // Partial scores, one plane per wave, only under the contraction split.
    __shared__ float lds_s[DS ? NW * KVT * 16 * DSPLIT_SPAD : 1];

    bf16x4 qreg[DW];
    if (qk_wave) {
        const bf16_t* const qbase = q + (size_t)tok * H * DH +
                                    (size_t)(hb_base + qk_ht * 16 + lm) * DH + lg * 4;
#pragma unroll
        for (int c = 0; c < DW; ++c) {
            qreg[c] = *reinterpret_cast<const bf16x4*>(qbase + (dc_base + c) * 16);
            pin_agpr(qreg[c]);
        }
    }

    f32x4 acc[HT][DVT];
#pragma unroll
    for (int h = 0; h < HT; ++h)
#pragma unroll
        for (int d = 0; d < DVT; ++d) acc[h][d] = f32x4{0.f, 0.f, 0.f, 0.f};

    float m_i[4], l_i[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m_i[i] = NEG_INF;
        l_i[i] = 0.f;
    }

    const int ld_col = lane * 8;
    const int uwid = __builtin_amdgcn_readfirstlane(wid);
    int slot_cur[COOP], slot_next[COOP];
    bf16x8 pf[COOP];

    // Row `r` of the virtual concatenation. Both the index fetch and the source
    // buffer follow from r, which is wave-uniform, so nothing per-lane is needed
    // to keep the two regions apart.
    auto load_slots = [&](int* dst, int kvb) {
#pragma unroll
        for (int i = 0; i < COOP; ++i) {
            const int r = min(kvb + uwid + i * NW, total - 1);
            dst[i] = r < p_len ? kv_indices_prefix[p_start + r]
                               : kv_indices_extend[e_start + r - p_len];
        }
    };
    auto load_rows = [&](const int* slots, int kvb) {
#pragma unroll
        for (int i = 0; i < COOP; ++i) {
            const int r = min(kvb + uwid + i * NW, total - 1);
            const bf16_t* const base = r < p_len ? unified_kv : kv_extend;
            // A sentinel row is masked out of the softmax, but the load still
            // has to land somewhere in bounds.
            const int s = SENT ? max(__builtin_amdgcn_readfirstlane(slots[i]), 0)
                               : __builtin_amdgcn_readfirstlane(slots[i]);
            pf[i] = *reinterpret_cast<const bf16x8*>(base + (size_t)s * DH + ld_col);
        }
    };

    const int num_tiles = (total + BK - 1) / BK;
    load_slots(slot_cur, 0);
    load_slots(slot_next, BK);
    load_rows(slot_cur, 0);

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int kvb = tile * BK;
        const int tile_kv = min(BK, total - kvb);

        __syncthreads();  // previous tile's LDS reads are retired
#pragma unroll
        for (int i = 0; i < COOP; ++i)
            *reinterpret_cast<bf16x8*>(&lds_k[(wid + i * NW) * KD + ld_col]) = pf[i];
        if (SENT) {
            // One dword per wave, plain stores: the reader ORs them, so there is
            // nothing to reset and no atomic to serialize on.
            unsigned long long wm = 0;
#pragma unroll
            for (int i = 0; i < COOP; ++i)
                if (slot_cur[i] >= 0) wm |= 1ull << (wid + i * NW);
            if (lane == 0) lds_vm[wid] = wm;
        }
        __syncthreads();
        if (tile + 1 < num_tiles) {
#pragma unroll
            for (int i = 0; i < COOP; ++i) slot_cur[i] = slot_next[i];
            load_rows(slot_cur, kvb + BK);
            load_slots(slot_next, kvb + 2 * BK);
        }

        unsigned long long vmask = 0ull;
        if (SENT) {
#pragma unroll
            for (int w = 0; w < NW; ++w) vmask |= lds_vm[w];
        }

        // ---- QK: S[16 heads][BK] for this wave's head tile ----
        // Keep the wave-uniform branch even though the contraction split makes
        // it always true: dropping it stretches `s` across the softmax and cost
        // 24% at H=128 through the register allocator.
        f32x4 s[KVT];
        if (qk_wave) {
            constexpr int QS = QSPLIT < DW ? QSPLIT : DW;
            f32x4 sp[KVT][QS];
#pragma unroll
            for (int j = 0; j < KVT; ++j)
#pragma unroll
                for (int u = 0; u < QS; ++u) sp[j][u] = f32x4{0.f, 0.f, 0.f, 0.f};

            // The wave keeps only ~6 ds_read in flight here against a cap of
            // 15, and retires to lgkmcnt(1) every step, so LDS latency is only
            // partly hidden. Staging K through a rotating prefetch buffer does
            // raise that, and is worth +3% at BK=32, but at BK=48 (KVT=3) the
            // fully unrolled form segfaults the gfx942 frontend, so the plain
            // read-then-use loop is what ships.
#pragma unroll
            for (int c = 0; c < DW; ++c) {
#pragma unroll
                for (int j = 0; j < KVT; ++j) {
                    const bf16x4 kb = *reinterpret_cast<const bf16x4*>(
                        &lds_k[(j * 16 + lm) * KD + (dc_base + c) * 16 + lg * 4]);
                    sp[j][c % QS] = mfma16(qreg[c], kb, sp[j][c % QS]);
                }
            }

#pragma unroll
            for (int j = 0; j < KVT; ++j) {
                s[j] = sp[j][0];
#pragma unroll
                for (int u = 1; u < QS; ++u) s[j] += sp[j][u];
            }
        }

        if constexpr (DS) {
            // Each wave holds a partial over its slice of D; the row is only a
            // score once all NW of them are added. The plane is padded by an odd
            // stride so the i-loop, which steps whole kv rows, walks the banks.
#pragma unroll
            for (int j = 0; j < KVT; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    lds_s[((wid * KVT + j) * 16 + lg * 4 + i) * DSPLIT_SPAD + lm] = s[j][i];
            __syncthreads();
            if (softmax_wave) {
#pragma unroll
                for (int j = 0; j < KVT; ++j)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float v = 0.f;
#pragma unroll
                        for (int w = 0; w < NW; ++w)
                            v += lds_s[((w * KVT + j) * 16 + lg * 4 + i) * DSPLIT_SPAD + lm];
                        s[j][i] = v;
                    }
            }
        }

        if (softmax_wave) {
            // Validity depends on (j, lm) only, so it is resolved once per tile
            // rather than once per head row.
            bool valid[KVT];
#pragma unroll
            for (int j = 0; j < KVT; ++j) {
                const int r = j * 16 + lm;
                valid[j] = r < tile_kv && (!SENT || ((vmask >> r) & 1ull));
            }

            // ---- online softmax, one row per head ----
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float sv[KVT];
                float lmax = NEG_INF;
#pragma unroll
                for (int j = 0; j < KVT; ++j) {
                    const float x = valid[j] ? s[j][i] * qk_scale : NEG_INF;
                    sv[j] = x;
                    lmax = fmaxf(lmax, x);
                }
                const float m_new = fmaxf(m_i[i], row_max16(lmax));
                const float alpha = __builtin_amdgcn_exp2f(m_i[i] - m_new);

                float psum = 0.f;
                const int h_tile = lg * 4 + i;
#pragma unroll
                for (int j = 0; j < KVT; ++j) {
                    // Guard the all-masked tile, where sv - m_new is inf-inf.
                    const float p = valid[j] ? __builtin_amdgcn_exp2f(sv[j] - m_new) : 0.f;
                    psum += p;
                    lds_p[qk_ht][h_tile][j * 16 + lm] = (bf16_t)p;
                }
                l_i[i] = l_i[i] * alpha + psum;
                m_i[i] = m_new;
                lds_alpha[qk_ht * 16 + h_tile] = alpha;
            }
        }  // softmax_wave

        __syncthreads();

        // ---- PV over this wave's dv slice ----
        f32x4 a4[HT];
        f32x4 amin = f32x4{1.f, 1.f, 1.f, 1.f};
#pragma unroll
        for (int h = 0; h < HT; ++h) {
            const int hbase = h * 16 + lg * 4;
            a4[h] = f32x4{lds_alpha[hbase + 0], lds_alpha[hbase + 1], lds_alpha[hbase + 2],
                          lds_alpha[hbase + 3]};
            amin = __builtin_elementwise_min(amin, a4[h]);
        }
        if (__any(fminf(fminf(amin[0], amin[1]), fminf(amin[2], amin[3])) < 1.f)) {
#pragma unroll
            for (int h = 0; h < HT; ++h)
#pragma unroll
                for (int d = 0; d < DVT; ++d) acc[h][d] *= a4[h];
        }

        auto gather_v = [&](int kc, int d) {
            const int dv = dv_base + d * 16 + lm;
            const int kv0 = kc * 16 + lg * 4;
            bf16x4 b;
#pragma unroll
            for (int i = 0; i < 4; ++i) b[i] = lds_k[(kv0 + i) * KD + dv];
            return b;
        };

#pragma unroll
        for (int kc = 0; kc < KVT; ++kc) {
            bf16x4 a[HT];
#pragma unroll
            for (int h = 0; h < HT; ++h)
                a[h] = *reinterpret_cast<const bf16x4*>(&lds_p[h][lm][kc * 16 + lg * 4]);

            bf16x4 b_cur = gather_v(kc, 0);
#pragma unroll
            for (int d = 0; d < DVT; ++d) {
                const bf16x4 b = b_cur;
                if (d + 1 < DVT) b_cur = gather_v(kc, d + 1);
#pragma unroll
                for (int h = 0; h < HT; ++h) acc[h][d] = mfma16(a[h], b, acc[h][d]);
            }
        }
    }

    // ---- epilogue: sink finalization, then normalize in place ----
    // m_i/l_i only ever existed in the wave that ran the softmax.
    if (softmax_wave)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float l = row_sum16(l_i[i]);
            if (lm == 0) {
                const int h_local = qk_ht * 16 + lg * 4 + i;
                const float sink = attn_sink[hb_base + h_local] * LOG2E_F;
                const float m_f = fmaxf(m_i[i], sink);
                const float scale = __builtin_amdgcn_exp2f(m_i[i] - m_f);
                const float l_f = l * scale + __builtin_amdgcn_exp2f(sink - m_f);
                lds_alpha[h_local] = scale / fmaxf(l_f, 1.0e-30f);
            }
        }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < HT; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int h_local = h * 16 + lg * 4 + i;
            const float inv = lds_alpha[h_local];
            bf16_t* op = out + ((size_t)tok * H + hb_base + h_local) * DH;
#pragma unroll
            for (int d = 0; d < DVT; ++d)
                op[dv_base + d * 16 + lm] = (bf16_t)(acc[h][d][i] * inv);
        }
}

// The contraction split is only ever worth it at HT=1, where there is no second
// head tile to give the idle waves; at HT>1 it would only add a reduction to a
// phase that is already spread over every wave.
template <int NW, int HT, int BK, int SENT>
void launch(dim3 grid, hipStream_t stream, const bf16_t* q, const bf16_t* unified_kv,
            const int* ip, const int* pp, const bf16_t* kv_extend, const int* ie,
            const int* pe, const float* sink, bf16_t* out, int H, float qk_scale) {
    constexpr int DS = (MLA_DSPLIT && HT == 1 && NKC % NW == 0) ? 1 : 0;
    mla_prefill_kernel<NW, HT, BK, SENT, DS><<<grid, NW * WARP, 0, stream>>>(
        q, unified_kv, ip, pp, kv_extend, ie, pe, sink, out, H, qk_scale);
}

}  // namespace

torch::Tensor mla_prefill_v4_bf16(torch::Tensor q, torch::Tensor unified_kv,
                                  torch::Tensor kv_indices_prefix,
                                  torch::Tensor kv_indptr_prefix, torch::Tensor kv,
                                  torch::Tensor kv_indices_extend,
                                  torch::Tensor kv_indptr_extend, torch::Tensor attn_sink,
                                  double softmax_scale, bool check_sentinel) {
    TORCH_CHECK(q.dim() == 3, "q must be [N, H, 512]");
    TORCH_CHECK(unified_kv.dim() == 2 && kv.dim() == 2, "kv sources must be 2-D [rows, 512]");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(unified_kv.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16,
                "both kv sources must be bf16");
    TORCH_CHECK(attn_sink.scalar_type() == at::kFloat, "attn_sink must be fp32");
    TORCH_CHECK(q.size(2) == DH && unified_kv.size(1) == DH && kv.size(1) == DH,
                "only D=512 is compiled");
    TORCH_CHECK(q.stride(2) == 1 && unified_kv.stride(1) == 1 && kv.stride(1) == 1,
                "Q/KV must be contiguous along D");

    const int N = (int)q.size(0);
    const int H = (int)q.size(1);
    TORCH_CHECK(H > 0 && H % 16 == 0, "H must be a positive multiple of 16, got ", H);
    TORCH_CHECK(kv_indptr_prefix.size(0) == N + 1 && kv_indptr_extend.size(0) == N + 1,
                "both indptr must be N+1");
    TORCH_CHECK(attn_sink.size(0) == H, "attn_sink must be [H]");

    auto out = torch::empty_like(q);
    if (N == 0) return out;

    int HT = H / 16;
    // Wave count and kv tile follow H, and the pair matters more than either
    // half. H=128 wants 8 waves and BK=32: eight head tiles fit one workgroup,
    // so the K tile is staged once per token instead of twice, and QK's MFMAs
    // land evenly on every wave.
    //
    // H=16 -- a TP=8 rank -- wants the opposite. There is only one head tile to
    // give out, so widening the workgroup only adds waves with nothing to do in
    // QK, and what the kernel is short of there is not MFMA throughput but
    // latency to hide: hardware counters put it at 0.99 LDS and 1.69 VALU per
    // MFMA, both *below* the Triton kernel it loses to. The fix is a second
    // resident workgroup, which means shrinking LDS, which means BK=16. That
    // pair measured 25.4 against 16.5 TFLOP/s for 8 waves at BK=32.
    const bool narrow = (H <= 16);
    const int nw = narrow ? 4 : 8;
    if (HT > nw) HT = nw;
    if (HT > MLA_MAXHT) HT = MLA_MAXHT;
    while (HT > 1 && (H % (HT * 16)) != 0) --HT;
    const int hblocks = H / (HT * 16);

    const float qk_scale = (float)softmax_scale * LOG2E_F;
    auto stream = c10::hip::getCurrentHIPStream().stream();
    const dim3 grid(N, hblocks, 1);

#define MLA_LAUNCH(NWV, BKV, HTV, SENTV)                                                  \
    launch<NWV, HTV, BKV, SENTV>(grid, stream, (const bf16_t*)q.data_ptr(),               \
                                 (const bf16_t*)unified_kv.data_ptr(),                    \
                                 kv_indices_prefix.data_ptr<int>(),                       \
                                 kv_indptr_prefix.data_ptr<int>(),                        \
                                 (const bf16_t*)kv.data_ptr(),                            \
                                 kv_indices_extend.data_ptr<int>(),                       \
                                 kv_indptr_extend.data_ptr<int>(),                        \
                                 attn_sink.data_ptr<float>(), (bf16_t*)out.data_ptr(), H, \
                                 qk_scale)
// One case per reachable HT. The instantiations must cover 1..NW exactly: a
// missing case that falls through to a smaller HT still launches with the
// hblocks the host computed for the larger one, so most of the heads are never
// written and the failure is silent garbage rather than a crash. That shipped
// once here; tests/test_correctness.py now exercises every reachable HT.
#define MLA_CASE(NWV, BKV, HTV, SENTV)                                            \
    case HTV:                                                                     \
        if constexpr (HTV <= NWV) { MLA_LAUNCH(NWV, BKV, HTV, SENTV); }           \
        break;
#define MLA_DISPATCH_HT(NWV, BKV, SENTV)                                          \
    switch (HT) {                                                                 \
        MLA_CASE(NWV, BKV, 1, SENTV) MLA_CASE(NWV, BKV, 2, SENTV)                 \
        MLA_CASE(NWV, BKV, 3, SENTV) MLA_CASE(NWV, BKV, 4, SENTV)                 \
        MLA_CASE(NWV, BKV, 5, SENTV) MLA_CASE(NWV, BKV, 6, SENTV)                 \
        MLA_CASE(NWV, BKV, 7, SENTV) MLA_CASE(NWV, BKV, 8, SENTV)                 \
        default: TORCH_CHECK(false, "no kernel instantiated for HT=", HT);        \
    }
#define MLA_DISPATCH_SENT(NWV, BKV)          \
    if (check_sentinel) {                    \
        MLA_DISPATCH_HT(NWV, BKV, 1)         \
    } else {                                 \
        MLA_DISPATCH_HT(NWV, BKV, 0)         \
    }
    if (narrow) {
        MLA_DISPATCH_SENT(4, BK_NARROW)
    } else {
        MLA_DISPATCH_SENT(8, BK_WIDE)
    }
#undef MLA_DISPATCH_SENT
#undef MLA_DISPATCH_HT
#undef MLA_CASE
#undef MLA_LAUNCH
    return out;
}
