// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// bf16 MLA paged-decode for gfx942 (MI300/MI308), unified_kv CSR contract.
//
// Build note: the optCompilerConfig entry for module_mla_decode_v4_bf16 passes
// `-mllvm -enable-post-misched=1`, cancelling the `=0` aiter applies globally.
// This kernel leans on the post-RA scheduler to hide MFMA and ds_read latency
// behind the software-pipelined tile loop, and leaving it off costs 19% across
// the whole shape grid while changing nothing about register allocation. The
// cancellation works because core.py sorts the flag list, so `=1` lands after
// `=0` and LLVM takes the later one -- verify it survived if that sort changes.
// Ported in spirit from aiter-taco aiter/ops/flydsl/v4_decode_bf16 (gfx950
// FlyDSL). gfx942 forces three departures from that kernel:
//   * v_mfma_f32_16x16x16bf16_1k instead of the gfx950 16x16x32 bf16 MFMA.
//   * No ds_read_tr16_b64; the PV B-operand is gathered with 4 ds_read_u16.
//     (A transposed LDS copy to avoid the gather was measured slower; see below.)
//   * 64 KiB LDS instead of 160 KiB, so BK is 32 single-buffered (the reference
//     runs BK=64 double-buffered at 146 KiB).
//
// Contract: a workgroup owns one query token, which attends over exactly its own
// CSR row kv_indices[kv_indptr[t] : kv_indptr[t+1]]. No relationship between the
// rows of different tokens is assumed, which is what makes MTP correct here --
// whatever causality the caller's index builder expressed, by trimming a row or
// by sliding its window or by picking a different top-k per token, is what the
// kernel applies. This matches both the Triton reference and the FlyDSL kernel.
//
// An earlier revision grouped QL=4 tokens per workgroup and streamed one tile
// for all of them, on the premise that an MTP group's rows are prefixes of the
// last token's row. That premise is false: only the last token's kv_indices was
// ever read, so under a sliding window (whose start moves, so the rows are not
// nested) or CSA (where each token picks its own compressed slots) the first
// three tokens attended over another token's slots, including their own future
// draft positions. The reuse it bought is preserved below by carrying the same
// four independent A operands over head tiles instead of tokens -- those really
// do share a row, being the same token -- at identical register, LDS-read, MFMA,
// grid and HBM cost.
//
// Shape note: a wave that owns a softmax row must see the whole BK tile, so QK
// runs one wave per head tile and PV re-cuts the same waves by dv, each covering
// all HT head tiles. NW=4 (1 wave/SIMD, 128 acc floats/lane at HT=4) is the
// measured optimum on MI308X. NW=8 does reach 2 waves/SIMD at 220 VGPRs without
// spilling but benchmarks ~2x slower; BK=16, and KPAD of 2 or 16 in place of 4,
// are also all slower. HT follows H at launch (see MAX_HT below).
//
// Against aiter's hand-written ASM decode kernel on the same shapes this runs at
// about half the throughput, and hardware counters put the whole difference in
// LDS: identical VALU per MFMA (3.41 vs 3.40) but 0.95 LDS instructions per MFMA
// against 0.42, with 48% bank conflicts against 0%, which shows up as 32x the
// cycles waiting on the LDS issue port. The cause is that QK gets no reuse out of
// its B operand at HT=1 -- every MFMA reads its own 8 bytes of K.
//
// Most of those bank conflicts turned out not to be the K tile at all but lds_p,
// whose unpadded row stride is degenerate for both of its accessors; PPAD fixes
// it and takes the conflict rate from 36% to 11% and bs=1 ctx=50k from 1065 to
// 935 us. What is left of the LDS issue port is 4.5% of wave cycles, so the
// bottleneck has moved off LDS and onto VALU, which gfx942 cannot overlap with
// MFMA inside a wave. The tile loop is 692 instructions for 128 MFMAs; the
// three largest VALU blocks were the 64-multiply accumulator rescale (now
// branched around, see below), 32 v_accvgpr_read shuttling the QK partials out
// of the AGPR file (now gone, see pin_agpr), and 32 v_perm_b32 packing the
// gathered PV B operand (structural; see gather_v).
//
// Widening a QK wave to two head tiles does fix that, and was measured under the
// previous mapping: LDS per MFMA falls to 0.54 and issue-wait from 33.5% to
// 21.9%. It still lost (1165 vs 1066 us at bs=1 ctx=50k) because a 256-float
// accumulator, a resident Q and the in-flight tile come to ~490 of the 512
// registers a wave has at 1 wave/SIMD. Splitting the files by hand -- Q and the
// tile pinned to AGPRs, accumulator held in VGPRs for the rescale, which is how
// aiter's ASM fits the same shape -- takes the spill from 71 VGPRs to 43 but
// does not clear it, and dropping the VGPR-accumulator flag is far worse (736
// bytes/lane of scratch) because then every rescale pays an AGPR round trip.
// QSPLIT=1 crashes the AGPR-rewrite pass. MLA_PIN_PF is left working for when
// that allocation can be pinned down; on this compiler it needs the hand
// assignment the ASM kernel does.
//
// The AGPR split frees 64 VGPRs and buys ~2%. Storing a second, transposed copy
// of the tile to widen the PV gather was also tried and reverted; see gather_v.
//
// Measured and rejected, so they do not get tried again: rotating 2-4 PV
// gathers in flight instead of one (the d loop is unrolled and the scheduler
// already spreads them; lgkmcnt peaks at 7 of its 15, so there is no
// backpressure to relieve either); writing the gather as paired dwords to coax
// ds_read_u16_d16 out of LLVM (it still emits ds_read_u16 + v_perm_b32, and
// forcing d16 would serialise the loads on one destination register); QSPLIT
// other than 4, re-swept after the above and still the optimum; PPAD=8, where
// 20*lm mod 32 collapses to 8 banks. s_setprio is not applicable at all here --
// launch_bounds pins this to 1 wave/SIMD, so there is no second wave to
// arbitrate against.
// bench/microbench.hip holds the roofline probes these choices came from.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

#include "mla.h"

#ifndef MLA_NW
#define MLA_NW 4
#endif
#ifndef MLA_BK
#define MLA_BK 32
#endif

namespace {

typedef __bf16 bf16_t;
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

constexpr int WARP = 64;
constexpr int NW = MLA_NW;
constexpr int NTHREADS = NW * WARP;
constexpr int DH = 512;       // kv_lora_rank
constexpr int BK_WIDE = MLA_BK;  // kv tile when a workgroup has the CU to itself
constexpr int BK_NARROW = 16;    // kv tile that lets two workgroups share a CU
constexpr int NKC = DH / 16;  // QK contraction steps
constexpr int DVW = DH / NW;  // dv slice per wave in PV
constexpr int DVT = DVW / 16;
#ifndef MLA_KPAD
#define MLA_KPAD 4
#endif
constexpr int KPAD = MLA_KPAD;
constexpr int KD = DH + KPAD;
// lds_p's row stride decides the bank spread of both of its accessors, and the
// unpadded stride (BK=32 halfwords = 16 dwords) is degenerate for both. The
// write indexes rows by h = 4*lg+i, so lg steps 4*16 = 64 dwords = 0 mod 32 and
// all four quarter-waves land on the same eight banks (4-way). The read indexes
// rows by lm, and lm*16 mod 32 is only {0, 16}, so 64 lanes crowd into 16 banks
// (8-way). Padding to 18 dwords fixes both at once: lg then steps 72 = 8 mod 32
// (four quarter-waves tile the 32 banks exactly), and 18*lm mod 32 walks all 16
// even banks because 9 is coprime with 16.
#ifndef MLA_PPAD
#define MLA_PPAD 4
#endif
constexpr int PPAD = MLA_PPAD;
// QK accumulates into QSPLIT partial sums so the MFMA pipeline sees
// KVT*QSPLIT independent chains instead of KVT; a single chain stalls on the
// ~40-cycle MFMA result latency. Each partial costs KVT*4 VGPRs, so past the
// point where the chains cover that latency the extra copies only buy spills.
#ifndef MLA_QSPLIT
#define MLA_QSPLIT 4
#endif
constexpr int QSPLIT = MLA_QSPLIT;

static_assert(DVW % 16 == 0, "dv slice must be a whole number of MFMA tiles");
constexpr int MAX_HT = NW;

// Everything the kv tile width decides. BK is a template parameter rather than a
// constant because it is the only handle on occupancy: lds_k alone is BK*KD*2
// bytes, so BK=32 puts a workgroup at 34 KiB and one per CU, while BK=16 halves
// that and lets two share. Which wins depends on whether there are enough
// workgroups to fill a second slot -- see the launch site.
template <int BK>
struct Tile {
    static constexpr int KVT = BK / 16;
    static constexpr int PD = BK + PPAD;
    static constexpr int COOP = BK * DH / (NTHREADS * 8);
    static_assert(BK % 16 == 0, "kv tile must be a whole number of MFMA tiles");
    static_assert(COOP >= 1, "kv tile too small for the workgroup");
};

// HT, the head tiles a workgroup covers, is the one thing that follows H: a rank
// under TP=8 holds 16 of DeepSeek-V4-Pro's 128 heads, i.e. a single tile. A
// softmax row spans the whole BK tile, so a QK wave must own an entire head
// tile, which caps HT at the wave count; waves past HT sit out QK and still
// carry their share of PV, exactly as the FlyDSL reference does.
//
// The wave count itself stays fixed. Letting it follow H instead was measured
// and is much worse: the workgroup stages the whole BK x DH tile through
// registers, so COOP = BK*DH/(NTHREADS*8) grows as the workgroup shrinks, and
// at one wave the 128-dword prefetch buffer pushes the QK partials into AGPRs
// (agpr 128, 123 v_accvgpr copies). Holding NTHREADS at 256 keeps COOP at 8 for
// every H, and the accumulator (HT*DVT*4) only gets smaller as HT does.

constexpr float NEG_INF = -3.0e38f;
constexpr float LOG2E_F = 1.4426950408889634f;
// The reduce kernel stages one rescale factor per split in LDS, so the split
// count is bounded by that array. The auto-selection can reach it on a part
// with many CUs when a single (seq, hblock) leaves the whole machine idle.
constexpr int MAX_KS = 128;

// DPP row rotate within a 16-lane row. __shfl_xor lowers to ds_bpermute, an LDS
// op with ~50 cycle latency in an 8-deep dependent chain; DPP is a VALU modifier
// costing ~2. A rotate butterfly (1,2,4,8) leaves the full reduction in every
// lane just like an xor butterfly.
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

// Pin a value to the AGPR file without emitting anything. The accumulator is
// rescaled by VALU every tile and VALU cannot address AGPRs, so it has to stay
// in VGPRs; Q is read-only and MFMA sources operands from the AGPR file just as
// happily, so parking Q there is what lets it stay resident alongside a large
// accumulator. aiter's hand-written kernel splits them the same way, holding its
// A operands in a[144:215] against an accumulator in v[32:255].
//
// This has to be an empty asm rather than an inline-asm MFMA: the hazard
// recognizer cannot see into an asm blob, so writing the MFMA by hand drops the
// s_nop wait states around it and silently corrupts the accumulator chain.
//
// Invariant: Q is the *only* thing in the AGPR file, so agpr_count is exactly
// HT*NKC*2 = 64 and the kernel contains zero v_accvgpr_read/write. Those copies
// are pure overhead on the path the matrix core waits behind -- an A/B operand
// is read out of AGPRs for free, but anything the VALU touches has to be moved
// across. The QK partials used to land there on top of Q (agpr_count 97, 32
// copies a tile) until the rescale hoist below changed the pressure profile
// enough for the allocator to leave them in VGPRs. That is the allocator's
// decision rather than something this file can state outright, so
// tests/test_isa_guard.py asserts it and must stay green.
__device__ __forceinline__ void pin_agpr(bf16x4& x) { asm("" : "+a"(x)); }
__device__ __forceinline__ void pin_agpr(bf16x8& x) { asm("" : "+a"(x)); }

// Whether the in-flight KV tile also parks in AGPRs. It is only ever moved
// global -> register -> LDS, so it never needs VALU either, but it competes with
// Q for the same file. Measured off: it takes agpr_count to 96 and puts 128
// v_accvgpr_read back in the module, so it is only ever worth revisiting if
// VGPR pressure becomes the binding limit again (it is not at HB=16, which
// sits at 316 VGPR + 64 AGPR of 512 with no spill).
#ifndef MLA_PIN_PF
#define MLA_PIN_PF 0
#endif

// MI308X is 4 XCDs of 20 CUs, each with its own 4 MB L2, and the dispatcher
// hands workgroup `L` to XCD `L % XCD`. The blocks that share data here are the
// hblocks of one (seq, split): they stream byte-identical KV rows and differ
// only in which heads of Q they hold. Under the default linear order those land
// on `hblk % 4`, i.e. spread over every XCD, so each XCD pulls its own copy of
// the tile from HBM instead of hitting a neighbour's L2.
//
// The remap below is the standard round-robin -> blocked transpose: XCD `c`
// ends up owning the contiguous logical range [c*per, (c+1)*per), and the
// caller decodes that range with hblk varying fastest, so a whole sharing group
// stays on one die. The N % XCD tail keeps its identity mapping, which only
// costs the last few blocks their grouping.
#ifndef MLA_XCD
#define MLA_XCD 4
#endif
__device__ __forceinline__ int xcd_swizzle(int L, int N) {
    if (MLA_XCD <= 1) return L;
    const int per = N / MLA_XCD;
    if (per == 0 || L >= per * MLA_XCD) return L;
    return (L % MLA_XCD) * per + (L / MLA_XCD);
}

// Tiles owned by split `pid_k`. Mirrors the reduce kernel's segment math so
// inactive splits can be masked without extra communication.
__device__ __forceinline__ void split_range(int BK, int kv_len, int KS, int pid_k,
                                            int& tile_start, int& tile_end) {
    const int num_tiles = (kv_len + BK - 1) / BK;
    const int tps = (kv_len + KS * BK - 1) / (KS * BK);
    tile_start = pid_k * tps;
    tile_end = min((pid_k + 1) * tps, num_tiles);
}

// FUSED=1 is the single-split path: the block already holds the whole softmax
// denominator, so it normalizes in place and writes bf16 straight out. That
// drops the fp32 partial round-trip through HBM, which at short contexts is the
// bulk of the traffic (4 MB of partials for 0.1 GFLOP of work at ctx=50).
template <int HT, int BK, int FUSED>
__global__ __launch_bounds__(NTHREADS, 1) void mla_decode_split_kernel(
    const bf16_t* __restrict__ q,        // [T, H, DH]
    const bf16_t* __restrict__ kv,       // [num_slots, DH]
    const int* __restrict__ kv_indices,  // [nnz]
    const int* __restrict__ kv_indptr,   // [T+1]
    const float* __restrict__ attn_sink,  // [H], FUSED only
    bf16_t* __restrict__ out,             // [T, H, DH], FUSED only
    float* __restrict__ m_partial,       // [T, KS, H]
    float* __restrict__ l_partial,       // [T, KS, H]
    float* __restrict__ acc_partial,     // [T, KS, H, DH]
    const int H, const int KS, const float qk_scale) {
    static_assert(HT >= 1 && HT <= MAX_HT, "one QK wave per head tile");
    constexpr int HB = HT * 16;  // heads per workgroup
    constexpr int KVT = Tile<BK>::KVT;
    constexpr int PD = Tile<BK>::PD;
    constexpr int COOP = Tile<BK>::COOP;

    // hblk decoded fastest so one sharing group is contiguous in logical space.
    const int ntok = gridDim.x, nhb = gridDim.y;
    int lin = xcd_swizzle(blockIdx.x + ntok * (blockIdx.y + nhb * blockIdx.z),
                          ntok * nhb * (int)gridDim.z);
    const int hblk = lin % nhb;
    lin /= nhb;
    const int tok = lin % ntok;
    const int pid_k = lin / ntok;

    const int tid = threadIdx.x;
    const int wid = tid >> 6;
    const int lane = tid & (WARP - 1);
    const int lm = lane & 15;
    const int lg = lane >> 4;

    // This token's own CSR row, and nothing else. Any masking the caller wanted
    // is already expressed in which slots it put in this row.
    const int kv_start = kv_indptr[tok];
    const int kv_len = kv_indptr[tok + 1] - kv_start;
    const int hb_base = hblk * HB;
    if (kv_len <= 0) {
        // Nothing to reduce afterwards on the fused path, so zero here.
        if (FUSED) {
            for (int i = tid; i < HB * DH; i += NTHREADS)
                out[((size_t)tok * H + hb_base + i / DH) * DH + i % DH] = (bf16_t)0.f;
        }
        return;
    }

    int tile_start, tile_end;
    split_range(BK, kv_len, KS, pid_k, tile_start, tile_end);
    if (tile_start >= tile_end) return;

    // A wave owns head tile `qk_ht` in QK and dv slice `dv_base` in PV. The two
    // roles cut the same work along different axes; at HT == NW every wave is
    // busy in both, and below that the surplus waves only do PV.
    const int qk_ht = wid;
    const bool qk_wave = wid < HT;
    const int dv_base = wid * DVW;

    __shared__ bf16_t lds_k[BK * KD];
    __shared__ bf16_t lds_p[HT][16][PD];
    __shared__ float lds_alpha[HB];

    // Q stays resident across the whole tile loop (A-operand layout: m = lane%16,
    // k = 4*(lane/16)+i). It is only ever consumed as an MFMA A operand, so it
    // is pinned to the AGPR file and the 64 registers it costs come out of a
    // budget the accumulator cannot use anyway.
    bf16x4 qreg[NKC];
    if (qk_wave) {
        const bf16_t* const qbase = q + (size_t)tok * H * DH +
                                    (size_t)(hb_base + qk_ht * 16 + lm) * DH + lg * 4;
#pragma unroll
        for (int c = 0; c < NKC; ++c) {
            qreg[c] = *reinterpret_cast<const bf16x4*>(qbase + c * 16);
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

    // Two-stage prefetch: slots run one tile ahead of the rows that use them,
    // so a row load never waits on an index load issued in the same iteration.
    const int ld_col = lane * 8;
    const int uwid = __builtin_amdgcn_readfirstlane(wid);
    int slot_cur[COOP], slot_next[COOP];
    bf16x8 pf[COOP];

    auto load_slots = [&](int* dst, int kvb) {
#pragma unroll
        for (int i = 0; i < COOP; ++i)
            dst[i] = kv_indices[kv_start + min(kvb + uwid + i * NW, kv_len - 1)];
    };
    auto load_rows = [&](const int* slots) {
#pragma unroll
        for (int i = 0; i < COOP; ++i) {
            pf[i] = *reinterpret_cast<const bf16x8*>(
                kv + (size_t)__builtin_amdgcn_readfirstlane(slots[i]) * DH + ld_col);
            if (MLA_PIN_PF) pin_agpr(pf[i]);
        }
    };

    load_slots(slot_cur, tile_start * BK);
    load_slots(slot_next, (tile_start + 1) * BK);
    load_rows(slot_cur);

    for (int tile = tile_start; tile < tile_end; ++tile) {
        const int kvb = tile * BK;
        const int tile_kv = min(BK, kv_len - kvb);

        __syncthreads();  // previous tile's LDS reads are retired
#pragma unroll
        for (int i = 0; i < COOP; ++i)
            *reinterpret_cast<bf16x8*>(&lds_k[(wid + i * NW) * KD + ld_col]) = pf[i];
        __syncthreads();
        if (tile + 1 < tile_end) {
#pragma unroll
            for (int i = 0; i < COOP; ++i) slot_cur[i] = slot_next[i];
            load_rows(slot_cur);
            load_slots(slot_next, (tile + 2) * BK);
        }

        // ---- QK: S[16 heads][BK] for this wave's head tile ----
        // Wave-uniform, and the barrier that closes the phase is outside it, so
        // the surplus waves skip straight to their PV share.
        if (qk_wave) {
        f32x4 sp[KVT][QSPLIT];
#pragma unroll
        for (int j = 0; j < KVT; ++j)
#pragma unroll
            for (int u = 0; u < QSPLIT; ++u) sp[j][u] = f32x4{0.f, 0.f, 0.f, 0.f};

#pragma unroll
        for (int c = 0; c < NKC; ++c) {
#pragma unroll
            for (int j = 0; j < KVT; ++j) {
                const bf16x4 kb = *reinterpret_cast<const bf16x4*>(
                    &lds_k[(j * 16 + lm) * KD + c * 16 + lg * 4]);
                sp[j][c % QSPLIT] = mfma16(qreg[c], kb, sp[j][c % QSPLIT]);
            }
        }

        f32x4 s[KVT];
#pragma unroll
        for (int j = 0; j < KVT; ++j) {
            s[j] = sp[j][0];
#pragma unroll
            for (int u = 1; u < QSPLIT; ++u) s[j] += sp[j][u];
        }

        // ---- online softmax, one row per head ----
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float sv[KVT];
            bool valid[KVT];
            float lmax = NEG_INF;
#pragma unroll
            for (int j = 0; j < KVT; ++j) {
                valid[j] = (j * 16 + lm) < tile_kv;
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
                // Guard the all-masked tile, where sv - m_new would be inf-inf.
                const float p = valid[j] ? __builtin_amdgcn_exp2f(sv[j] - m_new) : 0.f;
                psum += p;
                lds_p[qk_ht][h_tile][j * 16 + lm] = (bf16_t)p;
            }
            // alpha is uniform across the row, so each lane can carry its own
            // partial denominator and the 16-lane reduction happens once at the
            // end instead of every tile.
            l_i[i] = l_i[i] * alpha + psum;
            m_i[i] = m_new;
            lds_alpha[qk_ht * 16 + h_tile] = alpha;
        }
        }  // qk_wave

        __syncthreads();

        // ---- PV over this wave's dv slice ----
        // Rescaling the accumulator is 64 packed multiplies, the single largest
        // block of VALU in the loop, and gfx942 cannot overlap VALU with the
        // MFMAs that follow it. But alpha = exp2(m_old - m_new) is exactly 1.0
        // on every row whose running max did not move, and past the first
        // handful of tiles a 32-slot tile almost never beats the max of the
        // tens of thousands of scores already seen. Multiplying by 1.0 is the
        // identity, so skipping the block under a wave-uniform branch is
        // bit-exact, not an approximation. alpha is bounded above by 1, so
        // "any row moved" is just a minimum against 1.
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

        // B-operand V[kv = 16*kc + 4*(lane/16)+i][dv]; gfx942 has no LDS
        // transpose read, so gather the 4 kv rows by hand and run one step
        // ahead so the gather latency overlaps the MFMAs.
        //
        // Storing a second, transposed copy of the tile so this becomes one
        // ds_read_b64 was measured and is slower: QK needs [kv][d] and PV needs
        // [d][kv], both layouts resident need ~68 KiB against a 64 KiB budget,
        // and staging the transpose through LDS instead costs more LDS traffic
        // (1104 vs 848 bytes/lane/tile) than the saved instructions are worth.
        // The four halfwords cost two v_perm_b32 each gather to pack into the
        // B operand. Feeding them to ds_read_u16_d16 / _d16_hi instead would
        // drop them straight into place, but writing the gather as two dwords
        // built from paired loads does not get LLVM to emit those, and forcing
        // it would serialise the loads on the shared destination register
        // anyway; the packed form measured identical.
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
            // A-operand depends on (h, kc) only: hoist it out of the dv loop.
            bf16x4 a[HT];
#pragma unroll
            for (int h = 0; h < HT; ++h)
                a[h] = *reinterpret_cast<const bf16x4*>(&lds_p[h][lm][kc * 16 + lg * 4]);

            // One gather ahead. Rotating 2-4 in flight was measured and is
            // slightly worse: the d loop is fully unrolled, so the scheduler
            // already spreads these reads out on its own, and a hand-rolled
            // queue only pins registers and constrains it.
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

    // ---- epilogue ----
    if (FUSED) {
        // A row's m_i/l_i live in the wave that owned it in QK, but its acc is
        // spread over every wave's dv slice, so the per-row normalizer crosses
        // waves through LDS (reusing lds_alpha).
        if (qk_wave)
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
        return;
    }

    if (qk_wave) {
        const size_t mo = ((size_t)tok * KS + pid_k) * H;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float l = row_sum16(l_i[i]);
            if (lm == 0) {
                const int h = hb_base + qk_ht * 16 + lg * 4 + i;
                m_partial[mo + h] = m_i[i];
                l_partial[mo + h] = l;
            }
        }
    }
    {
        const size_t ao = ((size_t)tok * KS + pid_k) * H * DH;
#pragma unroll
        for (int h = 0; h < HT; ++h)
#pragma unroll
            for (int d = 0; d < DVT; ++d) {
                const int dv = dv_base + d * 16 + lm;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    acc_partial[ao + (size_t)(hb_base + h * 16 + lg * 4 + i) * DH + dv] =
                        acc[h][d][i];
            }
    }
}

// Combine splits, fold the attention sink, emit bf16 [T, H, DH].
template <int NT>
__global__ __launch_bounds__(NT, 1) void mla_decode_reduce_kernel(
    const float* __restrict__ m_partial, const float* __restrict__ l_partial,
    const float* __restrict__ acc_partial, const float* __restrict__ attn_sink,
    const int* __restrict__ kv_indptr, bf16_t* __restrict__ out, const int H, const int KS,
    const int BK, const float log2e) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;

    // Segment math must match the split kernel, which tiles over this token's
    // own row length.
    const int kv_len = kv_indptr[t + 1] - kv_indptr[t];
    bf16_t* op = out + ((size_t)t * H + h) * DH;
    if (kv_len <= 0) {
        for (int d = tid; d < DH; d += NT) op[d] = (bf16_t)0.f;
        return;
    }

    const int tps = (kv_len + KS * BK - 1) / (KS * BK);
    const int act = (kv_len + tps * BK - 1) / (tps * BK);

    __shared__ float sh_alpha[MAX_KS];
    __shared__ float sh_inv_l;

    if (tid == 0) {
        float m_max = NEG_INF;
        for (int k = 0; k < act; ++k)
            m_max = fmaxf(m_max, m_partial[((size_t)t * KS + k) * H + h]);
        float lc = 0.f;
        for (int k = 0; k < act; ++k) {
            const float a = __builtin_amdgcn_exp2f(m_partial[((size_t)t * KS + k) * H + h] - m_max);
            sh_alpha[k] = a;
            lc += l_partial[((size_t)t * KS + k) * H + h] * a;
        }
        const float sink = attn_sink[h] * log2e;
        const float m_final = fmaxf(m_max, sink);
        const float alpha_kv = __builtin_amdgcn_exp2f(m_max - m_final);
        const float l_final = lc * alpha_kv + __builtin_amdgcn_exp2f(sink - m_final);
        for (int k = 0; k < act; ++k) sh_alpha[k] *= alpha_kv;
        sh_inv_l = 1.0f / fmaxf(l_final, 1.0e-30f);
    }
    __syncthreads();

    const float inv_l = sh_inv_l;
    for (int d = tid; d < DH; d += NT) {
        float v = 0.f;
        for (int k = 0; k < act; ++k)
            v += acc_partial[(((size_t)t * KS + k) * H + h) * DH + d] * sh_alpha[k];
        op[d] = (bf16_t)(v * inv_l);
    }
}

int prev_pow2(int x) {
    int p = 1;
    while (p * 2 <= x) p *= 2;
    return p;
}

template <int HT, int BK>
void launch_split(bool fused, dim3 grid, hipStream_t stream, const bf16_t* q, const bf16_t* kv,
                  const int* kv_indices, const int* kv_indptr, const float* attn_sink,
                  bf16_t* out, float* m_partial, float* l_partial, float* acc_partial, int H,
                  int KS, float qk_scale) {
    auto k = fused ? mla_decode_split_kernel<HT, BK, 1> : mla_decode_split_kernel<HT, BK, 0>;
    hipLaunchKernelGGL(k, grid, dim3(NTHREADS), 0, stream, q, kv, kv_indices, kv_indptr, attn_sink,
                       out, m_partial, l_partial, acc_partial, H, KS, qk_scale);
}

int device_cu_count() {
    static int ncu = [] {
        hipDeviceProp_t p;
        (void)hipGetDeviceProperties(&p, 0);
        return p.multiProcessorCount;
    }();
    return ncu;
}

}  // namespace

// kv_indices/kv_indptr follow the unified_kv CSR contract: query token t attends
// over kv_indices[kv_indptr[t] : kv_indptr[t+1]] and nothing else. Causality,
// MTP included, is whatever the caller put in those rows.
torch::Tensor mla_decode_v4_bf16(torch::Tensor q, torch::Tensor unified_kv,
                                 torch::Tensor kv_indices, torch::Tensor kv_indptr,
                                 torch::Tensor attn_sink, double softmax_scale,
                                 int64_t kv_splits) {
    TORCH_CHECK(q.dim() == 3, "q must be [T, H, D]");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(unified_kv.scalar_type() == at::kBFloat16, "unified_kv must be bf16");
    TORCH_CHECK(q.size(2) == DH, "head_dim must be 512");
    TORCH_CHECK(unified_kv.size(1) == DH, "unified_kv last dim must be 512");
    TORCH_CHECK(attn_sink.scalar_type() == at::kFloat, "attn_sink must be fp32");

    const int T = q.size(0);
    const int H = q.size(1);
    TORCH_CHECK(H % 16 == 0 && H > 0, "H must be a positive multiple of the MFMA tile (16)");

    // Widest workgroup whose head tiles divide H. Wider is better: one KV tile
    // in LDS then feeds more independent PV A operands, which is where the B
    // operand gets its reuse. TP=8 on a 128-head model lands on HT=1.
    int HT = MAX_HT;
    while (HT > 1 && (H % (HT * 16)) != 0) --HT;
    const int HB = HT * 16;
    const int hblocks = H / HB;

    const int ncu = device_cu_count();

    int KS;
    if (kv_splits > 0) {
        TORCH_CHECK(kv_splits <= MAX_KS, "kv_splits must be <= ", MAX_KS);
        KS = (int)kv_splits;
    } else {
        const int base = std::max(1, T * hblocks);
        KS = std::max(1, prev_pow2(std::max(1, ncu / base)));
        // nnz/T is the mean row length, known without a device sync. Below ~4
        // tiles the split kernel's prologue and epilogue dominate, so splitting
        // buys no parallelism and costs a partial round-trip plus the reduce
        // launch; measured crossover on MI308X sits between 4 and 8 tiles.
        const int est_tiles =
            std::max(1, (int)((kv_indices.numel() / T + BK_WIDE - 1) / BK_WIDE));
        KS = est_tiles <= 4 ? 1 : std::min(KS, prev_pow2(est_tiles));
        KS = std::min(KS, MAX_KS);
    }
    const bool fused = KS == 1;

    // A workgroup at BK_WIDE needs 34 KiB of LDS, so only one fits per CU and
    // the NW-HT waves that sit out QK sit out the whole CU with them. BK_NARROW
    // halves lds_k and lets a second workgroup in, which covers those waves --
    // but only once there are more workgroups than CUs, and only at HT=1, where
    // the register file is slack enough for two to actually be resident. At
    // HT=4 the accumulator alone keeps occupancy at one whatever BK is, so the
    // narrow tile there just costs per-tile efficiency (measured 24% worse).
    const int blocks = T * hblocks * KS;
    const int BK = (HT == 1 && blocks > ncu) ? BK_NARROW : BK_WIDE;

    auto fopt = q.options().dtype(at::kFloat);
    const int ps = fused ? 0 : KS;  // partials are dead weight on the fused path
    auto m_partial = torch::empty({T, ps, H}, fopt);
    auto l_partial = torch::empty({T, ps, H}, fopt);
    auto acc_partial = torch::empty({T, ps, H, DH}, fopt);
    auto out = torch::empty({T, H, DH}, q.options());

    const float LOG2E = 1.4426950408889634f;
    const float qk_scale = (float)softmax_scale * LOG2E;
    auto stream = c10::hip::getCurrentHIPStream().stream();

    const dim3 grid(T, hblocks, KS);
#define MLA_LAUNCH(HTV, BKV)                                                                   \
    launch_split<HTV, BKV>(fused, grid, stream, (const bf16_t*)q.data_ptr(),                   \
                      (const bf16_t*)unified_kv.data_ptr(), kv_indices.data_ptr<int>(),        \
                      kv_indptr.data_ptr<int>(), attn_sink.data_ptr<float>(),                  \
                      (bf16_t*)out.data_ptr(), fused ? nullptr : m_partial.data_ptr<float>(),  \
                      fused ? nullptr : l_partial.data_ptr<float>(),                           \
                      fused ? nullptr : acc_partial.data_ptr<float>(), H, KS, qk_scale)
    switch (HT) {
        case 4: MLA_LAUNCH(4, BK_WIDE); break;
        case 3: MLA_LAUNCH(3, BK_WIDE); break;
        case 2: MLA_LAUNCH(2, BK_WIDE); break;
        default:
            if (BK == BK_NARROW)
                MLA_LAUNCH(1, BK_NARROW);
            else
                MLA_LAUNCH(1, BK_WIDE);
            break;
    }
#undef MLA_LAUNCH

    if (!fused)
        hipLaunchKernelGGL((mla_decode_reduce_kernel<256>), dim3(T, H), dim3(256), 0, stream,
                           m_partial.data_ptr<float>(), l_partial.data_ptr<float>(),
                           acc_partial.data_ptr<float>(), attn_sink.data_ptr<float>(),
                           kv_indptr.data_ptr<int>(), (bf16_t*)out.data_ptr(), H, KS, BK, LOG2E);

    return out;
}
