# FlyDSL a4w4 stage2 (gemm2) — token=16k optimization log

Investigation of the dsv4 FP4 **stage2 down-projection GEMM** (`moe_gemm2`) at
`token=16384`, on **MI355 / gfx950**. Records every optimization lever tried, the
hard profiling evidence, and the conclusion so the next person does not re-tread
the same dead ends.

## TL;DR

- **Baseline `t64x128x128` (mfma32k64, reduce, xcd0) is a robust local optimum.**
- gemm2 for this shape is **memory-latency-bound on the gmem A2/W2 loads**
  (ATT: `s_waitcnt` 45% + `buffer_load` 36% = **81%** of stall; MFMA only **0.1%**).
  It is **not** compute-, bandwidth-, occupancy-, or epilogue-overhead-bound.
- Every single-kernel lever (deepen pipeline, raise occupancy, bigger tile, L2
  swizzle, deep A2 prefetch) is **neutral or worse**. See the table below.
- The only lever that attacks the root cause (the A2 gmem round-trip) is a
  **cross-kernel gemm1↔gemm2 fusion** — a large, separate project.

## Shape

| | |
|---|---|
| model | dsv4, FP4/FP4, `QuantType.per_1x32` (MX-FP4), Silu |
| token | 16384 |
| model_dim (N of gemm2) | 7168 → **56 N-tiles** at tile_n=128 |
| inter_dim (K of gemm2) | 384 → **3 K-tiles** at tile_k=128 |
| expert / topk | 384 / 6 |
| gemm2 op | `out[T, model_dim] = A2[T, inter_dim] @ W2[model_dim, inter_dim]^T` |
| tuned kernel | `flydsl_moe2_afp4_wfp4_bf16_t64x128x128_reduce_mfma32k64_sbm128` |

Key structural fact: **K = inter_dim = 384 → only 3 K-tiles.** This is a
short-K, low-arithmetic-intensity GEMM; the K-loop is too short to build a
latency-hiding pipeline.

## Baseline profile (gemm2 @16384)

Kernel time (rocprofv3 kernel-trace, min over 10 iters): **683.73 µs**.
Static resources (final ISA): **VGPR 92, SGPR 46, LDS 21504 B, 0 spills.**

rocprof-compute System Speed-of-Light:

| metric | value | note |
|---|---|---|
| Wavefront Occupancy | **6.47 %** (529/8192) | very low |
| MFMA Utilization | **9.40 %** | compute is idle |
| VALU Utilization | 33.5 % | |
| IPC | 0.67 / 5 | |
| L2-Fabric **Read** BW | **5.8 %** (570 Gb/s) | **not** bandwidth-bound |
| L2-Fabric Write BW | 20.6 % (2030 Gb/s) | |
| L2-Fabric Read **Latency** | **1456 cycles** | high, unhidden |
| vL1D / L2 hit rate | 68.9 % / 70.1 % | |
| LDS per CU (gfx950) | **160 KB** | not the limiter |

Occupancy limiters (rocprof-compute 6.2, "Insufficient ..."):

| resource | % blocking |
|---|---|
| **SIMD VGPRs** | **39.24 %** ← occupancy is VGPR-capped when scheduling |
| CU LDS | 0.00 % |
| SIMD SGPRs / Waveslots / Barriers | 0.00 % |

**ATT stall breakdown** (`.rocprofv3/att_s2_t16k`, by Stall cycles):

| category | Stall % |
|---|---|
| `s_waitcnt` (waiting on loads) | **45.1 %** |
| `buffer_load` (gmem A2/W2 feed) | **36.3 %** |
| `s_barrier` | 8.9 % |
| global store/atomic (epilogue) | 4.7 % |
| other VALU | 3.1 % |
| `buffer_load`→LDS | 0.8 % |
| `ds_read` / `ds_write` (LDS) | 0.4 % / 0.4 % |
| **`v_mfma` (compute)** | **0.1 %** |

Reading: **81 % of the time is spent feeding A2/W2 from gmem and waiting on it**;
compute and epilogue are negligible. Combined with the 5.8 % HBM read BW, this is
**latency-bound, not bandwidth-bound**: few bytes, but each load has ~1456-cycle
latency and there is not enough in-flight work to hide it.

## Levers tried (all gemm2 min µs @16384 unless noted)

| lever | mechanism | result | why |
|---|---|---|---|
| **baseline** t64x128x128 | — | **683.7 µs** | optimum |
| `persist_n=2` | deepen pipeline / reuse A2 across N-tiles | **765.5 (+12 %)** | VGPR 92→132, occupancy collapses; also @64 +9 %, @1024 +1 % |
| tile_m=32 (≈2D wave grid) | raise occupancy (lower VGPR) | **773.2 (+13 %)** | occ 6.47→7.62 % (higher!) but MFMA↓, 2× WGs → 2× loads |
| tile_m=128 | bigger tile, fewer WGs, less redundant load | 674.4 (-1.4 %) @16384 | but **@4096 +13.6 %**, @32768 -0.8 %; VGPR 158; token-dependent, not a win |
| xcd=4 | L2-locality WG swizzle | **837.3 (+22 %)** | reordering hurts L2 reuse for this shape |
| A2-resident static K-loop (`persist_n=1`) | load all 3 K-tiles of A2 upfront (deepest prefetch) + drop ping-pong barriers | 689.8 (+0.9 %) | barrier stall is really the load-wait; upfront issue doesn't change total latency |

All variants keep correctness (cos ≈ 0.999997 vs torch reference).

### Why "raise occupancy" fails (the key counter-intuitive result)

Occupancy *allocation* is VGPR-capped (6.2 shows 39 % insufficient-VGPR), which
naively suggests "lower VGPR → more waves → hide latency → faster". But the
direct experiment (tile_m=32, VGPR 60, **higher** achieved occupancy 7.62 %) is
**+13 % slower** with **lower** MFMA util. More waves did not help because the
kernel is not starved for waves in a way that hides the load latency — it is
starved for *useful in-flight memory work per byte moved*, and the smaller tile
doubles the number of WGs (and redundant W2 loads). So occupancy is **not** the
performance lever here, and the 2D-wave-grid rewrite (which only raises
occupancy) was correctly abandoned before implementation.

### Why "bigger tile" barely helps

tile_m=128 halves the WG count (less total prologue/epilogue + redundant load)
and is marginally faster at large tokens (16384/32768), but VGPR balloons to 158
(~3 waves/SIMD) and it is clearly slower where WG count is already modest
(@4096 +13.6 %). Net: token-dependent, not worth adding as a default. (The
`tile_m=128` enumeration was added for the experiment and then reverted.)

## Root cause

gemm2 here is a **short-K (3 tiles), low-arithmetic-intensity GEMM that is
memory-latency-bound on its gmem inputs (A2 + W2)**. The single-kernel knobs
(pipeline depth, occupancy, tile size, WG swizzle, prefetch distance) only trade
these fixed costs against each other; none removes the fundamental per-load gmem
latency, so all are neutral-to-worse vs the tuned baseline.

## Only remaining lever: cross-kernel fusion

Eliminate the **A2 gmem round-trip** (stage1 writes A2 to gmem, stage2 reads it
back) by fusing gemm1→gemm2 so A2 never lands in gmem. That removes the A2 part
of the 81 % data-feed cost.

Difficulty: stage2 needs a token's **full inter_dim (all K=384) of A2** to
produce one output tile, but stage1 produces A2 tiled over inter_dim (its N). A
fused kernel must compute a full A2 row before starting stage2, with different
tile shapes/parallelism for the two GEMMs (flash-attention-style persistent
fusion). Large effort, uncertain payoff — treat as a separate project.

## Reproduce

```bash
# baseline gemm2 time (dsv4 fp4, token=16384)
HIP_VISIBLE_DEVICES=1 AITER_CONFIG_FMOE=aiter/configs/model_configs/dsv4_fp4fp4_tuned_fmoe.csv \
  rocprofv3 --kernel-trace --output-format csv -d /tmp/rp -- \
  python op_tests/_flydsl_prof.py --token 16384 --iters 10 --warmup 3
# -> aggregate 'moe2' kernel min duration from /tmp/rp/**/kernel_trace.csv

# correctness (use a token whose native block_m matches the tile, e.g. 4096)
AITER_CONFIG_FMOE=... python op_tests/_flydsl_prof.py --token 4096 --check

# occupancy / SOL / limiters
rocprof-compute profile -n g2 -k moe2 -b 2 3 6 -- python op_tests/_flydsl_prof.py --token 16384 --iters 3 --warmup 1
rocprof-compute analyze -p workloads/g2/MI355/ -b 2 3 6.2

# VGPR / LDS from ISA
FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/asm \
  python op_tests/_flydsl_prof.py --token 16384 --iters 1 --warmup 1
grep -E '\.vgpr_count|\.group_segment_fixed_size' /tmp/asm/mfma_moe2_*/21_final_isa.s
```

Gotcha: forcing a different gemm2 `tile_m`/`block_m` via a hand-edited
`tuned_fmoe.csv` **only validates at tokens whose native `block_m` already
matches** (e.g. 4096/16384/32768 use `block_m=128`). Editing `block_m` at a token
where stage1 was tuned for a different block breaks the stage1↔sort↔stage2
layout and gives cos≈0 — this is a test-harness artifact, not a kernel bug.

## Code state after this investigation

- **persist_n port** and **A2-resident static K-loop** landed in
  `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`, both **env-gated and
  default OFF** (byte-identical baseline when off):
  - `AITER_FLYDSL_STAGE2_PERSIST_N=<n>` — persist over N-tiles (measured negative).
  - `AITER_FLYDSL_STAGE2_A2_RESIDENT=1` — A2-resident static K-loop at persist_n=1
    (measured neutral).
  - Both are guarded to only engage on non-persistent, xcd=0, `inter_dim % tile_k == 0`
    shapes; otherwise they resolve to the original ping-pong path.
- `tile_m=128` gemm2 enumeration was added for the experiment and **reverted**.
- All `tuned_fmoe.csv` variants used for A/B were temporary (`/tmp`); the repo
  config is unchanged.
