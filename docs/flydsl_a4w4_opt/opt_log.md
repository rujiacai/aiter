# FlyDSL a4w4 MoE — optimization log

Running log of concrete, landed performance optimizations for the dsv4 FP4
(a4w4, mxfp4 per_1x32) 2-stage MoE on MI355 / gfx950. Each entry is a
self-contained, measured, low-risk change.

Background: profiling (rocprof-compute + ATT) showed gemm2 is
**memory-latency-bound on its gmem A2/W2 loads** — ~81% of stall is data-feed
(`s_waitcnt` + `buffer_load`), MFMA only ~0.1%, HBM BW ~6% (not bandwidth-bound),
occupancy VGPR-capped at ~6.5%. Tile/occupancy/pipeline levers explored and found
neutral-to-worse (kept for reference, not landed here): `persist_n` (deeper
pipeline, VGPR↑ → slower), 2D wave grid / `tile_m=32` (higher occupancy → still
slower), `tile_m=128` (bigger tile, token-dependent, net ~0), `xcd` L2 swizzle
(+22%). So the useful wins are cheap prologue/scheduling tweaks like the ones
below, not structural retiling.

Convention per entry: what / why / how / results (A/B) / correctness / risk.

---

## Opt #1 — stage2: overlap the prologue uniform-scalar loads

**File:** `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` (`compile_mixed_moe_gemm2`, per-tile prologue)

### What
Reorder the gemm2 (`moe_gemm2`) per-tile prologue so the three **uniform-scalar
control loads** — `num_valid_ids`, `expert id`, `first sorted token` — are issued
together and collapse into **one overlapped `vmcnt` wait** instead of two serial
waits.

### Why
ATT on gemm2 @token=16384 showed ~81% of stall is gmem data-feed
(`s_waitcnt` 45% + `buffer_load` 36%; MFMA only 0.1%). Inside that, the per-tile
prologue had **two serial vmcnt waits**:

1. `num_valid` load → `ReadfirstlaneOp` (needed to put the A2-scale buffer
   descriptor's `num_records` in SGPR, avoiding a waterfall) → this forced
   `num_valid`'s vmcnt wait **before** the expert / first-token loads were even
   issued (ATT: ~2.36M stall).
2. expert + first-token loads → their vmcnt wait (ATT: ~2.42M stall).

Because the `readfirstlane` (and the `sx_rsrc` build that consumes it) sat
between the `num_valid` load and the other two loads, the two latencies could not
overlap.

### How
For the default (non-persistent / fixed-`persist_m`) schedule:
- Issue `expert_i32` and `_first_tok` `buffer_load`s **first**.
- Then resolve `num_valid` (`readfirstlane`) + build the A2-scale resource
  (`sx_rsrc`) + `blk_valid`.
- The validity guard (`blk_valid && exp_valid && tile_has_tokens`) is the single
  sync point, so all three loads overlap into one wait.

Refactored into `_resolve_num_valid()` / `_build_sx_rsrc()` closures, gated by
`const_expr(not _persistent)`. The **persistent** schedule still resolves
`num_valid` eagerly before the loop (it sets the loop bounds) — that path is
byte-identical to before.

### Results — dsv4 fp4 gemm2 kernel time (rocprofv3 kernel-trace min, cache off)

| token | baseline | reorder | Δ |
|------:|---------:|--------:|----:|
| 256   |  95.80 |  93.64 | −2.3% |
| 512   | 102.36 | 102.08 | −0.3% |
| 1024  | 110.64 | 109.12 | −1.4% |
| 2048  | 161.44 | 156.96 | −2.8% |
| 4096  | 225.76 | 224.72 | −0.5% |
| 8192  | 371.56 | 361.76 | −2.6% |
| 16384 | 692.49 | 680.21 | −1.8% |
| 32768 |1300.85 |1290.45 | −0.8% |

Consistent small win across the whole sweep (−0.3% … −2.8%, avg ~−1.5%), **no
regression at any token**. Exact per-token magnitude is noisy (single-run min has
~1–2% run-to-run variance) but the direction is always faster.

### Correctness
`cos_sim = 0.999997` (unchanged from baseline), validated on both epilogue
paths: token=256 (atomic / tile_m=32) and token=4096 (reduce / tile_m=64).

### Risk
None. Pure source-level reorder, semantics-preserving, zero VGPR/occupancy cost,
default-on. Persistent path unchanged (const_expr-guarded).

### Notes / dead-ends checked
- **Dropping `readfirstlane` entirely is NOT a win: +10% slower** (745.85 vs
  678 µs @16384). The buffer descriptor `num_records` must be SGPR; without the
  explicit (cheap, uniform) readfirstlane the compiler emits a waterfall loop
  for the VGPR→SGPR descriptor. So `readfirstlane` stays; the reorder (not its
  removal) is the fix, and it already hides the wait.
- **stage1 (`moe_gemm1`) does not need this.** Its `num_valid` load has no
  forced early wait (no readfirstlane; `sx_rsrc` uses `sorted_m`, not
  `num_valid`; the value is used ~70 lines later; long K-loop of ~28 tiles).
  Same prologue load stalls only ~200 cyc in ATT vs 2.36M in stage2.
