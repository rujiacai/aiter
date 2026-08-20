# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Blockwise-fp8 MoE on prebuilt code objects.

The kernels are the same ones the FlyDSL path builds; here they are loaded from
``hsa/{arch}/moe_blk/*.co`` instead of being JIT-compiled, so a deployment needs
neither the kernel sources nor a FlyDSL install.

``co_name`` is shared with ``hsa/flydsl_export.py``: the shape, tile and the
smooth_scale flag are all baked into the binary, so a naming mismatch between
the exporter and this module would silently load a kernel built for a different
layout. Keeping one function is what prevents that.
"""

from __future__ import annotations

import csv
import functools
import os
import re

from aiter.ops.moe_op import moe_blk_stage1, moe_blk_stage2

# Opt-in while both routes exist: developing against the kernels needs FlyDSL,
# a customer build ships only the code objects. Flip the default to "1" (or set
# the env var in the launcher) for the release wheel.
_DEFAULT = "0"


def use_co_path() -> bool:
    return os.environ.get("AITER_MOE_BLK_CO", _DEFAULT) == "1"


# waves_per_eu=2 avoids the register-pressure cliff the wrapper default of 3
# falls off (4220 us vs 17036 us on the same config).
WAVES_PER_EU = 2

TUNED_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "moe_blk_tuned.csv"
)


_TILE_IN_CO_NAME = re.compile(r"_t(\d+)x(\d+)x(\d+)_w(\d+)")


def tiles_from_co_name(name: str) -> tuple[int, int, int, int]:
    """``(tile_m, tile_n, tile_k, waves_per_eu)`` encoded in a code object name."""
    m = _TILE_IN_CO_NAME.search(name)
    if not m:
        raise ValueError(f"not a moe_blk code object name: {name!r}")
    return tuple(int(g) for g in m.groups())


@functools.lru_cache(maxsize=1)
def _tuned_table() -> dict:
    """(token, model_dim, inter_dim, expert, topk) -> (stage1 tiles, stage2 tiles).

    Same layout as aiter/configs/tuned_fmoe.csv, so one row covers both stages
    and the tiles live in kernelName1/kernelName2 the way the CK and asm rows
    carry theirs. block_m is a single column, which is what keeps the two stages
    on the one block size moe_sorting laid the token ids out for.
    """
    table = {}
    if not os.path.exists(TUNED_CSV):
        return table
    with open(TUNED_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            key = tuple(
                int(r[c]) for c in ("token", "model_dim", "inter_dim", "expert", "topk")
            )
            block_m = int(r["block_m"])
            tiles = tuple(
                tiles_from_co_name(r[c]) for c in ("kernelName1", "kernelName2")
            )
            if any(t[0] != block_m for t in tiles):
                raise ValueError(
                    f"{TUNED_CSV}: block_m={block_m} disagrees with the tiles named "
                    f"in {r['kernelName1']} / {r['kernelName2']}"
                )
            table[key] = tiles
    return table


def _heuristic_tiles(token: int, inter_dim: int):
    """Fallback for shapes the tuner has not covered.

    Measured on gfx942 with the DSv4 shape: tile_m above 32 is dominated by
    per-expert padding, stage1 prefers tile_k=128 while stage2 prefers the widest
    tile_k its K (== inter_dim) can be split into evenly -- the ping-pong tail
    consumes two tiles, so the tile count has to stay even.
    """
    tile_m = 16 if token < 2048 else 32
    stage2_tile_k = 256 if inter_dim % 512 == 0 else 128
    return (tile_m, 128, 128, WAVES_PER_EU), (tile_m, 256, stage2_tile_k, WAVES_PER_EU)


def tiles_for(token: int, model_dim: int, inter_dim: int, expert: int, topk: int):
    """(stage1, stage2) tiles as ``(tile_m, tile_n, tile_k, waves_per_eu)``.

    Single source of truth for the runtime dispatch and hsa/flydsl_export.py: a
    code object is named after its tile, so if the two disagreed the runtime
    would ask for a file that was never built. Reads the tuned table first and
    falls back to the heuristic for shapes it does not cover, which keeps a new
    model runnable (if not optimal) before anyone tunes it.

    Both stages always come back on the same tile_m; _tuned_table rejects a row
    whose kernel names disagree with its block_m, and the heuristic derives one
    tile_m for the pair.
    """
    hit = _tuned_table().get((token, model_dim, inter_dim, expert, topk))
    return hit if hit else _heuristic_tiles(token, inter_dim)


def co_name(
    stage: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_per_eu: int,
    out_dtype: str = "bf16",
    smooth_scale: bool = False,
) -> str:
    smooth = "_smooth" if smooth_scale else ""
    return (
        f"moe_blk_stage{stage}_{out_dtype}"
        f"_d{model_dim}x{inter_dim}_e{expert}k{topk}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_w{waves_per_eu}{smooth}.co"
    )


def moe_blk_stage1_fwd(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    tile_m: int = 16,
    tile_n: int = 128,
    tile_k: int = 128,
    waves_per_eu: int = 2,
    out_dtype: str = "bf16",
    a1_scale=None,
    w1_scale=None,
    sorted_weights=None,
    swiglu_limit: float | None = None,
    smooth_scale=None,
    **_kwargs,
):
    """gate/up GEMM + activation. `out` is (tokens, topk, inter_dim) bf16."""
    E, n_total, model_dim = w1.shape
    inter_dim = n_total // 2
    moe_blk_stage1(
        out,
        hidden_states,
        w1,
        a1_scale,
        w1_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        smooth_scale,
        hidden_states.shape[0],
        inter_dim,
        model_dim,
        tile_n,
        # The kernel always clamps; +inf is the identity, so "no clamp" needs no
        # separate binary (see moe_2stage_blockscale's clamp_gate/clamp_up).
        float("inf") if not swiglu_limit else float(swiglu_limit),
        co_name(
            1, model_dim, inter_dim, E, topk, tile_m, tile_n, tile_k,
            waves_per_eu, out_dtype, smooth_scale is not None,
        ),  # fmt: skip
    )
    return out


def moe_blk_stage2_fwd(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    tile_m: int = 16,
    tile_n: int = 256,
    tile_k: int = 256,
    waves_per_eu: int = 2,
    out_dtype: str = "bf16",
    w2_scale=None,
    a2_scale=None,
    sorted_weights=None,
    **_kwargs,
):
    """down GEMM with atomic topk reduction; `out` must be pre-zeroed."""
    E, model_dim, inter_dim = w2.shape
    moe_blk_stage2(
        out,
        inter_states,
        w2,
        a2_scale,
        w2_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        inter_states.shape[0],
        model_dim,
        inter_dim,
        tile_n,
        co_name(
            2, model_dim, inter_dim, E, topk, tile_m, tile_n, tile_k,
            waves_per_eu, out_dtype,
        ),  # fmt: skip
    )
    return out


__all__ = [
    "WAVES_PER_EU",
    "co_name",
    "moe_blk_stage1_fwd",
    "moe_blk_stage2_fwd",
    "tiles_for",
    "use_co_path",
]
