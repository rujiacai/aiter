# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Blockwise-fp8 MoE on prebuilt code objects.

DeepSeek-style per-128x128 fp8 weights with per-1x128 fp8 activations, run as
two GEMM stages. The kernels ship as prebuilt binaries under
``hsa/{arch}/moe_blk/*.co`` rather than being compiled on the fly, so a
deployment carries no kernel sources and no compiler dependency.

Shape, tile and the smooth_scale flag are all baked into each binary and encoded
in its file name, so ``co_name`` here has to agree exactly with the name the
binary was published under -- disagreeing would load a kernel built for a
different layout without any error.

Enable with ``AITER_MOE_BLK_CO=1``.
"""

from __future__ import annotations

import csv
import functools
import os
import re

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.moe_op import moe_blk_stage1, moe_blk_stage2

# On by default: blockwise fp8 routes to the code objects unless
# AITER_MOE_BLK_CO=0 sends it back to the stock asm/CK kernels.
_DEFAULT = "1"


def use_co_path() -> bool:
    return os.environ.get("AITER_MOE_BLK_CO", _DEFAULT) == "1"


# waves_per_eu=2 avoids the register-pressure cliff the wrapper default of 3
# falls off (4220 us vs 17036 us on the same config).
WAVES_PER_EU = 2

TUNED_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "moe_blk_tuned.csv"
)


@functools.lru_cache(maxsize=1)
def co_dir() -> str:
    """Directory the launcher will look the code objects up in.

    Read from AITER_ASM_DIR rather than derived from __file__, because that is
    what aiter_hip_common.h resolves against: an installed (non-develop) tree
    puts them under aiter_meta/hsa and AITER_META_DIR can move them again.
    Guessing here instead would report every shape as unpublished and quietly
    turn the whole route off.
    """
    from aiter.jit.core import AITER_ASM_DIR

    return os.path.join(os.environ.get("AITER_ASM_DIR") or AITER_ASM_DIR,
                        get_gfx(), "moe_blk")  # fmt: skip


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

    A code object is named after its tile, so asking for a tile that was never
    published means asking for a file that does not exist. Reads the tuned table
    first and falls back to the heuristic for shapes it does not cover, which
    keeps an untuned model runnable (if not optimal).

    Both stages always come back on the same tile_m; _tuned_table rejects a row
    whose kernel names disagree with its block_m, and the heuristic derives one
    tile_m for the pair.
    """
    hit = _tuned_table().get((token, model_dim, inter_dim, expert, topk))
    return hit if hit else _heuristic_tiles(token, inter_dim)


@functools.lru_cache(maxsize=None)
def have_co_for(
    token: int, model_dim: int, inter_dim: int, expert: int, topk: int
) -> bool:
    """Whether this shape's code objects were actually published.

    Only the shapes someone exported binaries for can take this route; anything
    else has to fall back to the stock kernels, because the name tiles_for
    resolves would otherwise point at a file that does not exist and the load
    would fail at the first call. Stage1 is checked in both smooth_scale
    variants since that flag is a runtime decision made after dispatch.
    """
    tiles = tiles_for(token, model_dim, inter_dim, expert, topk)
    wanted = [
        co_name(2, model_dim, inter_dim, expert, topk, *tiles[1]),
        *(
            co_name(1, model_dim, inter_dim, expert, topk, *tiles[0], smooth_scale=sm)
            for sm in (False, True)
        ),
    ]
    return all(os.path.exists(os.path.join(co_dir(), n)) for n in wanted)


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
    "TUNED_CSV",
    "WAVES_PER_EU",
    "co_dir",
    "co_name",
    "have_co_for",
    "moe_blk_stage1_fwd",
    "moe_blk_stage2_fwd",
    "tiles_for",
    "tiles_from_co_name",
    "use_co_path",
]
