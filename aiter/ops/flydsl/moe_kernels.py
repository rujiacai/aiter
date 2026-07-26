# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

import functools
import os
import re

from typing import Dict, Optional
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility import dtypes

import flydsl.compiler as flyc
import torch

_KERNEL_PARAMS: Dict[str, Dict] = {}

_SUFFIX_RE = re.compile(r"(?P<fq>_fq)?(?:_sbm(?P<sbm>\d+))?$")
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_moe(?P<stage>[12])_a(?P<a>.+?)_w(?P<b>.+?)_"
    r"(?P<out>bf16|f16)_t(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)"
    r"(?:_(?P<rest>.*))?$"
)

_MFMA16_ALIASES = {"16", "16x16", "16x16x128", "mfma16", "mfma16k128"}
_MFMA32_ALIASES = {"32", "32x32", "32x32x64", "mfma32", "mfma32k64"}
_LDS_LIMIT_BYTES_BY_GFX = {
    "gfx942": 64 * 1024,
    "gfx950": 160 * 1024,
}


def _current_gfx() -> str:
    return str(get_gfx()).split(":", 1)[0].lower()


def _expand_per_tensor_scale(scale: Optional[torch.Tensor], rows: int, cols: int):
    """Expand compact per-tensor/per-expert scales to FlyDSL row-scale layout."""
    if scale is None:
        return None
    flat = scale.view(-1)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() == rows:
        return flat
    if cols > 1 and flat.numel() * cols == rows:
        return flat.view(-1, 1).expand(-1, cols).contiguous().view(-1)
    return flat


def _stage2_mfma_variant_tag(tile_k: int, a_dtype: str, b_dtype: str) -> str:
    """Return the FP4 stage2 MFMA variant tag needed in kernel names."""
    if a_dtype != "fp4" or b_dtype != "fp4":
        return ""
    if int(tile_k) == 128:
        return "mfma32k64"

    variant = os.environ.get("FLIR_MOE_STAGE2_MFMA", "16x16x128").strip().lower()
    if not variant or variant in _MFMA16_ALIASES:
        return ""
    if variant in _MFMA32_ALIASES:
        return "mfma32k64"
    raise ValueError(
        "FLIR_MOE_STAGE2_MFMA must be '16x16x128' or '32x32x64', "
        f"got {variant!r}"
    )


def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
    sort_block_m: int = 0,
    fuse_fp4_quant: bool = False,
) -> str:
    """Construct kernel name: ``flydsl_moe{stage}_a{a}_w{b}_{out}_t{M}x{N}x{K}[_{mode}][_fq][_sbm{S}]``."""
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if fuse_fp4_quant:
        name += "_fq"
    if sort_block_m > 0 and sort_block_m != tile_m:
        name += f"_sbm{sort_block_m}"
    return name


def _parse_flydsl_kernel_name(name: str) -> Optional[Dict]:
    match = _KERNEL_NAME_RE.match(name)
    if match is None:
        return None

    stage = int(match.group("stage"))
    a_dtype = match.group("a")
    b_dtype = match.group("b")
    params: Dict = {
        "stage": stage,
        "a_dtype": a_dtype,
        "b_dtype": b_dtype,
        "out_dtype": match.group("out"),
        "tile_m": int(match.group("tm")),
        "tile_n": int(match.group("tn")),
        "tile_k": int(match.group("tk")),
        "MPerBlock": int(match.group("tm")),
        "use_async_copy": False,
        "waves_per_eu": 1 if b_dtype == "fp4" else 0,
        "b_nt": 2,
    }
    if stage == 1:
        params.update({"k_batch": 1, "gate_only": False, "persist_m": 1})
    else:
        params.update(
            {
                "mode": "atomic",
                "sort_block_m": 0,
                "persist": False,
                "k_batch": 1,
                "persist_m": 1,
            }
        )

    for token in (match.group("rest") or "").split("_"):
        if not token:
            continue
        if token == "async":
            params["use_async_copy"] = True
        elif stage == 2 and token in ("atomic", "reduce"):
            params["mode"] = token
        elif token in ("mfma16k128", "mfma32k64", "mfma16k32"):
            params["mfma_variant"] = token
        elif token.startswith("w") and token[1:].isdigit():
            params["waves_per_eu"] = int(token[1:])
        elif token.startswith("bnt") and token[3:].isdigit():
            params["b_nt"] = int(token[3:])
        elif token.startswith("kb") and token[2:].isdigit():
            params["k_batch"] = int(token[2:])
        elif token.startswith("pm") and token[2:].isdigit():
            params["persist_m"] = int(token[2:])
        elif token.startswith("bp") and token[2:].isdigit():
            params["b_pool_depth"] = int(token[2:])
        elif token.startswith("xp") and token[2:].isdigit():
            params["x_pool_depth"] = int(token[2:])
        elif stage == 1 and token == "go":
            params["gate_only"] = True
        elif stage == 1 and token == "fq":
            params["fuse_fp4_quant"] = True
        elif stage == 2 and token == "persist":
            params["persist"] = True
        elif stage == 2 and token.startswith("pn") and token[2:].isdigit():
            # persist_n: number of consecutive N-tiles each workgroup serially
            # sweeps (per-WG N-loop length), reusing the M-block's activation X
            # across them. See compile_moe_gemm2.
            params["persist_n"] = int(token[2:])
        elif token.startswith("sbm") and token[3:].isdigit():
            params["sort_block_m"] = int(token[3:])
        elif token == "fq":
            params["fuse_fp4_quant"] = True
        elif token == "rgx":
            params["remap"] = "gx"
        elif token == "roff":
            params["remap"] = "off"
        elif token == "axy":
            params["splitk_axis"] = "y"
        elif token == "xnt":
            params["x_nt"] = 2
        elif token == "snt":
            params["scale_nt"] = 2
        elif token == "ont":
            params["out_nt"] = 2
        elif stage in (1, 2) and token.startswith("skw") and token[3:].isdigit():
            # StreamK num_wg (persistent grid size); must precede the sk<mode>
            # check since "skw..." also starts with "sk".
            params["streamk"] = True
            params["streamk_num_wg"] = int(token[3:])
        elif stage in (1, 2) and token.startswith("sk") and token[2:] in _SK_TAG_TO_MODE:
            params["streamk"] = True
            params["streamk_mode"] = _SK_TAG_TO_MODE[token[2:]]
        else:
            return None
    return params


def get_flydsl_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name. Strips ``_fq`` / ``_sbm{N}`` suffixes transparently."""
    params = _KERNEL_PARAMS.get(name)
    if params is not None:
        return params
    m = _SUFFIX_RE.search(name)
    if m and m.group(0):
        base_name = name[: m.start()]
        params = _KERNEL_PARAMS.get(base_name)
        if params is not None:
            extra: Dict = {}
            if m.group("fq"):
                extra["fuse_fp4_quant"] = True
            if m.group("sbm") is not None:
                extra["sort_block_m"] = int(m.group("sbm"))
            return {**params, **extra}
    return _parse_flydsl_kernel_name(name)


def _x_load_supported(
    a_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    use_async_copy: bool,
) -> bool:
    """Mirror X-load compile-time constraints before creating tune tasks."""
    elem_bytes = 2 if a_dtype in ("fp16", "bf16", "int4_bf16") else 1
    total_threads = (tile_n // 32) * 64
    bytes_x_per_tile = tile_m * tile_k * elem_bytes
    if bytes_x_per_tile % total_threads != 0:
        return False

    bytes_per_thread_x = bytes_x_per_tile // total_threads
    if a_dtype in ("fp16", "bf16", "int4_bf16"):
        return bytes_per_thread_x % 16 == 0
    if use_async_copy:
        return bytes_per_thread_x % 16 == 0
    return bytes_per_thread_x % 4 == 0


@functools.lru_cache(maxsize=1)
def _device_lds_limit_bytes() -> int:
    """Return the supported gfx target's per-workgroup LDS limit in bytes."""
    gfx = _current_gfx()
    if gfx not in _LDS_LIMIT_BYTES_BY_GFX:
        raise RuntimeError(f"FlyDSL MoE LDS filtering does not support {gfx!r}.")
    return _LDS_LIMIT_BYTES_BY_GFX[gfx]


def _tune_csv(env_name: str, default: list[str]) -> list[str]:
    """Read a comma-separated tuning candidate list from env (else default)."""
    raw = os.environ.get(env_name, "")
    if not raw.strip():
        return default
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _moe_knob_variants(kb: int):
    """Yield (name_tag, params) for the dispatch/cache tuning knobs.

    Defaults keep a single canonical variant (empty tag, no params) so the tuned
    candidate set is byte-identical unless widened via env:
      AITER_TUNE_MOE_REMAP=gy,gx,off   AITER_TUNE_MOE_SPLITK_AXIS=z,y
      AITER_TUNE_MOE_X_NT=0,2          AITER_TUNE_MOE_SCALE_NT=0,2
      AITER_TUNE_MOE_OUT_NT=0,2
    """
    remaps = _tune_csv("AITER_TUNE_MOE_REMAP", ["gy"])
    axes = _tune_csv("AITER_TUNE_MOE_SPLITK_AXIS", ["z"])
    xnts = _tune_csv("AITER_TUNE_MOE_X_NT", ["0"])
    snts = _tune_csv("AITER_TUNE_MOE_SCALE_NT", ["0"])
    onts = _tune_csv("AITER_TUNE_MOE_OUT_NT", ["0"])
    seen = set()
    for rm in remaps:
        for ax in axes:
            # splitk_axis only changes the grid when kb>1; collapse to "z" otherwise
            # to avoid emitting duplicate kernels under different names.
            eff_ax = ax if kb > 1 else "z"
            for xn in xnts:
                for sn in snts:
                    for on in onts:
                        tag, p = "", {}
                        if rm == "gx":
                            tag += "_rgx"; p["remap"] = "gx"
                        elif rm == "off":
                            tag += "_roff"; p["remap"] = "off"
                        if eff_ax == "y":
                            tag += "_axy"; p["splitk_axis"] = "y"
                        if int(xn):
                            tag += "_xnt"; p["x_nt"] = 2
                        if int(sn):
                            tag += "_snt"; p["scale_nt"] = 2
                        if int(on):
                            tag += "_ont"; p["out_nt"] = 2
                        if tag in seen:
                            continue
                        seen.add(tag)
                        yield tag, p


def _pool_variants():
    """Yield (name_tag, params) for stage1/stage2 B/X weight/activation prefetch-pool depths.

    Default candidate set = no-pool + the bp3/xp3 double pool (the tuned winner:
    ~-3 to -5.5% at mid/large tokens). Widen/override via env, e.g.
      AITER_TUNE_MOE_POOL=0,bp3,xp3,bp3xp3,bp4xp4
    Tokens: "0" (none), "bp{N}", "xp{N}", "bp{N}xp{M}". Tag order matches the
    kernel module_name (_bp{N} before _xp{N}).
    """
    import re as _re
    specs = _tune_csv("AITER_TUNE_MOE_POOL", ["0", "bp3xp3"])
    seen = set()
    for s in specs:
        tag = ""
        p: Dict = {}
        mb = _re.search(r"bp(\d+)", s)
        if mb and int(mb.group(1)) >= 2:
            p["b_pool_depth"] = int(mb.group(1))
            tag += f"_bp{mb.group(1)}"
        mx = _re.search(r"xp(\d+)", s)
        if mx and int(mx.group(1)) >= 2:
            p["x_pool_depth"] = int(mx.group(1))
            tag += f"_xp{mx.group(1)}"
        if tag in seen:
            continue
        seen.add(tag)
        yield tag, p


def _persist_m_candidates() -> list[int]:
    """Yield persist_m tuning candidates (workgroup-merge along M).

    persist_m is the inverse of split-K: each WG serially sweeps persist_m
    M-blocks (fewer, heavier WGs). Defaults to a small set so large-token configs
    can be tuned out of the box; override via env, e.g.
      AITER_TUNE_MOE_PERSIST_M=1,2,4,8
    """
    vals = _tune_csv("AITER_TUNE_MOE_PERSIST_M", ["1", "2", "4", "8"])
    out: list[int] = []
    for v in vals:
        try:
            iv = int(v)
        except ValueError:
            continue
        if iv >= 1 and iv not in out:
            out.append(iv)
    return out or [1]


def _persist_n_candidates() -> list[int]:
    """Yield persist_n tuning candidates (N-merge along the output/model_dim).

    persist_n = how many consecutive N-tiles each WG serially sweeps, reusing
    the M-block's activation X across them (X is independent of the output dim).
    Default OFF: returns [] so the candidate set is byte-identical unless widened
    via env, e.g.
      AITER_TUNE_MOE_PERSIST_N=1,2,4,8,16
    Only values > 1 emit extra `_pn{n}` variants; per-shape divisibility (n must
    divide model_dim/tile_n) is checked by the caller when the shape is known.
    """
    vals = _tune_csv("AITER_TUNE_MOE_PERSIST_N", ["0"])
    out: list[int] = []
    for v in vals:
        try:
            iv = int(v)
        except ValueError:
            continue
        if iv > 1 and iv not in out:
            out.append(iv)
    return out


# StreamK dispatch modes <-> name-token aliases (aliases must be underscore-free
# so they survive the "_"-split kernel-name tokenizer).
_SK_MODE_TO_TAG = {
    "mfocus": "mfocus",
    "affine": "affine",
    "flat": "flat",
    "lockstep": "lockstep",
    "lockstep_mouter": "mouter",
    "xcd_nsplit": "nsplit",
}
_SK_TAG_TO_MODE = {v: k for k, v in _SK_MODE_TO_TAG.items()}


def _streamk_candidates():
    """Yield (name_tag, params) StreamK dispatch candidates for stage2 tuning.

    StreamK swaps the fixed 2D grid for a persistent grid of ``streamk_num_wg``
    WGs that sweep the output tiles in a chosen dispatch order (``mode``). It is a
    distinct dispatch axis: atomic-only, mutually exclusive with split-K /
    persist_m / persist_n. Default OFF -> returns [] so the candidate set is
    byte-identical. Enable via:
      AITER_TUNE_MOE_STREAMK=1                 # all modes, auto num_wg (occ2)
      AITER_TUNE_MOE_STREAMK=mfocus,affine     # specific modes
    Persistent-grid size is swept as a single knob -- OCCUPANCY (num_wg = num_cu *
    occ), which sets the in-flight wave count, is GPU-portable, and always yields a
    CU (hence NUM_XCD) multiple:
      AITER_TUNE_MOE_STREAMK_OCC=2,4,8
    Absent/0 => auto (streamk_default_num_wg). Names get a ``_sk{mode}`` token
    (+ ``_skw{N}`` when occ is pinned; N is the resolved absolute num_wg).
    """
    vals = _tune_csv("AITER_TUNE_MOE_STREAMK", ["0"])
    tokens = [str(v).strip().lower() for v in vals if str(v).strip()]
    if not tokens or tokens[0] in ("0", "off", "false", "no"):
        return []
    if tokens[0] in ("1", "all", "true", "yes", "on"):
        modes = list(_SK_MODE_TO_TAG)
    else:
        modes = [t for t in tokens if t in _SK_MODE_TO_TAG]
    # Single grid-size knob: occupancy (num_wg = num_cu * occ). Absent => [0] (auto).
    wgs: list[int] = []
    _occ_vals = _tune_csv("AITER_TUNE_MOE_STREAMK_OCC", [])
    if _occ_vals:
        from aiter.jit.utils.chip_info import get_cu_num
        _ncu = int(get_cu_num())
        for o in _occ_vals:
            try:
                iocc = int(o)
            except ValueError:
                continue
            if iocc > 0 and _ncu * iocc not in wgs:
                wgs.append(_ncu * iocc)
    if not wgs:
        wgs = [0]
    out = []
    for m in modes:
        for wg in wgs:
            tag = f"_sk{_SK_MODE_TO_TAG[m]}" + (f"_skw{wg}" if wg else "")
            out.append(
                (tag, {"streamk": True, "streamk_mode": m, "streamk_num_wg": wg})
            )
    return out


def _a2c_tune_enabled() -> bool:
    """Whether the autotuner should try a2_compact (expert-major compacted a2).

    Unlike the per-kernel knobs above, a2_compact is a COUPLED stage1<->stage2
    property (stage1 writes compacted, stage2 reads compacted -- they must match),
    so it is NOT enumerated as an independent per-stage name token. Instead the
    tuner re-profiles the best (stage1, stage2) PAIR with a2_compact and keeps it
    only when it is correct AND faster than the token-major pair.

    Default OFF -> the tuned result is byte-identical to before. Enable via:
      AITER_TUNE_MOE_A2C=1
    """
    vals = _tune_csv("AITER_TUNE_MOE_A2C", ["0"])
    return any(
        str(v).strip().lower() in ("1", "true", "yes", "on") for v in vals
    )


def compute_blk_valid_start(
    sorted_token_ids, num_tokens: int, block_m: int, topk: int,
    num_valid_ids=None,
):
    """Exclusive prefix-sum of per-M-block valid-row counts, for the compacted
    (expert-major, no-padding) a2 layout used by AITER_MOE_A2_COMPACT.

    A sorted position is *valid* exactly as the stage kernels decide it: fused
    ``t = id & 0xFFFFFF`` with ``t < num_tokens`` AND slot ``s = id >> 24`` with
    ``s < topk``, AND (when given) the position is below ``num_valid_ids`` (the
    padded tail of the array is zero-filled, i.e. ``t == 0``, and must not be
    counted). Valid rows are a contiguous prefix within each block, so the
    compacted row of valid sorted position ``p = b*block_m + r`` is
    ``blk_valid_start[b] + r`` and the layout is gapless.

    Args:
      sorted_token_ids: 1-D int32 tensor (fused token|slot).
      num_tokens: real token count (token-id sentinel threshold).
      block_m: sorting block size (== stage tile_m's M-block size).
      topk: routing top-k (slot sentinel threshold).
      num_valid_ids: optional int32 tensor/int bounding the processed region.
    Returns:
      (blk_valid_start[num_blocks] int32, total_valid int) on the same device.
    """
    import torch

    n = sorted_token_ids.numel()
    num_blocks = (n + block_m - 1) // block_m
    fused = sorted_token_ids.to(torch.int64)
    tok = fused & 0xFFFFFF
    slot = fused >> 24
    valid = (tok < int(num_tokens)) & (slot < int(topk))
    if num_valid_ids is not None:
        if torch.is_tensor(num_valid_ids):
            nvi = num_valid_ids.reshape(-1)[0].to(torch.int64)
        else:
            nvi = int(num_valid_ids)
        pos = torch.arange(n, device=sorted_token_ids.device)
        valid = valid & (pos < nvi)
    valid = valid.to(torch.int32)
    # pad to full blocks then per-block valid count
    if valid.numel() < num_blocks * block_m:
        pad = num_blocks * block_m - valid.numel()
        valid = torch.cat([valid, torch.zeros(pad, dtype=torch.int32, device=valid.device)])
    per_block = valid.view(num_blocks, block_m).sum(dim=1)  # [num_blocks] int32
    start = torch.zeros(num_blocks, dtype=torch.int32, device=sorted_token_ids.device)
    if num_blocks > 1:
        start[1:] = torch.cumsum(per_block, dim=0)[:-1].to(torch.int32)
    total_valid = int(per_block.sum().item())
    return start, total_valid


def _stage2_mfma_variants(tk: int, a_dtype: str, b_dtype: str):
    """Yield (name_tag, params-overlay) for the fp8 stage2 MFMA-ISA choice.

    For a given tile_k, several MFMA instructions are valid; this exposes that as
    a tuning dimension so the tuner can pick the best-latency ISA per shape:
      - auto / 16x16x128 : the wide `mfma_scale_f32_16x16x128_f8f6f4` when
                           tk % 128 == 0 (else the kernel falls back to 16x16x32).
      - 16x16x32         : force `mfma_f32_16x16x32_fp8_fp8` even at tk % 128 == 0.
                           Finer-grained -> lets pooling/soft-pipeline hide more of
                           the VALU->MFMA dependency stall (anti-starvation).

    Default (env unset) yields only the auto variant, so the candidate set stays
    byte-identical unless widened via AITER_TUNE_MOE_MFMA=auto,16x16x32. Only ISA
    valid for the given tile_k (and platform, gfx950 for the wide scale MFMA) are
    emitted.
    """
    if b_dtype == "fp4":
        # fp4 keeps its dedicated MFMA path (see _stage2_mfma_variant_tag).
        yield "", {}
        return
    specs = _tune_csv("AITER_TUNE_MOE_MFMA", ["auto"])
    seen: set[str] = set()
    for s in specs:
        s = s.strip().lower()
        if s in ("", "auto", "16x16x128", "mfma16k128"):
            key = "auto"
        elif s in ("16x16x32", "mfma16k32"):
            # Only meaningful where auto would pick the wide 16x16x128 (tk % 128
            # == 0); at tk % 128 != 0 auto already emits 16x16x32, so forcing it
            # would just duplicate that candidate.
            if (tk % 128) != 0:
                continue
            key = "mfma16k32"
        else:
            raise ValueError(
                "AITER_TUNE_MOE_MFMA tokens must be in "
                "{auto,16x16x128,16x16x32}, got " + repr(s)
            )
        if key in seen:
            continue
        seen.add(key)
        if key == "auto":
            yield "", {}
        else:
            yield f"_{key}", {"mfma_variant": key}
    if not seen:
        yield "", {}


def _async_copy_candidates() -> list[bool]:
    gfx = _current_gfx()
    if gfx == "gfx942":
        #TODO: gfx942 not support buffer load from global memory to LDS directly, so we only support false
        return [False]
    if gfx == "gfx950":
        return [False, True]
    raise RuntimeError(f"FlyDSL MoE async-copy enumeration does not support {gfx!r}.")


def _lds_within_limit(
    a_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_per_eu: int,
    *,
    use_cshuffle_epilog: bool = True,
    lds_limit_bytes: Optional[int] = None,
) -> bool:
    """Filter candidates that exceed the supported gfx target's LDS limit."""
    if lds_limit_bytes is None:
        lds_limit_bytes = _device_lds_limit_bytes()
    elem_bytes = 2 if a_dtype in ("fp16", "bf16", "int4_bf16") else 1
    lds_x_bytes = 2 * int(tile_m) * int(tile_k) * int(elem_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0
    lds_tid_bytes = int(tile_m) * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    if waves_per_eu >= 1:
        # Match compile_moe_gemm{1,2}: waves_per_eu reserves minimum LDS to
        # influence occupancy. With 160KB total LDS, waves_per_eu=1 raises
        # allocation to ~82KB.
        min_lds = (160 * 1024) // (waves_per_eu + 1) + 1
        lds_total_bytes = max(lds_total_bytes, min_lds)
    return lds_total_bytes <= lds_limit_bytes


def get_flydsl_stage1_kernels(
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    model_dim: Optional[int] = None,
    n_dim: Optional[int] = None,
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage1 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"

    tile_ns = [32, 64, 128, 256] if is_fp4 else [64, 128, 256]
    tile_ks = [256] if is_fp4 else [128, 256]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [32, 64, 128]
    waves_per_eus = [1, 2, 3, 4] if is_fp4 else [0, 1, 2, 3, 4]
    # Non-fp4 split-K is supported by moe_gemm_2stage.compile_moe_gemm1 (the GEMM
    # writes raw gate/up partials with atomic-add; silu/mul is deferred to the
    # post-kernel reduction). It requires the CShuffle epilogue (tile_n % 128 == 0),
    # guarded below. The autotuner only picks these when they win, so the default
    # (best) performance is preserved.
    k_batches = [1, 2, 4, 8, 16] if is_fp4 else [1, 2, 4]
    b_nts = [0, 2] if is_fp4 else [0, 2]
    async_copies = _async_copy_candidates()
    for tm in tile_ms:
        stage1_tile_ns = tile_ns
        if is_fp4:
            if tm in [16, 32]:
                stage1_tile_ns = [32, 64, 128]
            else:
                stage1_tile_ns = [64, 128, 256]
        for tn in stage1_tile_ns:
            if n_dim is not None and n_dim % tn != 0:
                continue
            use_cshuffle_epilog = None if is_fp4 or tn % 128 == 0 else False
            for tk in tile_ks:
                if model_dim is not None and model_dim % tk != 0:
                    continue
                for async_copy in async_copies:
                    if not _x_load_supported(a_dtype, tm, tn, tk, async_copy):
                        continue
                    for wpe in waves_per_eus:
                        if (
                            not is_fp4
                            and not _lds_within_limit(
                                a_dtype,
                                tm,
                                tn,
                                tk,
                                wpe,
                                use_cshuffle_epilog=(
                                    True
                                    if use_cshuffle_epilog is None
                                    else bool(use_cshuffle_epilog)
                                ),
                            )
                        ):
                            continue
                        for kb in k_batches if wpe == 3 else [1]:
                            if is_fp4:
                                if tk == 512 and kb == 16:
                                    continue
                                if tn // 32 * 64 > 256:
                                    continue
                            elif kb > 1:
                                # Non-fp4 split-K stage1 requires the CShuffle epilogue
                                # (raw partials are shuffled to contiguous half2 frags
                                # for packed atomic-add). It is only enabled when
                                # tile_n % 128 == 0 (use_cshuffle_epilog resolves True).
                                if tn % 128 != 0:
                                    continue
                            # Split-K stage1 requires:
                            #   model_dim % k_batch == 0 and
                            #   (model_dim // k_batch) % tile_k == 0
                            # When model_dim is provided by the tuner, skip
                            # kernels that can never compile for that shape.
                            if model_dim is not None and kb > 1:
                                if model_dim % kb != 0:
                                    continue
                                # The stage1 ping-pong loop has a fixed 2-tile
                                # tail, so the per-slice tile count must be even
                                # (an odd count silently drops the middle tile).
                                if (model_dim // kb) % (2 * tk) != 0:
                                    continue
                            gate_onlys = [False, True] if kb > 1 and is_fp4 else [False]
                            for bnt in b_nts:
                                for go in gate_onlys:
                                    name = flydsl_kernel_name(
                                        1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                                    )
                                    if async_copy:
                                        name += "_async"
                                    if is_fp4 and wpe != 1:
                                        name += f"_w{wpe}"
                                    elif not is_fp4 and wpe > 0:
                                        name += f"_w{wpe}"
                                    if kb != 1:
                                        name += f"_kb{kb}"
                                    if bnt != 2:
                                        name += f"_bnt{bnt}"
                                    if go:
                                        name += "_go"
                                    base = {
                                        "stage": 1,
                                        "a_dtype": a_dtype,
                                        "b_dtype": b_dtype,
                                        "out_dtype": out_dtype,
                                        "tile_m": tm,
                                        "tile_n": tn,
                                        "tile_k": tk,
                                        "MPerBlock": tm,
                                        "use_cshuffle_epilog": use_cshuffle_epilog,
                                        "use_async_copy": async_copy,
                                        "waves_per_eu": wpe,
                                        "k_batch": kb,
                                        "b_nt": bnt,
                                        "gate_only": go,
                                    }
                                    # fp4 stage1 goes through the mixed kernel, which
                                    # does not take the dispatch/cache knobs.
                                    if is_fp4:
                                        kernels[name] = base
                                        continue
                                    # fp8 MFMA-ISA choice (16x16x128 scale vs
                                    # 16x16x32) is a tuning dimension, gated per
                                    # tile_k. Default env => auto only, so the
                                    # candidate set is byte-identical unless widened
                                    # via AITER_TUNE_MOE_MFMA (shared with stage2).
                                    for mvtag, mvp in _stage2_mfma_variants(
                                        tk, a_dtype, b_dtype
                                    ):
                                        v_name = name + mvtag
                                        v_base = {**base, **mvp}
                                        # persist_m (workgroup-merge along M) is the
                                        # inverse of split-K, so it only applies when
                                        # k_batch==1. The tuner gates split-K vs persist
                                        # by token count.
                                        _pms = _persist_m_candidates() if kb == 1 else [1]
                                        for ktag, kp in _moe_knob_variants(kb):
                                            for pm in _pms:
                                                pmtag = f"_pm{pm}" if pm != 1 else ""
                                                kernels[v_name + ktag + pmtag] = {
                                                    **v_base,
                                                    **kp,
                                                    "persist_m": pm,
                                                }
                                            # B/X prefetch pool: reuse the shared
                                            # `_pool_variants()` as tunable name tokens
                                            # (`_bpN`/`_xpN`/`_bpNxpN`) exactly like stage2
                                            # (driven by AITER_TUNE_MOE_POOL). Only for small N
                                            # tiles (tile_n<=128) with register room -- gate+up
                                            # doubles the accumulators, so large tiles spill.
                                            # k_batch==1, persist_m==1. Skip "" (no-pool already
                                            # emitted above).
                                            if kb == 1 and tn <= 128:
                                                for ptag, pp in _pool_variants():
                                                    if not pp:
                                                        continue
                                                    kernels[v_name + ktag + ptag] = {
                                                        **v_base,
                                                        **kp,
                                                        "persist_m": 1,
                                                        **pp,
                                                    }
                                            # StreamK dispatch variants (stage1;
                                            # AITER_TUNE_MOE_STREAMK, default off).
                                            # kb==1 only (mutually exclusive with
                                            # split-K/persist). Inherits tile/wpe/bnt/
                                            # mfma/async; adds mode x num_wg (+ pool
                                            # for tn<=128). Dropped by the tuner's cos
                                            # check if a combo is invalid.
                                            if kb == 1:
                                                for sktag, skp in _streamk_candidates():
                                                    kernels[v_name + ktag + sktag] = {
                                                        **v_base, **kp,
                                                        "persist_m": 1, **skp,
                                                    }
                                                    if tn <= 128:
                                                        for ptag, pp in _pool_variants():
                                                            if not pp:
                                                                continue
                                                            kernels[
                                                                v_name + ktag + ptag + sktag
                                                            ] = {
                                                                **v_base, **kp,
                                                                "persist_m": 1,
                                                                **pp, **skp,
                                                            }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    k_dim: Optional[int] = None,
    n_dim: Optional[int] = None,
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [128, 256] if is_fp4 else [128, 256]
    tile_ks = [128, 256] if is_fp4 else [64, 128, 256]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [32, 64, 128]
    modes = ["atomic", "reduce"]
    waves_per_eus = [0] if is_fp4 else [0, 1, 2, 3, 4]
    b_nts = [0] if is_fp4 else [0, 2]
    async_copies = _async_copy_candidates()

    for tm in tile_ms:
        for tn in tile_ns:
            if n_dim is not None and n_dim % tn != 0:
                continue
            for tk in tile_ks:
                if k_dim is not None and k_dim % tk != 0:
                    continue
                mfma_variant_tag = _stage2_mfma_variant_tag(tk, a_dtype, b_dtype)
                if is_fp4 and tk == 128:
                    # tile_k=128 uses the dedicated MFMA32/K64 stage2 path.
                    # Keep the candidate set deliberately small until layout
                    # coverage is broadened.
                    if a_dtype != "fp4" or tm not in (32, 64) or tn != 128:
                        continue
                for mode in modes:
                    for async_copy in async_copies:
                        if is_fp4 and tk == 128 and async_copy:
                            continue
                        if not _x_load_supported(a_dtype, tm, tn, tk, async_copy):
                            continue
                        for wpe in waves_per_eus:
                            if not is_fp4 and not _lds_within_limit(
                                a_dtype, tm, tn, tk, wpe
                            ):
                                continue
                            for bnt in b_nts:
                                # Non-fp4 split-K stage2: K=inter_dim is partitioned
                                # across k_batch workgroups that atomic-accumulate
                                # their partials (requires atomic mode). Only valid
                                # when inter_dim (k_dim) is divisible appropriately.
                                if (not is_fp4) and mode == "atomic" and wpe == 3:
                                    _stage2_kbs = [1, 2, 4]
                                else:
                                    _stage2_kbs = [1]
                                for kb in _stage2_kbs:
                                    if kb > 1 and k_dim is not None:
                                        if k_dim % kb != 0:
                                            continue
                                        if (k_dim // kb) % tk != 0:
                                            continue
                                    base_name = flydsl_kernel_name(
                                        2,
                                        a_dtype,
                                        b_dtype,
                                        out_dtype,
                                        tm,
                                        tn,
                                        tk,
                                        mode,
                                    )
                                    if async_copy:
                                        base_name += "_async"
                                    if mfma_variant_tag:
                                        base_name += f"_{mfma_variant_tag}"
                                    if is_fp4 and wpe != 1:
                                        base_name += f"_w{wpe}"
                                    elif not is_fp4 and wpe > 0:
                                        base_name += f"_w{wpe}"
                                    if bnt != 2:
                                        base_name += f"_bnt{bnt}"
                                    if kb != 1:
                                        base_name += f"_kb{kb}"
                                    base_params = {
                                        "stage": 2,
                                        "a_dtype": a_dtype,
                                        "b_dtype": b_dtype,
                                        "out_dtype": out_dtype,
                                        "tile_m": tm,
                                        "tile_n": tn,
                                        "tile_k": tk,
                                        "mode": mode,
                                        "MPerBlock": tm,
                                        "use_async_copy": async_copy,
                                        "waves_per_eu": wpe,
                                        "b_nt": bnt,
                                        "k_batch": kb,
                                    }
                                    if mfma_variant_tag:
                                        base_params["mfma_variant"] = mfma_variant_tag
                                    # fp4 stage2 goes through the mixed kernel, which
                                    # does not take the dispatch/cache knobs.
                                    if is_fp4:
                                        kernels[base_name] = base_params
                                        continue
                                    # fp8 MFMA-ISA choice (16x16x128 scale vs
                                    # 16x16x32) is a tuning dimension, gated per
                                    # tile_k. Default env => auto only, so the
                                    # candidate set is byte-identical unless widened
                                    # via AITER_TUNE_MOE_MFMA.
                                    for mvtag, mvp in _stage2_mfma_variants(
                                        tk, a_dtype, b_dtype
                                    ):
                                        v_name = base_name + mvtag
                                        v_params = {**base_params, **mvp}
                                        # persist_m (workgroup-merge along M) is the
                                        # inverse of split-K, so it only applies when
                                        # k_batch==1. The tuner gates split-K vs persist
                                        # by token count.
                                        _pms = _persist_m_candidates() if kb == 1 else [1]
                                        for ktag, kp in _moe_knob_variants(kb):
                                            for pm in _pms:
                                                pmtag = f"_pm{pm}" if pm != 1 else ""
                                                # B/X prefetch pool applies to ALL stage2
                                                # paths (default, persist_m>1, split-K): the
                                                # pool is a compile-time ring over the same
                                                # constexpr K-loop, which is emitted inside the
                                                # persist-loop body and is split-K slice-aware
                                                # (uses k_start/num_k_tiles_py). Measured gains
                                                # ~+5-8% at 16k/40k for pm4 and kb2 (X pool
                                                # still helps since activations aren't reused
                                                # across M-blocks). kernel has no pm/kb gate on
                                                # _use_pool; this just makes them candidates.
                                                _pvars = list(_pool_variants())
                                                for ptag, pp in _pvars:
                                                    kernels[v_name + ktag + pmtag + ptag] = {
                                                        **v_params,
                                                        **kp,
                                                        "persist_m": pm,
                                                        **pp,
                                                    }
                                                    # persist_n (N-merge) variants:
                                                    # each WG sweeps `pn` consecutive
                                                    # N-tiles, reusing the M-block's X.
                                                    # Only for the plain pm==1/kb==1
                                                    # path; default env unset => no
                                                    # extra variants (candidate set
                                                    # unchanged). pn must divide the
                                                    # N-tile count (model_dim/tile_n)
                                                    # when the shape is known.
                                                    if pm == 1 and kb == 1:
                                                        for pn in _persist_n_candidates():
                                                            if (
                                                                n_dim is not None
                                                                and (n_dim // tn) % pn != 0
                                                            ):
                                                                continue
                                                            kernels[
                                                                v_name + ktag + pmtag + ptag + f"_pn{pn}"
                                                            ] = {
                                                                **v_params,
                                                                **kp,
                                                                "persist_m": pm,
                                                                **pp,
                                                                "persist_n": pn,
                                                            }
                                                    # StreamK dispatch variants
                                                    # (AITER_TUNE_MOE_STREAMK; default
                                                    # off). Mutually exclusive with
                                                    # split-K/persist, so only on the
                                                    # pm==1/kb==1 path; works for both
                                                    # atomic and reduce (full-K per
                                                    # tile -> no cross-WG K-partials).
                                                    # Hung here => inherits the swept
                                                    # tile/wpe/bnt/mfma/async/pool
                                                    # knobs; the sweep adds mode x
                                                    # num_wg on top. Wrong combos are
                                                    # dropped by the tuner's cos check.
                                                    if pm == 1 and kb == 1:
                                                        for sktag, skp in _streamk_candidates():
                                                            kernels[
                                                                v_name + ktag + pmtag + ptag + sktag
                                                            ] = {
                                                                **v_params,
                                                                **kp,
                                                                "persist_m": pm,
                                                                **pp,
                                                                **skp,
                                                            }
    return kernels


def _register_all_configs():
    """Pre-populate _KERNEL_PARAMS with all supported configs at import time."""
    for a in ("fp8", "fp4", "fp16"):
        for b in ("fp4",):
            for out in ("bf16", "f16"):
                _KERNEL_PARAMS.update(get_flydsl_stage1_kernels(a, b, out))
                _KERNEL_PARAMS.update(get_flydsl_stage2_kernels(a, b, out))
    for a in ("fp8", "int8"):
        for b in (a,):
            for out in ("bf16", "f16"):
                _KERNEL_PARAMS.update(get_flydsl_stage1_kernels(a, b, out))
                _KERNEL_PARAMS.update(get_flydsl_stage2_kernels(a, b, out))


_register_all_configs()


def compile_flydsl_moe_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    act: str = "silu",
    use_g1u1: bool = True,
    persist_m: int = 1,
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    use_cshuffle_epilog: Optional[bool] = None,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_only: bool = False,
    remap: Optional[str] = None,
    splitk_axis: Optional[str] = None,
    x_nt: Optional[int] = None,
    scale_nt: Optional[int] = None,
    out_nt: Optional[int] = None,
    b_pool_depth: int = 0,
    x_pool_depth: int = 0,
    mfma_variant: Optional[str] = None,
    a2_compact: bool = False,
    streamk: bool = False,
    streamk_num_wg: int = 0,
    streamk_mode: str = "mfocus",
):
    """Compile stage1 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1

        return compile_mixed_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            act=act,
            use_g1u1=use_g1u1,
            persist_m=persist_m,
            fuse_fp4_quant=fuse_fp4_quant,
            fuse_sort_scale=fuse_sort_scale,
            use_async_copy=use_async_copy,
            k_batch=k_batch,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            gate_only=gate_only,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm1

        return compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            act=act,
            use_g1u1=use_g1u1,
            use_cshuffle_epilog=use_cshuffle_epilog,
            use_async_copy=use_async_copy,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            k_batch=k_batch,
            persist_m=persist_m,
            remap=remap,
            splitk_axis=splitk_axis,
            x_nt=x_nt,
            scale_nt=scale_nt,
            out_nt=out_nt,
            b_pool_depth=b_pool_depth,
            x_pool_depth=x_pool_depth,
            mfma_variant=mfma_variant,
            a2_compact=a2_compact,
            streamk=streamk,
            streamk_num_wg=streamk_num_wg,
            streamk_mode=streamk_mode,
        )


def compile_flydsl_moe_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
    persist_m: int = 1,
    sort_block_m: int = 0,
    use_async_copy: bool = False,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    mfma_variant: Optional[str] = None,
    k_batch: int = 1,
    remap: Optional[str] = None,
    splitk_axis: Optional[str] = None,
    x_nt: Optional[int] = None,
    scale_nt: Optional[int] = None,
    out_nt: Optional[int] = None,
    b_pool_depth: int = 0,
    x_pool_depth: int = 0,
    persist_n: int = 0,
    a2_compact: bool = False,
    streamk: bool = False,
    streamk_num_wg: int = 0,
    streamk_mode: str = "mfocus",
):
    """Compile stage2 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

        return compile_mixed_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
            persist_m=persist_m,
            sort_block_m=sort_block_m,
            use_async_copy=use_async_copy,
            mfma_variant=mfma_variant,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm2

        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
            use_async_copy=use_async_copy,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            k_batch=k_batch,
            persist_m=persist_m,
            remap=remap,
            splitk_axis=splitk_axis,
            x_nt=x_nt,
            scale_nt=scale_nt,
            out_nt=out_nt,
            b_pool_depth=b_pool_depth,
            x_pool_depth=x_pool_depth,
            persist_n=persist_n,
            mfma_variant=mfma_variant,
            a2_compact=a2_compact,
            streamk=streamk,
            streamk_num_wg=streamk_num_wg,
            streamk_mode=streamk_mode,
        )


# Private helpers


_DLPACK_SAFE = (torch.uint8, torch.float16, torch.bfloat16, torch.float32)


def _view_safe(t: torch.Tensor) -> torch.Tensor:
    """View as uint8 if dtype is not dlpack-safe, otherwise return as-is."""
    return (
        t.view(torch.uint8)
        if t is not None and t.numel() > 0 and t.dtype not in _DLPACK_SAFE
        else t
    )


def _s1_args_fp4(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    out_scale_sorted,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
    dev,
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    return (
        _view_safe(out),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        empty_f32,
        out_scale_sorted,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        torch.cuda.current_stream(),
    )


def _s1_args_std(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
    blk_valid_start,
):
    return (
        out,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        blk_valid_start,
        torch.cuda.current_stream(),
    )


def _s2_args_fp4(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
    dev,
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    return (
        _view_safe(target),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        empty_f32,
        token_num,
        n_in,
        k_in,
        blocks,
        torch.cuda.current_stream(),
    )


def _s2_args_std(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
    blk_valid_start,
):
    return (
        target,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        n_in,
        k_in,
        blocks,
        blk_valid_start,
        torch.cuda.current_stream(),
    )


def _run_compiled(exe, args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    # flydsl>=0.1.2 exposes flydsl.compiler.compile; older versions (e.g.
    # 0.1.1.dev409) only provide jit wrappers and execute via exe(*args).
    if not hasattr(flyc, "compile"):
        exe(*args)
        return

    cf = getattr(exe, "_aiter_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._aiter_cf = cf
    else:
        cf(*args)


@functools.cache
def _get_compiled_silu_fq(inter_dim: int, topk: int):
    """Compile and cache the fused silu_and_mul + mxfp4 quant + scale-sort kernel."""
    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module

    return build_silu_and_mul_fq_module(inter_dim, topk)


# Public API


def flydsl_moe_stage1(
    a: torch.Tensor,
    w1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    act: str = "silu",
    use_g1u1: Optional[bool] = None,
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    persist_m: int = 0,
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    use_cshuffle_epilog: Optional[bool] = None,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_only: bool = False,
    remap: Optional[str] = None,
    splitk_axis: Optional[str] = None,
    x_nt: Optional[int] = None,
    scale_nt: Optional[int] = None,
    out_nt: Optional[int] = None,
    b_pool_depth: int = 0,
    x_pool_depth: int = 0,
    mfma_variant: Optional[str] = None,
    a2_compact: bool = False,
    blk_valid_start: Optional[torch.Tensor] = None,
    streamk: bool = False,
    streamk_num_wg: int = 0,
    streamk_mode: Optional[str] = None,
):
    """Fused MOE stage1 GEMM.

    a: (token_num, model_dim)
    w1:
      - g1u1: (E, 2*inter_dim, model_dim) pre-shuffled
      - g1u0: (E, inter_dim, model_dim) pre-shuffled
    For fp4 stage1, `w1`/`w1_scale` must use the same preshuffle layout as
    `shuffle_weight(..., (16, 16))` and `e8m0_shuffle(...)`.

    When fuse_sort_scale=True, the kernel writes e8m0 scales in sorted tiled
    layout directly, avoiding a separate moe_mxfp4_sort call.

    When k_batch>1 (split-K), the kernel writes partial sums via atomics into
    a zeroed buffer, then runs post-activation:
      - g1u1: reduce gate/up partials with activation+mul
      - g1u0: reduce proj partials then apply unary activation

    When gate_only=True (requires use_g1u1 and k_batch>1), each workgroup computes only
    one B-tile stream (no gate/up interleaving).  The grid X doubles so
    that by_n naturally covers both gate and up regions.

    Returns:
        Basic:                      out
        fuse_sort_scale:            (out, out_scale_sorted)
    """
    token_num = a.shape[0]
    E = w1.shape[0]

    if use_g1u1 is None:
        logical_out_inter_dim = None
        if out is not None:
            logical_out_inter_dim = out.shape[-1] * (2 if fuse_fp4_quant else 1)
        if logical_out_inter_dim is not None:
            if w1.shape[1] == 2 * logical_out_inter_dim:
                use_g1u1 = True
                inter_dim = logical_out_inter_dim
            elif w1.shape[1] == logical_out_inter_dim:
                use_g1u1 = False
                inter_dim = logical_out_inter_dim
            else:
                raise ValueError(
                    f"Unable to infer g1u mode from w1.shape={tuple(w1.shape)} "
                    f"and out.shape={tuple(out.shape)}"
                )
        else:
            # Preserve the historical direct-call behavior: when the caller does
            # not provide enough shape information, assume the legacy g1u1 path.
            if (w1.shape[1] % 2) != 0:
                raise ValueError(
                    f"Unable to infer g1u1 inter_dim from odd w1.shape[1]={w1.shape[1]}"
                )
            use_g1u1 = True
            inter_dim = w1.shape[1] // 2
    else:
        use_g1u1 = bool(use_g1u1)
        if use_g1u1:
            if (w1.shape[1] % 2) != 0:
                raise ValueError(
                    f"g1u1 stage1 expects w1.shape[1] to be even, got {w1.shape[1]}"
                )
            inter_dim = w1.shape[1] // 2
        else:
            inter_dim = w1.shape[1]

    if act == "swiglu" and not use_g1u1:
        raise ValueError("swiglu stage1 requires use_g1u1=True")

    model_dim = a.shape[1]

    if a_dtype == "fp4":
        model_dim = model_dim * 2

    torch_out_dtype = (
        dtypes.fp4x2
        if fuse_fp4_quant
        else dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
    )
    _is_splitk = k_batch > 1

    dev = a.device
    # streamk (stage1): persistent-grid dispatch, N axis = inter_dim. flat/affine
    # need the even-split wg_unit_start schedule (n_tiles = inter_dim/tile_n,
    # k_tiles=1) carried in the arg_blk_valid_start slot; lockstep/mfocus/nsplit
    # compute their dispatch in-kernel. See compile_moe_gemm1.
    _sk_mode = (
        streamk_mode
        if streamk_mode is not None
        else os.environ.get("AITER_MOE_STREAMK_MODE", "mfocus")
    ).lower()
    _sk_num_wg = 0
    _sk_wg_start = None
    if streamk:
        from .kernels.streamk_schedule import (
            streamk_default_num_wg,
            flydsl_streamk_schedule,
        )
        _sk_num_wg = (
            int(streamk_num_wg)
            if streamk_num_wg and int(streamk_num_wg) > 0
            else streamk_default_num_wg()
        )
        if _sk_mode in ("flat", "affine"):
            _sk_s1_n_tiles = inter_dim // tile_n
            _sk_wg_start = flydsl_streamk_schedule(
                num_valid_ids, _sk_num_wg, tile_m, _sk_s1_n_tiles, 1
            )
    _splitk_fq = _is_splitk and fuse_fp4_quant

    _splitk_out_cols = inter_dim * (2 if use_g1u1 else 1)
    if _splitk_fq and not use_g1u1:
        raise ValueError("split-K fused fp4 quant currently requires use_g1u1=True")
    if _splitk_fq and act not in ("silu", "swiglu"):
        raise ValueError("split-K fused fp4 quant only supports silu/swiglu stage1")

    if out is None:
        if fuse_fp4_quant:
            out = torch.empty(
                (token_num, topk, inter_dim // 2), dtype=torch_out_dtype, device=dev
            )
        else:
            out = torch.empty(
                (token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev
            )

    if _is_splitk:
        torch_tmp_out_dtype = dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
        tmp_out = torch.zeros(
            (token_num, topk, _splitk_out_cols), dtype=torch_tmp_out_dtype, device=dev
        )
    else:
        tmp_out = None

    flat_a_scale = _expand_per_tensor_scale(a1_scale, token_num, 1)
    if flat_a_scale is None:
        flat_a_scale = torch.empty(0, device=dev)
    flat_w_scale = _expand_per_tensor_scale(w1_scale, E * w1.shape[1], w1.shape[1])
    if flat_w_scale is None:
        flat_w_scale = torch.empty(0, device=dev)
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )

    _need_quant = fuse_fp4_quant or _splitk_fq
    _need_sort = _need_quant and (fuse_sort_scale or _splitk_fq)

    _sort_block_m = max(32, tile_m)
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = (
        min(token_num * topk * _sort_block_m, sorted_token_ids.shape[0])
        // _sort_block_m
    )
    _grid_y = min(_dense_blks, _all_blks)

    _persist_m = persist_m if persist_m > 0 else 1

    # Allocate sorted-scale buffer with padding for tiled layout
    scale_cols = inter_dim // 32
    sorted_size = max(
        sorted_token_ids.shape[0], sorted_expert_ids.shape[0] * _sort_block_m
    )
    padded_rows = (sorted_size + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    out_scale_sorted_flat = (
        torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)
        if _need_sort
        else torch.empty(0, dtype=torch.uint8, device=dev)
    )

    # split-K GEMM kernel does not fuse quant; the fused silu_and_mul_fq kernel
    # handles activation + quant + scale-sort after the GEMM completes.
    _gemm_fq = fuse_fp4_quant and not _is_splitk
    _gemm_fss = fuse_sort_scale and not _is_splitk

    _kernel_out = tmp_out if _is_splitk else out
    is_fp4 = b_dtype == "fp4"
    _n_in = inter_dim * (2 if (is_fp4 and use_g1u1) else 1)
    # _n_in is used by kernel launch to compute gx.
    # Keep historical behavior:
    # - use_g1u1=True (gated): fp4 path uses 2*inter_dim, non-fp4 uses inter_dim
    # - use_g1u1=False  (non-gated): always inter_dim
    if use_g1u1:
        _n_in = inter_dim * 2 if is_fp4 else inter_dim
    else:
        _n_in = inter_dim
    _k_in = model_dim

    if is_fp4:
        args = _s1_args_fp4(
            _kernel_out.view(-1),
            a.view(-1),
            w1.view(-1),
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            out_scale_sorted_flat.view(-1),
            token_num,
            _n_in,
            _k_in,
            _grid_y,
            dev,
        )
    else:
        _bvs1 = _sk_wg_start if _sk_wg_start is not None else blk_valid_start
        if _bvs1 is None:
            _bvs1 = torch.zeros(1, dtype=torch.int32, device=a.device)
        args = _s1_args_std(
            _kernel_out.view(-1),
            a.view(-1),
            w1.view(-1),
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            _grid_y,
            _bvs1,
        )

    exe = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        use_g1u1=use_g1u1,
        persist_m=_persist_m,
        fuse_fp4_quant=_gemm_fq,
        fuse_sort_scale=_gemm_fss,
        use_async_copy=use_async_copy,
        use_cshuffle_epilog=False
        if use_cshuffle_epilog is None and tile_n % 128 != 0
        else use_cshuffle_epilog,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_only=gate_only,
        remap=remap,
        splitk_axis=splitk_axis,
        x_nt=x_nt,
        scale_nt=scale_nt,
        out_nt=out_nt,
        b_pool_depth=b_pool_depth,
        x_pool_depth=x_pool_depth,
        mfma_variant=mfma_variant,
        a2_compact=a2_compact,
        streamk=streamk,
        streamk_num_wg=_sk_num_wg,
        streamk_mode=_sk_mode,
    )
    _run_compiled(exe, args)

    if _splitk_fq:
        _silu_fq = _get_compiled_silu_fq(inter_dim, topk)
        num_sorted_rows = sorted_token_ids.shape[0]
        _run_compiled(
            _silu_fq,
            (
                tmp_out.view(-1, inter_dim * 2),
                out.view(-1).view(torch.uint8),
                out_scale_sorted_flat,
                sorted_token_ids,
                num_valid_ids,
                token_num,
                num_sorted_rows,
                torch.cuda.current_stream(),
            ),
        )
    elif _is_splitk:
        if use_g1u1:
            if act == "gelu":
                from aiter.ops.activation import gelu_and_mul

                gelu_and_mul(out.view(-1, inter_dim), tmp_out.view(-1, inter_dim * 2))
            else:
                from aiter.ops.activation import silu_and_mul

                silu_and_mul(out.view(-1, inter_dim), tmp_out.view(-1, inter_dim * 2))
        else:
            tmp_view = tmp_out.view(-1, inter_dim).to(torch.float32)
            if act == "gelu":
                out.copy_(
                    torch.nn.functional.gelu(tmp_view, approximate="tanh")
                    .to(out.dtype)
                    .view_as(out)
                )
            else:
                out.copy_(
                    torch.nn.functional.silu(tmp_view).to(out.dtype).view_as(out)
                )

    if fuse_fp4_quant:
        from aiter.utility.dtypes import fp8_e8m0

        out_scale_sorted = out_scale_sorted_flat.view(fp8_e8m0).view(
            padded_rows, padded_cols
        )
        return out, out_scale_sorted

    return out


def flydsl_moe_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    mode: str = "atomic",
    w2_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    sort_block_m: int = 0,
    persist: Optional[bool] = None,
    use_async_copy: bool = False,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    mfma_variant: Optional[str] = None,
    zero_intermediate: bool = True,
    k_batch: int = 1,
    persist_m: int = 1,
    remap: Optional[str] = None,
    splitk_axis: Optional[str] = None,
    x_nt: Optional[int] = None,
    scale_nt: Optional[int] = None,
    out_nt: Optional[int] = None,
    b_pool_depth: int = 0,
    x_pool_depth: int = 0,
    persist_n: int = 0,
    a2_compact: bool = False,
    blk_valid_start: Optional[torch.Tensor] = None,
    streamk: bool = False,
    streamk_num_wg: int = 0,
    streamk_mode: Optional[str] = None,
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2). Supports atomic/reduce modes.

    When k_batch>1 (split-K), the K=inter_dim contraction is partitioned across
    k_batch workgroups (grid.z) that atomic-accumulate their partials into the
    output. Requires atomic mode (mode != "reduce").

    a: (token_num, topk, inter_dim), w1: (E, model_dim, inter_dim) pre-shuffled.
    Returns (token_num, model_dim).

    sort_block_m: block_size used by moe_sorting / stage1. When 0 (default),
        assumed equal to tile_m. When set, stage2 can use a different tile_m
        from sorting/stage1.
    persist: if True, use persistent round-robin mode (grid_y=cu_num);
        if False, use legacy persist_m mode; if None, auto-select.
    """

    E = w2.shape[0]
    model_dim = w2.shape[1]
    if a2_compact:
        # compacted a2 is 2-D [total_valid, inter_dim]; the real token count comes
        # from `out` ([token, model_dim]) since inter_states.shape[0]=total_valid.
        assert out is not None, "a2_compact requires `out` (source of token_num)"
        token_num = out.shape[0]
        inter_dim = inter_states.shape[-1]
    else:
        token_num = inter_states.shape[0]
        inter_dim = inter_states.shape[2]

    accumulate = mode != "reduce"

    if a_dtype == "fp4":
        inter_dim = inter_dim * 2

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    if out is None:
        alloc_fn = torch.zeros if accumulate else torch.empty
        out = alloc_fn(
            (token_num, model_dim), dtype=torch_out_dtype, device=inter_states.device
        )

    dev = inter_states.device
    # streamk: precompute per-WG unit ranges on-device (schedule kernel) and carry
    # them in the arg_blk_valid_start slot (streamk is mutually exclusive with
    # a2_compact). num_wg defaults to num_cu when unset. n_tiles=model_dim/tile_n,
    # k_tiles=inter_dim/tile_k (same as the in-kernel unit decode).
    _sk_num_wg = 0
    _sk_mode = (
        streamk_mode
        if streamk_mode is not None
        else os.environ.get("AITER_MOE_STREAMK_MODE", "mfocus")
    ).lower()
    _sk_wg_start = None
    if streamk:
        from .kernels.streamk_schedule import (
            streamk_default_num_wg,
            flydsl_streamk_schedule,
        )
        _sk_num_wg = (
            int(streamk_num_wg)
            if streamk_num_wg and int(streamk_num_wg) > 0
            else streamk_default_num_wg()
        )
        # lockstep computes its (m_block, n_tile) dispatch IN-KERNEL from
        # blockIdx.x + num_valid (no schedule kernel).  flat / affine need the
        # even-split wg_unit_start boundaries: run the tiny schedule kernel with
        # k_tiles=1 so each unit is one full-K output tile (matches the in-kernel
        # decode u -> (n_tile=u%n_tiles, m_block=u//n_tiles)); carry it in the
        # arg_blk_valid_start kernarg slot.
        if _sk_mode in ("flat", "affine"):
            _sk_n_tiles = model_dim // tile_n
            _sk_wg_start = flydsl_streamk_schedule(
                num_valid_ids, _sk_num_wg, tile_m, _sk_n_tiles, 1
            )
    flat_a_scale = _expand_per_tensor_scale(a2_scale, token_num * topk, 1)
    if flat_a_scale is None:
        flat_a_scale = torch.empty(0, device=dev)
    flat_w_scale = _expand_per_tensor_scale(w2_scale, E * model_dim, model_dim)
    if flat_w_scale is None:
        flat_w_scale = torch.empty(0, device=dev)
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(sorted_token_ids.shape, dtype=torch.float32, device=dev)
    )

    _sbm = sort_block_m if sort_block_m > 0 else tile_m
    if _sbm == tile_m:
        m_blocks = min(sorted_expert_ids.shape[0], token_num * topk)
    else:
        total_sorted = sorted_expert_ids.shape[0] * _sbm
        m_blocks = (total_sorted + tile_m - 1) // tile_m
    # if persist is True:
    #     _persist_m = -1
    # elif persist is False:
    #     # _persist_m = 4 if m_blocks > 256 else 1
    #     _persist_m = 1  # _persist_m = 1 is better for g1u0
    # else:
    #     _persist_m = -1 if m_blocks > 256 else 1
    _persist_m = persist_m if persist_m and persist_m > 0 else 1

    is_fp4 = b_dtype == "fp4"
    _n_in = model_dim
    _k_in = inter_dim

    target = out
    if not accumulate:
        # Reduce mode writes one row per *valid* (token, slot) into a
        # [token*topk, model_dim] scratch and then sums over topk. When some
        # slots can be left unwritten (EP: slots routed to non-local experts;
        # or variable-length dispatch) the scratch must be zero-initialized,
        # otherwise those uninitialized rows corrupt the sum. When every slot is
        # written (dense, non-EP) the GEMM fully overwrites the scratch, so the
        # (large) zeroing can be skipped via zero_intermediate=False.
        alloc_reduce = torch.zeros if zero_intermediate else torch.empty
        target = alloc_reduce(
            (token_num * topk * model_dim,),
            device=out.device,
            dtype=out.dtype,
        )

    if is_fp4:
        args = _s2_args_fp4(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
            dev,
        )
    else:
        # arg_blk_valid_start is a trailing kernarg on every stage2 kernel (ABI6).
        # Used by a2_compact, and by streamk flat/affine (carries wg_unit_start);
        # lockstep streamk computes its dispatch in-kernel so it leaves this unused.
        # Pass a 1-elem dummy when none of those apply.
        _bvs = _sk_wg_start if _sk_wg_start is not None else blk_valid_start
        if _bvs is None:
            _bvs = torch.zeros(1, dtype=torch.int32, device=inter_states.device)
        args = _s2_args_std(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
            _bvs,
        )

    exe = compile_flydsl_moe_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        persist_m=_persist_m,
        sort_block_m=sort_block_m,
        use_async_copy=use_async_copy,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        mfma_variant=mfma_variant,
        k_batch=k_batch,
        remap=remap,
        splitk_axis=splitk_axis,
        x_nt=x_nt,
        scale_nt=scale_nt,
        out_nt=out_nt,
        b_pool_depth=b_pool_depth,
        x_pool_depth=x_pool_depth,
        persist_n=persist_n,
        a2_compact=a2_compact,
        streamk=streamk,
        streamk_num_wg=_sk_num_wg,
        streamk_mode=_sk_mode,
    )
    _run_compiled(exe, args)

    if not accumulate:
        torch.sum(target.view(token_num, topk, model_dim), dim=1, out=out)

    return out
