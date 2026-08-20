#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Export FlyDSL-compiled MoE kernels as standalone AMDGPU code objects.

Each configuration is compiled into an isolated FlyDSL cache, the ELF is lifted
out of the cached artifact's ``gpu.binary``, and it is written next to the
hand-written asm kernels under ``hsa/{arch}/`` together with a manifest CSV.

Nothing in the output references FlyDSL. The ``.co`` is an ordinary code object
that ``AiterAsmKernel`` can ``hipModuleLoadData``, and the manifest carries the
launch metadata (kernel symbol, kernarg layout, LDS, workgroup size) read back
out of the compiled binary rather than assumed by hand.

Adding a shape means adding a row to the spec; the kernels are untouched.

    # default spec, current GPU's arch
    python hsa/flydsl_export.py

    # explicit shapes, dry run to see the matrix first
    python hsa/flydsl_export.py --shape 6144,256 --shape 7168,512 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import pathlib
import re
import shutil
import sys
import tempfile

# The cache dir has to be redirected before aiter/flydsl import so the compiles
# below cannot collide with (or be short-circuited by) a developer's own cache.
_TMP_CACHE = tempfile.mkdtemp(prefix="flydsl_export_")
os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = _TMP_CACHE
os.environ.setdefault("AITER_AOT_IMPORT", "1")

REPO = pathlib.Path(__file__).resolve().parent.parent

# (model_dim, inter_dim, expert, topk). All four are baked into the code object,
# so a new model shape needs a new entry here until they become runtime kernel
# arguments. Tiles are NOT listed: they are derived per token bucket from
# moe_blk.tiles_for, the same function the runtime dispatch uses.
DEFAULT_SHAPES = [
    (6144, 256, 256, 8),
    (6144, 2048, 16, 8),
    (6144, 256, 257, 9),
    (6144, 2048, 17, 9),
]
# Every token the tuned table has a row for, plus two representatives of the
# heuristic fallback that covers the untuned tail (it only splits on token<2048).
DEFAULT_TOKEN_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048]

MANIFEST_FIELDS = [
    "stage", "model_dim", "inter_dim", "expert", "topk",
    "tile_m", "tile_n", "tile_k", "waves_per_eu", "out_dtype", "smooth_scale",
    "arch", "co_name", "kernel_name",
    "kernarg_size", "lds_bytes", "workgroup_size", "vgpr_count", "sgpr_count",
]  # fmt: skip


def mlir_unescape(s: str) -> bytes:
    """Decode an MLIR string literal. ``\\XX`` is hex there, not octal."""
    out, i, n = bytearray(), 0, len(s)
    simple = {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}
    while i < n:
        if s[i] != "\\":
            out.append(ord(s[i]))
            i += 1
            continue
        pair = s[i + 1 : i + 3]
        if len(pair) == 2 and all(c in "0123456789abcdefABCDEF" for c in pair):
            out.append(int(pair, 16))
            i += 3
        else:
            out.append(ord(simple.get(s[i + 1], s[i + 1])))
            i += 2
    return bytes(out)


def parse_artifact(ir_text: str) -> dict:
    """Pull the code object plus its launch metadata out of a compiled artifact."""
    blob = re.search(r'"((?:[^"\\]|\\.){512,})"', ir_text, re.S)
    if blob is None:
        raise RuntimeError("no embedded binary found in gpu.binary")
    elf = mlir_unescape(blob.group(1))
    if elf[:4] != b"\x7fELF":
        raise RuntimeError(f"extracted blob is not an ELF (magic {elf[:4]!r})")

    head = ir_text[: ir_text.find("metadata =") + 600]
    name = re.search(r'#gpu\.kernel_metadata<"([^"]+)"', head)
    sig = re.search(r"!llvm\.func<void \(([^)]*)\)>", head)
    arch = re.search(r'#rocdl\.target<chip = "([^"]+)"', head)
    if not (name and sig and arch):
        raise RuntimeError("could not parse kernel metadata")

    def meta(key, default=0):
        m = re.search(rf"\b{key} = (-?\d+)", head)
        return int(m.group(1)) if m else default

    args = [a.strip() for a in sig.group(1).split(",") if a.strip()]
    # ptr<1> is 8 bytes and i32 is 4; the kernarg segment is their sum, which is
    # exactly what the launcher has to pack.
    kernarg = sum(8 if a.startswith("ptr") else 4 for a in args)
    return {
        "elf": elf,
        "kernel_name": name.group(1),
        "arch": arch.group(1),
        "arg_types": args,
        "kernarg_size": kernarg,
        "lds_bytes": meta("group_segment_fixed_size"),
        "workgroup_size": meta("max_flat_workgroup_size"),
        "vgpr_count": meta("vgpr_count"),
        "sgpr_count": meta("sgpr_count"),
    }


def compile_config(stage: int, cfg: dict, seen: set[pathlib.Path]) -> dict:
    """Compile one config and return the parsed artifact for the new cache entry.

    Goes through the AOT precompile helper rather than ``compile_flydsl_moe_*``
    directly: those only build the jit object, and MLIR compilation (hence the
    cache write we read back) does not happen until the kernel is invoked with
    real arguments.
    """
    import pickle

    from aiter.aot.flydsl.common import compile_only_env
    from aiter.aot.flydsl.moe import _precompile_to_cache

    with compile_only_env():
        _precompile_to_cache(
            stage=stage,
            model_dim=cfg["model_dim"],
            inter_dim=cfg["inter_dim"],
            experts=cfg["expert"],
            topk=cfg["topk"],
            tile_m=cfg["tile_m"],
            tile_n=cfg["tile_n"],
            tile_k=cfg["tile_k"],
            a_dtype="fp8",
            b_dtype="fp8blk",
            out_dtype=cfg["out_dtype"],
            act="silu",
            waves_per_eu=cfg["waves_per_eu"],
            swiglu_limit=cfg.get("swiglu_limit"),
            enable_smooth_scale=bool(cfg["smooth_scale"]),
        )

    new = [p for p in pathlib.Path(_TMP_CACHE).rglob("*.pkl") if p not in seen]
    if len(new) != 1:
        raise RuntimeError(f"expected exactly 1 new cache artifact, got {len(new)}")
    seen.add(new[0])
    with open(new[0], "rb") as fh:
        return parse_artifact(pickle.load(fh)._ir_text)


def co_filename(stage: int, cfg: dict) -> str:
    """Delegate to the runtime's naming so the two can never disagree."""
    from aiter.ops.moe_blk import co_name

    return co_name(
        stage,
        cfg["model_dim"],
        cfg["inter_dim"],
        cfg["expert"],
        cfg["topk"],
        cfg["tile_m"],
        cfg["tile_n"],
        cfg["tile_k"],
        cfg["waves_per_eu"],
        cfg["out_dtype"],
        bool(cfg["smooth_scale"]),
    )


def build_specs(args) -> list[tuple[int, dict]]:
    """One entry per code object the dispatch can ask for.

    Driven by the same tiles_for the runtime uses, evaluated over every tuned
    token so the exported set is exactly what can be requested. Distinct tokens
    that resolve to the same tile collapse into one binary.
    """
    from aiter.ops.moe_blk import tiles_for

    shapes = args.shape or DEFAULT_SHAPES
    seen, specs = set(), []
    for (md, idim, e, k), token in itertools.product(shapes, args.token_bucket):
        for stage, (tm, tn, tk, w) in enumerate(
            tiles_for(token, md, idim, e, k), start=1
        ):
            for smooth in args.smooth:
                # stage2 has no activation, so smooth_scale is stage1-only.
                if stage == 2 and smooth:
                    continue
                cfg = {
                    "model_dim": md, "inter_dim": idim, "expert": e, "topk": k,
                    "tile_m": tm, "tile_n": tn, "tile_k": tk,
                    "waves_per_eu": args.waves if args.waves is not None else w,
                    "out_dtype": args.out_dtype, "smooth_scale": int(smooth),
                }  # fmt: skip
                key = (stage, *cfg.values())
                if key in seen:
                    continue
                seen.add(key)
                specs.append((stage, cfg))
    return specs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape", action="append", type=lambda s: tuple(int(x) for x in s.split(",")),
        metavar="MODEL_DIM,INTER_DIM,EXPERT,TOPK",
        help="repeatable; defaults to the four shipped model shapes",
    )  # fmt: skip
    p.add_argument(
        "--token-bucket", type=int, nargs="+", default=DEFAULT_TOKEN_BUCKETS,
        help="representative token counts; tiles are derived from these",
    )  # fmt: skip
    p.add_argument("--waves", type=int, default=None, help="default: moe_blk.WAVES_PER_EU")
    p.add_argument("--smooth", type=int, nargs="+", default=[0, 1])
    p.add_argument("--out-dtype", default="bf16")
    p.add_argument("-o", "--outdir", default=None, help="default hsa/{arch}/moe_blk")
    p.add_argument("--dry-run", action="store_true", help="list the matrix, compile nothing")
    args = p.parse_args()

    specs = build_specs(args)
    print(f"{len(specs)} configs to export")
    if args.dry_run:
        for stage, cfg in specs:
            print(f"  stage{stage}  {co_filename(stage, cfg)}")
        return 0

    seen: set[pathlib.Path] = set()
    rows, failed = [], []
    outdir = None
    for i, (stage, cfg) in enumerate(specs, 1):
        name = co_filename(stage, cfg)
        try:
            art = compile_config(stage, cfg, seen)
        except Exception as exc:  # keep going; one bad tile should not stop a release
            print(f"  [{i}/{len(specs)}] FAIL {name}: {exc}")
            failed.append((name, str(exc)))
            continue
        if outdir is None:
            outdir = pathlib.Path(
                args.outdir or (REPO / "hsa" / art["arch"] / "moe_blk")
            )
            outdir.mkdir(parents=True, exist_ok=True)
        (outdir / name).write_bytes(art["elf"])
        rows.append(
            {
                "stage": stage, **cfg, "arch": art["arch"], "co_name": name,
                "kernel_name": art["kernel_name"],
                "kernarg_size": art["kernarg_size"], "lds_bytes": art["lds_bytes"],
                "workgroup_size": art["workgroup_size"],
                "vgpr_count": art["vgpr_count"], "sgpr_count": art["sgpr_count"],
            }  # fmt: skip
        )
        print(
            f"  [{i}/{len(specs)}] {name}  {len(art['elf']):>6}B  "
            f"kernarg={art['kernarg_size']} lds={art['lds_bytes']} "
            f"vgpr={art['vgpr_count']}"
        )

    if rows:
        manifest = outdir / "manifest.csv"
        with open(manifest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(rows)
        total = sum(len(r) for r in [b"" for b in rows]) or sum(
            (outdir / r["co_name"]).stat().st_size for r in rows
        )
        print(f"\nwrote {len(rows)} .co ({total / 1024:.0f} KiB) + {manifest}")
    if failed:
        print(f"{len(failed)} failed:")
        for n, e in failed:
            print(f"  {n}: {e[:100]}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_TMP_CACHE, ignore_errors=True)
