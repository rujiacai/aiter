"""统一低比特 MoE 对比（FlyDSL CDNA3 / gfx942）——同一份 raw 数据、同一个 golden、可比。

比较 a16w4 / a8w8 / a8w4(fold) / a8w4_aligned 四种方式（都过同一个 `aiter.fused_moe`）：
- **共享原始数据**：固定 seed 生成 raw bf16 `input/w1/w2 + 路由`，各方法从同源量化。
- **统一 golden**：一个 bf16 全精度 `torch_moe`，各方法都对它比 cos → 直接可比"哪种最准"。
- **子进程隔离**：每个方法在**独立子进程**跑。a8w4 fold 与 aligned 共享 `compile_moe_gemm1`
  的 in-process `lru_cache` key（aligned flag 在核内读 `AITER_A8W4_ALIGNED`，不是 cache 参数），
  同进程顺序跑会复用先编译的二进制 → 结果错。子进程用**同 seed** 确定性重建同一份 raw 数据。
- 方法→env+prep 映射与 aiter-moe-benchmark skill（gfx942 FlyDSL 后端）保持一致。

triton 暂不含（独立实现族，后续单独接入 `aiter/ops/triton/moe`）。

默认 shape = DeepSeek-V4-Pro TP8 (no fuse): model_dim=7168, inter_dim=384, E=384, topk=6。

用法：
    # 聚合（默认 dsv4-pro tp8；自动为每个方法起子进程）:
    PYTHONPATH=/data/aiter python op_tests/test_moe_lowbit_compare.py -t 128 -t 4096 -t 16384
    # 内核正确性口径（复刻 skill，应 ~0.99+）:
    PYTHONPATH=/data/aiter python op_tests/test_moe_lowbit_compare.py --ref dequant -t 4096
    # 覆盖 shape（如 dsv4-flash tp8）:
    PYTHONPATH=/data/aiter python op_tests/test_moe_lowbit_compare.py \
        --model-dim 4096 --inter-dim 256 -E 256 --topk 6 -t 4096
    # 单方法（聚合器内部调用；也可手跑）:
    PYTHONPATH=/data/aiter python op_tests/test_moe_lowbit_compare.py \
        --run-one --method a8w4_aligned -t 4096
"""
import argparse
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, fused_moe, torch_moe
from aiter.test_common import run_perftest

torch.set_default_device("cuda")

# ── 方法注册：name -> (FlyDSL env, 激活精度, prep 种类)。与 skill 的 _FLYDSL_ENV 一致。──
_ALL_FLYDSL_ENV = (
    "AITER_FLYDSL_A16W4",
    "AITER_FLYDSL_A8W4",      # Phase-0 mxfp8 (a8w8)
    "AITER_FLYDSL_A8W4_W4",   # Phase-1 真 4-bit
    "AITER_A8W4_ALIGNED",     # Phase-1 A+B aligned（叠加在 A8W4_W4 上）
)
METHODS = {
    #  name          : (要置 1 的 env,                        prep 种类)
    "a16w4":          (("AITER_FLYDSL_A16W4",),                "a16w4"),
    "a8w8":           (("AITER_FLYDSL_A8W4",),                 "a8w8"),
    "a8w4":           (("AITER_FLYDSL_A8W4_W4",),              "a8w4"),
    "a8w4_aligned":   (("AITER_FLYDSL_A8W4_W4", "AITER_A8W4_ALIGNED"), "a8w4_aligned"),
}
# gfx942 MI308X delivered-TF（与 skill MFU 表一致）。MFU = 达成 TFLOPS / peak。
_PEAK_TF = {"bf16": 203.0, "fp8": 406.0}
_COMPUTE_DTYPE = {"a16w4": "bf16", "a8w8": "fp8", "a8w4": "fp8", "a8w4_aligned": "fp8"}


def _set_method_env(method: str) -> None:
    """只置本方法的 FlyDSL flag，其余清零（子进程内 exactly-one 路径）。"""
    on = set(METHODS[method][0])
    for env in _ALL_FLYDSL_ENV:
        os.environ[env] = "1" if env in on else "0"


def gen_raw(seed, token, model_dim, inter_dim, E, topk, dtype=torch.bfloat16):
    """固定 seed 生成 raw bf16 数据 + 路由（g1u1）。各方法/子进程调用得到完全相同的数据。"""
    torch.manual_seed(seed)
    inp = torch.randn((token, model_dim), dtype=dtype)
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)  # gate+up
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    return inp, w1, w2, topk_weights, topk_ids


def _quant_mxfp4(w, E):
    """raw bf16 权重 -> mxfp4 (fp4x2 码 + E8M0 scale)，per-1x32。"""
    q = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    wq, ws = q(w, quant_dtype=dtypes.fp4x2)
    wq = wq.view(w.shape[0], w.shape[1], w.shape[2] // 2)  # fp4x2: 2 码/字节
    return wq, ws


# e2m1 码本（低 3 位幅值 + bit3 符号）；用于把 mxfp4 反量化回 bf16 做 dequant 参考。
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _mxfp4_dequant_bf16(wq, ws, N, K):
    """mxfp4 (fp4x2 码 + E8M0 scale) -> bf16 权重 (E,N,K)。与 test_flydsl_moe_a16w4 一致。"""
    E = wq.shape[0]
    u = wq.view(torch.uint8)
    lo = (u & 0x0F).long()
    hi = ((u >> 4) & 0x0F).long()
    codes = torch.empty((E, N, K), dtype=torch.long, device=wq.device)
    codes[..., 0::2] = lo
    codes[..., 1::2] = hi
    scale = torch.pow(2.0, ws.view(torch.uint8).view(E, N, K // 32).float() - 127.0)
    table = _E2M1.to(wq.device)
    return (table[codes] * scale.repeat_interleave(32, dim=2)).to(torch.bfloat16)


def _prep_weights(method, w1_qt, w1_scale, w2_qt, w2_scale, E, n1, k1, n2, k2):
    """按方法选 host prep（与 skill _prepare_flydsl_weights 一致）。返回 (w1f,w1sf,w2f,w2sf)。"""
    from aiter.ops.flydsl.moe_kernels import (
        prep_a16w4_weight,
        prep_a16w4_scale,
        prep_a8w4_weight_scale,
        prep_a8w4_w4,
        prep_a8w4_w4_aligned,
    )

    if method == "a16w4":
        return (
            prep_a16w4_weight(w1_qt, n1, k1),
            prep_a16w4_scale(w1_scale, n1, k1),
            prep_a16w4_weight(w2_qt, n2, k2),
            prep_a16w4_scale(w2_scale, n2, k2),
        )
    if method == "a8w8":  # Phase-0 mxfp8：mxfp4 重铸 fp8（权重 8-bit）
        w1f, w1sf = prep_a8w4_weight_scale(w1_qt, w1_scale, E, n1, k1)
        w2f, w2sf = prep_a8w4_weight_scale(w2_qt, w2_scale, E, n2, k2)
        return w1f, w1sf, w2f, w2sf
    # a8w4 / a8w4_aligned：真 4-bit；aligned 用 shuffle_weight_NK(16,32) 布局
    _prep = prep_a8w4_w4_aligned if method == "a8w4_aligned" else prep_a8w4_w4
    w1f, w1sf = _prep(w1_qt, w1_scale, n1, k1)
    w2f, w2sf = _prep(w2_qt, w2_scale, n2, k2)
    return w1f, w1sf, w2f, w2sf


def run_one(method, token, model_dim, inter_dim, E, topk, seed, iters, warmup, ref):
    """单方法：置 env → 同 seed 建 raw → 量化 → golden(按 ref) → prep → fused_moe → cos+latency。

    ref 决定 cos 的口径（两者都用同一份 raw 数据、可比）：
      - "golden"  : 对 **bf16 全精度未量化权重** 的 MoE → 衡量端到端**量化精度**(含 mxfp4 权重损失)。
      - "dequant" : 对 **同一份 mxfp4 权重反量化回 bf16** 的 MoE → 排除权重量化误差,
                    衡量**内核正确性**(复刻 skill 的 torch_moe_reference 口径, 应 ~0.99+)。
    """
    _set_method_env(method)
    inp, w1, w2, tw, ti = gen_raw(seed, token, model_dim, inter_dim, E, topk)

    n1, k1 = inter_dim * 2, model_dim
    n2, k2 = model_dim, inter_dim
    w1_qt, w1_scale = _quant_mxfp4(w1, E)
    w2_qt, w2_scale = _quant_mxfp4(w2, E)

    if ref == "dequant":
        # 用 kernel 实际吃的**同一份量化权重**反量化做参考 → cos 不含权重量化误差。
        w1_g = _mxfp4_dequant_bf16(w1_qt, w1_scale, n1, k1)
        w2_g = _mxfp4_dequant_bf16(w2_qt, w2_scale, n2, k2)
    else:  # golden: 全精度未量化权重
        w1_g, w2_g = w1, w2
    golden = torch_moe(inp, w1_g, w2_g, tw, ti, activation=aiter.ActivationType.Silu)

    w1f, w1sf, w2f, w2sf = _prep_weights(
        method, w1_qt, w1_scale, w2_qt, w2_scale, E, n1, k1, n2, k2
    )

    def _fn():
        return fused_moe(
            inp, w1f, w2f, tw, ti,
            quant_type=aiter.QuantType.per_1x32,
            activation=aiter.ActivationType.Silu,
            w1_scale=w1sf, w2_scale=w2sf,
        )

    out = _fn()
    torch.cuda.synchronize()
    cos = F.cosine_similarity(
        out.float().flatten(), golden.float().flatten(), dim=0
    ).item()

    _, us = run_perftest(_fn, num_iters=iters, num_warmup=warmup)

    flops = 2 * token * topk * model_dim * (2 * inter_dim + inter_dim)  # g1u1 gemm1+gemm2
    tflops = flops / (us * 1e-6) / 1e12
    peak = _PEAK_TF[_COMPUTE_DTYPE[method]]
    mfu = 100.0 * tflops / peak
    w_mb = (w1f.numel() * w1f.element_size() + w2f.numel() * w2f.element_size()) / 1e6
    # 机器可解析的一行（聚合器抓这行）
    print(f"RESULT|{method}|{token}|{cos:.4f}|{us:.1f}|{tflops:.1f}|{mfu:.1f}|{w_mb:.2f}")
    return cos, us


def aggregate(methods, tokens, model_dim, inter_dim, E, topk, seed, iters, warmup, ref):
    """为每个 (method, token) 起独立子进程（同 seed → 同 raw 数据），聚合成一张表。"""
    base = [
        sys.executable, os.path.abspath(__file__), "--run-one",
        "--model-dim", str(model_dim), "--inter-dim", str(inter_dim),
        "-E", str(E), "--topk", str(topk), "--seed", str(seed),
        "--iters", str(iters), "--warmup", str(warmup), "--ref", ref,
    ]
    child_env = os.environ.copy()
    for env in _ALL_FLYDSL_ENV:  # 清掉父进程可能残留的 flag，子进程自己置
        child_env.pop(env, None)

    rows = {}  # (method, token) -> (cos, us, tflops, mfu, w_mb)
    for method in methods:
        for t in tokens:
            cmd = base + ["--method", method, "-t", str(t)]
            print(f"[spawn] {method} token={t} ...", flush=True)
            p = subprocess.run(cmd, env=child_env, capture_output=True, text=True)
            line = next(
                (ln for ln in p.stdout.splitlines() if ln.startswith("RESULT|")), None
            )
            if line is None:
                tail = "\n".join((p.stdout + p.stderr).splitlines()[-6:])
                print(f"  [FAIL] {method} t={t}\n{tail}")
                rows[(method, t)] = None
                continue
            _, m, tk, cos, us, tfl, mfu, wmb = line.split("|")
            rows[(method, t)] = (float(cos), float(us), float(tfl), float(mfu), float(wmb))

    # ── 汇总表 ──
    _ref_desc = "全精度(量化精度)" if ref == "golden" else "dequant(内核正确性, 复刻skill)"
    print(f"\n{'='*78}\ndsv4-like MoE 低比特对比  "
          f"dim=({model_dim},{inter_dim}) E={E} topk={topk} seed={seed}  ref={ref} [{_ref_desc}]\n{'='*78}")
    header = f"{'token':>7} | " + " | ".join(f"{m:>16}" for m in methods)
    for metric, label, fmt in [
        ("cos", f"cos vs {ref}", "{:.4f}"),
        ("us", "latency us", "{:.1f}"),
        ("mfu", "MFU %", "{:.1f}"),
        ("wmb", "weight MB", "{:.1f}"),
    ]:
        idx = {"cos": 0, "us": 1, "tflops": 2, "mfu": 3, "wmb": 4}[metric]
        print(f"\n--- {label} ---\n{header}")
        for t in tokens:
            cells = []
            for m in methods:
                r = rows.get((m, t))
                cells.append("     ERROR      " if r is None else f"{fmt.format(r[idx]):>16}")
            print(f"{t:>7} | " + " | ".join(cells))
    # 相对 a8w4 fold 的加速（若两者都在）
    if "a8w4" in methods and "a8w4_aligned" in methods:
        print("\n--- a8w4_aligned vs a8w4 (fold) 加速 ---")
        for t in tokens:
            rf, ra = rows.get(("a8w4", t)), rows.get(("a8w4_aligned", t))
            if rf and ra:
                print(f"  token={t:>6}: {rf[1] / ra[1]:.2f}x")


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-one", action="store_true", help="内部：跑单方法并打印 RESULT 行")
    p.add_argument("--method", choices=list(METHODS), help="--run-one 时的方法")
    p.add_argument("--methods", nargs="+", default=list(METHODS),
                   help=f"聚合模式跑哪些方法（默认全部）：{list(METHODS)}")
    p.add_argument("-t", "--tokens", type=int, action="append", default=None,
                   help="token 数（可多次）")
    # 默认 = DeepSeek-V4-Pro TP8 (no fuse-shared-expert):
    #   model_dim=7168, inter_dim = full_inter_dim(3072) // tp8 = 384,
    #   routed_experts=384, topk=6, g1u1 Silu.
    p.add_argument("--model-dim", type=int, default=7168)
    p.add_argument("--inter-dim", type=int, default=384)
    p.add_argument("-E", "--experts", type=int, default=384)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--ref", choices=["golden", "dequant"], default="golden",
                   help="cos 参考口径: golden=全精度(量化精度) | dequant=反量化权重(内核正确性, 复刻skill)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    tokens = args.tokens or [256, 4096]
    if args.run_one:
        assert args.method, "--run-one 需要 --method"
        run_one(args.method, tokens[0], args.model_dim, args.inter_dim,
                args.experts, args.topk, args.seed, args.iters, args.warmup, args.ref)
    else:
        aggregate(args.methods, tokens, args.model_dim, args.inter_dim,
                  args.experts, args.topk, args.seed, args.iters, args.warmup, args.ref)
