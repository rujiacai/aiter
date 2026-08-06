# FlyDSL a8w4 MoE 性能优化记录（MI308X / gfx942）

针对 a8w4 Phase-1 路径（fp8 激活 × 4-bit 存储的 mxfp4 权重，e2m1→fp8 在 kernel 内解包）的一轮端到端优化。
e2e 几何平均 **1.42×**，小 batch 最高 **2.16×**，精度不变。

- 复现命令：`AITER_A8W4_ALIGNED=1 python aiter/ops/flydsl/test_flydsl_moe_a8w4.py --sweep --full`
- 形状：`model_dim=7168, inter_dim=384, E=384, topk=6`
- 硬件：MI308X（gfx942，80 CU，192GB HBM3），ROCm 7.x

`--sweep` 计时口径为 device kernel time（`run_perftest` 的 torch.profiler 路径），不含 host 侧
Python 开销；a8w4 的时间**包含**两次 HIP fp8 量化。

---

## 1. 结果

同一台机器、同一次会话下的 A/B（baseline 用 `git stash` 回退到改动前重跑）：

| token | baseline (us) | 优化后 (us) | 加速 |
|------:|-------------:|-----------:|-----:|
| 1     | 105.7  | 48.9    | 2.16× |
| 2     | 109.9  | 52.8    | 2.08× |
| 4     | 125.0  | 71.5    | 1.75× |
| 8     | 192.5  | 113.0   | 1.70× |
| 16    | 346.3  | 199.6   | 1.73× |
| 32    | 521.7  | 320.8   | 1.63× |
| 64    | 842.3  | 592.2   | 1.42× |
| 128   | 1141.1 | 825.5   | 1.38× |
| 256   | 1279.6 | 917.9   | 1.39× |
| 512   | 1332.7 | 940.5   | 1.42× |
| 1024  | 1394.8 | 1263.3  | 1.10× |
| 2048  | 1969.3 | 1813.6  | 1.09× |
| 4096  | 3173.3 | 2917.1  | 1.09× |
| 8192  | 5779.6 | 5150.0  | 1.12× |
| 16384 | 11131.0| 9894.4  | 1.12× |
| 32768 | 21764.5| 19345.6 | 1.13× |

几何平均 1.42×（token ≤ 512 为 1.65×，token ≥ 1024 为 1.11×）。

精度未变：全部 token 上 `a8_cos`（对 bf16 全精度 golden）仍为 0.9986–0.9991。
对 triton a16w4 的领先从 0.73–1.49× 提升到 1.59–2.49×。

---

## 2. 瓶颈定位

### 2.1 先分项，别猜

把 `a8_e2e` 拆成四段单独计时（`opt_logs/bench_parts.py --mode breakdown`），baseline 配置
（tile = 32×128×128）：

| token | quant1 | stage1 | quant2 | stage2 |
|------:|-------:|-------:|-------:|-------:|
| 1     | 2.2  | 81.2   | 2.2   | 13.3   |
| 64    | 3.2  | 532.6  | 2.6   | 309.2  |
| 1024  | 10.2 | 837.2  | 10.8  | 524.7  |
| 16384 | 126.0| 5902.1 | 227.2 | 4857.4 |

结论：两次 fp8 量化合计只占约 3%，**stage1 约 60%、stage2 约 35%**，优化目标明确。

### 2.2 硬件计数器

用 `rocprofv3` 对 stage1 单独采样（`opt_logs/prof_s1.py`）：

```
moe_gemm1_0  vgpr=104 agpr=0 lds=8192
  SQ_INSTS_VALU             11,102,812
  SQ_INSTS_MFMA              1,271,424      -> VALU / MFMA = 8.73
  SQ_VALU_MFMA_BUSY_CYCLES  20,342,784      -> / MFMA = 16.0
  SQ_WAIT_ANY               14,610,389
```

每条 MFMA 要配 **8.7 条 VALU** 指令，这是第一个危险信号。

> 关于 `SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA = 16`：这个计数器的单位口径有歧义（按数据手册
> 的 dense fp8 峰值反推，一条 `v_mfma_f32_16x16x32_fp8_fp8` 应为 32 cycle）。因此下面**没有**单靠
> cycle 模型下结论，而是用 2.3 的对照实验来定性。

### 2.3 决定性的对照实验：mxfp4 vs mxfp8

aiter 里同时存在 Phase-0 的 `mxfp8` 路径：权重在 host 侧就已经重铸成 fp8 存储，
**完全没有 in-kernel 解包**，但权重 HBM 字节翻倍。两条路径共用同一套 compute/epilogue。
所以两者的差值就是"解包 VALU vs 双倍带宽"的直接对比（`--mode unpack`，tile = 32×64×128）：

| token | s1 mxfp4 | s1 mxfp8 | s2 mxfp4 | s2 mxfp8 |
|------:|---------:|---------:|---------:|---------:|
| 64    | 464.8  | 501.1  | 353.0  | 411.4  |
| 1024  | 760.2  | 844.5  | 603.3  | 710.6  |
| 4096  | 1691.5 | 1721.5 | 1517.7 | 1730.9 |
| 16384 | 5365.1 | 5305.9 | 5767.4 | 6386.2 |

**两者几乎同速**。这一条同时否定了两个朴素假设：

- 不是纯带宽受限 —— 否则 mxfp8 应该慢一倍（token=16384 时权重流量从 8.45 GB 涨到 16.9 GB）；
- 也不是"解包无所谓" —— 否则去掉解包的 mxfp8 应该明显更快。

真实情况是两者各自撞到不同的墙：**mxfp4 卡在解包 VALU，mxfp8 卡在带宽**，而恰好撞在同一个时间点。
token=16384 时 mxfp4 侧的权重带宽约 1.6 TB/s，远低于 MI308X 的 HBM 峰值，**带宽还有大量余量**，
所以砍 VALU 是有效方向。

### 2.4 反汇编热循环

FlyDSL 可以 dump 最终 ISA（`opt_logs/dump_isa.py`）：

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir> python ...
# -> <dir>/<kernel>/21_final_isa.s
```

baseline（tile = 32×128×128）热循环，一轮 = 2 个 K-tile：

```
valu 477   mfma 64   vmem 53   lds 10   salu 83
  s_nop 43   s_waitcnt 34   scratch_load_dwordx2 3
.vgpr_count: 168   .vgpr_spill_count: 38
```

三个问题一次暴露：

1. **38 个 VGPR spill，其中 3 条 `scratch_load` 直接落在热循环里**；
2. `s_nop 43` —— MFMA 结果被紧接着的下一条 VALU 读取，每次都吃满 MFMA 延迟；
3. `.vgpr_count = 168` 恰好是 `waves_per_eu=3` 的上限（512/3 ≈ 170），说明寄存器分配器是被这个
   上限**逼着 spill** 的。

VALU 构成 —— 以最终配置 `32×64×128` 的热循环为例（一轮 32 条 MFMA / 16 个 B operand，共 237 条
VALU；`32×128×128` 各项等比翻倍，占比相同）：

| 项 | 指令数 | 占比 |
|---|---:|---:|
| e2m1→fp8 解包（perm-LUT，8 条/operand） | 128 | 54% |
| MFMA 后的 per-32 scale FMA（2.4 条/MFMA） | 76 | 32% |
| bf16 scale 提取 | 16 | 7% |
| 地址计算 | 17 | 7% |

---

## 3. 优化项

### 3.1 延迟解包（defer unpack）

**问题**：aligned 路径在 `load_b_tile` 里就把 e2m1 展开成 fp8。展开后每个 K32 operand 是一个 i64
（2 个 VGPR），而打包态只要 1 个 dword。更糟的是流水线里当前 tile 和预取 tile 同时存活，这部分
循环携带状态（`scf.for` 的 `iter_args`）被放大一倍。

**做法**：`iter_args` 里改为携带打包态 `(r0, r1, sc)`，进 `compute_tile` 后再展开。展开位置放在
`(ku, ni)` 层级、`mi` 循环之前 —— 因为 `m_repeat` 条 MFMA 共用同一个 B operand，不能放进 `mi` 里
重复解包。

**效果**（tile = 32×128×128）：

| | spill 总数 | 热循环 scratch | s_nop | s_waitcnt | salu |
|---|---:|---:|---:|---:|---:|
| 改前 | 38 | 3 | 43 | 34 | 83 |
| 改后 | 22 | 0 | 19 | 14 | 39 |

指令总数没变，但停顿显著减少，stage1 提升约 5–8%。

代码：`moe_gemm_2stage.py`，开关 `AITER_A8W4_DEFER_UNPACK`（默认 1）。

### 3.2 scale-FMA 流水化

**问题**：a8w4 的每条 K32 MFMA 都先写进一个零累加器，再用一条 f32 FMA 把 per-32 的 E8M0 scale
乘进去：

```
p  = mfma(a0, b0, zero)
acc = fma(sc0, p, acc)      # 紧接着就读 p
```

紧邻读取 MFMA 结果 ⇒ 全额 MFMA 延迟停顿，就是上面那 43 条 `s_nop`。

**做法**：加一个 FIFO（`_make_scale_fma_pipe`），把 scale-FMA 延后 `depth` 条 MFMA 再回收。
因为是 FIFO，**同一个累加器上的加法顺序不变**，f32 结果与内联版本逐位相同 —— 这一点是这个改动
可以放心做的前提。

`depth=4` 实测最好（16 个 VGPR 的在途 partial 是可以接受的代价）。开关
`AITER_A8W4_FMA_DEPTH`（默认 4，设 0 恢复旧行为）。

### 3.3 tile 形状（贡献最大）

原来 stage1/stage2 都固定 `(32, 128, 128)`。逐项实测后拆成三个独立决策，收进新函数
`moe_kernels.a8w4_tiles()`，由 `fused_moe` 分发路径和 sweep 测试共用。

**(a) stage1 `tile_n`: 128 → 64**

各 tile 配置的寄存器实测：

| tile (m,n,k) | vgpr | spill |
|---|---:|---:|
| (32, 64, 128)  | **102** | **0** |
| (32, 128, 128) | 168 | 22 |
| (64, 64, 128)  | 168 | 86 |
| (64, 64, 64)   | 162 | 0 |
| (128, 64, 64)  | 168 | 144 |

`tile_n=64` 让每个 wave 只有一个 N 累加器（`num_acc_n=1`），102 VGPR 无 spill，占用率 5 waves/SIMD。

**(b) stage2 `tile_n`: 128 → 256**

stage2 没有 gate/up 这一对累加器，寄存器宽裕（tn128 = 82 VGPR，tn256 = 141 VGPR，都无 spill），
反而更喜欢宽 tile —— workgroup 数和 per-WG epilogue 都减半。token=16384 时 4694 → 4432 us。

**(c) `tile_m`: 按每专家平均 token 数在 16/32 之间切换**

stage1 + stage2 合计（us）：

| token | tile_m=16 | tile_m=32 | tile_m=64 |
|------:|----------:|----------:|----------:|
| 1     | **52.7**  | 59.1   | 101.0  |
| 16    | **179.8** | 310.5  | 483.0  |
| 64    | **540.6** | 746.9  | 1292.9 |
| 256   | **854.1** | 1150.8 | 1964.1 |
| 1024  | 1241.0    | 1252.0 | 2041.1 |
| 4096  | 3606.0    | **2898.6** | 3258.0 |
| 16384 | 12141.7   | **10067.9**| 10349.6|
| 32768 | 23583.6   | **19105.6**| 20861.7|

专家没填满时（每专家不足 16 token），32 行的 M-tile 一半以上是 padding，MFMA 白跑；专家填满后，
更大的 tile_m 摊薄权重流量反而更划算。判据取 `tokens_per_expert < 16`（16 正是 MFMA 的 M 粒度）。

**(d) `tile_k` 保持 128**：它能整除所有 128 对齐的形状（256 会丢掉 `inter_dim=384` 这类的 K 尾巴），
且在合法处实测也更慢。

### 3.4 `waves_per_eu = 4`（stage1）

`(32,64,128)` 本来就只用 102 VGPR，要 4 waves/SIMD 不花任何代价：

| token | wpe=3 | wpe=4 |
|------:|------:|------:|
| 64    | 461 | **454** |
| 256   | 731 | **709** |
| 1024  | 766 | **752** |
| 16384 | 5373| **5209** |

写进 `get_flydsl_stage1_kernels_mxfp4_fp8` 的 params（常量
`A8W4_STAGE1_WAVES_PER_EU`），经 `fused_moe.py` 的 `parsed.get("waves_per_eu", 3)` 生效。

---

## 4. 试过但没用的方向

记录负面结果，避免重复踩。

**`tile_m=64` / `128`（提高 `m_repeat` 摊薄解包）** —— 理论上最有吸引力：解包成本是
`8 VALU / m_repeat` per MFMA，`m_repeat` 从 2 到 4 能砍掉约 27% 的 VALU。但实测在**所有** token 下
都更慢。原因是累加器数量随 `tile_m` 线性增长，`m_repeat=4` 时在 `waves_per_eu=3` 的 168 VGPR 上限
下 spill 86 个。放开上限（`wpe=0/1/2`）后 spill 消失但占用率掉到 2–3，我把
`waves_per_eu ∈ {0,1,2,3,4} × FMA depth ∈ {0,2,4,8}` 全组合扫了一遍，`tile_m=64` 最好也只有
5773 us，仍输给 `tile_m=32` 的 5209 us。

**`tile_k=64`** —— 想用更短的 K 步降低 B 的循环携带状态，实测在所有 token 下都不如 128。

**改用 mxfp8 权重（彻底去掉解包）** —— 见 2.3，带宽换 VALU 是零和的。

**在 kernel 内把 scale 折进 perm-LUT** —— 思路是：E8M0 scale 是 2 的幂，把 `2^r` 加到 fp8 的
指数域等价于对 LUT 的每个字节做整数加法，这样多条 MFMA 就能直接链式累加、省掉 per-K32 的 FMA。
逐项算过账：构造移位 LUT 需要约 4–9 条 VALU/operand，而省下的 FMA 只有 `2 × m_repeat` 条，
在 `m_repeat=2` 下净亏。只有 `m_repeat ≥ 8` 才划算，而那个 tile 根本立不住。此外 `r < -7` 时字节加法
会借位、且 code 0（+0.0）必须保持 0x00，正确性代价也不小。

---

## 5. 验证

- `--stage all -t 1 -t 16 -t 64 -t 512 -t 1024 -t 4096`：全部 PASS；
- 全量 sweep 的 `a8_cos` 与 baseline 逐 token 一致（0.9986–0.9991）；
- 换形状 `model_dim=4096, inter_dim=512, E=256, topk=6` 跑通，cos 0.9989–0.9992；
- `test_flydsl_moe_a16w4.py` 回归 PASS（stage1/stage2 与 a8w4 共用同一个 kernel builder）。

---

## 6. 后续方向

按预期收益排序：

1. **scale 张量 bf16 → f32**。现在每次 `extract_bf16_scale` 要 2 条 VALU 从一个 dword 里拆出两个
   bf16，占热循环 VALU 的 7%。改成 f32 存储可以完全省掉，代价是 scale 流量翻倍 —— 而 2.3 已经证明
   带宽有余量。需要改 host prep 和 `_load_groupwise_scale` 的 f32 分支。
2. **重排 aligned 权重布局**，让一条 `buffer_load_dwordx4` 覆盖 4 个连续 K32 block（当前布局下
   相邻 k0 在 lane 内相隔 256B，只能用 `dword` 逐个加载）。可把 B 的访存指令数降到 1/4，同时减少
   地址计算 VALU。需要 host 侧 `shuffle_weight_NK` 配套改。
3. **stage2 的短 K 问题**。`inter_dim=384 / tile_k=128` 只有 3 个 K-tile，prologue/epilogue 摊不开：
   token=16384 时 stage2 的实测吞吐（122 TFLOP/s）只有 stage1（192 TFLOP/s）的三分之二。
   可以考虑为短 K 单独走一条不做 ping-pong 的路径。

---

## 附：分析脚本

均在 `opt_logs/`（未纳入版本控制）：

| 脚本 | 用途 |
|---|---|
| `bench_parts.py --mode breakdown` | e2e 分项计时（quant1 / stage1 / quant2 / stage2） |
| `bench_parts.py --mode sweep1\|sweep2` | 单 stage 的 (tile_m, tile_n, tile_k) 扫描 |
| `bench_parts.py --mode combo` | 联合选 tile_m（两个 stage 共享 moe_sorting 的 block_m） |
| `bench_parts.py --mode unpack` | mxfp4 vs mxfp8 对照 |
| `bench_parts.py --mode wpe` | `waves_per_eu` × tile 扫描 |
| `dump_isa.py --stage 1 --tile-m ...` | dump 最终 ISA 并统计指令构成 / VGPR / spill |
| `prof_s1.py` | 供 `rocprofv3 --pmc` 采样的最小复现 |

注意两点：`dump_isa.py` 必须配 `FLYDSL_RUNTIME_ENABLE_CACHE=0`，否则命中磁盘缓存不会重新编译；
`bench_parts.py` 里由环境变量控制的开关（`AITER_A8W4_FMA_DEPTH` 等）在 import 时读取，切换需要
新起进程。
