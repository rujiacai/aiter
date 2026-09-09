# FlyDSL blockwise-fp8 MoE：stage2 带宽优化记录

日期：2026-09-08
机器：MI308X（gfx942，80 CU，sclk 1850 / mclk 1300，SPX/NPS1，ROCm 7.14）
路径：`AITER_FLYDSL_BLKFP8=1` 的 2stage FlyDSL blockwise fp8（`fp8blk`，128×128 权重 scale + 1×128 激活 scale）

负载：vLLM GLM-5.3 EP16 decode 的真实 dump
（`fmoe_dump_20260908/dump/bs*/bs*_mstep51_dp0_layer*.pt`，见 `op_tests/test_fmoe_vllm_dump.py`）
hidden 6144 / inter 2048 / 256 全局专家 top-8 / EP16 → 每 rank 16 个本地专家。

---

## 0. 结论速览

| 项 | 结果 |
| --- | --- |
| stage2 收益 | bs16 **-20.0%**、bs64 **-25.2%**、bs128 **-20.0%**、bs224 **-20.5%** |
| stage1 收益 | bs16 **-5.7%**、bs64 **-10.2%**、bs128 **-3.5%**、bs224 **-4.0%**（§3.5 的 x-load 钉序 + §3.6 的 NT） |
| e2e 收益 | bs16 **-10.8%**、bs64 **-14.5%**、bs128 **-9.3%**、bs224 **-9.9%** |
| 有效带宽（stage2, bs64） | 2.33 → **2.69 TB/s**（§3.1+§3.2 后的采集点，见 §3.4；§3.6 的 NT 又把 gemm2 压到 66.0 µs，但它同时改变了 L2 命中，不能按固定流量折算） |
| 生效方式 | 默认生效，无需 env |
| 正确性 | 5 个 bs × 5 个 layer，`zero_rows=0`、`relL2≈0.024`（参考实现口径见 §6） |
| 旁证 | 另一个 dump（EP8 / 32 专家）e2e 435.6 → **391.8 µs** |

五处改动：

1. **stage2 `tile_k` 256 → 128**（配置，`aiter/fused_moe.py`）—— 主要收益
2. **A2 scale 的 load 提到 B burst 之前**（kernel，`moe_2stage_blockscale.py`）—— 次要收益，且在 `tile_k=256` 下收益更大
3. **A-tile（x）的 load 用 `sched_barrier` 钉在 burst 头部**（kernel，两个 stage 各一行）—— 见 §3.5，唯一同时改善 stage1 的一项
4. **stage1 的 W 用 non-temporal 加载，按 token 数开关**（配置 + kernel）—— 见 §3.6，小 bs 上 e2e -6~8%，大 bs 上必须关掉
5. **共享同一 16B kpack 的两条 B load 合并成一条**（kernel，`mfma_preshuffle_pipeline.py`）—— 见 §3.7，省掉一条只命中 L1 的冗余 `dwordx4`

---

## 1. 起点：为什么怀疑指令而不是流量

bs64/layer3 的分项（device avg，`AITER_LOG_MORE=1`）：

| kernel | us | 占比 |
| --- | ---: | ---: |
| moe_gemm1 | 152.3 | 59% |
| moe_gemm2 | 86.2 | 34% |
| a2 quant | 5.5 | 2% |
| sorting ×2 | 12.3 | 5% |

两个 GEMM 都是**按专家整片流式读权重**，激活/输出/scale 合计不到 3 MB（0.3%）：

- stage1 = 17 blocks × 25.17 MB(W1) = 427.8 MB
- stage2 = 17 blocks × 12.58 MB(W2) = 213.9 MB

（17 而不是 16：`tile_m=16` 下最热专家 27 个 token 要占 2 个 M-block，它的 W 被读两遍。）

按 4 TB/s 折算，stage1 地板 106.9 µs（剩 45.4）、stage2 地板 53.5 µs（剩 32.7）。
stage2 离上限更远（62% vs 70%），且 HBM 事务已经 99.99% 是 128B —— 流量侧没有可压缩的，
所以问题在**发射与等待**，需要指令级证据。

## 2. ATT 定位：一条 `s_waitcnt` 吃掉 10%

采集（注意 `FORCE_GRID_Y`，否则指标会被 22 倍的空 WG 稀释，见 §5）：

```bash
FORCE_GRID_Y=17 rocprofv3 --att --att-target-cu 1 \
  --att-library-path /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib \
  --kernel-include-regex moe_gemm2 --kernel-trace -d <out> -o att \
  -- python /tmp/replay_vllm_dump.py
```

按指令家族聚合：

| 家族 | latency | stall |
| --- | ---: | ---: |
| `buffer_load` | 43.2% | 48.6% |
| `s_waitcnt` | 28.3% | 33.6% |
| VALU | 16.0% | 8.7% |
| `s_barrier` | 4.6% | 5.3% |
| `v_mfma` | 4.1% | 1.9% |
| `global_atomic_pk_add_bf16` | 0.5% | 0.5% |

`buffer_load + s_waitcnt` = **71.5% latency / 82% stall**。atomic epilogue 只有 0.5%（此前
的猜测「atomic 是 stage2 的短板」被否掉）。

热循环的实际调度：

```
0x0230c  buffer_load_dwordx4 v[136:139], v84, s[36:39]          ┐ 16 条 W2（16 KB/wave）
   ...                                                           │
0x0238c  buffer_load_dwordx2 v[212:213], v228, s[28:31] off:56   │  4 条 A2-scale
   ...                                                           ┘  共 21 条背靠背发出
0x023b4  s_waitcnt vmcnt(3)    lat=87228 hit=60 → 1453 cyc/次   <<< 单条占全 kernel 10%
0x023b8  v_cndmask_b32_e32 v78, 0, v212, vcc    ← select(ok, scale, 0)
```

`vmcnt(3)` 要等 21 条里的 **18 条**回来才放行，等于把双缓冲作废：wave 刚把下一拍的 load
发出去就在原地等它们，等待期间不再产生任何新的内存请求。循环里其他 wait 是正常的
（`vmcnt(21)/(20)/(19)`，允许刚发的继续在飞）。

**为什么它必须等这么紧**：wait 之后第一条指令消费的 `v212` 正是 burst **最末尾**那条
A2-scale load 的目标寄存器 —— `_a_blk_scales()` 里的 `select(a_blk_scale_ok, scale, 0.0)`。
scale 排在最后发，却最先被用，顺序反了。

**A2-scale 的加载方式本身也浪费**：地址来自 `row_base_blk = bx_m + lane_div_16*4`，一个 wave
里只有 4 个不同的 `lane_div_16`，配上 `ii` 的 4 个取值，真正需要的只有 **16 个 f32（64 B）**，
却发了 `4 × 64 lanes × 8B = 2048 B` 的请求 —— **32 倍冗余**。这解释了 coalescing 只有 28.5%
（gemm1 是 39%）。它不增加 HBM 流量（92% 被 vL1D 吸收，HBM 读仍是 205 MB ≈ 纯权重），但占
VMEM 发射槽并制造了那条 1453 cycle 的强依赖。

配套的 SOL 指标（同一次采集）：

| 指标 | 值 |
| --- | --- |
| Wavefront occupancy | 431.6 / 2560 = **16.9%** |
| VGPR + AGPR | 112 + 128 = **240 / 512 → 2 waves/SIMD** |
| VMEM latency | 987 cycles |
| Wave cycles 拆分 | Active **11.5%** / Dependency wait 39.7% / Issue wait 48.8% |

只有 2 个 wave/SIMD，一个 wave 陷进 1453 cycle 的 drain 时只剩 1 个顶着，HBM 立刻饿。

## 3. 改动与归因

### 3.1 stage2 `tile_k` 256 → 128（主要收益）

`aiter/fused_moe.py`，`inter_dim % 256 == 0` 时取 128。

`tile_k` 减半把每 wave 的 VMEM burst 从 16 条 dwordx4 砍到 8 条，
`kblk_per_tile` 从 2 变 1（A2-scale 从 8 条降到 4 条），
并且 B tile 的寄存器占用减半 —— 见 §3.3 的 VGPR 112 → 32。

### 3.2 A2-scale 提到 B burst 之前（次要收益）

`moe_2stage_blockscale.py`：把 `_a_blk_scales()` 从 `compute_tile()` 内部提出来，变成
`load_a_scales(base_k)`，在循环体里**先于**下一拍的 `load_b_tile()` 调用，通过新增的
`a_scales=` 参数传进 `compute_tile()`（`a_scales=None` 时回退到原地加载，保证其它调用点安全）。

> 试过但**更慢**的做法：把 A2-scale 像 B tile 一样放进 scf.for 的 loop-carried state
> 做整拍预取。bs64/tile_k=128 下 77.3 → 88.0 µs —— 循环携带增加的寄存器压力和边界
> copy 超过了预取收益。最终采用的是「同一拍内提前发射」这个更轻的版本。

### 3.3 归因矩阵（gemm2 device avg µs，layer3）

用独立的 `FLYDSL_RUNTIME_CACHE_DIR` 逐格重编，避免同名 kernel 命中旧二进制：

| bs | 原始 | 仅 tile_k=128 | 仅 scale 前置 | **两者** | 总收益 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 60.1 | 55.9 | 56.5 | **54.5** | -9.3% |
| 64 | 88.2 | 75.9 | 81.7 | **76.5** | -13.3% |
| 128 | 114.2 | 96.4 | 105.1 | **95.2** | -16.6% |
| 224 | 163.7 | 136.5 | 151.4 | **134.7** | -17.7% |

- `tile_k=128` 单独：-7.0% / -13.9% / -15.6% / -16.6%（主力）
- scale 前置单独：-6.0% / -7.4% / -8.0% / -7.5%（稳定）
- 两者**不完全叠加**：`tile_k=128` 已经把 burst 砍半、scale 降到 4 条，drain 本来就浅了，
  此时 scale 前置只再贡献 0~1.8%。保留它的理由是：`inter_dim % 256 != 0` 的 shape 仍会
  落到 `tile_k=256`，那条路径上它值 7~8%。

### 3.5 A-tile（x）load 钉序：唯一对 stage1 也有效的一项

优化 3.1/3.2 之后重采 ATT，两个 stage 都剩一条同源的大项：

| | 指令 | cyc/次 | 占该 kernel ATT latency |
| --- | --- | ---: | ---: |
| stage1 | `s_waitcnt vmcnt(0)` | 1185 | **17.4%** |
| stage2 | `s_waitcnt vmcnt(2)` | 839 | **10.5%** |

两条 wait 的下一条指令都是 `ds_write_b64`，也就是 `store_x_tile_to_lds`。反汇编里
A-tile 的那条 `buffer_load_dwordx2` 被排到了 burst 的**最后**（stage1）或倒数第三
（stage2）：

```text
0x01e74  buffer_load_dwordx2  s[28:31]   ← A tile，burst 最后一条
0x01e80  s_waitcnt vmcnt(0)   lat=436220 hit=368   ← 后面没 load 了，只能全排空
0x01e84  ds_write_b64                    ← store_x_tile_to_lds
```

Python 里的程序顺序是 `load_x_tile()` 在 `load_b_tile()` **之前**的，是 LLVM 把它下沉了：
x 的唯一消费者就是 `store_x_tile_to_lds`，把 load 挪到 use 附近能缩短活跃区间、省 2 个
VGPR。局部正确，全局灾难 —— 因为 `vmcnt` 是**按发射序退休的单一计数器**，要等最新那条
load 就只能连同它之前的全部一起排空，把刚发出去的 B 预取全丢掉。

修法是在两个 `load_x_tile()` 末尾各加一行调度区边界：

```python
_BLK_XPIN_MASK = 0x02 | 0x04 | 0x08 | 0x80  # VALU | SALU | MFMA | DS
...
rocdl.sched_barrier(_BLK_XPIN_MASK)
```

mask 的语义是「**仍然允许**跨越的指令类别」，所以这样只钉住 VMEM，VALU/SALU/MFMA/DS
的重排自由度完全保留。三个变体实测（bs64）：

| 变体 | gemm1 | gemm2 |
| --- | ---: | ---: |
| 不加 | 152.1 | 74.2 |
| `sched_barrier(0)`（全挡） | 155.2 | **86.7** |
| `sched_barrier(0x8E)`（只挡 VMEM） | **145.8** | **72.3** |

全挡把 gemm2 打回优化前水平 —— 它连 MFMA/LDS 的交错一起挡了。这个对照反过来也确认了
根因就是**发射位置**本身。

跨 bs 一致（e2e）：bs16 181.5→172.3、bs64 246.0→239.0、bs128 292.9→284.0、
bs224 412.9→399.3。

### 3.4 改动后的 profile 对照（bs64，stage2）

| 指标 | 改动前 | 改动后 |
| --- | ---: | ---: |
| VGPR | 112 | **32** |
| AGPR | 128 | 128 |
| 总寄存器 | 240（2 waves/SIMD） | **160（3.2 waves/SIMD）** |
| Wavefront occupancy | 16.9% | **23.4%** |
| vL1D coalescing | 28.5% | **39.1%** |
| vL1D 总请求 | 22.07M | 19.21M |
| MFMA util | 12.1% | 15.2% |
| Wave cycles | 106.1M | **81.3M** |
| Active cycles 占比 | 11.5% | 14.4% |
| HBM 读请求（128B） | 1,602,695 | 1,606,280（流量不变 ✓） |

ATT 里 `s_waitcnt vmcnt(3)` **消失**，最坏 wait 从 1453 → 597 cyc/次。

有效带宽：205.6 MB / 76.3 µs = **2.69 TB/s**（改动前 205.6 / 88.2 = 2.33），占 4 TB/s 的 67%。

### 3.6 stage1 的 W 用 non-temporal 加载（按 token 数开关）

W 的 `buffer_load` 带上 `sc0/sc1/nt` 之后不在 L2 留驻。这只在**每个专家的 W 大约只被读一遍**
时是净收益；一旦 `tile_m=16` 让热专家跨多个 M-block，它的 W1 会被重读，此时告诉 cache 丢掉
这些行等于扔掉真实的复用。实测每专家 M-block 数：

| rows | 64 | 256 | 512 | 768 | 896 |
| --- | ---: | ---: | ---: | ---: | ---: |
| blocks/expert | 1.00x | 1.06x | **1.50x** | 2.00x | 2.25x |

所以策略是按 token 数（= `hidden_states.shape[0]`，EP 下是容量行数）设阈值，
`aiter/fused_moe.py` 里 `_bnt1 = 2 if token <= 256 else 0`，只作用于 stage1。

全程 NT vs 全程 cached（layer3，device avg µs）：

| bs | rows | gemm1 cached | gemm1 NT | gemm2 cached | gemm2 NT | e2e cached | e2e NT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 102.5 | **98.4** | 51.6 | **47.5** | 172.2 | **161.6** |
| 64 | 256 | 145.5 | **136.3** | 74.6 | **66.3** | 241.0 | **221.5** |
| 128 | 512 | **169.6** | 176.6 | 91.5 | **83.2** | 283.8 | 282.9 |
| 224 | 896 | **240.5** | 269.2 | 130.2 | **121.5** | **398.8** | 418.9 |

gemm1 的走向和 blocks/expert 完全对应：1.0~1.06x 时 NT 赢 4~6%，1.5x 起反转（+4.1%），
2.25x 时亏 11.9%。阈值 256 正好落在反转点前。

值得单独记一笔的是 **gemm2 也动了 7~11%，而 `_bnt1` 根本没传给 stage2**（stage2 的 partial
不设 `b_nt`，恒为 cached）。这是**跨 kernel 的 L2 效应**：stage1 走 NT 就不把 W1 留在 L2，
stage2 于是拿到一个更干净的缓存。bs128 上 gemm2 省的 8.3 µs 几乎正好抵掉 gemm1 亏的 7.0 µs，
这就是该点 e2e basically 打平（282.9 vs 283.8）的原因 —— 阈值放在 256 还是 512 在 bs128 上
是硬币，但 bs224 上必须是 0，否则亏 20 µs。

> 推论（未实施）：既然 stage2 自己用 NT 也可能有收益，而它现在恒为 cached，
> 给 stage2 单独接一个 `b_nt` 策略是一个待验证的方向。

### 3.7 共享 kpack 的两条 B load 合并

`kpack_bytes=16` 时，`load_b_pack_k32(ki)` 和 `load_b_pack_k32(ki+1)` 的地址都由 `ki//2`
算出，**是同一个 16B kpack** —— 也就是两条 `buffer_load_dwordx4` 读的是同一份数据，第二条
必然只命中 L1。新增 `load_b_pack_k64()`（`mfma_preshuffle_pipeline.py`）发一条 16B load，
再从 `i32x4` 里拆出两个 K32 的 i64 片段：

```python
b16 = _buffer_load_vec(..., vec_elems=16 // elem_bytes, cache_modifier=...)
b_i32x4 = vector.bitcast(T.i32x4, b16)
return (_pack_i32_pair_to_i64(d0, d1, ...), _pack_i32_pair_to_i64(d2, d3, ...))
```

kernel 侧用 `load_b_pair()` 包一层：`kpack_bytes==16` 且非 int4 时走合并路径，否则回退到两条
K32 load，所以 int4 / 8B kpack 的调用点行为不变。这不改变 HBM 流量（第二条本来就是 L1 命中），
省的是 VMEM 发射槽和 burst 长度 —— 和 §3.1 缩短 burst 是同一个方向。

## 4. 试过但没有收益的（负结果）

| 尝试 | 结果 |
| --- | --- |
| stage2 `tile_n` 256 → 128 | 更差（89.0 vs 86.5）；64 没有对应 kernel（名字解析失败） |
| stage2 `waves_per_eu` 3 | 只在 bs64 赢 2 µs，bs16 更差、bs128/224 持平 → 保留 2；wpe4 崩（96.8） |
| stage1 `waves_per_eu` 1/3/4 | 默认 2 最优（153.8 vs 163.6 / 159.4 / 189.0） |
| 启用 `hot_loop_scheduler()` 的 `sched_*` 交错提示 | no-op（76.3 → 76.6，噪声内）。该函数原本就是 `sched_barrier(0); return` 的死代码，已改为显式注明并保留 |
| `tile_m` 16 → 32 | 灾难性：gemm2 76 → **336 µs** |
| `AITER_BLKFP8_FMA_DEPTH` 0/2/8/12 | 默认 4 已在最优区间（0 明显差：82.2） |
| A2-scale 放进 loop-carried state | 更慢（77.3 → 88.0），见 §3.2 |
| stage1 也做 scale 前置 | 0 收益（105.7/151.8/250.6 vs 105.4/151.8/250.0）。stage1 有 gate+up 两个 B tile，scale 占比小得多，MFMA 也够长来掩盖延迟。改动保留仅为与 stage2 对称 |
| stage2 `use_async_copy`（global→LDS） | 该 kernel family 未实现，`compile_moe_gemm2` 不接这个参数 |

另外一个**有收益但没有采纳**的：stage1 `tile_n` 128 → 64 在 bs16/bs64 上让 gemm1
151.7 → 145.3（-4.2%），但 bs128 起就反转（174.9 → 192.2，+9.9%），bs224 更差
（249.8 → 266.8）。翻转点在 rows 256~512 之间。

根因值得记一笔：现有启发式是

```python
_tile_n1 = 64 if token <= 8 else 128
```

而这里的 `token` 是 `hidden_states.shape[0]`，在 MORI EP 下等于**容量行数**（bs64 → 256，
bs224 → 896），不是真实 token 数（115 / 359），更不是真正决定 grid 的**活跃 M-block 数**。
所以这条启发式在 EP 下永远走不到小 token 分支。要正确修它需要 host 侧拿到活跃 block 数，
而那个值只存在于 device 上的 `num_local_tokens` / `num_valid_ids` 里 —— 读它就要 host sync。
暂不改；要人工覆盖就直接改 `_tile_n1` 那一行（这个 family 还没进 fmoe CSV tuner，
进了之后应该由 tuned 行来决定）。

## 5. 采集这个负载时的两个坑

1. **grid 被容量放大 22 倍**：这个 shape `resolve_flydsl_gemm_grid_y` 给出 grid_y=384，
   而 `num_valid_ids[0]=272` 只需 **17** 个 M-block。空 WG 对**墙钟几乎没有影响**
   （之前实测 grid_y 94 → 31，gemm1 247.2/249.5 → 246.5/246.5，在噪声内），但会把
   rocprof/ATT 的 per-wave 指标稀释到没法读（每 wave 只有 88.9 条指令，真实值约 1872）。
   采集时务必用 `FORCE_GRID_Y` 收紧（`/tmp/replay_vllm_dump.py` 里的 monkeypatch）。
2. **做 A/B 时给每格设独立的 `FLYDSL_RUNTIME_CACHE_DIR`**。FlyDSL 的 disk cache key 由
   `_jit_function_cache_key()` 算出，除了工具链指纹和源码，还会递归收集闭包里的标量值
   （`_collect_closure_scalar_vals`），所以 `tile_m/tile_n/tile_k/b_nt` 这类编译期参数
   **确实**会进 key（实测：同一 cache 目录里切换 `b_nt` 会重编出第二份二进制）。
   但纯 Python 层面的临时 patch（直接改一行常量再跑）不一定被覆盖，所以扫描时逐格隔离
   cache 目录是更省心的做法。最终版本里所有 tile/wpe 的 env 开关都已删除，
   手测出的默认值直接写死在 `aiter/fused_moe.py` 里。

## 6. 复现

```bash
# 分项计时 + 正确性（默认即为优化后配置）
AITER_FLYDSL_BLKFP8=1 AITER_LOG_MORE=1 \
  python op_tests/test_fmoe_vllm_dump.py -b 64 -l 3
```

本文所有 A/B 都不再有 env 开关（见 §5 第 2 点），要复现某一格就直接改
`aiter/fused_moe.py` 里 `get_2stage_cfgs()` 的 fp8blk 分支对应的那一行，
并给该次运行设一个独立的 `FLYDSL_RUNTIME_CACHE_DIR`：

| 想对照的项 | 改哪一行 | 改成 |
| --- | --- | --- |
| §3.1 stage2 `tile_k` | `_tile_k2 = 128` | `256` |
| §3.6 NT 权重加载 | `_bnt1 = 2 if token <= 256 else 0` | `0`（全 cached）/ `2`（全 NT）|
| §4 stage1 `tile_n` | `_tile_n1 = 64 if token <= 8 else 128` | `64` |
| §4 `waves_per_eu` | `kn1`/`kn2` 末尾的 `+ "_w2"` | `"_w1"` / `"_w3"` |
| §3.5 x-load 钉序 | `moe_2stage_blockscale.py` 的 `_BLK_XPIN_MASK` | `0`（全挡）；或注掉两处 `sched_barrier` |

`relL2 ≈ 0.024` 是参考实现的口径差异，不是 kernel 误差：`op_tests/test_fmoe_vllm_dump.py`
的 torch 参考全程 fp32，没有模拟「stage1 输出再量化成 fp8」这一步。该值在所有 bs / layer 上
稳定在 0.023~0.024，`zero_rows` 恒为 0。

权重是随机生成的（dump 没存权重，386 MB/层），但按同样的 128×128 块量化 + `shuffle_weight(w,(16,16))`
预排布，所以 GEMM 的时间与真实权重一致；决定流量的**活跃专家数来自真实的 topk_ids**。

## 7. 最终数据

优化前 = §3.3 的「原始」列（同一套 harness 实测）。

| bs | rows | recv | gemm1 前 | gemm1 后 | gemm2 前 | gemm2 后 | e2e 前 | e2e 后 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 16 | 104.7 | **98.7** | 60.1 | **48.1** | 182.7 | **163.0** |
| 64 | 256 | 115 | 152.2 | **136.7** | 88.2 | **66.0** | 258.8 | **221.2** |
| 128 | 512 | 221 | 175.2 | **169.1** | 114.2 | **91.4** | 312.3 | **283.2** |
| 224 | 896 | 359 | 250.4 | **240.4** | 163.7 | **130.2** | 442.3 | **398.6** |

小 bs 上「后」列比 §3.3/§3.4 的对照更好，差额来自 §3.6 的 NT（它在 bs128/224 上关闭，
所以那两行与只有 §3.1~§3.5 时基本相同）。

与手写 ASM 1stage（`fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256`）的对比：
bs16 FlyDSL **163.0 vs ASM 215.6（快 24%）**；bs64 **221.2 vs 234.4（快 5.6%**，优化前是慢 10.9%，
§3.6 之前是慢 3.0%）；bs128 283.2 vs 246.7（慢 14.8%），bs192/224 同样仍落后。
bs128 的差距分析见 §8。

跨层（bs64，优化后）：

| layer | gemm1 | gemm2 | e2e | zero_rows | relL2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 151.8 | 76.3 | 249.3 | 0 | 0.0243 |
| 20 | 151.7 | 68.5 | 241.1 | 0 | 0.0228 |
| 40 | 150.1 | 78.8 | 249.3 | 0 | 0.0227 |
| 60 | 153.3 | 68.7 | 242.2 | 0 | 0.0231 |
| 77 | 154.6 | 69.5 | 244.8 | 0 | 0.0231 |

旁证（另一个 dump：EP8 / 32 本地专家 / `max_m=128`，`test_fused_moe_dispatch_dump.py`）：

| | 优化前 | 优化后 |
| --- | ---: | ---: |
| gemm1 | 283 | **246.1** |
| gemm2 | 139 | **124.7** |
| e2e | 435.6 | **391.8** |

回归验证：`op_tests/flydsl_tests/test_flydsl_moe_blockscale.py`（token 16/128 × tile_k 128/256）
stage1 全 pass；stage2 的 1.7% 容差 warning 是**既有**的（HEAD 版 kernel 给 1.8% / 13778 个元素，
`logits_diff` 与 `max abs delta` 完全一致）。

## 8. 距离 4 TB/s 还差什么

stage2 现在 2.69 TB/s，4 TB/s 对应 51.4 µs（现在 76.3），还有 **25 µs**。
优化后 ATT 的剩余大头：

| 指令 | latency 占比 | cyc/次 |
| --- | ---: | ---: |
| `s_barrier` | **11.1%** | 172 |
| `s_waitcnt vmcnt(2)` | **10.7%** | 597 |
| `buffer_load_dwordx4`（W2，真实 HBM 读） | 42.4% 家族合计 | 238~595 |

两者同源，都绑在 **A tile 的 gmem → wait → LDS → barrier → ds_read** 这条串行链上：

- `tile_k=128` 让 MFMA 每拍的工作量减半，能盖住 VMEM 延迟（~1208 cycles）的窗口也随之变窄，
  于是 `store_x_tile_to_lds` 前那次等 A 的 wait 从隐藏变成暴露（597 cyc）。
- 同时 K-tile 数翻倍 → `gpu.barrier()` 次数翻倍（ATT 命中 400 → 720）。

也就是说 `tile_k=128` 的净收益（-13%）是「burst 变浅」赚的，减去「A 链暴露 + barrier 翻倍」亏的。
要继续往上走，需要动这条链，两个方向（都是结构性改动，未实施）：

1. **A tile 加深预取 / 增加 LDS 级数**：现在是 2 级 ping-pong，A 只提前一拍。做成 3 级可以让
   等 A 的 wait 落在两拍之外。LDS 只用了 8 KB（利用率 3.8%），空间充足。
2. **A tile 绕过 LDS**：`tile_m=16`、A tile 只有 2 KB。若 gmem load 能直接落到 MFMA 需要的
   lane 布局（每 lane 取自己那行的 K 切片），LDS 和 barrier 一起省掉。代价是 16 个散开的行地址，
   coalescing 会更差。

stage1 的头号问题是另一条：ATT 显示 **`s_waitcnt vmcnt(0)` 全排空，1042 cyc/次，占 19.3%**
（比 stage2 最坏情况还严重），以及 W-scale 的**单 dword per-lane load**（top-10 里占 3 条、约 10%，
和 A2-scale 同一类 32 倍冗余，但 W-scale 已经随 B tile 预取，位置没问题，问题在粒度）。
`waves_per_eu` 已验证默认 2 最优，所以这个全排空得靠改 pipeline，未实施。
