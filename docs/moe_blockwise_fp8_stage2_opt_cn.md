# FlyDSL blockwise-fp8 MoE：带宽优化记录

日期：2026-09-08（§3.1~§3.7，起点是 stage2）／2026-09-09（§3.8~§3.9，scale 加载，两个 stage）
机器：MI308X（gfx942，80 CU，sclk 1850 / mclk 1300，SPX/NPS1，ROCm 7.14）
路径：`AITER_FLYDSL_BLKFP8=1` 的 2stage FlyDSL blockwise fp8（`fp8blk`，128×128 权重 scale + 1×128 激活 scale）

负载：vLLM GLM-5.3 EP16 decode 的真实 dump
（`fmoe_dump_20260908/dump/bs*/bs*_mstep51_dp0_layer*.pt`，见 `op_tests/test_fmoe_vllm_dump.py`）
hidden 6144 / inter 2048 / 256 全局专家 top-8 / EP16 → 每 rank 16 个本地专家。

---

## 0. 结论速览

| 项 | 结果 |
| --- | --- |
| gemm1 收益 | bs16 **-12.4%**、bs64 **-15.4%**、bs128 **-9.2%**、bs224 **-12.6%** |
| gemm2 收益 | bs16 **-24.6%**、bs64 **-28.6%**、bs128 **-24.9%**、bs224 **-27.2%** |
| e2e 收益 | bs16 **-16.3%**、bs64 **-18.6%**、bs128 **-14.2%**、bs224 **-17.3%** |
| 对手写 ASM 1stage | bs16 **快 29%**、bs64 **快 10%**；bs128 仍慢 8.6%（优化前慢 25.9%） |
| fabric 读带宽占用 | gemm1 **93.8%**、gemm2 70.5% 的实测上限（2662 GB/s，见 §8 —— 不是早先以为的 4 TB/s） |
| 生效方式 | 默认生效，无需 env |
| 正确性 | 6 个 bs × 5 个 layer，`zero_rows=0`、`relL2≈0.023~0.024`（参考实现口径见 §6） |
| 旁证 | 另一个 dump（EP8 / 32 专家）e2e 435.6 → **391.8 µs**（§3.1~§3.5 之后测的，未含后续两轮） |

七处改动，前五处是第一轮（指令与调度），后两处是第二轮（scale 加载的粒度与冗余）：

1. **stage2 `tile_k` 256 → 128**（配置，`aiter/fused_moe.py`）—— 第一轮主要收益
2. **A2 scale 的 load 提到 B burst 之前**（kernel，`moe_2stage_blockscale.py`）—— 次要收益，且在 `tile_k=256` 下收益更大
3. **A-tile（x）的 load 用 `sched_barrier` 钉在 burst 头部**（kernel，两个 stage 各一行）—— 见 §3.5，第一轮唯一同时改善 stage1 的一项
4. **stage1 的 W 用 non-temporal 加载，按 token 数开关**（配置 + kernel）—— 见 §3.6，小 bs 上 e2e -6~8%，大 bs 上必须关掉
5. **共享同一 16B kpack 的两条 B load 合并成一条**（kernel，`mfma_preshuffle_pipeline.py`）—— 见 §3.7，省掉一条只命中 L1 的冗余 `dwordx4`
6. **A block-scale 整块预载进 LDS**（kernel，两个 stage）—— 见 §3.8，gemm1 -4~6%
7. **W block-scale 每个 scale 块只发一条 load**（kernel，两个 stage）—— 见 §3.9，gemm2 -4~7%、gemm1 再 -2%

后两处合起来把 load 指令数砍掉 gemm1 **34%** / gemm2 **47%**，其中单 dword 降 72% / 83%。

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

### 3.8 A block-scale 整块预载进 LDS

第二轮的起点是一个反直觉的观察：ATT 里**单 dword 的 scale load 条数比 16B 的权重 load 还多**，
而且贵得多。

| bs128 | `dwordx4`（真实权重） | `dword`（scale） |
| --- | --- | --- |
| gemm1 | 7680 条 / 27.4% | **7840 条 / 18.0%** |
| gemm2 | 4096 条 / 24.8% | **4416 条 / 15.5%** |

按资源寄存器拆开，贵的那一组是 **activation scale**（gemm1 里 3 条静态指令就占 12.2%，
171~320 cyc/次），而 W scale 反而便宜（20~35 cyc/次，合计 3.6%）—— 因为 W scale 的地址在
wave 内是均匀的，硬件合并成一次 cacheline 请求且常驻。

**A scale 为什么贵**：它的布局是 `(tokens, K/128)`，索引 `token_id * num_k_blocks + kb`。
MFMA 16x16x32 的 lane 映射让一个 lane 的 4 个累加器 f32 对应 4 个连续 M 行
（`row = bx_m + mi*16 + lane_div_16*4 + ii`），所以一拍要发 4 条 per-lane load，
每条 64 个 lane 只取到 4 个不同值（**16 倍 lane 冗余**）。更要命的是行步长 =
`num_k_blocks * 4B`，stage1 是 **192 字节** > 128，那 16 个 f32 落在 **16 条不同 cacheline**
上，每条 128B 只有 4B 有用。而同一行的所有 k-block scale 其实是**连续的** —— K 循环跑 48 拍，
每拍都去碰同样那 16 条线、每次只取 4 字节，两拍之间 wave 要流过 8~16 KB 的 W，把 16 KB 的
vL1D 冲干净，于是**每拍重新 miss**。

**改法**：K 循环之前一次性把整个 M-block 的表读进 LDS（`preload_a_scales_to_lds()`），
K 循环里改成 `ds_read`。三个设计点：

- **线程映射**：`d = tx + i*256`，`row = d / num_k_blocks`、`kb = d % num_k_blocks` ——
  相邻线程走**同一行的相邻 k-block**，正是唯一连续的方向，整张表变成 24 条顺序 cacheline 读。
- **LDS 按 `(kb, row)` 转置**：一个 lane 需要的 4 个连续行落在同一个 16B 里，
  `4 条 ds_read_b32` 变成 **1 条 `ds_read_b128`**。代价是预载时 ds_write 有 bank 冲突，
  但那是一次性的。
- **预算闸门**：表大小 `num_k_blocks * tile_m * 4B`（stage1 3 KB、stage2 1 KB），
  超过 8 KB 就自动回退到原来的逐拍 gather，避免大 `tile_m` / 大 `model_dim` 让 LDS
  取代寄存器成为 occupancy 瓶颈（8 KB 对应 8 WG/CU，而寄存器只允许 ~3.2）。

padding 行照旧写 0，保持「补齐行累加恰好为 0」的不变量，epilogue 不用改。`load_a_scales()`
的返回值接口没变，所以 `compute_tile` 一行未动；旧的逐拍 gather 路径完整保留。

单独这一项（bs128）：gemm1 169.1 → **162.3**（-4.0%）、gemm2 91.4 → 91.1。
gemm2 收益小是因为它的 `num_k_blocks=16`，行步长只有 64 字节，**两行就挤在一条 cacheline 里**，
整张表 1 KB / 8 条线，本来就容易留在 L1。

值得记一笔的副作用：ATT 里单 dword 少了 3740 条、占比 18.0% → 8.4%，但**总 latency 只降 2.7%**，
因为 X（activation tile）那条 `dwordx2` 的占比从 1.5% 涨到 12.2% —— 以前是 A scale 在吃内存
延迟、把 X 挡在后面，现在 A scale 走了，X 的 wait 就暴露成新的头号项。典型的「消掉一个 stall、
露出下一个」，墙钟的真实收益（-4.0%）来自 fabric 上少读的那些分散 cacheline。

### 3.9 W block-scale 每个 scale 块只发一条 load

`load_b_tile()` 的调用点是 `for ku in k_unroll: for ni in num_acc_n:`，gate+up 合计
**每拍 16 条** load —— 取的全是**同一个 f32**：

- **`ku` 那层**：`scale_blk_k` 是 64B 微步的整数倍，所以一个 scale 块内所有 `ku` 算出同一个 `kb`。
  而且 compute 只读 `b_*_tile_in[ku_first][ni][2]`，`ku_first` 恒为 0 ——
  **`ku=1` 那一半加载完从来没被用过**。
- **`ni` 那层**：`nb = (n_blk*16 + n_intra) >> log2(scale_blk_n)`。一个 wave 拥有连续的
  `tile_n/4` 列，起点是 `tile_n/4` 对齐的（expert offset 和 `by*tile_n` 都是 `scale_blk_n`
  的倍数，wave 项是 `tile_n/4` 的倍数），这样的切片不跨 scale 块的充要条件就是
  **`scale_blk_n % (tile_n/4) == 0`**。stage1 是 128%32、stage2 是 128%64，都成立。

所以每个 scale 块只加载一次、复用到所有 (ku, ni)。条件写成 `_wsc_uniform_ni` 在**外层编译期**
求值（因此也进 FlyDSL 的 disk cache key），不满足时自动回退到逐 `ni` 加载。
`(b0, b1, sc)` 的元组形状保留，所以 `_flatten_b_tile` / `_unflatten_b_tile` 和 `compute_tile`
都不用改 —— 同一个 SSA 值复用到所有槽位。

单独这一项（bs128）：gemm1 162.3 → **159.1**、gemm2 91.1 → **85.8**（-5.9%）。
gemm2 收益反而更大，因为它 `num_acc_n=4`（stage1 只有 2），冗余倍数是 stage1 的两倍。

### 3.10 第二轮的 profile 对照（bs128）

| | 原始 | +§3.8 | +§3.9 |
| --- | ---: | ---: | ---: |
| gemm1 load 总条数 | 16500 | 12740 | **10820**（-34%） |
| gemm1 其中单 dword | 7840 | 4100 | **2180**（-72%） |
| gemm2 load 总条数 | 9056 | 7008 | **4788**（-47%） |
| gemm2 其中单 dword | 4416 | 2400 | **756**（-83%） |
| gemm2 ATT 总 latency | 1442236 | 1448768 | **1285428**（-11%） |

| 指标 | gemm1 前 → 后 | gemm2 前 → 后 |
| --- | --- | --- |
| VGPR / AGPR | 28 / 132 → **4** / 132 | 32 / 128 → **12** / 132 |
| 总寄存器 | 160 → **136** | 160 → **144** |
| Wavefront occupancy | 34.1% → **37.0%** | 33.8% → 29.5% |
| L2-Fabric 读带宽 | 92.4% → **93.8%** | 80.5% → 70.5% |
| MFMA util | 20.0% → **23.1%** | 19.7% → 19.8% |
| LDS | 4096 B → 7168 B | 8192 B → 9216 B |

gemm2 的 occupancy 和 fabric 占比都**降了**却更快，这不矛盾 —— fabric 利用率不是目标，
字节/时间才是：`1875 GB/s × 85.8 µs = 161 MB`，改前是 `2143 × 91.4 = 196 MB`，
**少读了 35 MB**（gemm1 同理少约 19 MB）。那些分散的 A-scale gather 和冗余 W-scale load
是**真实的 HBM 流量**，不只是延迟。

另外 gemm1 的 VGPR 掉到只剩 4 个，几乎全部状态都进了 AGPR（132）。总寄存器 136 仍在 128 之上，
所以还没跨过 4 wave/SIMD 的门槛 —— 想再进一步得从那 132 个 AGPR 下手。

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

第二轮（bs128 专项）另外几个负结果：

| 尝试 | 结果 |
| --- | --- |
| `tile_m` 16 → 32 减权重重读 | traffic 只降 19%（重读本来已被 L2 吃掉大半），但 `m_repeat` 翻倍让累加器翻倍、occupancy 从 32% 掉到 **17%**，fabric 带宽掉 25% → 净亏（gemm1 170 → 185）。配合 `tile_n` 收到 64 想还原累加器数量也没救回来（176） |
| grid 顺序改成 M 快维度（提 L2 命中率） | 明确负结果：gemm1 **+28%**、gemm2 **+36%**。N 做快维度正是把并发读铺开到所有 HBM channel 的原因，拿它换局部性会让 fabric 带宽崩掉 |
| XCD swizzle | 不需要：`num_n=48`，`48 mod 4 == 0`，同 n 不同 m 本来就落在同一个 XCD |
| 收紧 `grid_y`（砍掉 21.4 倍空 WG） | 墙钟无变化（gemm1 169.5~170.2、gemm2 91.2~92.2，扫 24~513 全在噪声内）。**注意**：在 rocprofv3 下测会看到假的 4% 收益，那是 profiler 的 per-dispatch 开销随 WG 数放大 —— 见 §5 第 1 点 |
| bs128 上重扫全部 tile/wpe 旋钮 | 全部饱和：stage1 `tile_k` 128 最优（256 慢 18 µs）、`wpe` 2/3 打平（4 慢 74 µs）、stage2 `tile_n` 256 最优（128 慢 18 µs）、`tile_m` 16/32 打平 |

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

## 5. 采集这个负载时的三个坑

1. **grid 被容量放大 20 倍以上**：`resolve_flydsl_gemm_grid_y` 是按容量行数给的，
   真实 M-block 数来自 `num_valid_ids`：

   | bs | rows | `num_valid_ids` | 真实 M-block | 启动 grid_y | 放大 |
   | ---: | ---: | ---: | ---: | ---: | ---: |
   | 16 | 64 | 176 | 11 | 289 | 26.3x |
   | 64 | 256 | 272 | 17 | 385 | 22.6x |
   | 128 | 512 | 384 | 24 | 513 | 21.4x |
   | 224 | 896 | 576 | 36 | 705 | 19.6x |

   空 WG 对**墙钟没有影响**（bs128 上扫 grid_y 24~513，gemm1 稳定 169.5~170.2、
   gemm2 91.2~92.2，全在噪声内），但会把 rocprof/ATT 的 per-wave 指标稀释同样的倍数
   （`Instructions per wavefront` 显示 ~100，真实值约 2100）。
   采集时务必用 `FORCE_GRID_Y` 收紧到真实块数（`/tmp/replay_vllm_dump.py` 里的 monkeypatch）；
   已验证它不改变被测对象。
2. **不要用 rocprofv3 的 kernel trace 去比较 WG 数不同的配置**。在 `--kernel-trace` 下测
   grid_y 收紧会看到 gemm2 -4% / gemm1 -3~7% 的"收益"，那是假的 —— profiler 的 per-dispatch
   开销随 WG 数放大（同一配置下 gemm1 绝对值 169 → 192，被抬高 13%）。
   换成 aiter 自带的 per-kernel device 计时（`AITER_LOG_MORE=1`）差异立刻消失。
3. **做 A/B 时给每格设独立的 `FLYDSL_RUNTIME_CACHE_DIR`**。FlyDSL 的 disk cache key 由
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
| §3.8 LDS A-scale 表 | `moe_2stage_blockscale.py` 的 `_use_asc_lds`（两处） | `False`（回退到逐拍 gather） |
| §3.9 W-scale 折叠 | `moe_2stage_blockscale.py` 的 `_wsc_uniform_ni`（两处） | `False`（回退到逐 `ni` 加载；`ku` 那层的折叠无条件保留） |

`relL2 ≈ 0.024` 是参考实现的口径差异，不是 kernel 误差：`op_tests/test_fmoe_vllm_dump.py`
的 torch 参考全程 fp32，没有模拟「stage1 输出再量化成 fp8」这一步。该值在所有 bs / layer 上
稳定在 0.023~0.024，`zero_rows` 恒为 0。

权重是随机生成的（dump 没存权重，386 MB/层），但按同样的 128×128 块量化 + `shuffle_weight(w,(16,16))`
预排布，所以 GEMM 的时间与真实权重一致；决定流量的**活跃专家数来自真实的 topk_ids**。

## 7. 最终数据

优化前 = §3.3 的「原始」列（同一套 harness 实测）。

| bs | rows | recv | gemm1 前 | gemm1 后 | gemm2 前 | gemm2 后 | e2e 前 | e2e 后 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 16 | 104.7 | **91.7** | 60.1 | **45.3** | 182.7 | **153.0** |
| 32 | 128 | 51 | — | 126.9 | — | 55.3 | — | 200.0 |
| 64 | 256 | 115 | 152.2 | **128.8** | 88.2 | **63.0** | 258.8 | **210.6** |
| 128 | 512 | 221 | 175.2 | **159.1** | 114.2 | **85.8** | 312.3 | **268.0** |
| 192 | 768 | 300 | — | 197.8 | — | 107.1 | — | 332.0 |
| 224 | 896 | 359 | 250.4 | **218.8** | 163.7 | **119.1** | 442.3 | **366.0** |

两轮的分解（e2e）：第一轮 §3.1~§3.7 到 bs16 163.0 / bs64 221.2 / bs128 283.2 / bs224 398.6，
第二轮 §3.8~§3.9 再推到上表的 153.0 / 210.6 / 268.0 / 366.0。

与手写 ASM 1stage（`fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256`）的对比：
bs16 FlyDSL **153.0 vs ASM 215.6（快 29%）**；bs64 **210.6 vs 234.4（快 10.2%**，
优化前慢 10.9%）；bs128 **268.0 vs 246.7（慢 8.6%**，优化前慢 25.9%），bs192/224 同样仍落后。
bs128 剩余差距的分析见 §8。

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

## 8. 上限在哪，以及 bs128 剩余的 8.6%

**先纠正一个前提。** 本文早期版本按「实测上限 4 TB/s」折算还差多少，那个数是错的。
rocprof-compute 给出的 **L2-Fabric 读带宽上限是 2662 GB/s**，而 gemm1 现在跑到
**2498 GB/s = 93.8%**。也就是说 gemm1 已经**贴着这台机器的 fabric 读上限**，
它是纯 traffic 决定的：时间 ≈ 字节数 / 2.5 TB/s，唯一的提速方式是少读字节。
这也解释了为什么 §4 里所有「减少权重重读」的尝试都失败 —— 它们省下的字节都小于
付出的带宽代价。

gemm2 是 70.5%，还有余量，但它的字节数已经被第二轮砍掉 35 MB，继续压要靠结构改动。

### bs128 对 ASM 的 8.6% 差距怎么分

| | FlyDSL | ASM |
| --- | ---: | ---: |
| gemm1 + gemm2 | 159.1 + 85.8 = 244.9 | 232（单个融合 kernel） |
| a2 quant kernel | 7.5 | 0 |
| sorting ×2 | 15.0 | 15.0 |
| 合计 | 268.0 | 246.7 |

差距 21.3 µs 里，**7.5 µs 是 ASM 根本没有的中间量化 kernel**，剩下 ~13 µs 是两个 GEMM
对一个融合 kernel 的结构性劣势（两次权重流、两次 launch ramp）。

### 现在的头号瓶颈：barrier / 同步

两轮优化把 scale 加载从瓶颈里去掉之后，gemm1 的 ATT 变成这样：

| 家族 | latency 占比 | 说明 |
| --- | ---: | --- |
| `buffer_load` | 44.2% | 真实权重流，贴着 fabric 上限，不可压 |
| `s_waitcnt` | **24.0%** | 其中 `vmcnt(8)` 520 cyc/次 |
| **`s_barrier`** | **12.8%** | 176 cyc × 1920 次，单条最大项 |
| VALU | 8.6% | 第一轮时是 14.2%，scale 乘法的占比已被摊薄 |

gemm2 同理，`s_barrier` 7.6%。两者同源，都绑在
**A tile 的 gmem → wait → LDS → barrier → ds_read** 这条串行链上：`tile_k=128`（§3.1）
让 K-tile 数翻倍，`gpu.barrier()` 次数也就翻倍。这是 §3.1 净收益里被抵掉的那部分。

两个方向（都是结构性改动，未实施）：

1. **A tile 加深预取 / 增加 LDS 级数**：现在是 2 级 ping-pong，A 只提前一拍。做成 3 级可以让
   等 A 的 wait 落在两拍之外。LDS 现在用 7~9 KB（含 §3.8 的 scale 表），64 KB 里空间充足。
2. **A tile 绕过 LDS**：`tile_m=16`、A tile 只有 2 KB。若 gmem load 能直接落到 MFMA 需要的
   lane 布局（每 lane 取自己那行的 K 切片），LDS 和 barrier 一起省掉。代价是 16 个散开的行地址，
   coalescing 会更差。

### 其余几条，按性价比排

1. **把 a2 量化融进 stage1 的 epilogue** —— 7.5 µs，风险低。stage1 的 `tile_n=128` 正好等于
   per_1x128 的量化组大小，所以每个 block 已经完整拥有自己那些行的一个量化组，amax 归约
   完全在块内、不需要跨块通信。顺带把 a2 的 bf16 往返降成 fp8 单次写。
   `MOEMetadata` 里已经有 `fuse_quant` 字段。
2. **压 AGPR 到 128 以下** —— 两个 kernel 的 VGPR 已经只有 4/12，是 132 个 AGPR 把总数顶在
   136/144，卡在 3.2 wave/SIMD。累加器理论上只需 ~32 个，多出来的很可能是循环携带的 B tile。
   压到总数 <128 能拿到 4 wave/SIMD（+25% occupancy）。
3. **`v_pk_mul_f32` 替掉 scale 乘法** —— `v_pk_fma_f32` 已经是打包的，但 `sa[ii] * sw[ni]`
   那 4 条标量乘没有。第一轮时值 4.8%，现在 VALU 整体已降到 8.6%，收益有限。
4. **gemm2 的 W scale 也可以试 NT** —— §3.6 的 `b_nt` 只喂给 stage1，stage2 恒为 cached，
   而实测 stage1 开 NT 会通过 L2 间接让 gemm2 快 7~11%（见 §3.6），说明 stage2 自己用 NT
   可能也有收益，待验证。
