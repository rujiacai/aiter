# FlyDSL stage1 追赶 CK 与 PR3987 gateup：逐 feature 优化记录（token=32768）

> shape: token=32768, model_dim=4096, inter_dim=192, expert=193, topk=9, bf16 输出，
> fp8(e4m3fnuz) per_tensor 权重与激活
> 硬件: MI308X（gfx942，80 CU），HIP 7.2.53211，torch 2.9.1+rocm7.2.3
> 软件: aiter `moe_opt_0727` @ 01e74c06；flydsl 0.1.2
> 驱动: `moe_stage1_opt/run.sh`，数据落在 `moe_stage1_opt/results/`
> **PTL 关**（见 1.2），所有绝对值不可与 PTL 开的数据混用

这份记录和 `moe_stage2_opt/moe_stage2_reduce_parity_32k.md` 是同一套方法，
换成 stage1：在**旧 FlyDSL stage1 内核**上一个 feature 一个 feature 地加，
直到追平参照。

## TL;DR

| | stage1 | e2e（stage2 也优化到头） |
|---|---|---|
| 起点 | 3714.1 | 7317.0 |
| **五个 feature 之后** | **2657.4** | **6270.6** |
| CK stage1（生产默认） | 2969.5 | 6574.6 |
| PR3987 flydsl gateup | 2615.2 | 6190.7（整条流水线，cos 0.997831） |

**stage1 −1056.7 us（1.40×），e2e −1046.5 us（1.167×）；比生产默认的 CK 快 304 us，
离 PR3987 差 1.3%，而精度高两个数量级（cos 0.999995 全程不变）。**

五个 feature 里 **f4 一个占了 62%**，而它只是 kernel 名加四个字符（`_bnt0`）。

结构：

- 第一章：测试怎么做、起点和两个参照各是多少、tile 怎么选的；
- 二~四章、六~七章：每章一个 feature，做了什么、值多少、为什么；
- 第五章：和 PR gateup 的指令级对比——**四个被排掉的解释和一个找到的原因**；
- 第八章：汇总，stage1 口径 + e2e 口径 + 和 PR 整条流水线比；
- 第九章：还没做的，以及量过之后确认不值得做的。

---

## 一、测试情况与基线

### 1.1 起点和两个参照

| | stage1 内核 | stage1 us |
|---|---|---|
| **起点 `base`** | 旧 `flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16` | **3714.1** |
| **参照 `ck`** | `moe_ck2stages_gemm1_256x64x64x128_...`（当前生产默认） | **2969.5** |
| **目标 `target`** | PR3987 的 `moe_2stage_gateup_prefill_1x4`（flydsl gateup） | **2615.2** |

- `base → ck` 差距 **744.6 us（1.251×）**
- `base → target` 差距 **1098.9 us（1.420×）**
- **tile 无需对齐：PR 用的就是同一套几何**，见第五章

`target` 是本轮在 `/data/aiter_pr`（PR 分支 `luocheng/moe_gemm_308`，自带 flydsl 0.2.4
的 venv `/data/pr_env`）上复现的，**同样是 PTL 关**。另一条会话先前采到 2609.9，
本轮 2615.2，差 0.2%；同一批里它测到的我方 CK stage1 是 2965.5，本轮 2969.5，
差 0.1%——两边可以直接比。

> **精度上有一处不对等**：PR 整条流水线的 `cos=0.997831`，我方是 `0.999995`，
> 差两个数量级。本文所有 feature 都保持 `cos=0.999995` 逐位不变，
> 如果最终要整体切到 PR 的方案，这个精度差得先弄清楚。

### 1.2 PTL：这批数据是在关着的状态下采的

PTL（Peak TOPS Limiter）关掉整机慢约 22%，而且各档慢的比例不同，
所以 PTL 开/关的数据绝对不能混。本节点上：

- GPU 0–3：`amdsmi_get_gpu_ptl_state` 返回 True
- GPU 4–7：返回 False / `NOT_SUPPORTED`

跑在 GPU 4 上。核对方式是拿 stage2 那份记录的同一个 case 对一下：
它 PTL 开是 e2e 7828.8、关是 9673.1，本轮同配置测到 9712.6，落在"关"这一档，
偏差 0.4%。所以这是一台 PTL 关的卡，且状态与那份记录的"关"列一致。

`moe_stage2_opt/run.sh` 在 PTL 关时会直接拒绝跑；stage1 这份不拒绝，
因为这半台机器上打不开，只在开头把状态打出来。

### 1.3 方法

- **headline 是 stage1 内核时间**，不是 e2e。stage1 只占这个 case 的 ~30%，
  用 e2e 当 headline 会把每个 feature 的收益除以三，埋进运行间波动里。
  取 `AITER_LOG_MORE=1` 打出的 ROCTracer 表里 `moe_gemm1_0` 的 `device_time_avg`。
- e2e 同时记录，用来确认 stage1 的收益没有在别处付回去。
  **这一列带 tracer**（约 +0.6%），只能各 stage 之间横向比。
- 每个 stage 跑 3 次取中位数。组内全距 <0.05%（例如 base 三次 3712.2/3713.4/3714.1）。
- **归因用的那一组里 stage2 全程钉死**在
  `flydsl_moe2_..._t64x128x64_reduce_persist_bnt0`，一个 knob 都不开。
  它不是最快的 stage2，这是故意的：它没有任何 env knob，
  stage1 的 knob 不可能和它串味。
- **另有一组 `--stage2-opt`**：把 `moe_stage2_opt` 那条梯子跑到头的 knob 集加上，
  再走一遍同样的 stage1 梯子，量"两级都优化完"的 e2e（8.2）。
  两组都留着，因为它们的口径不同，而且差异本身有信息（8.3）。
- **`block_m` 全程钉死 64。** 它喂给 `moe_sorting`，改了会连带改变 stage2 和归约
  看到的 padding 行数——那样得到的 stage1"收益"里会掺进另外两个算子少干的活。
  64 也正是 CK 参照和 pr1x4 stage2 用的值。
- **防呆**：配置里的 kernel 名写错时 `fused_moe` 会**静默回落**到别的配置，
  run.sh 每次都核对实际跑的 `kernelName1` 与配置一致，不一致报 `CONFIG MISS`。

### 1.4 起点是怎么选的：tile 扫描

`aiter/configs` 里 32k 这一档的 stage1 没有 FlyDSL 条目（生产用 CK），
所以起点得自己选。stage1 的 `n_dim` 是 `inter_dim=192`，
枚举器要求 `n_dim % tile_n == 0`，所以 `tile_n ∈ {32, 64}`；`tile_m` 固定 64（=block_m）。
全部合法组合（cos 都是 0.999995）：

| kernelName1 | stage1 us |
|---|---|
| `t64x32x256` | 34061.0 |
| `t64x32x128` | 13107.1 |
| `t64x64x256` | 8609.8 |
| `t64x64x256_bnt0` | 7916.3 |
| `t64x64x256_n16` | 5075.7 |
| `t64x64x128` | 4462.0 |
| `t64x64x128_bnt0` | 3936.4 |
| **`t64x64x128_n16`** | **3715.0** ← 起点 |
| `t64x64x128_n16_w2` | 3714.2 |

`waves_per_eu`（`_w1`~`_w4`）在这个 shape 上完全没有作用（4468.5~4469.1，全距 0.6 us）。
`tile_k=64` 也试过（`t64x64x64_n16` = 4758.7），更差——它把 LDS 降到 8448 字节、
occupancy 拉得很高，但 K 循环的开销涨得更多。**这一条很重要：它说明
occupancy 在这个内核上不是可以单独换取收益的东西**，后面 f2 的设计要绕开这一点。

起点取"最好的合法 tile"而不是随便一个，是为了让后面每一行都是真正的代码改动，
而不是本来调个 tile 就能拿到的东西。

### 1.5 基线在哪儿慢：base vs ck 的计数器

`rocprofv3 --pmc`，只采 stage1 这一个 kernel，每 dispatch 平均：

| counter | base | ck | base/ck |
|---|---|---|---|
| `SQ_INSTS_MFMA` | 57,753,600 | 57,753,600 | **1.000** |
| `SQ_WAVES` | 57,600 | 57,612 | 1.000 |
| `SQ_INSTS_VMEM_WR` | **887,904** | **112,800** | **7.87** |
| `SQ_WAIT_ANY` | 637,115,006 | 290,542,146 | **2.19** |
| `MeanOccupancyPerCU` | **11.40** | **15.35** | 0.74 |
| `MfmaUtil` | 53.28 | 69.56 | 0.77 |
| `SQ_INSTS_VALU` | 84,113,880 | 105,524,400 | 0.80 |
| `SQ_INSTS_SALU` | 8,632,020 | 11,682,072 | 0.74 |
| `MemUnitStalled` | 0.54 | 0.15 | 3.6 |

三条读数，直接决定了 f1 和 f2：

1. **MFMA 条数和 wave 数逐位相同。** 两边算的是同一个 GEMM、用的是同一个 MFMA 形状，
   差距全在访存和调度，不在算力。
2. **`SQ_INSTS_VMEM_WR` 是 CK 的 7.87 倍**，几乎正好 8 倍。
   stage1 一共要写 `token*topk*inter_dim` 个 bf16；887,904 条指令 × 64 lane
   ≈ 每 lane 每条指令写 1 个 bf16。**我们在做 16 位标量存储，CK 在做 128 位。** → f1
3. **occupancy 11.40 vs 15.35。** 11.40 ≈ 12 = 3 workgroup × 4 wave。
   而 VGPR 只用了 106（512/106 = 4.8，允许 4 个），**卡在 LDS 上**。 → f2

VALU/SALU 我们反而比 CK **少**，所以不是发射受限；bank conflict 我们是 0。

---

## 二、Feature 1：让 CShuffle 在 tile_n=64 上重新可用

**stage1 −171.5 us（补上 base→ck 差距的 23.0%）**

### 2.1 优化前：整条 CShuffle 被一个整除条件关掉了

stage1 有两条 epilogue：CShuffle（先把结果写进 LDS，换一种线程映射读回来再宽存）
和 direct（直接每个元素一条 store）。默认应该是 CShuffle，但实际跑的是 direct，
因为 host 侧在 tile_n 不是 128 的倍数时把它整个关掉：

```1822:1824:aiter/ops/flydsl/moe_kernels.py
        use_cshuffle_epilog=False
        if use_cshuffle_epilog is None and tile_n % 128 != 0
        else use_cshuffle_epilog,
```

`tile_n % 128` 这个条件不是任意的，它是从 `c_shuffle_epilog` 的一个校验倒推出来的：

```147:150:aiter/ops/flydsl/kernels/mfma_epilogues.py
    if (int(tile_n) % (int(cshuffle_nlane) * int(e_vec))) != 0:
        raise ValueError(
            f"tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_vec}, got tile_n={tile_n}"
        )
```

Step 2 每一轮把 (tile_m, tile_n) 这个矩形切成
`cshuffle_mlane` 行 × `cshuffle_nlane * e_vec` 列，所以列宽必须整除 tile_n。
stage1 写死了 `e_vec=4`，`cshuffle_nlane` 用默认的 32，列宽就是 128——
于是 `tile_n=64` 非法，只能回落 direct。

而 `inter_dim=192` 决定了 `tile_n` 只能是 32 或 64（1.4），**两个都不是 128 的倍数**。
也就是说在这个 shape 上，CShuffle 从来没有被启用过。

### 2.2 优化后：nlane 跟着 e_vec 收窄

同样是每轮 2048 个元素，摆成 32 行 × 64 列而不是 8 行 × 256 列，
`tile_n=64` 就合法了。`cshuffle_nlane` 不再钉死 32，而是搜一个最大的合法值：

```aiter/ops/flydsl/kernels/moe_gemm_2stage.py
            for _cand in range(min(32, int(tile_n) // _e_vec), 0, -1):
                if int(tile_n) % (_cand * _e_vec):
                    continue
                if int(total_threads) % _cand:
                    continue
                if int(tile_m) % (int(total_threads) // _cand):
                    continue
                _cshuffle_nlane = _cand
                break
```

搜索而不是直接取 `tile_n // e_vec`：后者在 `tile_n=64` 恰好合法，
但换个宽度就不一定（192/8 = 24，既不整除 256 也不整除 64）。

两个 knob：

```bash
FLYDSL_MOE_STAGE1_NLANE_FIT=1   # nlane 跟着 e_vec 走
FLYDSL_MOE_STAGE1_EVEC=8        # 每 lane 8 个 bf16 = 128 位
```

`e_vec=8` 时 `nlane=8`、`mlane=256/8=32`，`tile_m=64` 整除 32，合法。

host 侧那个条件也跟着改成：开了 NLANE_FIT 就把决定权还给内核。

### 2.3 ISA：存储宽度确实变了

`FLYDSL_DUMP_IR=1`，数 `moe_gemm1_0/17_final_isa.s`：

| | `buffer_store_short` | `buffer_store_dwordx2` | `buffer_store_dwordx4` | `ds_write_b16` |
|---|---|---|---|---|
| base（direct） | **16** | 0 | 0 | 0 |
| `e_vec=4`（nlane=16） | 0 | **4** | 0 | 16 |
| `e_vec=8`（nlane=8） | 0 | 0 | **2** | 16 |

16 条 16 位存储 → 2 条 128 位存储。中间那档 `e_vec=4` 实测 3559.4 us，
比 `e_vec=8` 的 3541.9 慢 17.5 us，所以取 8。

### 2.4 计数器：这一刀切得很干净

| counter | base | f1 | f1/base | f1/ck |
|---|---|---|---|---|
| `SQ_INSTS_VMEM_WR` | 887,904 | **110,832** | **0.125** | **0.983** |
| `SQ_INSTS_SALU` | 8,632,020 | 3,061,020 | 0.355 | 0.262 |
| `MemUnitStalled` | 0.54 | 0.18 | 0.342 | 1.21 |
| `SQ_INSTS_LDS` | 18,964,500 | 20,090,532 | 1.059 | 1.073 |
| `SQ_INSTS_MFMA` | 57,753,600 | 57,753,600 | 1.000 | 1.000 |

`SQ_INSTS_VMEM_WR` 降到 **0.125 = 恰好 1/8**，且**与 CK 的 112,800 只差 1.7%**——
写侧从此不再是差距的一部分。SALU 一起降了 65%（direct 那条路径每个元素都要算一次地址）。
代价是 LDS 指令 +5.9%（多了 Step 1 的 16 条 `ds_write_b16` 和 Step 2 的读回）。

MFMA 条数逐位不变，确认改的全在 epilogue。

---

## 三、Feature 2：把 lds_tid 塞进 CShuffle 用不到的那半 X 区

**stage1 −170.7 us（累计补上 45.9%）**

### 3.1 优化前：256 个字节卡掉了 1/4 的 occupancy

f1 之后 LDS 布局是：

| 区域 | 字节 | 用途 |
|---|---|---|
| `[0, 16384)` | 16384 | X ping-pong（ping 8192 + pong 8192），主循环用 |
| `[0, 8192)` | (别名) | `lds_out`，CShuffle 暂存区，epilogue 用 |
| `[16384, 16640)` | 256 | `lds_tid`，`moe_sorting` 打包的 sorted id |
| **合计** | **16640** | |

gfx942 每 CU 64 KB LDS：`65536 / 16640 = 3.94` → **每 CU 只能放 3 个 workgroup**，
每个 4 个 wave，共 12 wave。实测 `MeanOccupancyPerCU = 11.40`，对得上。

如果是 16384：`65536 / 16384 = 4.0` → **4 个 workgroup = 16 wave**，
正好是 CK 的 15.35。而 VGPR 只用 106（`512/106 = 4.8`），**寄存器允许 4 个**，
唯一挡路的就是这 256 字节。

关键在于 `lds_out` 只有 8192 字节，X 区有 16384——**epilogue 期间
`[8192, 16384)` 这 8 KB 是空的**，`lds_tid` 只需要其中 256 字节。

### 3.2 优化后

```bash
FLYDSL_MOE_STAGE1_LDSTIGHT=1
```

`lds_tid` 从 `max(lds_x, lds_out)` 挪到 `lds_out` 之后：

```aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    if _ldstight:
        _lds_tid_byte_off = lds_out_bytes
        lds_total_bytes = max(lds_x_bytes, lds_out_bytes + lds_tid_bytes)
```

代价是**填充时机必须往后挪**。原来 `lds_tid` 在 prologue 之前就填好，
整个主循环给它的 global load 打掩护；现在那块字节在主循环期间属于 X pong buffer，
只能等主循环用完：

```aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                if _ldstight:
                    # lds_tid aliases bytes the X ping-pong just finished
                    # reading, so wait for every wave to be out of them before
                    # writing.  Visibility of the write is covered by the
                    # barrier c_shuffle_epilog emits before its Step 1, which is
                    # the first thing that reads lds_tid back.
                    gpu.barrier()
                    _fill_lds_tid()
```

两个同步点都要对：

- **写之前**要一个 barrier，等所有 wave 读完 X（自己加）；
- **写之后**要一个 barrier 才能被别的 wave 读到——这个不用自己加，
  `c_shuffle_epilog` 在 Step 1 之前本来就有一个，而 Step 1 正是第一个读 `lds_tid` 的地方。

区域上也不冲突：Step 1 写 `lds_out [0, 8192)`，`lds_tid` 在 `[8192, 8448)`，
Step 2 同时读这两块，互不重叠。

只在 CShuffle 开着、且 `lds_out + lds_tid <= lds_x` 时才生效；
direct 路径 `lds_out = 0`，挪过去就会和 X 撞，所以自动跳过。

### 3.3 效果

ISA 里 `.group_segment_fixed_size` 从 16640 变成 **16384**，VGPR 106 → 98。
stage1 3541.9 → **3371.2**。cos 仍然是 0.999995，逐位不变。

计数器确认它做的正是预期的那一件事，别的一件都没动：

| counter | f1 | f2 | ck | f2/ck |
|---|---|---|---|---|
| `MeanOccupancyPerCU` | 11.49 | **15.27** | 15.35 | **0.995** |
| `SQ_INSTS_MFMA` | 57,753,600 | 57,753,600 | 57,753,600 | 1.000 |
| `SQ_INSTS_LDS` | 20,090,532 | 20,090,532 | 18,724,800 | 1.073 |
| `SQ_INSTS_VMEM_RD` | 12,084,900 | 12,084,900 | 11,110,800 | 1.088 |
| `SQ_INSTS_VMEM_WR` | 110,832 | 110,832 | 112,800 | 0.983 |
| `MfmaUtil` | 56.11 | 58.54 | 69.56 | 0.842 |
| `MemUnitStalled` | 0.18 | 0.16 | 0.15 | 1.017 |

**LDS / VMEM 指令数逐位不变**（f1 与 f2 完全相同）——它只改了摆放，没改任何访存行为。
occupancy 11.49 → 15.27，**追平 CK 的 15.35**。

> `SQ_WAIT_ANY` 从 586M 涨到 640M，方向看着不对，其实是口径问题：
> 这个计数器按 wave 累加，f2 同时驻留的 wave 多了 33%，等待的 wave-cycle 自然更多，
> 而墙钟时间是降的。跨 occupancy 变化时不要直接比它的绝对值。

### 3.4 为什么这一刀值，`tile_k=64` 那一刀不值

1.4 里试过 `tile_k=64`：LDS 降到 8448、occupancy 高得多，结果**慢了 1200 us**。
两件事看起来都是"拿 LDS 换 occupancy"，结论却相反，区别在于**有没有副作用**：

- `tile_k=64` 把 K 循环的 tile 数翻倍，每个 tile 的固定开销（barrier、
  prologue/epilogue prefetch、B 的重新加载）都多摊一遍——occupancy 涨了，
  但每 wave 要干的活也涨了。
- f2 只动了 256 个字节的**摆放位置**，MFMA、访存、循环结构一个都没变，
  唯一的代价是 `lds_tid` 那一次 global load 从"被主循环掩护"变成"在 epilogue 口上"。

所以 occupancy 本身是有价值的，前提是不要用干活变多去换。

---

## 四、Feature 3：per-tensor 激活 scale 提到入口

**stage1 −54.9 us（累计补上 53.3%）**

这一条是**第五章那次指令级对比直接指出来的**，不是猜的：我们的 `SQ_INSTS_VMEM_RD`
比 PR gateup 高 8.6%，换算成每 wave 是 209.8 vs 193.1，**多 16.7 次**。
而 CShuffle Step 1 每个线程正好处理 16 行（`m_repeat=4 × ii=4`），每行读一次激活 scale。

### 4.1 优化前：每行都去读同一个 float

epilogue 每行都按 token 号去 buffer 里取激活 scale：

```aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                            sx = (... buffer_ops.buffer_load(
                                        sx_rsrc, t2, vec_width=1, dtype=T.f32
                                    ), ...)
```

在 **per-tensor** 量化下这 16 次读的是**同一个值**。之所以还要按 token 索引，
是因为 host 侧把那个标量**展开成了 token_num 长的数组**：

```1724:1724:aiter/ops/flydsl/moe_kernels.py
        flat_a_scale = _expand_per_tensor_scale(a1_scale, token_num, 1)
```

也就是说 32768 个 float 里存的是同一个数，内核再逐行去读它。

### 4.2 优化后

```bash
FLYDSL_MOE_STAGE1_SCALAR_ASCALE=1
```

host 不再展开，直接把那一个元素传下去；内核在 epilogue 入口读一次：

```aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                sx_scalar = (
                    buffer_ops.buffer_load(
                        sx_rsrc, fx.Int32(0), vec_width=1, dtype=T.f32
                    )
                    if (scalar_a_scale and not is_f16_or_bf16)
                    else fx.Float32(1.0)
                )
```

原来那个 `arith.select(t_valid, load, 0.0)` 里的置零可以一起去掉：
padding 行的结果只写进 LDS，Step 2 的全局写本来就带 `t_valid` 谓词，永远不会落盘。

**这个 knob 是有前提的，用错会静默算错**：它假设 scale 真的是 per-tensor，
而内核分辨不出来（展开之后两种量化看起来一模一样）。所以校验放在 host 侧
还看得见原始张量的地方，`a1_scale.numel() != 1` 直接报错——
和 stage2 的 `FLYDSL_MOE_STAGE2_SCALAR_ASCALE` 是同一套做法。

`scalar_a_scale` 会改变 `scale_x` 的预期长度，所以它进了 module cache key（`_sas`），
否则换个设置会悄悄复用上一个二进制。

### 4.3 效果

静态 ISA 里 `buffer_load` 从 215 条降到 200 条（少的 15 条就是那 16 次减去入口的 1 次）。
stage1 3370.6 → **3315.7**。cos 仍是 0.999995。

收益（−54.9）比 f1/f2（各约 −171）小一个量级，符合预期：这 16 次读的是同一个
cache line，本来就基本全命中 L1，省掉的是发射和延迟链，不是访存带宽。

---

## 五、和 PR gateup 对齐后的指令级对比

### 5.1 tile 本来就是对齐的

PR 那一行配置叫 `fused_moe_gfx942__64_128_256_True`，四个字段是
`BLOCK_M_BLOCK_N_BLOCK_K_use_prefill`。但**第三个字段是死的**——调用点把它注释掉了：

```/data/aiter_pr/aiter/fused_moe_gfx942.py
            BLOCK_TILE_SIZE_K=None,  # kcfgs.BLOCK_K,
```

所以 `TILE_K` 走 fp8 的默认值 **128**，不是名字里的 256。
再把 N 的口径对齐（PR 的 `N=2*inter_dim=384`，`BN=128` → `contiguous_n = BN//2 = 64`
个输出通道，正好等于我们的 `tile_n=64`）：

| | PR gateup | 我们 |
|---|---|---|
| tile_m / BM | 64 | 64 |
| 每个 N-tile 的输出通道 | 64 | 64 |
| TILE_K | 128 | 128 |
| workgroup | 256 线程 | 256 线程 |
| N 方向 tile 数 | 3 | 3 |
| MFMA | `v_mfma_f32_16x16x32_fp8_fp8` | 同 |
| LDS | 16384 B | 16384 B（f2 之后） |

**已经完全对齐了**，一个参数都不用改。所以剩下的差距和 tile 选择无关。

本轮在 `/data/aiter_pr` 上复现了一遍：gateup **2615.2 us**
（另一条会话记的 2609.9，差 0.2%），e2e 6190.7、cos 0.997831。

### 5.2 动态计数器：活一样多，就是在等

| counter | 我们 f2 | PR gateup | f2/PR |
|---|---|---|---|
| `SQ_INSTS_MFMA` | 57,753,600 | 57,753,600 | **1.000** |
| `SQ_WAVES` | 57,600 | 57,612 | 1.000 |
| `MeanOccupancyPerCU` | 15.27 | 15.36 | **0.994** |
| `SQ_INSTS_VALU` | 82,039,260 | 78,015,300 | 1.052 |
| `SQ_INSTS_LDS` | 20,090,532 | 18,513,300 | 1.085 |
| `SQ_INSTS_VMEM_RD` | 12,084,900 | 11,124,900 | 1.086 |
| `SQ_INSTS_VMEM_WR` | 110,832 | 112,800 | 0.983 |
| `SQ_LDS_IDX_ACTIVE` | 150,938,856 | 148,021,800 | 1.020 |
| `SQ_LDS_BANK_CONFLICT` | 1,804,800 | 902,400 | 2.000 |
| **`GRBM_GUI_ACTIVE`** | 19,626,913 | 14,863,984 | **1.320** |
| **`SQ_WAIT_ANY`** | 640,352,894 | 285,712,477 | **2.241** |
| **`MfmaUtil`** | 58.54 | 76.79 | **0.762** |

读法：**MFMA 条数、wave 数、occupancy 三项完全一致，各类指令数只多 2~9%，
但忙周期多 32%。** `MfmaUtil` 的比值 0.762 和时间比值（2615/3371 = 0.776）几乎相等
——**剩下的差距就是 MFMA 管线的空转率，别的都不是**。

### 5.3 排掉的三个解释

**(a) 不是指令 cache。** 我们的内核 K 循环**完全展开**，2563 条指令（约 20 KB）；
PR 保留真循环，601 条（约 5 KB）。看起来像 icache 问题，实测不是：

| | 我们 f2 | PR gateup |
|---|---|---|
| `SQC_ICACHE_REQ` | 31,021,632 | 31,206,936 |
| `SQC_ICACHE_MISSES` | **240** | **96** |
| `SQ_IFETCH` | 31,021,632 | 31,206,936 |

3100 万次取指里 miss 两位数，两边取指次数还几乎相同。**展开没有带来取指代价。**

**(b) 不是指令发射排队。**

| | 我们 f2 | PR gateup | 比值 |
|---|---|---|---|
| `SQ_WAIT_INST_ANY` | 729,286,515 | 728,849,870 | **1.001** |
| `SQ_WAIT_INST_LDS` | 12,247,617 | 37,982,150 | 0.32 |
| `SQ_ACTIVE_INST_ANY` | 141,703,959 | 138,822,550 | 1.021 |

等**发射**的周期两边一模一样（1.001），PR 等 LDS 发射反而比我们多。
所以多出来的 `SQ_WAIT_ANY` 全是**数据依赖等待**，不是排队。

到这里能确定的是：同样的指令、同样的并行度、同样的取指、同样的发射排队，
我们的 wave 花在**等数据回来**上的时间是 PR 的 2.24 倍。等的是哪一种数据，见 5.4。

**(c) 也不是软件流水线。** 这一条本文先前判错过，记在这里。当时看到我们
**完全展开**（2563 条指令）而 PR 是**真循环**（601 条），就归因到"流水线质量"。
把两边的 ISA 按 `s_barrier` 切成单个 K-tile 段来数，这个解释站不住：

| 每个 K-tile 段 | 我们 f3（展开 31 份） | PR gateup（循环 1 份） |
|---|---|---|
| 段内容 | 32 MFMA / 8 ds_read / 6 buffer_load / 2 ds_write | **完全相同** |
| 整队 `lgkmcnt(0)` 次数 | 30 段是 1 次，1 段是 6 次 | 3 段是 1 次，1 段是 4 次 |
| `s_waitcnt` 条数（中位） | 12 | 13 |
| MFMA 连续段最长（中位） | 4 | 4 |
| MFMA 连续段个数（中位） | 14 | 14 |

**两边的稳态调度基本一样**：都是"取一次内存 + 2~4 条 MFMA"的均匀交错，
都只在段尾整队一次然后 `s_barrier`。展开出来的 31 份也没有互相变坏。
流水线不是差距所在。

### 5.4 真正的差距：权重加载带了 `nt`

把两边稳态段的指令逐条打出来，唯一对不上的是**权重那 4 条 `buffer_load` 的修饰符**：

```asm
; 我们
buffer_load_dwordx4   v[74:77], v34, s[28:31], 0 offen offset:2048 nt
buffer_load_dwordx4   v[62:65], v34, s[28:31], 0 offen offset:3072 nt
; PR
buffer_load_dwordx4   v[62:65], v54, s[12:15], 0 offen
buffer_load_dwordx4   v[58:61], v54, s[12:15], 0 offen offset:1024
```

`nt`（non-temporal）告诉 L2"这份数据不会再用，优先淘汰"。对这个 shape 这是
**恰好相反**的事实：`moe_sorting` 把 4800 个 M-block 分给 193 个专家，
平均每个专家约 **25 个 block**，它们读的是**同一份** 1.5 MB 权重。
带上 `nt`，这 25 次里每一次都得回 HBM。

这就是第六章的 f4，也解释了 5.2 里那些"活一样多却慢 26%"的读数：
等的是 HBM 延迟，不是 LDS，也不是流水线。

> 方法上的教训：`nt` 由 kernel 名里的 `_bnt0` 控制，1.4 的 tile 扫描**扫过它**
> （`t64x64x128_bnt0` = 3936.4，比同档的 4462.0 快 526 us），但当时 `_n16` 单独更快
> （3715.0），我就只留了 `_n16`，**从没试过两个一起开**。逐个 knob 对着 base 比，
> 会漏掉这种"两个都要才最好"的组合——和 stage2 f1 里 sorted 布局与 FASTVALID 的
> 协同是同一类坑。

### 5.5 静态 ISA 上还能直接对上的两处（引出第七章）

静态条数不能跨"展开 vs 循环"直接比，但两处是结构性的，看得出来：

| | 我们 f2 | PR gateup |
|---|---|---|
| epilogue Step 1 写 LDS | **16 × `ds_write_b16_d16_hi`** | **2 × `ds_write2st64_b64`** |
| LDS bank conflict（动态） | 1,804,800 | 902,400 |

Step 1 我们是 16 次 16 位标量写，PR 是 2 次 64 位配对写。原因是累加器朝向：
A-first 下一个 lane 在一个 (mi, ni) 里拿到的 4 个值分属 4 个**不同的行**，
LDS 上不连续，只能一个一个写；B-first 下它拿到的是同一行的 4 个**相邻通道**，
一次 64 位写就够。这也正好解释了 bank conflict 差 2 倍。

这是下一个可做的 feature（stage2 的 f5 就是这么干的），但它要改 MFMA 朝向，
不是一个 knob。

---

## 六、Feature 4：去掉权重加载的 `nt` 标记

**stage1 −657.3 us（累计补上 base→ck 差距的 141%，即已经反超 CK）**

### 6.1 一个后缀的事

```
kernelName1: flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16  ->  ..._n16_bnt0
```

`b_nt` 默认 2（带 `nt`），`_bnt0` 关掉它。没有代码改动，只是 kernel 名多四个字符。

### 6.2 为什么值这么多

`nt` 的语义是"这行数据不会被复用，L2 里优先淘汰"。stage1 的 B 是**专家权重**，
而 MoE 的分块方式决定了它一定会被复用：4800 个 M-block / 193 个专家 ≈
**每个专家 25 个 block 读同一份 1.5 MB 权重**。`nt` 把这 25 次全变成 HBM 访问。

计数器（每 dispatch 平均）：

| counter | 带 `nt`（f3） | 不带（f4） | 变化 |
|---|---|---|---|
| `TCC_HIT_sum` | 11,487,897 | **42,590,780** | **3.71×** |
| `TCC_MISS_sum` | 76,555,901 | 45,451,360 | 0.59× |
| `TCC_EA0_RDREQ_sum`（发往 HBM） | 75,671,729 | 44,567,189 | **0.59×** |
| **L2 命中率** | **13.0%** | **48.3%** | |
| `SQ_WAIT_ANY` | 555,868,639 | 360,376,406 | 0.65× |
| `MfmaUtil` | 59 | **76** | PR 是 76.79 |
| `GRBM_GUI_ACTIVE` | 19,385,059 | **15,018,228** | PR 是 14,863,984 |

L2 命中率从 13% 抬到 48%，发往 HBM 的读请求少了 3100 万次/dispatch。
`MfmaUtil` 从 59 跳到 **76**，正好是 PR 的 76.79；忙周期 15.02M 对 PR 的 14.86M，
**差 1%**。5.2 那张"活一样多却慢 26%"的表，到这里全部对上了。

### 6.3 精度没动

cos 仍然是 0.999995。`nt` 只是缓存替换策略的提示，不改变读到的数据。

---

## 七、Feature 5：B-first 累加器 + lds_out 行填充（**指令对齐了，时间没动**）

**在 f4 之前 stage1 ±0 us；f4 之后值 −2.4 us（填充部分值 14.6 us，见 9.2b）。**
这一章保留下来是因为它当时的**零值本身**排掉了一整类解释——
以及后来它变得不为零，这件事本身又是一课。

### 7.1 做了什么

5.4 指出的两处结构差异，这一节全部消掉：

```bash
FLYDSL_MOE_STAGE1_BFIRST=1    # 交换 MFMA 两个源操作数
FLYDSL_MOE_STAGE1_LDSPAD=8    # lds_out 每行多 8 个元素
```

**B-first**：`v_mfma` 的两个源操作数交换之后结果转置，一个 lane 从"一个通道 × 4 行"
变成"一行 × 4 个连续通道"。于是 Step 1 的写变成连续的：

```aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                                    if _bfirst:
                                        gate_list[acc_idx] = mfma_k64(
                                            gate_list[acc_idx],
                                            b_gate_packs0[ni],
                                            b_gate_packs1[ni],
                                            a0,
                                            a1,
                                        )
```

权重 scale 跟着改成 4 宽加载（4 个通道现在是连续的），Step 1 改成一次 4 元素的向量存。

**LDSPAD**：B-first 之后相邻 lane 写的是**相邻的行**，行距正好 `tile_n*2 = 128` 字节
= LDS 32 个 bank 的整宽，于是 16 个 lane 全落在同一个 bank 上，16 路冲突。
PR 的注释预言了这件事（它用 XOR swizzle 解，我们用填充，效果一样）：

```/data/aiter_pr/aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
            # bank-aligned, so an unswizzled 64-bit store is 16-way
            # bank-conflicted; the swizzle spreads it
```

填充 8 个元素后 `lds_out` 是 `2×64×72 = 9216` 字节，加 `lds_tid` 仍小于 X 区，
所以 f2 的 16384 字节和 4 个 workgroup 都保住了。

### 7.2 指令上完全对齐了

| counter | f3 | f4（无填充） | **f4（填充 8）** | PR gateup |
|---|---|---|---|---|
| `SQ_INSTS_LDS` | 19,188,132 | 18,511,332 | 18,511,332 | **18,513,300** |
| `SQ_LDS_IDX_ACTIVE` | 149,134,056 | 159,962,856 | **148,213,398** | **148,021,800** |
| `SQ_LDS_BANK_CONFLICT` | 1,804,800 | **13,536,000** | **1,786,542** | 902,400 |
| `SQ_INSTS_VALU` | 78,373,260 | 77,696,460 | — | 78,015,300 |
| `SQ_INSTS_VMEM_RD` | 11,238,900 | 11,238,900 | — | 11,124,900 |
| **`GRBM_GUI_ACTIVE`** | 19,279,271 | **18,722,019** | **18,752,972** | 14,863,984 |

Step 1 的 ISA 从 **16 条 `ds_write_b16_d16_hi`** 变成 **4 条 `ds_write_b64`**。
`SQ_INSTS_LDS` 与 PR 的比值变成 **1.000**，`SQ_LDS_IDX_ACTIVE` 也基本相等。

### 7.3 但时间一点没动，这本身是结论（**在当时那个瓶颈下**）

三档时间：f3 = 3315.1，f4 无填充 = 3315.9，f4 填充 8 = 3320.1。**没有收益，甚至略负。**

无填充那一档尤其干净：它把 LDS 指令数降到了 PR 的水平，却同时制造了
**7.5 倍的 bank 冲突**（1.80M → 13.54M），两者相抵，时间不变；
填充之后冲突消掉 **11.7M**、LDS 活跃周期少 **11.7M**，忙周期从
18,722,019 变成 18,752,972——**变化为零**。

当时的结论是：**整个 CShuffle epilogue 不在关键路径上**——
LDS 存储宽度、bank 冲突、LDS 活跃周期三项全部对齐 PR，一微秒都没换回来。
这解释了为什么 f1 值 172 us 而这一章值 0：f1 省的是**全局存储**
（写回路径当时是瓶颈），这一章省的是 **LDS**，而 LDS 这一侧当时有富余。

**但这个结论只在当时那个瓶颈下成立。** f4 去掉 `nt` 之后 MFMA 利用率从 59%
抬到 76%，epilogue 重新进入关键路径，同样的填充变成值 14.6 us（9.2b）。
所以本章的正确读法不是"epilogue 永远不值钱"，而是：

> **一个优化值多少钱，取决于当时哪一段是瓶颈。瓶颈换了以后，
> 先前测出"无效"的东西必须重测。**

这也是当时那句"剩下的 700 us 全部在 K 主循环的软件流水线里"错在哪：
epilogue 不是瓶颈这一点是对的，但由此推断瓶颈在流水线是错的——
它在访存，见 5.4。

---

## 八、汇总

### 8.1 stage1 内核时间（stage2 钉死在未优化档）

这是做归因用的那一组：stage2 全程是 `reduce_persist_bnt0` 且一个 knob 都不开，
所以每一行的变化只能来自 stage1。us，3 次中位数，PTL 关。

| stage | stage1 us | vs base | vs ck | 累计补上 base→ck | 说明 |
|---|---|---|---|---|---|
| `base` | 3714.1 | — | +744.6 | — | 旧 FlyDSL stage1，最好的合法 tile |
| `f1` | 3542.2 | −171.9 | +572.7 | 23.1% | CShuffle 重新可用 + 128 位存储 |
| `f2` | 3375.2 | −338.9 | +405.7 | 45.5% | LDS 16640→16384，3→4 workgroup |
| `f3` | 3317.2 | −396.9 | +347.7 | 53.3% | per-tensor 激活 scale 提到入口 |
| `f4` | **2659.8** | **−1054.3** | **−309.7** | **141.6%** | 去掉权重加载的 `nt`（L2 命中 13%→48%） |
| `f5` | **2657.4** | **−1056.7** | **−312.1** | **141.9%** | B-first + lds_out 填充 |
| `ck` | 2969.5 | −744.6 | — | 100% | CK stage1（生产默认） |
| `target` | 2615.2 | −1098.9 | −354.3 | — | PR3987 flydsl gateup |

五个 feature 都**没有动精度**：七档 cos 全是 0.999995。
**f4 就已经比 CK 快 309.7 us，离 PR gateup 只差 42.2 us（1.6%）。**

### 8.2 e2e：stage1 与 stage2 的优化叠加之后

`./run.sh --stage2-opt` 把 `moe_stage2_opt` 那条梯子跑到头的 knob 集
（它的 f8，同一个 `kernelName2`）加到每一档上，再走一遍同样的 stage1 梯子。
stage2 那边的收益（4962 → 1795 us）不属于本文，这里只看**两级叠加之后 e2e 长什么样**：

| stage | stage1 | stage2 | **e2e** | e2e vs base | 说明 |
|---|---|---|---|---|---|
| `base` | 3713.2 | 1795.1 | **7317.0** | — | 两级都是起点 |
| `f1` | 3546.5 | 1806.7 | 7205.5 | −111.5 | |
| `f2` | 3431.4 | 1825.3 | 7145.9 | −171.2 | |
| `f3` | 3386.4 | 1827.3 | 7112.0 | −205.0 | |
| `f4` | 2654.7 | 1786.0 | **6272.9** | **−1044.1** | |
| `f5` | 2653.7 | 1785.9 | **6270.6** | **−1046.5** | |
| `ck` | 2962.8 | 1785.8 | 6574.6 | −742.4 | CK stage1 + 优化后的 stage2 |

**e2e 7317.0 → 6270.6（−1046.5 us，1.167×）**，并且比"CK stage1 + 同样优化过的
stage2"（6574.6）还快 **304.0 us**。stage1 省下的 1059.5 us 里有 1046.5 落到了 e2e 上，
基本一比一。

> 两组的 stage1 数字有 ±7 us 的漂移（例如 f2 是 3375.2 对 3431.4）。
> 差在哪见 8.3 —— 那不是漂移，是真的。

### 8.3 一个跨内核的副作用：stage1 的 `nt` 也在拖慢 stage2

8.2 的 stage2 那一列不是常数：base 1795.1 → f2 1825.3 → f3 1827.3 → f4 1786.0。
**f2/f3 把 stage2 拖慢了约 30 us**，而 f4 又让它回到 1786。
这一段单独复测过一遍，逐档复现（base 1794.6 / f2 1823.6 / f3 1822.5 / f4 1786.3），
**不是噪声**。

stage1 的 knob 不可能直接改到 stage2 的代码，所以这只能是**通过 L2 传递的**：
f1~f3 阶段权重加载还带着 `nt`，stage1 把 L2 搅得很乱（命中率只有 13%）；
f2 又把 stage1 的 occupancy 从 3 个 workgroup 抬到 4 个，同时在飞的 wave 更多，
搅得更厉害，于是 stage2 启动时能从 L2 拿到的 `a2` 更少。f4 去掉 `nt` 之后
L2 恢复正常，stage2 也跟着回到 1786。

**教训**：逐内核的归因会漏掉这种跨内核的缓存耦合。f2 在 8.1 里值 −167，
在 8.2 里只值 −59.6（因为它同时把 stage2 拖慢了 18），
两个数都对，只是口径不同——所以两张表都得留着。

### 8.4 和 PR3987 整条流水线比

| | 我们 f5 + 优化后的 stage2 | PR3987 流水线 |
|---|---|---|
| stage1 | 2653.7（flydsl，本仓库） | 2615.2（flydsl gateup） |
| stage2 | 1785.9 | 1853.2（pr1x4 down） |
| **e2e** | **6270.6** | **6190.7** |
| **cos** | **0.999995** | **0.997831** |

e2e 差 **79.9 us（1.3%）**，而我们的精度高两个数量级。
stage2 我们反而快 67 us；差距全在 stage1 那 42 us（9.1）加上流水线其余部分的零头。

> PR 的 e2e 里量化和归约的实现与我们不同（它自带 absmax/quant 与 `sorted_sum`），
> 所以这一行是"两条完整流水线"的比较，不是逐算子比较。

---

## 九、还没做的

### 9.1 离 PR gateup 只剩 42 us（1.6%），计数器全部对齐

f5 的完整计数器（每 dispatch 平均），和 PR gateup 逐项对比：

| counter | 我们 f5 | PR gateup | 比值 |
|---|---|---|---|
| `SQ_INSTS_MFMA` | 57,753,600 | 57,753,600 | 1.000 |
| `SQ_WAVES` | 57,600 | 57,612 | 1.000 |
| `MeanOccupancyPerCU` | 15.31 | 15.36 | 0.997 |
| `SQ_INSTS_LDS` | 18,511,332 | 18,513,300 | 1.000 |
| `SQ_LDS_IDX_ACTIVE` | 148,213,398 | 148,021,800 | 1.001 |
| `SQ_INSTS_VALU` | 77,805,780 | 78,015,300 | 0.997 |
| `SQ_INSTS_VMEM_RD` | 11,238,900 | 11,124,900 | 1.010 |
| `SQ_INSTS_VMEM_WR` | 110,832 | 112,800 | 0.983 |
| `SQ_INSTS_SALU` | 3,201,564 | 5,403,936 | 0.592 |
| `TCC_HIT_sum` | 42,539,464 | 42,292,302 | 1.006 |
| `TCC_MISS_sum` | 45,503,444 | 46,070,676 | 0.988 |
| **L2 命中率** | **48.3%** | **47.9%** | 我们略好 |
| `MfmaUtil` | 75.82 | 76.79 | 0.987 |
| `SQ_BUSY_CYCLES` | 59,664,989 | 59,124,892 | 1.009 |
| **`GRBM_GUI_ACTIVE`** | **15,007,104** | **14,863,984** | **1.010** |
| **stage1 us** | **2657.4** | **2615.2** | **1.016** |
| `SQ_LDS_BANK_CONFLICT` | 1,786,542 | 902,400 | **1.980** |
| `MemUnitStalled` | 0.19 | 0.13 | 1.474 |
| `SQ_WAIT_ANY` | 360,167,450 | 285,712,477 | 1.261 |

忙周期只差 **1.0%**，MFMA 利用率差 1.3%，L2 我们还略好一点。
还明显不同的只剩三项，都不足以解释 42 us：

- **bank 冲突仍是 2 倍**（1.79M vs 0.90M）。填充只能把冲突压到这一档，
  PR 用的是 XOR swizzle，摊得更均匀。9.2(b) 的填充扫描显示再加宽填充没有收益，
  所以要再降只能换 swizzle。
- `SQ_WAIT_ANY` 高 26%，但忙周期只高 1%——说明这些等待基本被其它 wave 盖住了。
- `SQ_INSTS_SALU` 我们**少** 40%，不是代价。

**结论：已经收敛。** 42 us（1.6%）没有单独一项能解释，而且两边跑在
**不同的 flydsl 版本**（0.1.2 vs 0.2.4，后端不同）、不同的 host 流水线上，
这个量级接近跨仓库比较能分辨的下限。再往下要靠 ATT 看停顿分布，不要再靠计数器猜。

顺带一提，PR 的 gateup 还多做了一件事：它的 `act_quant_type="ptpc"`，
每个 token 一个激活 scale，比我们的 per-tensor 标量多一次 gather。
它多做这件事还快 1.6%。

### 9.2 复查过、确认没有剩余空间的两件事

**(a) tile 选择在新的访存状态下仍然最优。** §1.4 的扫描是带着 `nt` 做的，
也就是在一个访存挨饿的状态下选出来的，f4 之后前提变了，所以重扫了一遍：

| kernelName1（都带 f1+f2+f3） | stage1 us |
|---|---|
| `t64x32x128_n16_bnt0` | 3779.0 |
| `t64x64x256_n16_bnt0` | 2909.2 |
| `t64x64x128_bnt0`（n32） | 2857.0 |
| **`t64x64x128_n16_bnt0`** | **2657.4** |
| `t64x64x128_n16_bnt0_w2` | 2659.2 |

结论没变，`t64x64x128_n16` 仍是最好的。`tile_k=256` 依然差（LDS 32768 字节，
occupancy 掉到 2 个 workgroup）。

**(b) LDS 填充已经到底。** 但这里有个**状态变化值得记住**：

| bfirst + LDSPAD | 访存受限时（f3 之前） | 现在（f4 之后） |
|---|---|---|
| 0 | 3315.9 | **2672.2** |
| 4 | 3321.4 | **2657.6** |
| 8 | 3318.8 | 2658.3 |
| 16 | — | 2657.8 |

第七章测出填充**一分钱不值**，那是对的——**在当时那个访存受限的状态下**。
f4 把 MFMA 利用率从 59% 抬到 76% 之后，epilogue 重新进入关键路径，
同一个填充变成值 **14.6 us**。填充 4 就到底，再宽没有收益。

**这条是本文最该记住的方法论**：一个优化值多少钱，取决于当时哪一段是瓶颈；
瓶颈换了以后，先前测出"无效"的东西要重测。

### 9.3 少读一遍 A：PR 没做，而且量过之后不值得做

**PR 在这一点上没有做任何事**，和我们一模一样，所以这条不是"追平 PR"，
是"超过 PR"。先把 PR 的做法钉死：

```/data/aiter_pr/aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)      # div_up(384, 128) = 3
        ...
            ).launch(grid=(num_n_blocks, task_num, 1), block=(256, 1, 1), stream=stream)
```

`blk_n = gpu.block_idx.x` 在 gateup 内核里只用于给 B 和输出做下标，
**没有任何 N 方向的循环**；A 由每个 WG 自己 gather 一整块
（`a_idx.copy(buf_cp_atom_r, k_next, a_cp_frag)` 在流水线 stage 里）。
所以 3 个 N-block 各读一遍完整的 A tile。`SQ_INSTS_VMEM_RD` 两边只差 1%，
从计数器侧也印证了这一点。

**但量过之后，这条大概率不值得做。** 我们离带宽瓶颈还很远：

| | 值 |
|---|---|
| HBM 读（`TCC_EA0_RDREQ_sum` × 64B） | 2.86 GB / dispatch |
| 用时 | 2657.4 us |
| **实测读带宽** | **1.07 TB/s** |
| 本卡 HBM 峰值（`vram_max_bandwidth`） | **5325 GB/s** |
| **占峰值** | **20%** |
| `MemUnitStalled` | **0.19%** |

唯一数据只有 438 MB（A 134 MB + 权重 304 MB），实际读了 2.86 GB，放大 6.5 倍
——但**放大本身不疼**，因为带宽只用掉两成，访存单元几乎不停顿。
省掉 A 的 3 遍重读，省的是我们本来就不缺的带宽。

真正的限制是 `MfmaUtil` 76%，而**PR 也是 76.79%**——两边都顶在同一个天花板上，
那不是 tiling 能改的。所以 `tile_n=192` + `LDSCHUNK` 这条先搁置：
它要付出 occupancy（`lds_out` 24576 字节 → 2 个 workgroup，必须配 LDSCHUNK 才不亏），
换回来的却是过剩的带宽。要做也该排在 ATT 之后，先搞清楚 76% 那 24% 空转在等什么。

（f1 的 nlane 搜索本来就是为 `tile_n=192` 留的口子，真要试的话那一侧已经就绪。）

### 9.4 PR3987 的 gateup 在本仓库跑不起来

`target` 那 2609.9 是在 `/data/aiter_pr` 上采的，**不是在本仓库**。
本仓库里 `moe_2stage_gateup_prefill_1x4` 是**死代码**：
commit 6b1cb649 只把 stage2 的 `down_prefill_1x4` 移植到了 flydsl 0.1.2，
gateup 那条路径没有移植，也没有任何 host 侧接线。具体缺的：

- `fx.struct` / `fx.union` / `fx.Array` 在 0.1.2 上不存在，
  gateup 的 `SharedStorage`（`sorted_lds` 与 A ping-pong 的 union）用的就是它们；
- `_gemm_1x4`（约 260 行，A 的 LDS ping-pong + B 的 direct g2r）**只有 gateup 在用**，
  down 那条是另写的，所以这 260 行一行都没移植过，
  而 0.1.2 表示不了 `f8E4M3FNUZ`，整条 fp8 数据通路要按 down 的做法换成 u8/u32 别名；
- `kernelName1` 解析里 `pr1x4` 只对 stage 2 生效，没有 stage1 的 host wrapper。

还有第四小块：`Vec` 在 0.1.x 上是个只会抛异常的占位类，而 gateup 会在
`_silu_pair_bf16` / `_apply_1x4_fp8_dequant` 里用 `Vec.from_elements`。
这一块**很便宜**——0.1.x 的寄存器向量本身就是 `ArithValue`（它被注册成了
VectorType 的 value caster），所以打包标量就是一句：

```python
        @staticmethod
        def from_elements(items, dtype):
            items = list(items)
            return fx.vector.from_elements(T.vec(len(items), dtype.ir_type), items)
```

本轮验证过它能编译，但**没有并入提交**：真正移植之前它没有任何调用方，
而上面那三条才是大头。记在这里省得下次再研究一遍。

**升级 flydsl 到 0.2.x 这条路试过，走不通**：aiter 自己的
`kernels/splitk_hgemm.py` 用了 0.1.x 才有的 `flydsl.compiler.protocol.fly_values`，
0.2.0 和 0.2.4 都没有这个符号，整个 `aiter.ops.flydsl` 起不来
（`aiter/ops/flydsl/__init__.py` 里还有一道 `0.1.2` 的硬版本校验，
绕过它之后就撞上上面这个 ImportError）。

所以要在本仓库拿到 target，得做一次和 6b1cb649 同等规模的移植。

---

## 十、怎么复现

```bash
cd /data/aiter/moe_stage1_opt

./run.sh --list                    # 看每个 stage 是什么
./run.sh --repeats 3               # 8.1 那一组：stage2 钉死，量 stage1
./run.sh --repeats 3 --stage2-opt  # 8.2 那一组：stage2 也优化到头，量 e2e
./run.sh base f4                   # 只跑这两个
./run.sh --repeats 1 --counters    # 加一趟 rocprofv3 计数器

GPU=5 ./run.sh                     # 换卡（默认 4；0-3 是别人的）
```

结果追加进 `results/ladder.csv`（每次调用一个 session id，`mode` 列区分
`s2base` / `s2opt`），计数器进 `results/counters.csv`。
（`results/stage1.csv` 是加 `stage2_us` / `mode` 两列之前的旧 schema，留着不再写入。）
每行同时记 `stage1_us` / `stage2_us` / `e2e_us`，8.3 那个跨内核效应就是这么看出来的。

单独试一个 kernel 名不进 ladder：

```bash
./probe.sh flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16 flydsl_moe1_afp8_wfp8_bf16_t64x64x256
```

看 ISA：

```bash
FLYDSL_MOE_STAGE1_NLANE_FIT=1 FLYDSL_MOE_STAGE1_EVEC=8 FLYDSL_MOE_STAGE1_LDSTIGHT=1 \
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa AITER_CONFIG_FMOE=/tmp/s1.csv HIP_VISIBLE_DEVICES=4 \
  python test_qmoe_multi.py --token 32768 --model-dim 4096 --inter-dim 192 \
    --expert 193 --topk 9 --activation silu --dtype bf16 --use-g1u1 1 \
    --doweight-stage1 0 --quant fp8 --quant-type per_tensor
# 产物: /tmp/isa/moe_gemm1_0/17_final_isa.s
grep -E "\.group_segment_fixed_size|\.vgpr_count" /tmp/isa/moe_gemm1_0/17_final_isa.s
```

### 本轮改到的代码

| 文件 | 改了什么 |
|---|---|
| `aiter/ops/flydsl/kernels/moe_gemm_2stage.py` | f1 的 CShuffle 几何搜索、f2 的 LDS 布局与 `lds_tid` 填充时机、f3 的 `scalar_a_scale`、f4 的 B-first MFMA 与 `lds_out` 行填充；四个都进了 module cache key |
| `aiter/ops/flydsl/moe_kernels.py` | `_stage1_cshuffle_default()`（开了 NLANE_FIT 就不再因 `tile_n % 128` 关掉 CShuffle）+ f3 的 host 侧不再展开 per-tensor scale，并校验它确实是 per-tensor |
| `aiter/ops/flydsl/kernels/mfma_epilogues.py` | `mfma_epilog` 透传 `bfirst` / `lds_out_stride`（原本只有直接调 `c_shuffle_epilog` 的 stage2 用得上） |

`moe_gemm_2stage_gfx942.py` 本轮**没有改动**：那个文件只有 pr1x4 stage2 会 import，
本文的测量一次都没碰到它。（曾经在里面补过一个 `Vec.from_elements` shim，
因为无调用方已撤销，做法记在 9.4。）

六个 knob 默认全关，不开就是改动前的行为（`base` 实测 3714.1，与改动前逐位一致）。
**要复现最终结果，全套是这样：**

```bash
# kernel 名（f4 在这里，是最值钱的一个 feature，却只是四个字符）
#   kernelName1 = flydsl_moe1_afp8_wfp8_bf16_t64x64x128_n16_bnt0

export FLYDSL_MOE_STAGE1_NLANE_FIT=1      # f1  −171.9
export FLYDSL_MOE_STAGE1_EVEC=8           # f1
export FLYDSL_MOE_STAGE1_LDSTIGHT=1       # f2  −167.0
export FLYDSL_MOE_STAGE1_SCALAR_ASCALE=1  # f3  −58.0（只在 per-tensor 量化下可用，否则报错）
export FLYDSL_MOE_STAGE1_BFIRST=1         # f5  −2.4（要配 CShuffle）
export FLYDSL_MOE_STAGE1_LDSPAD=8         # f5（B-first 必配，否则 16 路 bank 冲突）
```

只想要 95% 的收益又不想碰任何 knob 的话：**光把 kernel 名换成 `_bnt0`
就值 657 us**（f4 相对 f3），其余四个 feature 加起来 399 us。

### 复现 PR gateup（`/data/aiter_pr`，自带 flydsl 0.2.4 的 venv）

```bash
cd /data/aiter_pr && env AITER_LOG_MORE=1 AITER_USE_SYSTEM_TRITON=1 \
  PYTHONPATH=/data/aiter_pr \
  AITER_CONFIG_FMOE=/data/aiter_pr/pr_gfx942_tuned_fmoe.csv HIP_VISIBLE_DEVICES=4 \
  /data/pr_env/bin/python test_qmoe_multi.py --token 32768 --model-dim 4096 \
    --inter-dim 192 --expert 193 --topk 9 --activation silu --dtype bf16 \
    --use-g1u1 1 --doweight-stage1 0 --quant fp8 --quant-type per_tensor
```

`AITER_USE_SYSTEM_TRITON=1` 是必须的：那个仓库的 gluon 内核要求 triton>=3.6.0，
环境里是 3.5.1，不设这个变量连 `import aiter` 都过不去。
