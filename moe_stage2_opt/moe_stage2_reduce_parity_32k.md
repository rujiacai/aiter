# 旧 stage2 内核追赶 pr1x4：逐 feature 优化记录（token=32768）

> shape: token=32768, model_dim=4096, inter_dim=192, expert=193, topk=9, bf16 输出，
> fp8(e4m3fnuz) per_tensor 权重与激活
> 硬件: MI308X（gfx942:sramecc+:xnack-，80 CU），HIP 7.2.53211，torch 2.9.1+rocm7.2.3
> 软件: aiter `moe_opt_0727` @ 6b1cb649；flydsl 0.1.2
> 驱动: `moe_stage2_opt/run.sh`，数据落在 `moe_stage2_opt/results/`

本文是一份**进行中的优化记录**，不是一次性的对比报告。目标是在**旧内核**上一个 feature
一个 feature 地加，直到追平新内核。

| 章 | 内容 |
|---|---|
| 一 | **术语**（partial / sorted 行号 / padding 行 / epilogue 等）、测试怎么做、起点和目标 |
| **二 ~ 九** | **每章一个 feature（f1 ~ f8）**：做了什么、怎么写的、值多少、为什么 |
| 十 | 汇总表，e2e 和逐算子两个口径，每加一个 feature 一行 |
| 十一 | 怎么复现 |
| 十二 | **优化方法论**：怎么定位下一个瓶颈，以及踩过的测量坑 |

八个 feature 章的写法是统一的：**先讲优化前卡在哪（附源码）**，再讲**改法（附源码）**，
然后是 ISA / 计数器验证和 e2e 效果。只想知道某个 feature 干了什么，看每章的前两节就够。

第十二章是从实战里抽出来的**通用技巧**，和某一个具体 feature 无关：它记录 f4 之后
"下一步该优化什么"是怎么一步步定位出来的（指令分类对比 → ISA 定位 → 候选筛选），
以及几个反复踩到的测量陷阱。**要开新一轮优化，先读那一章。**
它当时的几处判断被后来的实测修正了，正文都标了出来——保留下来是因为**判断错在哪**
比结论本身更有参考价值。

---

## 一、测试情况与基线

### 1.0 术语

全文反复出现、且容易望文生义的几个词，先钉死。以本文的 shape 为例
（token=32768，topk=9，expert=193，model_dim=4096，block_m=64）：

**partial（部分和）**
　MoE 里一个 token 被路由到 9 个专家，**每个专家都算出一份完整的 4096 维结果**，
　最终输出是这 9 份的加权和。而 stage2 的 GEMM 一次只算一个 `(token, 专家)` 组合，
　产出的是没求和的半成品——这就是 partial。它需要一个中间缓冲存下全部
　`32768 × 9 = 294,912` 份，之后再归约。**「partial 缓冲」就是这个中间缓冲，
　每行是一个 `(token, 槽)` 的 4096 维结果。**

**归约（reduce）**
　把每个 token 的 9 份 partial 加起来，得到最终的 `32768 × 4096` 输出。
　它是 stage2 之后一个独立的 kernel。所谓 **reduce 模式**就是「先写 partial、再归约」；
　对应的 **atomic 模式**是让 stage2 直接原子累加到输出上，不要中间缓冲（见 1.1
　为什么本文选 reduce）。

**moe_sorting / sorted 行号**
　GEMM 要按专家分块（一个 workgroup 只能用一份权重），所以有个前置步骤把全部
　294,912 个 `(token, 槽)` 对**按专家号排序**，产出 `sorted_token_ids`。排完之后
　同一个专家的行全连在一起，一个 workgroup 就能处理连续的 64 行、共用一份权重。
　**「sorted 行号」= 某个 `(token, 槽)` 对在这个排序后数组里的下标。**

**padding 行 / 哨兵**
　排序时每个专家的行数要补齐到 `block_m = 64` 的倍数，补出来的行填哨兵值。
　它们不对应任何真实的 `(token, 槽)`，**写出去会污染真实数据**——除非能证明没人会读回
　（这正是 Feature 1 做的事）。`num_valid_ids` 记录真实行数。

**epilogue**
　GEMM 主循环结束、结果还在累加器里之后，到写出全局内存之间的那一段：缩放、类型转换、
　算地址、（可能的）LDS 重排、存储。本文八个 feature 里有六个动的是这一段。

**累加器布局：`mi` / `ii` / `ni` / `lane`**
　一条 `v_mfma_f32_16x16x32` 只产出 `16(M) × 16(N)`，所以本文的
　`tile_m=64 × n_per_wave=32` 要拆成 `m_repeat = 64/16 = 4` 个行块 ×
　`num_acc_n = 32/16 = 2` 个列块 = **8 个累加器，每个是一个 `f32x4`**。
　定位一个输出元素要四个下标：

```
行 = mi*16 + (lane/16)*4 + ii          mi ∈ [0,4)   ii ∈ [0,4)
列 = wave_n_id*32 + ni*16 + lane%16    ni ∈ [0,2)   lane%16 ∈ [0,16)
```

　`m_repeat`/`num_acc_n` 分别是行、列方向的块数；`mi`/`ni` 是块号；
　`ii` 和 `lane` 在块内定位。每线程每 N-tile 处理 `4 × 4 × 2 = 32` 个元素，
　乘以 256 线程正好 `64 × 128 = 8192`，一个 tile。

　**要记住的一条：`ii` 走行、`ni` 走列——一个 lane 手上那 4 个 `f32` 是
　「同一列的 4 个连续行」。** 这条几何贯穿全文：f2 只能向量化算术却动不了 LDS 写、
　f5 把朝向翻过来之后那 4 个值才变成 4 个连续通道，根都在这里。

**CShuffle**
　epilogue 里的一次 LDS 中转：先把累加器按一种线程映射写进 LDS（Step 1），barrier，
　再按另一种映射读回来（Step 2）。目的是让最终的全局写是合并的。见第六、九章。
　**注意这两步是被 barrier 隔开的两段代码**（`write_row_to_lds` / `store_pair`），
　第三章那三个 knob 就分处两边。

**N-tile / persist**
　输出的 4096 维被切成 32 个 `tile_n = 128` 的块，逐块处理，每块叫一个 N-tile。
　`persist` 是指一个 workgroup 把这 32 个 N-tile 全包了（而不是每块一个 workgroup），
　好处是 X 只需搬进 LDS 一次。本文的内核名带 `persist` 就是这个模式。

### 1.1 起点和目标

两端都是 `fused_moe` 的完整 e2e，stage1 和 moe_sorting 完全相同
（CK `256x64x64x128`、`block_m = block_m2 = 64`），只有 stage2 不同：

| | stage2 内核 | e2e |
|---|---|---|
| **起点 `base`** | 旧 `flydsl_moe2_..._t64x128x64_reduce_persist_bnt0`，未做任何改动 | **7859.3 us** |
| **目标 `target`** | 新 `flydsl_moe2_..._t64x128x64_pr1x4_bnt0` + Triton 归约 | **6220.1 us** |
| | | **差距 1639.2 us（1.264×）** |

起点选 `reduce` 而不是 `atomic`，是因为新内核结构上就不做原子累加——它每个 sorted slot
独占一行写进 padded 缓冲，再单独归约。选 reduce 才有共同的比较基础。这个选择本身几乎不花钱：
同 tile 下 atomic 与 reduce 的 stage2 总代价（GEMM + 归约 + 输出预清零）差在 ±100 us 以内。

### 1.2 方法

- **e2e**：`test_qmoe_multi.py` 的 `[PERF] e2e fused_moe`，无 profiler。每个 stage 跑 3 次取中位数，
  组内全距 <0.3%。
- **逐算子**：`AITER_LOG_MORE=1` 让 `aiter/test_common.py:391` 打出 ROCTracer 的 kernel 表，
  取 `device_time_avg`（单次 us）。这一遍**单独跑**，因为 tracer 会把 e2e 抬高约 0.6%，
  headline 数字必须来自干净的运行。
- **变量控制**：所有 env knob 每次运行前全部 `unset` 再按 stage 重新设置。knob 串了不会报错，
  只会给出一个悄悄错掉的数字。
- **防呆**：每次运行都核对 `[aiter] [fused_moe] using 2stage (... kernelName2='...')`。
  配置里的 kernel 名写错时 fused_moe 会**静默回落**到别的配置，这是唯一能发现的途径。
- **PTL 必须确认是开的**，见下。

#### PTL：比其它所有噪声大一个数量级的坑

**PTL（Peak TOPS Limiter）是 MI300 系列的一个算力上限控制器，关掉之后整机慢约 22%。**
它是机器级设置、重启会丢，而且**没有任何征兆**——kernel 名对、`cos` 对、组内全距照样 <0.1%，
只有绝对值整体抬高。本文有一整批数据是这么废掉的。

```bash
# 采之前先查，8 张卡应该都是 True（run.sh 现在会自动查，关着直接拒绝跑）
python3 -c "
from amdsmi import *
amdsmi_init()
for i,h in enumerate(amdsmi_get_processor_handles()):
    print(i, amdsmi_get_gpu_ptl_state(h), amdsmi_get_gpu_ptl_formats(h))
amdsmi_shut_down()"

# 关着的话打开（整机设置，会影响这台机器上所有人）
amd-smi set -g all --ptl-status 1
amd-smi set -g all --ptl-format VECTOR,F8
```

名字反直觉：PTL 是"限制器"，但**关掉反而更慢**。按 AMD SMI 文档，启用 PTL 时要指定两种数据
格式，*被指定的格式拿到 peak performance*；关掉时没有任何格式被标记，整机跑保守档位。
本 case 两个 GEMM 都是 fp8 MFMA，所以格式对里要有 `F8`。

本文所有数字采于 `formats=(6, 5)`。这个 API 返回的是枚举下标，按 `amd-smi set -h` 列出的
顺序 `I8, F16, BF16, F32, F64, F8, VECTOR` 解，就是 **VECTOR + F8**。
换格式对没有测过，`F8,BF16` 会不会更好是个开放问题。

实测同一份代码在两种状态下的差别，以及它对**归因**的影响：

| | base | f1 | f2 | target | f2 已补上 | 剩余 |
|---|---|---|---|---|---|---|
| PTL 开 | 7828.8 | 7005.8 | 6788.6 | 6194.3 | **63.6%** | 594.3 |
| PTL 关 | 9673.1 | 8347.4 | 8002.8 | 6689.8 | **56.0%** | 1313.0 |

**不是乘个系数就能换算**——四档慢的比例不同（+23% / +19% / +18% / +8%），
所以结论也跟着变。两种状态的数据绝对不能混。

好在**指令数计数器不受影响**：`SQ_INSTS_*` 在两批采集里逐位相同（3.4）。
所以 PTL 出问题时，指令口径的分析仍然有效，只有时间口径要重来。

### 1.3 基线逐算子分解

`base`，`AITER_LOG_MORE=1`，us/次：

| kernel | us | 占比 |
|---|---|---|
| `moe_gemm2_0`（stage2 GEMM） | **3801.8** | 48.0% |
| `ck::kernel_moe_gemm`（stage1） | 2425.1 | 30.6% |
| `_topk_sum_kernel`（归约） | 729.9 | 9.2% |
| `_quant_from_per_tensor_amax_kernel` | 464.3 | 5.9% |
| 其余（量化、sorting、elementwise） | 494.0 | 6.2% |
| **合计** | **7915.1** | |

stage2 GEMM 一个就占了一半，所以优化都围绕它。

---

## 二、Feature 1：partial 存储从 (token, slot) 改成 sorted 行序

**e2e −824.2 us（补上总差距的 50.3%）；stage2 GEMM −957.4 us**

### 2.0 一图看懂

用一个能手算验证的小例子：**4 个 token、topk=2、3 个专家、`block_m=4`**。
路由是 `t0→(e0,e1)`、`t1→(e0,e2)`、`t2→(e1,e2)`、`t3→(e1,e2)`。

`moe_sorting` 把 8 个 `(token, 槽)` 对按专家号排好，每个专家补齐到 4 的倍数，得到 12 行：

```
sorted 行号   |  0    1    2    3    4    5    6    7    8    9   10   11
这一行是谁    |t0s0 t1s0  PAD  PAD t0s1 t2s0 t3s0  PAD t1s1 t2s1 t3s1  PAD
属于哪个专家  | e0   e0   e0   e0   e1   e1   e1   e1   e2   e2   e2   e2
谁来算        | WG0  WG0  WG0  WG0  WG1  WG1  WG1  WG1  WG2  WG2  WG2  WG2
```

一个 workgroup 负责连续 4 行、共用一份专家权重。**PAD 是补出来的哨兵行，不对应任何真实 token。**

#### 改之前：partial 按 `(token, 槽)` 排，8 行

行号 = `t * topk + s`。看最后一行——**同一个 WG 的输出跳着落在整个缓冲里**：

```
partial 行号 |  0    1    2    3    4    5    6    7
这一行是谁   |t0s0 t0s1 t1s0 t1s1 t2s0 t2s1 t3s0 t3s1
谁来写       | WG0  WG1  WG0  WG2  WG1  WG2  WG1  WG2
                ↑          ↑
                └─ WG0 只写 0 和 2，中间隔着 WG1 的行
```

WG0 手上是 sorted 行 0~3，落到 partial 却是 {0, 2}；WG1 是 {1, 4, 6}；WG2 是 {3, 5, 7}。
**行号跟 workgroup 的下标毫无关系，只能一行一行查出来。** PAD 行则被谓词挡掉、不写。

#### 改之后：partial 按 sorted 行号排，12 行

行号 = sorted 行号 = `bx_m + row_in_tile`，就是 workgroup 和线程的下标本身：

```
partial 行号 |  0    1    2    3    4    5    6    7    8    9   10   11
谁来写       | WG0  WG0  WG0  WG0  WG1  WG1  WG1  WG1  WG2  WG2  WG2  WG2
是否 PAD     |  -    -   PAD  PAD   -    -    -   PAD   -    -    -   PAD
```

**每个 WG 写一段连续区间，起点就是它自己的 `bx_m`。** 代价是缓冲从 8 行涨到 12 行
（多出来的就是 PAD 行），且 PAD 行也照写——写进去的垃圾没人读，理由见下。

#### 代价挪到了归约那头

行号换了编码，归约就得换取法：

```
改之前   out[t] = partial[t*topk + 0] + partial[t*topk + 1]      每 token 两行相邻，连续相加
改之后   out[t] = partial[loc[t][0]] + partial[loc[t][1]]        按反查表 gather

反查表 loc[token][槽]（host 侧新增一个 Triton kernel 把 sorted_token_ids 反过来建出）
         t0: [0, 4]      t1: [1, 8]      t2: [5, 9]      t3: [6, 10]
                 ↑ 就是上图 sorted 数组里 (t0,s1) 所在的行号
```

PAD 行（2、3、7、11）**不在 loc 表里**，所以永远读不到——这就是"写进去无害"的依据。

#### 真正省下来的是一条依赖链

上面是布局，下面才是性能的来源。旧路径里 store 的**地址和谓词都吊在同一个 LDS 读上**：

```
改之前
              ┌─→ 解码 t, s ─→ 地址 = t*topk + s ────┐
  LDS 读 ─────┤                                      ├─→ store（必须等 LDS 返回）
  (打包 id)   └─→ 谓词 = row<num_valid && t<T && s<K ┘

改之后
  bx_m + row_in_tile ─→ 地址 ─→ store（不等任何东西）
                       谓词：无
  LDS 读 ─→ 没有任何消费者 ─→ 被编译器整个删掉
```

代码上就是这一行的差别（完整上下文见 2.2）：

```python
# 改之前：先读 LDS 解出 (t, s)，再拼地址
fused2 = memref.load(lds_tid, [row_local])       # ← store 等的就是它
t = fused2 & mask24_i32
s = fused2 >> 24
row_byte_base = out_base_idx + (t * topk + s) * (model_dim * out_elem_bytes)

# 改之后：行号直接是下标
row_byte_base = out_base_idx + row * (model_dim * out_elem_bytes)   # row = bx_m + row_in_tile
```

两个 knob 正好各砍掉一个消费者，**必须一起开**——只砍一个，另一个还吊着那条 LDS 读：

| knob | 砍掉的消费者 |
|---|---|
| `AITER_FLYDSL_STAGE2_SORTED_PARTIAL=1` | 地址 |
| `FLYDSL_MOE_STAGE2_FASTVALID=1` | 谓词 |

### 2.1 优化前：每一行输出都被同一个 LDS 读卡住

旧 epilogue 里，每个输出行要先从 LDS 取回 `moe_sorting` 打包的 sorted id，
再从里面解出 token 号 `t` 和 topk 槽号 `s`：

```4858:4861:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                    def precompute_row(*, row_local, row):
                        fused2 = memref.load(lds_tid, [row_local])
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
```

解出来的 `(t, s)` 有**两个下游消费者**，这一点是这个 feature 的全部关键：

**消费者一：算存储地址。** partial 缓冲的布局是 `(token, topk, model_dim)`，
所以行号必须是 `t*topk + s`：

```4902:4905:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                            else:
                                row_byte_base = out_base_idx + ts_idx * fx.Index(
                                    model_dim * out_elem_bytes
                                )
```

**消费者二：算存储谓词。** `moe_sorting` 把每个专家补齐到 `block_m` 的倍数，
补出来的行填的是哨兵值，写出去会污染真实数据，所以每行要查三个条件：

```4862:4873:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        if _epi["masked"]:
                            row_i32 = arith.index_cast(T.i32, row)
                            row_valid0 = arith.cmpi(
                                arith.CmpIPredicate.ult, row_i32, num_valid_i32
                            )
                            t_ok = arith.cmpi(
                                arith.CmpIPredicate.ult, t, tokens_i32
                            )
                            s_ok = arith.cmpi(
                                arith.CmpIPredicate.ult, s, topk_i32_v
                            )
                            row_valid = row_valid0 & t_ok & s_ok
```

结果是：**地址和谓词都挂在同一个 LDS 读的依赖链上**，存储必须等它。
ISA 上体现为 288 条 `s_cbranch_execz`（32 个 N-tile × 9 条）加逐行的 cmp/cndmask。

### 2.2 优化后是怎么写的

两个 knob 各砍掉一个消费者，**必须一起开**：

```bash
AITER_FLYDSL_STAGE2_SORTED_PARTIAL=1    # 砍掉消费者一
FLYDSL_MOE_STAGE2_FASTVALID=1           # 砍掉消费者二
```

#### (a) 地址不再依赖 `(t, s)`

partial 缓冲改成**一行一个 sorted slot**，行号直接就是 sorted 行号 `row`，
而 `row = bx_m + row_in_tile` 纯粹来自 block/线程下标，**不需要读 LDS**：

```4894:4901:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                            elif _SORTED_PARTIAL:
                                # Partial rows keep sorted order, so a workgroup's stores
                                # cover one contiguous row range instead of scattering over
                                # the whole buffer; the reduce gathers via the inverted
                                # index (see _fused_post.build_sorted_partial_index).
                                row_byte_base = out_base_idx + row * fx.Index(
                                    model_dim * out_elem_bytes
                                )
```

host 侧要跟着换三样东西。缓冲按 sorted 行数（含 padding）而不是 `token*topk` 开，
并额外建一张反查表：

```2212:2229:aiter/ops/flydsl/moe_kernels.py
    if not accumulate:
        if _stage2_sorted_partial():
            # Sorted-row layout: one partial row per sorted (token, slot) slot,
            # including moe_sorting's per-expert padding, so the buffer is sized by
            # sorted rows rather than token_num*topk.
            from aiter.ops.flydsl._fused_post import build_sorted_partial_index

            target = torch.empty(
                (sorted_token_ids.numel() * model_dim,),
                device=out.device,
                dtype=out.dtype,
            )
            _sorted_partial_loc = build_sorted_partial_index(
                sorted_token_ids,
                num_valid_ids,
                token_num=token_num,
                topk=topk,
            )
```

归约也跟着从"每 token 的 9 行连续相加"换成"按反查表 gather 再相加"：

```2309:2321:aiter/ops/flydsl/moe_kernels.py
        if _sorted_partial_loc is not None:
            # Sorted-row partials are not contiguous per token; gather through the
            # inverted index instead of the plain topk-slab sum.
            from aiter.ops.flydsl._fused_post import fused_topk_sum_gather

            fused_topk_sum_gather(
                out,
                target,
                _sorted_partial_loc,
                token_num=token_num,
                topk=topk,
                model_dim=model_dim,
            )
```

这也正是新内核的布局——它每个 block 直接写 `p_output + e_idx * BLOCK_M * N`，没得选；
`AITER_PR1X4_TRITON_REDUCE=1` 之后两边用的是**同一个** `_topk_sum_gather_kernel`。

#### (b) 谓词整个消失

padding 行写进 partial 缓冲是**无害的**——归约按反查表取，根本不会碰到那些行。
所以掩码可以整个去掉：

```4874:4876:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        else:
                            # fast-valid block: every row stores unconditionally.
                            row_valid = None
```

这不是精度换性能。编译期把两条 epilogue 都编进去，运行时选一条：

```5014:5026:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                    elif _fast_valid_block:
                        # blk_all_valid is uniform across the workgroup (depends only on
                        # bx_m). moe_sorting pads each expert to a tile_m multiple, so
                        # sentinel padding only ever occupies a block's tail rows. Hence
                        # if the block's LAST row is a real (token, slot) pair, every row
                        # in the block is real and we can run a masking-free epilogue.
                        # ... (hoist cache lookup)
                        if blk_all_valid is None:
                            last_row_idx = bx_m + fx.Index(tile_m - 1)
```

因为 padding 只会出现在一个 block 的尾部，**最后一行真实就说明整块真实**。
cos 与掩码版逐位一致（两边都是 0.999995）。

#### (c) 两个一起开之后，那个 LDS 读没有消费者了

`(t, s)` 的两个下游都没了。`precompute_row` 返回的 `fused2` 变成死值
（`store_pair` 只用 `row_byte_base`），`write_row_to_lds` 里那份也是死的
——bf16 输出下 `sx` 是常量 1.0，本来就不读 `(t, s)`。**整条 LDS 读被 DCE 掉，
存储不再等任何东西。**

这就是为什么这两个 knob 是**一个 feature 而不是两个**：只砍掉一个消费者，
另一个还吊着那条 LDS 读，存储照样等。

### 2.3 e2e 上的 2×2 析因

每格 5~6 次取中位数。**这四格是一批独立的手工运行**（D 格等价于 f1，7085.5 与第十章那一
session 的 7035.1 不同批），只在本节内部可比：

| | 掩码开（默认） | `FASTVALID=1` |
|---|---|---|
| **ts 布局**（默认） | A **7881.8** | C **7835.8**（−45.9） |
| **sorted 布局** | B **7552.5**（−329.3） | D **7085.5**（−796.3） |

```
单独效应之和   −329.3 + −45.9 = −375.2 us
实际联合效应                    −796.3 us
交互项（协同）                  −421.0 us
```

条件效应更直观：

| 加 `FASTVALID` 的收益 | |
|---|---|
| 在 ts 布局下 | −45.9 us（全距 33~39 us，就是噪声） |
| 在 sorted 布局下 | **−467.0 us（10.2 倍）** |

协同项比两个单独效应加起来还大。拆开做哪一半都拿不到大头。

> 代码里 `FASTVALID` 那段注释说"这个 shape 上 epilogue VALU 被访存停顿掩盖，没有收益，默认 OFF"
> （`moe_gemm_2stage.py:2404-2412`）——在它被测量的 ts 布局下这是**对的**，−45.9 us 确实是噪声。
> 它只是没预料到换布局之后前提就不成立了。该改的不是注释，是让默认值跟着布局走。

### 2.4 profiler 证据

#### 2.4.0 这些数据是怎么来的

四格（A/B/C/D）都是**实际跑出来的**，不是推算的。每格两类数据：

**(i) 硬件计数器**，`rocprofv3 --pmc`，只采 `moe_gemm2_0` 这一个 kernel。
18 个计数器一趟装不下，分三组跑三趟；`--warmup 1 --iters 3` 是为了压缩采集时间
（计数器采集会把每次 dispatch 拖慢很多，但计数值本身不受影响）。
表里的数值是**每次 dispatch 的平均**（总值除以 dispatch 数，四格都是 6 次）。

```bash
# 以 C 格（只开 FASTVALID）的第一组计数器为例，其余同理
FLYDSL_MOE_STAGE2_FASTVALID=1 \
AITER_CONFIG_FMOE=moe_stage2_opt/configs/old.csv HIP_VISIBLE_DEVICES=0 \
rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_VMEM_WR SQ_INSTS_VMEM_RD \
                SQ_INSTS_LDS SQ_INSTS_SALU \
  --kernel-include-regex moe_gemm2_0 -d /tmp/pmc/C_g1 -o r -- \
  python test_qmoe_multi.py --token 32768 --model-dim 4096 --inter-dim 192 \
    --expert 193 --topk 9 --activation silu --dtype bf16 --use-g1u1 1 \
    --doweight-stage1 0 --quant fp8 --quant-type per_tensor \
    --warmup 1 --iters 3 --run perf
```

输出是 SQLite（`rocpd` 格式），计数值在 `rocpd_pmc_event` 表里，
名字要 join `rocpd_info_pmc`。

**(ii) 最终 ISA**，`FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=...`，
产物在 `<dir>/moe_gemm2_0/17_final_isa.s`。用来解释计数器为什么会那样动。

```bash
FLYDSL_MOE_STAGE2_FASTVALID=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa_C \
AITER_CONFIG_FMOE=moe_stage2_opt/configs/old.csv HIP_VISIBLE_DEVICES=0 \
  python test_qmoe_multi.py ...同上
```

> `rocprof-compute` 3.4.0 在这台机器上跑不出 join 后的数据（v3→v2 csv 转换报
> `Agent_Id` 类型不匹配，最终 `[join_prof] No data available`），所以直接用 rocprofv3 采原始计数器。
> 缺的十几个 python 依赖补齐后问题仍在转换逻辑里，没有继续追。

#### 完全没变的量：工作量和访存流量

| counter | A base | B sorted | C fastvalid | D 两个都开 |
|---|---|---|---|---|
| `SQ_INSTS_MFMA` | 28,876,800 | 28,876,800 | 28,876,800 | 28,876,800 |
| `SQ_INSTS_VMEM_WR` | 9,440,448 | 9,440,448 | 9,440,448 | 9,440,448 |
| `SQ_WAVES` | 19,204 | 19,204 | 19,204 | 19,204 |
| `SQ_LDS_BANK_CONFLICT` | 38,502,400 | 38,502,400 | 38,502,400 | 38,502,400 |
| `TCP_TCC_WRITE_REQ_sum` | 37,748,736 | 37,748,736 | 37,748,736 | 37,748,736 |
| `TCC_EA0_WRREQ_sum` | 37,748,744 | 37,748,736 | 37,748,742 | 37,748,737 |
| `TCC_EA0_WRREQ_64B_sum` | 37,748,744 | 37,748,736 | 37,748,742 | 37,748,737 |
| `TCC_HIT_sum` | 59,836,201 | 57,634,786 | 59,625,982 | 58,770,200 |
| `TCC_MISS_sum` | 24,576,364 | 24,925,816 | 24,618,662 | 24,686,614 |

这张表是**否定性证据**，而且很硬：

1. **MFMA / wave 数 / LDS bank conflict 一模一样** —— GEMM 本体一个字节没动，
   全部变化都发生在 epilogue。
2. **`TCC_EA0_WRREQ_64B_sum` 恒等于 `TCC_EA0_WRREQ_sum`，四格全是 100% 64B 满请求**，
   而且 `TCP_TCC_WRITE_REQ_sum` 四格完全相同。
   **写侧的合并度根本没变。** L2 命中率也只动了 <4%。

所以"散射写变连续写、访存变快"这个直觉解释是**错的**——一行 `model_dim=4096` 的 bf16
本来就由相邻 lane 连续写出，换的只是写到哪一行，请求粒度不受影响。收益不在访存侧。

#### 真正变了的量：epilogue 的指令和等待

| counter | A base | B sorted | C fastvalid | D 两个都开 |
|---|---|---|---|---|
| `GRBM_GUI_ACTIVE`（GPU 忙周期） | 27,956,949 | 24,548,091 | 27,049,695 | **20,240,627** |
| `SQ_WAIT_ANY` | 396,937,447 | 376,188,429 | 320,181,995 | **256,115,897** |
| `SQ_INSTS_VALU` | 247,092,896 | 224,841,708 | 175,869,388 | **162,007,504** |
| `SQ_INSTS_SALU` | 40,765,324 | 40,895,308 | 16,442,620 | 23,591,212 |
| `SQ_INSTS_LDS` | 38,616,624 | **38,616,624** | 36,305,968 | **33,995,312** |
| `SQ_INSTS_VMEM_RD` | 24,910,404 | 24,910,404 | 25,512,004 | 25,512,004 |

归一到 A：

| counter | B sorted | C fastvalid | D 两个都开 | 单独效应之和 | 协同 |
|---|---|---|---|---|---|
| `GRBM_GUI_ACTIVE` | 0.878 | 0.968 | **0.724** | 0.846 | **−0.122** |
| `SQ_WAIT_ANY` | 0.948 | 0.807 | **0.645** | 0.755 | +0.110 |
| `SQ_INSTS_VALU` | 0.910 | 0.712 | **0.656** | 0.622 | +0.034 |
| `SQ_INSTS_LDS` | **1.000** | 0.940 | **0.880** | 0.940 | −0.060 |

`GRBM_GUI_ACTIVE` 降到 0.724，与逐算子表里 `moe_gemm2_0` 的 2844.4/3801.8 = 0.748 吻合，
说明这几趟 PMC 采样和正式计时是一致的。

三条读数：

**(1) `SQ_INSTS_LDS` 在 B 上一条都没少（38,616,624 与 A 逐位相等）。**
换布局这件事对 LDS 指令**一条都没动**——这符合预期，`SORTED_PARTIAL` 只改存储地址怎么算，
不碰 LDS 那一侧。C 和 D 才降，原因见 2.4.1，**不是** DCE。

**(2) C 砍掉了 28.8% 的 VALU 和 59.7% 的 SALU，忙周期却只降 3.2%。**
少发 7100 万条 VALU 指令几乎白干——说明这个 epilogue **不是发射受限**，
指令条数根本不预测时间。这恰好解释了代码注释里"VALU 被访存停顿掩盖"的观察。

**(3) 忙周期的协同项是 −0.122**（实际 0.724 vs 单独效应之和 0.846），
和 e2e 上 −421/−796 的比例一致，两个口径互相印证。

#### 2.4.1 `SQ_INSTS_LDS` 为什么在 C 上就降了

先交代背景，这段 epilogue 在做什么。它分两步：

- **Step 1**：把 MFMA 累加器里的结果转成 bf16，写进一块 LDS 暂存区（`lds_out`）。
  ISA 上是 `ds_write_b16_d16_hi`，每个 N-tile 32 条。
- **Step 2**：换一种线程映射从 `lds_out` 把结果读回寄存器（这一步叫 CShuffle，
  目的是让写全局内存时相邻 lane 写相邻地址），然后 `global_store_dword` 写出去。

除此之外，每一行输出还要从另一块 LDS（`lds_tid`）读回 `moe_sorting` 打包的 sorted id
——就是 2.1 说的那个"哨兵读"。所以 LDS 读有**两拨**：CShuffle 读回 + 哨兵读。

`FASTVALID` 会把两条 epilogue 都编进去（运行时选一条），所以 C 的 ISA 里有两份代码。
按"基本块里有没有 `s_cbranch_execz`"把它们分开，得到：

| C 的 ISA 分簇 | `ds_write_b16` | `ds_read_b128` | `ds_read_b32` | `ds_read2_b32` | `global_store_dword` |
|---|---|---|---|---|---|
| 含 `execz`（掩码路径） | 1024 | 512 | 256 | 256 | 512 |
| 不含（快路径） | 1024 | 128 | 0 | 384 | 480 |
| **A 的全部指令**（只有掩码一条路径） | **1024** | **512** | **256** | **256** | **512** |

第三行是这个分簇方法的**验证**：A 只编了掩码那一条路径，它的全部计数和 C 里"含 execz"那一簇
**逐项完全相同**。说明分簇没分错，剩下那一簇就是快路径。

结论：**快路径的 LDS 读是 512 条，掩码路径是 1024 条，正好一半。**

原因在汇编里看得很直白。掩码路径每次只处理**一行**：

```asm
; C 的掩码路径，LBB0_6（每个 N-tile 有 8 个这样的小块）
ds_read_b32   v45, v47 offset:28704      ; 读 1 个 sorted id
v_and_b32     v48, 0xffffff, v45         ; t = id & 0xFFFFFF
v_cmp_gt_u32  s[0:1], s33, v48           ; t < tokens ?
v_cmp_gt_u32  s[4:5], s6,  v45           ; s < topk ?   (0x9000000 = 9<<24)
s_and_b64     s[4:5], s[0:1], vcc        ; 三个条件与起来
s_and_saveexec_b64 s[0:1], s[4:5]        ; 写进 exec 掩码
s_cbranch_execz .LBB0_8                  ; 整组都无效就跳过
v_lshrrev_b32 v45, 24, v45               ; s = id >> 24
v_mad_u32_u24 v84, v48, 9, v45           ; ts = t*9 + s
ds_read2_b32  v[98:99], v45 offset1:32   ; CShuffle 读回
global_store_dword v[84:85], v98, off nt ; 写出
```

**每一行都要单独读一次 id、单独算一次谓词、单独设一次 `exec`。** 而 `s_and_saveexec_b64`
和 `s_cbranch_execz` 是**基本块的边界**——编译器不能把跨越它们的访存合并到一起，
因为下一块能不能执行取决于上一块算出来的掩码。于是 8 行就是 8 条 `ds_read_b32`。

快路径没有谓词，整段塌成一个直线基本块，编译器可以一次把多行的 id 一起读进来：

```asm
; C 的快路径，LBB0_22
s_barrier
ds_read_b128  v[44:47],  v43 offset:28672   ; 一次读 4 个 sorted id
ds_read_b128  v[98:101], v43 offset:28736   ; 再读 4 个
v_lshrrev_b32   v48, 24, v44   ; \
v_mul_u32_u24   v44,  9, v44   ;  | 第 1 行的 ts = t*9+s
v_add_lshl_u32  v44, v44, v48, 2 ; /
v_lshrrev_b32   v48, 24, v45   ; \
v_mul_u32_u24   v45,  9, v45   ;  | 第 2 行，直线展开，没有分支
v_add_lshl_u32  v45, v45, v48, 2 ; /
...                             ; 第 3、4 行同理
buffer_load_dword v48,  v44, s[28:31], 0 offen
buffer_load_dword v84,  v45, s[28:31], 0 offen
...
```

**8 条 `ds_read_b32` 变成 2 条 `ds_read_b128`**（一条读 4 个 dword），
CShuffle 读回也从 16 条 `ds_read_b128` 重排成 4 条 `ds_read_b128` + 12 条 `ds_read2_b32`。
每个 N-tile 的读从 32 条降到 16 条。

所以 C 那 2,310,656 条 LDS 指令的减少，是**谓词把基本块切碎、挡住了访存合并**；
去掉谓词，合并就做得了。这是个纯代码生成效应，**和 2.2(c) 说的那条 `(t,s)` 依赖链无关**。

> 早先本文这里写的是"只有 D 才降，所以是依赖链的证据"——那是错的，C 也降，
> 而且降幅正好是 D 的一半。`SQ_INSTS_LDS` 里混了两种效应，不能单独拿来证明 DCE。
> 唯一还成立的是 B 与 A 逐位相等这一条。

#### 2.4.2 两个还没解释的地方

**(a) 计数器的协同项方向不一致。** `SQ_WAIT_ANY` 和 `SQ_INSTS_VALU` 的协同项是**正的**
（+0.110 / +0.034），只有忙周期是负的（−0.122）。也就是 B 和 C 各自减少的等待/指令加起来
比 D 还多，但时间反而是 D 最短。这两个计数器按 wave 累加，和墙钟时间之间还隔着调度和重叠。

**(b) 静态 ISA 推不出实测的比值 —— 但快路径命中率本身没有问题。**

按 2.4.1 那张表，快路径每次执行 1536 条 LDS 指令、掩码路径 2048 条。如果快路径命中 94%，
C/A 应该是 0.76 左右；实测是 **0.940**，反推只有约 24% 走了快路径。

这个反推是**错的**，问题出在静态模型而不是 guard。直接在 host 侧用真实的 `moe_sorting`
输出把 `blk_all_valid` 算一遍（token=32768/E=193/topk=9/block_m=64，路由用测试同款
`fused_topk`）：

```python
aiter.moe_sorting_fwd(tid, tw, sorted_ids, sorted_w, sorted_eid, num_valid, moe_buf, E, BM)
nv = int(num_valid[0])
s   = sorted_ids[:nb*BM].view(nb, BM)
t, k = s & 0xFFFFFF, s >> 24
row = torch.arange(nb*BM, device=s.device, dtype=torch.int32).view(nb, BM)
valid = (row < nv) & (t < M) & (k < TOPK)
last_ok, allv = valid[:, -1], valid.all(dim=1)     # guard 的判断 vs 真值
```

| | 块数 | 占比 |
|---|---|---|
| `blk_all_valid` 命中（只看最后一行） | 4517 / 4800 | **94.10%** |
| 整块逐行全有效（真值） | 4517 | 94.10% |
| **误判**（命中但实际含 padding） | **0** | |
| **漏判**（未命中但实际全有效） | **0** | |
| 纯 padding 块 | 93 | 1.94% |
| 部分有效块 | 190 | 3.96% |

**快路径实际命中 94.10%，而且 guard 零误判零漏判**——"最后一行真实 ⇒ 整块真实"这个推理
在这个 shape 上是精确的。所以 `FASTVALID` 从命中率这一侧**没有剩余空间可挖**。

对不上的是 `SQ_INSTS_LDS` 与静态计数的关系：连 A 自己也只测到 2010.9 条/wave，
而它的静态计数是 2083（只有一条路径，本该相等）。也就是说这个计数器的口径和
"每 wave 执行一遍静态代码"这个模型有系统性偏差，**不要拿它做绝对数量的推算**，
只能看四格之间的相对变化。

顺带一个查这个问题时挖出来的、值得记住的细节：**`num_valid_ids = 301248`，
比真实行数 `token*topk = 294912` 多 6336**。也就是说 `num_valid_ids` 是补齐到块边界之后的数，
它**里面**就含 6336 行哨兵。所以 epilogue 的掩码光查 `row < num_valid` 不够，
必须再查 `t < tokens` 和 `s < topk`——2.1 那三个条件一个都不能省。
全部 padding 行 12288（4.00%）= 5952 行在 `num_valid` 之外 + 6336 行在里面但是哨兵。

综合起来，目前的数据能确定的是：**收益不在访存侧（写请求四格完全相同）、
全部发生在 epilogue（MFMA/wave/bank conflict 四格相同）**；
但还不足以把机制钉死到具体哪条依赖链。要钉死需要 ATT（thread trace）看 store 的实际发射时刻，
留到后面有需要再做。

### 2.5 效果

| | base | f1 | 差 |
|---|---|---|---|
| `moe_gemm2_0` | 3801.8 | **2844.4** | **−957.4** |
| 归约 | 729.9（`_topk_sum_kernel`） | 783.9（`_topk_sum_gather_kernel`）+ 9.9（invert） | +63.9 |
| e2e | 7859.3 | **7035.1** | **−824.2** |

GEMM 本体省 957 us，归约那边因为要 gather 多花 64 us，净赚 824 us。

---

## 三、Feature 2：epilogue 的 scale 与地址从逐元素重算改成提前算一次

**e2e −239.7 us（累计补上总差距的 64.9%）；stage2 GEMM −261.5 us**

三个 env knob，和 Feature 1 一样**必须一起开**：

```bash
FLYDSL_MOE_STAGE2_SCALAR_ASCALE=1   # 激活 scale 只读一次
FLYDSL_MOE_STAGE2_VEC_SCALE=1       # 缩放按 f32x4 整体做
FLYDSL_MOE_STAGE2_BUFSTORE=1        # 输出走 per-block buffer 描述符
```

### 3.0 一句话与整合视图

**一句话：epilogue 里有三样东西被放在了比它们实际变化频率更低的层级上重算，这个 feature 把它们各自提回该待的层级。**

三个 knob 不是三件独立的事，它们打的是同一个靶子：**epilogue 里「把累加器缩放、转换、写出去」这条路上的重复计算。**

但要注意它们**不在同一个函数里**——epilogue 是 CShuffle，被 barrier 切成两半：

```
Step 1  write_row_to_lds :  累加器 → 缩放 → 转 bf16 → 写进 LDS      ← (a) (b) 在这
        ──────────── barrier ────────────
Step 2  换一套线程映射读回 LDS → store_pair → 写全局                ← (c) 在这
```

中间转一道 LDS 是为了让**最终的全局写是合并的**：MFMA 出来的累加器布局（一个 lane 拿同一列的
4 个连续行）和全局写想要的布局（连续通道）是正交的，直接写会非常散。

下面把三处改动**并排放在一起**看清楚它们在改什么——注意这是示意，`(c)` 那一行真实位于
barrier 之后的 `store_pair`，不在 `for ni` 循环里（真实代码见 3.1 和 3.2）：

```python
# ═══════════ 改之前 ═══════════
# 对每个输出行 (mi, ii) 都做一遍：
fused2 = memref.load(lds_tid, [row_in_tile])   # 读 LDS 拿打包 id
t2, s2 = fused2 & 0xFFFFFF, fused2 >> 24       # 解码"这一行是谁"
ts2    = t2 * topk + s2
sx     = buffer_load(sx_rsrc, ts2)             # (a) 逐行读激活 scale —— 其实每次读到的是同一个数

for ni in range(num_acc_n):                    # 再对每个通道块：
    v = vector.extract(acc[mi*num_acc_n + ni], [ii])   # (b) 从 f32x4 里抠出 1 个元素
    v = v * (sx * tw * sw_vals[ni])            #     标量乘（每 (mi,ii) 共 5 条）
    vector.store(v1, lds_out, alignment=2)     #     写 LDS，一次 2 字节
# ─── barrier，换线程映射读回 LDS ───
    llvm.StoreOp(frag, inttoptr(64 位地址))     # (c) 在 store_pair 里：裸 64 位指针，
                                               #     每条在途 store 的地址占一对 VGPR


# ═══════════ 改之后 ═══════════
# 内核入口，整个 workgroup 只做一次：
sx_scalar    = buffer_load(sx_rsrc, 0)                       # (a) scale 读一次就够
blk_out_rsrc = MakeBufferRsrc(base = out_base + bx_m*model_dim*2,   # (c) 描述符只覆盖本 block
                              num_records = tile_m*model_dim*2)     #     的 512 KB 窗口

# 对每个 (mi, ni)——注意 ii 那一层没了：
svec = tw_vec * splat(sx_scalar * sw_vals[ni])
got  = acc[mi*num_acc_n + ni] * svec           # (b) 整个 f32x4 一次乘完，缓存起来
                                               #     ii=0..3 直接 extract 缓存，不再重算
vector.store(v1, lds_out, alignment=2)         #     写 LDS 仍是 2 字节，f2 没动它
# ─── barrier，换线程映射读回 LDS ───
buffer_store(frag, blk_out_rsrc, 32 位偏移)     # (c) 在 store_pair 里：基址在 SGPR，
                                               #     每 lane 只留 32 位偏移
```

三处改动各自消掉的是**不同层级**的重复：

| knob | 消掉哪个层级的重复 | 具体 |
|---|---|---|
| `SCALAR_ASCALE` | **跨行**：一个常量被当成 294,912 个数逐行读 | 提到 kernel 入口读一次；连带 `(t,s)` 解码和 `ts2` 整条链一起消失 |
| `VEC_SCALE` | **跨元素**：能整体做的乘法被拆成 4 次标量乘 | f32x4 一次乘完并缓存，每 N-tile 从 80 条乘法降到 34 条 |
| `BUFSTORE` | **跨 lane**：workgroup 级不变的基址被当成每 lane 的 64 位量 | block 级描述符，基址落进 SGPR，每 lane 只留 32 位偏移 |

**为什么必须一起开**：`SCALAR_ASCALE` 是使能者。它拔掉 `(t,s)` 那条链之后，
`VEC_SCALE` 才可能整块向量化——逐行的 `sx` 要塞进 f32x4 就得凑齐 4 行的 `(t,s)`，
而凑齐 4 行正好要把刚拔掉的链请回来。代码里是直接 `and` 掉的，不是测出来的（见 3.3）。

### 3.1 优化前：epilogue 被逼成逐行 + 逐元素

旧 epilogue 每处理一个输出元素，要做三件与"这一行是谁"绑定的事：

**(a) 读一个 per-row 的激活 scale。** host 侧的
`moe_kernels._resolve_a_scale_for_fused_init(a2_scale, token_num*topk, dev)` 会把
per-tensor 的**标量广播成 294912 个元素的数组**，内核再逐行读回来。为了索引它，
每行都要解出 `(t, s)` 再算 `ts2 = t*topk + s`：

```python
fused2 = memref.load(lds_tid, [row_in_tile])
t2 = fused2 & mask24_i32
s2 = fused2 >> 24
...
ts2 = t2 * topk_i32_v + s2
sx  = buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=T.f32)
```

**(b) 逐元素取累加器。** `acc[acc_idx]` 是一个 f32x4，但代码一次只取一个。

这一段 f2 之前的代码**今天还在源码里**——后来的 feature 都是往 `write_row_to_lds` 里加分支，
没有删掉老路径。所以想对比 f2 前后，直接读同一个函数的两条分支就行：

| 行 | 判断条件 | 属于 |
|---|---|---|
| 4716 | `if _bfirst:` | **f5**（B-first），读 f2 时整段跳过 |
| 4746 | `sx_scalar` 有值 且 `_vec_scale` | **f2 之后**（(a)+(b) 都开），见 3.2 |
| 4761 | `sx_scalar` 有值，`_vec_scale` 关 | f2 中间态（只开 (a)） |
| **4791** | 都不满足，fallthrough | **f2 之前**，就是下面这段 |

> 同样地，`_scaled_acc`（3.2b）里 `if _wscalar:` 属于 **f7**、`if _bfirst:` 属于 **f5**，
> f2 的原始形态是那条 `else` 分支。读某一个 feature 时按这个表把别的分支滤掉。

```4841:4856:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            sw = sw_vals[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            if is_int8:
                                v = arith.sitofp(T.f32, v)
                            v = v * (sx_row * sw)
                            v_out = _cvt_out(v)

                            lds_idx = row_base_lds + col_local
                            vec1_out = T.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)
```

逐行（下标的含义见 1.0 的「累加器布局」）：

| 行 | 在做什么 |
|---|---|
| `for ni in range_constexpr(...)` | 遍历本 wave 的 2 个列块。`range_constexpr` 是**编译期展开**，IR 里是两份直线代码，没有循环 |
| `col_local = col_base_local + ni*16` | 这一列在 tile 内的下标。`col_base_local = n_tile_base + lane%16`（`mfma_epilogues.py:159`），合起来 `wave_n_id*32 + ni*16 + lane%16`，覆盖 `[0,128)` |
| `sw = sw_vals[ni]` | 这一列的权重 scale，**是标量**——一个 lane 在 N 方向只占一列 |
| `acc_idx = mi*num_acc_n + ni` | 把 `(mi, ni)` 压平成一维，取到那个 f32x4 |
| `vector.extract(..., [ii])` | 从 f32x4 里取第 `ii` 个，也就是**第 `ii` 行** |
| `if is_int8: sitofp` | int8 输入时累加器是 i32，要先转 f32；fp8 路径不走 |
| `v * (sx_row * sw)` | 两条标量乘。`sx_row = sx * tw` 已在 ni 循环外折好（见上方注释），否则还要再多 |
| `_cvt_out(v)` | f32 → bf16，用**截断**而非 RNE 舍入：`bitcast → >>16 → trunci → bitcast`，3 条（RNE 那套 5 条/元素曾被 ATT 标成热点） |
| `vector.store(v1, ..., alignment=2)` | 降下去是 `ds_write_b16`，**一次 2 字节** |

最后一行值得单独说：**为什么只能写 2 字节。** 一个 lane 手上那 4 个值（`ii=0..3`）落在
**4 个不同的 LDS 行**上（行距 = `lds_out_stride`），而沿 `ni` 方向列距是 16，也不连续。
**两个方向都不连续，所以只能一个元素一个元素写。** 这不是实现偷懒，是累加器朝向决定的几何——
f2 能向量化的只有那条乘法，这条写它动不了，要等 f5 把朝向翻过来（见第六章）。

ISA 上每行是这个模式，2 个输出元素要 5 条标量乘法：

```asm
v_mul_f32_e32 v47, v82, v47   ; sx * tw       ← 跨 32 个 N-tile 重复算，其实不变
v_mul_f32_e32 v49, v51, v47   ; sx_row * sw   (ni=0)
v_mul_f32_e32 v47, v1,  v47   ; sx_row * sw   (ni=1)
v_mul_f32_e32 v49, v26, v49   ; acc * scale   (ni=0)
v_mul_f32_e32 v47, v22, v47   ; acc * scale   (ni=1)
```

**(c) 用 64 位裸指针存储。** reduce 模式的 partial 缓冲可能超过 4 GiB，而 buffer 描述符的
`num_records` 只有 32 位，所以这条路径退回 `llvm.StoreOp` + `llvm.inttoptr`
（`moe_gemm_2stage.py:4262-4272`），每条在途 store 的地址要占一对 VGPR。

三件事共用同一条依赖链：**sorted id → `(t, s)` → 地址/scale 索引**。

### 3.2 优化后是怎么写的

下面是三处改动各自的真实代码；整合在一起看的版本见 3.0。

#### (a) 激活 scale 提到入口

per-tensor 时那个数组里每个元素都一样，读一次就够：

```3234:3239:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
            # Per-tensor activation scale: one load for the whole workgroup.
            sx_scalar = None
            if _scalar_ascale and sx_rsrc is not None:
                sx_scalar = buffer_ops.buffer_load(
                    sx_rsrc, fx.Int32(0), vec_width=1, dtype=T.f32
                )
```

`write_row_to_lds` 里随之整条 `(t, s)` 解码和 `ts2` 计算都不需要了。

> **只在 per-tensor 下成立。** 内核无从判断 host 传进来的是标量广播还是真正的
> per-token scale，所以这是 opt-in；在 per-token 下它会**静默算错**。

#### (b) 缩放按 f32x4 整体做

##### 先看清楚这个 f32x4 是什么

取 wave 0 的 lane 5（`lane/16 = 0`、`lane%16 = 5`），它的 `acc[mi=0, ni=0]`：

```
                          列 (通道) →
              0   1   2   3   4   5   6  ...  15
            ┌────────────────────────────────────
     行 0   │                   ●                  ← ii=0
     行 1   │                   ●                  ← ii=1
     行 2   │                   ●                  ← ii=2
     行 3   │                   ●                  ← ii=3
     行 4   │                   ·
      ↓     │                   ·
                                ↑
                          全都在第 5 列
```

**这 4 个数是「同一列的 4 个连续行」**（下标含义见 1.0 的累加器布局）。

##### 三个 scale 因子在这 4 个格子上怎么变

每个输出元素最终要乘 `sx × sw × tw` 三样：

```
                ii=0     ii=1     ii=2     ii=3
 acc (f32x4)  [  a0   ,   a1   ,   a2   ,   a3  ]   ← 行 0/1/2/3 的第 5 列

 sx  激活scale[  SX   ,   SX   ,   SX   ,   SX  ]   per-tensor  → 全一样 ✓
 sw  权重scale[  SW5  ,   SW5  ,   SW5  ,   SW5 ]   逐"列"变    → 同一列 → 全一样 ✓
 tw  路由权重 [  TW0  ,   TW1  ,   TW2  ,   TW3 ]   逐"行"变    → 4 个都不同 ★
```

| | 变量 | 来自 | 粒度 |
|---|---|---|---|
| 激活 scale | `sx` | `arg_scale_x`（A2 scale，fp8 反量化） | per-tensor（本 feature 的前提） |
| 权重 scale | `sw` | `arg_scale_w`（W2 scale） | per-channel，`sw_vals[ni]` |
| 路由权重 | `tw` | `sorted_weights` | per-row，`doweight_stage2` 时才有 |

> `sx` 只在**输入是 fp8/int8** 时存在；f16/bf16 输入下 `sx_rsrc = None`、`sx` 恒为 1.0。

**注意这里不是"per-tensor 所以 scale 相同、算一次就够"**——最终乘的 scale 在 4 个元素上
**并不相同**（`tw` 逐行变）。三个因子里两个在向量内是常量，只有一个在变。

##### 改之前：4 条独立的标量链

```
ii=0:  SX ─┬─× TW0 ──→ sx_row ──× SW5 ──→ scale ──× a0 ──→ 结果0     3 条标量乘
ii=1:  SX ─┼─× TW1 ──→ sx_row ──× SW5 ──→ scale ──× a1 ──→ 结果1     3 条
ii=2:  SX ─┼─× TW2 ──→ sx_row ──× SW5 ──→ scale ──× a2 ──→ 结果2     3 条
ii=3:  SX ─┴─× TW3 ──→ sx_row ──× SW5 ──→ scale ──× a3 ──→ 结果3     3 条
                                  ↑
                          SX×SW5 明明不变，却算了 4 遍
```

##### 改之后：不变的提到外层，变的拼成向量

```
【每个 N-tile 折一次】  SX × SW5  ──→ splat ──→ SXSW = [ S, S, S, S ]     ← 不随行变

【每个 (mi,ni) 折一次】 TW_vec = [ TW0, TW1, TW2, TW3 ]                    ← 随行变，拼成向量
                        svec   = TW_vec × SXSW      ← 1 条向量乘
                        got    = acc    × svec      ← 1 条向量乘

【ii = 0..3】           extract(got, ii)            ← 不再有任何乘法
```

所以省的是两处，**都不是"值相同所以只算一次"**：

1. **把不变的因子提到外层**——`SX × SW5` 跨 4 行不变，从算 4 遍变成每 N-tile 算 1 遍；
2. **把变的因子拼成向量**——`TW` 是 4 个不同的值，但一条 `v_pk_mul_f32` 能处理 2 个 f32，
   所以 f32x4 乘一次只要 2 条机器指令，而不是 4 条标量乘。

##### 对应代码

```4611:4623:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        _sw_x = [
                            (
                                sw_vals[ni]
                                * vector.from_elements(
                                    T.vec(4, T.f32), [sx_scalar] * 4
                                )
                                if _bfirst
                                else vector.from_elements(
                                    T.vec(4, T.f32), [sx_scalar * sw_vals[ni]] * 4
                                )
                            )
                            for ni in range(num_acc_n)
                        ]
```

A-first 走 `else` 那支：`splat(sx_scalar * sw_vals[ni])`，就是图里的 `[S,S,S,S]`。
这是**普通 Python 列表推导**，emit 时执行一次，整个 N-tile 只发出 `num_acc_n = 2` 条乘法。

`_sw_x[ni]` 必须先 splat 成向量——**MLIR 的 `arith.mulf` 不做 vector×scalar 广播**，
否则报 `'arith.mulf' op requires the same type for all operands and results`。

```4678:4701:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                                else:
                                    tws = []
                                    for jj in range_constexpr(4):
                                        if tw_pf is not None:
                                            tws.append(tw_pf[(mi * 4) + jj])
                                        else:
                                            # `row` is this call's row; the sibling rows of the
                                            # same mi differ by (jj - ii).
                                            tws.append(
                                                buffer_ops.buffer_load(
                                                    sorted_w_rsrc,
                                                    row + fx.Index(jj - ii),
                                                    vec_width=1,
                                                    dtype=T.f32,
                                                )
                                            )
                                    tw_vec = vector.from_elements(T.vec(4, T.f32), tws)
                                svec = tw_vec * _sw_x[ni]
                            else:
                                svec = _sw_x[ni]
                            a = acc[mi * num_acc_n + ni]
                            if is_int8:
                                a = arith.sitofp(T.vec(4, T.f32), a)
                            got = a * svec
```

和改之前逐处对照：

| 图里的 | 改之前 | 改之后 |
|---|---|---|
| 取 `TW` | 4829 单个 `tw_pf[mi*4+ii]` | 4682 四个 `tw_pf[mi*4+jj]`，`jj=0..3` |
| `SX × SW` | 4850 内联，每次重算 | 4611 提到 N-tile 层，只算 1 次 |
| `× TW` | 4840 标量 | 4695 向量 |
| `× acc` | 4850 标量（extract 后乘） | 4701 向量（整块乘） |
| `ii` 展开 | 4841 每次全算 | 4632 缓存命中，只剩 extract |
| 写 LDS | 4856 `alignment=2` | 4759 `alignment=2`（**不变**） |

##### 缓存键里为什么可以不带 `ii`

```4632:4634:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        key = (bool(_epi["masked"]), mi, ni)
                        got = _vec_scale_cache.get(key)
                        if got is None:
```

`write_row_to_lds` 每个输出行被调一次，固定 `mi` 下 `ii = 0..3` 共 4 次，
**四次要的是同一个 f32x4 的四个元素**：

```
调用 1 (ii=0) → _scaled_acc(mi,ni,0)  未命中 → 算出 got=[g0,g1,g2,g3] 存缓存 → extract(got,0)
调用 2 (ii=1) → _scaled_acc(mi,ni,1)  命中 ───────────────────────────────→ extract(got,1)
调用 3 (ii=2) → _scaled_acc(mi,ni,2)  命中 ───────────────────────────────→ extract(got,2)
调用 4 (ii=3) → _scaled_acc(mi,ni,3)  命中 ───────────────────────────────→ extract(got,3)
```

之所以安全，是因为 **`_scaled_acc` 的返回值根本不依赖 `ii`**。`ii` 在函数体里只出现一处，
就是 `tw_pf` 为空时的兜底 `row + fx.Index(jj - ii)`，而这个偏移恰好互相抵消：

| 谁来调 | `row` 是 | `jj - ii`（jj=0..3） | 实际取的行 |
|---|---|---|---|
| `ii=0` | base+0 | 0, 1, 2, 3 | base+0 ~ base+3 |
| `ii=2` | base+2 | −2, −1, 0, 1 | base+0 ~ base+3 |

不管谁来调，取到的都是同一组 4 行，结果恒等。

> **这不是运行时缓存。** `_vec_scale_cache` 是普通 Python 字典，`range_constexpr` 也是编译期
> 展开——命中缓存等于那几条乘法**根本没被写进 IR**，不是运行时跳过。同理 `mi`/`ii`/`ni`
> 全是 Python `int`，最终 ISA 里是 16 份展开的直线代码，没有循环也没有查表。

##### f2 没有减少调用次数

一个容易产生的误解：既然四次要的是同一个 f32x4，是不是该合并成一次调用？
**f2 做不到，因为 A-first 下这 4 个值落在 4 个不同的 LDS 行上，地址不连续，
每行必须单独写一次 `ds_write_b16`。**

```
f2 之前：调用 16 次，每次算完整标量链              → 80 条乘法
f2 之后：调用 16 次，其中 12 次命中缓存只做 extract → 34 条乘法
         ↑ 次数没变，LDS 写指令也没变（1024 条）
```

`default_epilog`（`mfma_epilogues.py:75-81`）那个 `for mi: for ii:` 双层循环在 f2 前后
**一字未改**。真正把它塌成每 `mi` 一次（`_step1`，16 → 4 次调用）的是 **f5 的 B-first**，
那时一个 lane 的 4 个值变成同一行的 4 个连续通道、地址连续了，一次能写完。见第六章。

按每个 N-tile 数（`tile_m=64`、`num_acc_n=2`）：标量形式是每 `(mi, ii)` 5 条乘法共 **80 条**；
向量形式是 `2 (sx*sw) + 16 (tw_vec*sw) + 16 (acc*scale)` = **34 条**。

#### (c) 输出走 per-block buffer 描述符

sorted-row 布局下（也就是 Feature 1 的前提），一个 workgroup 的行就是连续区间
`[bx_m, bx_m + tile_m)`，所以描述符只要覆盖这一小片：

```python
_blk_base_idx = out_base_idx + bx_m * fx.Index(model_dim * out_elem_bytes)
_blk_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), _blk_i64_raw)
blk_out_rsrc = _rocdl_dialect.MakeBufferRsrcOp(
    ir.Type.parse("!llvm.ptr<8>"), _blk_ptr, _mk_i16(0),
    _mk_i64(int(tile_m) * int(model_dim) * int(out_elem_bytes)),
    _mk_i32(_mk_buf_flags()),
).result
```

`bx_m` 是 workgroup 级 uniform，所以那次 64 位乘加落进 SGPR，之后每 lane 只需 32 位偏移。
窗口是 `64 × 4096 × 2 B = 512 KB`，**与 token 数无关**——所以这条路比原来的全局 64 位寻址
更安全，不是更危险，3.1(c) 引用的 4 GiB 限制在按 block 切片之后根本不存在。

### 3.3 为什么算一个 feature

**这一节的数据来自一批独立的手工运行**（f1 基线是 7066.0，与第十章那一 session 的 7035.1
不是同一批），只在本节内部可比，**不要和第十章的绝对值混着读**。5 次取中位，全部叠在 f1 之上：

| 配置 | e2e | vs f1 |
|---|---|---|
| f1 | 7066.0 | — |
| `SCALAR_ASCALE` | 6951.7 | −114.3 |
| `SCALAR_ASCALE` + `BUFSTORE` | 6926.8 | −139.2 |
| `SCALAR_ASCALE` + `VEC_SCALE` | 6922.4 | −143.6 |
| 三个全开 | **6828.5** | **−237.5** |

```
BUFSTORE 单独（SC → SC+BUF）    −24.9    ← 噪声量级，见下
VEC      单独（SC → SC+VEC）    −29.3    ← 噪声量级，见下
实际联合                       −123.2
```

> **这张表里只有 `SCALAR_ASCALE` 的 −114.3 和联合的 −123.2 站得住。**
> `BUFSTORE` 和 `VEC_SCALE` 各自那 −24.9 / −29.3 落在噪声里：`results/e2e.csv` 里 f1 在不同
> session 的中位数跨度有 60 us，比这两个数都大。2.3 节判 `FASTVALID` 的 −45.9 us 是"噪声"，
> 同一把尺子，这两个也只能判噪声。
>
> 所以**不要**拿它们去算交互项。本文早先由此得出"交互项 −69.0，比两者之和还大"，
> 那是在噪声上做减法，已删。
>
> 不依赖这两个小数、仍然站得住的结论是：**`SCALAR_ASCALE` 单独就值 −114.3，三个一起
> 值 −123.2**——中间只差 9 us，说明后两个 knob 在缺少前者时几乎交不出东西。至于这是
> "被前者使能"还是"它们本来就没多少收益"，这批数据分不开，要分开需要同 session 多轮重复。

依赖结构（这一条不依赖上面的数字，是代码层面的硬约束，两条都能在源码里读到）：

```
SCALAR_ASCALE ──> VEC_SCALE     `_vec_scale = _scalar_ascale and env(...)`，直接 and 掉
                                （moe_gemm_2stage.py:2990）

Feature 1 的 ──> BUFSTORE       `_bufstore = env(...) and _SORTED_PARTIAL and ...`
sorted 布局                      只有 sorted 布局才有连续小窗口可收（3.2c）
                                （moe_gemm_2stage.py:2962）
```

注意 `BUFSTORE` 的前置是 **Feature 1 的 sorted 布局**，不是 `SCALAR_ASCALE`——
三个 knob 里只有 `VEC_SCALE` 被 `SCALAR_ASCALE` 硬门控。

`SCALAR_ASCALE` 把 `ts2 = t*topk + s` 那条链连根拔掉，于是 epilogue 不再需要逐行的标量
上下文，`VEC_SCALE` 才可能整块向量化——逐行的 `sx` 要进 f32x4 就得凑齐 4 行，
而凑齐 4 行正好要把那条链请回来。这是代码里 `and` 出来的，不是测出来的。

这和 Feature 1 是同一种形态——两个 feature 都在拆同一条依赖链
（`sorted id → (t,s) → 地址/谓词/scale 索引`），链上留任何一个消费者，标量化就还在。

### 3.4 效果

时间（与第十章同一 session）：

| | f1 | f2 | 差 |
|---|---|---|---|
| `moe_gemm2_0` | 2844.4 | **2582.9** | **−261.5** |
| e2e | 7035.1 | **6795.4** | −239.7 |

**归因看 GEMM 那一行。** e2e 的 −239.7 被 stage1 在 f2 那一档 +27.8 us 的漂移压低了，
真实收益是 stage2 GEMM 的 −261.5；理由见 10.3。`cos` 全程 0.999995，与 base 一致。

每 wave 指令数（动态计数器，`SQ_INSTS_* / SQ_WAVES`）：

| | base | f1 | **f2** | target | f1→target 差 | f2 补上 |
|---|---|---|---|---|---|---|
| VALU | 12,867 | 8,436 | **6,731** | 3,886 | 4,550 | **37%** |
| LDS | 2,011 | 1,770 | 1,645 | 391 | 1,379 | 9% |
| SALU | 2,123 | 1,228 | **703** | 136 | 1,093 | **48%** |
| VMEM 读 | 1,297 | 1,328 | **828** | 198 | 1,131 | **44%** |
| VMEM 写 | 492 | 492 | 492 | 125 | 366 | 0% |
| MFMA | 1,504 | 1,504 | 1,504 | 1,504 | 0 | — |
| **合计** | **20,293** | **14,759** | **11,902** | **6,240** | **8,519** | **34%** |

这张表在 PTL 关掉和打开的两批采集里**逐位相同**——指令数不受时钟状态影响。
这反过来也说明 PTL 只改吞吐、不改工作量（PTL 是什么、为什么要盯它，见 1.2）。

f2 自己的计数器还给了 12.7 那条"occupancy 不是杠杆"一个直接证据：

| | f1 | f2 | 变化 |
|---|---|---|---|
| 每 wave 指令合计 | 14,759 | 11,902 | **−19.4%** |
| `GRBM_GUI_ACTIVE`（GPU 忙周期） | 20,227,637 | 18,134,001 | **−10.3%** |
| `MeanOccupancyPerCU` | 7.613 | 7.690 | **+1.0%** |

**f2 砍掉近两成指令、时间降了一成，而 occupancy 纹丝不动。** 斜率正好是"砍 2% 指令 ≈ 快 1%"。
反过来说：如果 occupancy 真是杠杆，f2 这个 feature 就不该有收益。

> 采集这批数时踩过一次坑：某一趟的 `SQ_INSTS_MFMA` 采成了 34,698,273 而不是 28,876,800。
> 四个 stage 跑的是同一个 GEMM，MFMA 数必须相同，不同就说明那趟被扰动了、所有每-MFMA 的
> 派生值都不能用。`run.sh` 现在会自动检查并告警。


## 四、Feature 3：循环不变量外提 + 输出宽度

**e2e −316.3 us（累计补上总差距的 83.7%）**

```bash
FLYDSL_MOE_STAGE2_HOIST_PF=1    # 路由权重 / fast-valid guard 不再逐 N-tile 重读
FLYDSL_MOE_STAGE2_HOIST_X=1     # X 的 LDS 读不再逐 N-tile 重做
FLYDSL_MOE_STAGE2_EVEC=4        # 输出存储宽度 4 B -> 8 B
```

三个 knob，**前两个是同一件事**（把 N 循环的不变量在 emit 时缓存复用），第三个是修一个
默认值疏漏。三者两两之间都基本可加，归在一个 feature 里是因为**前两个机制同源**（见 4.4）。

前两个来自第十二章那轮定位里的两个热点；`HOIST_X` 是做的过程中发现的第三处，
当时以为要做结构改造，实际上同一个手法就够。

### 4.1 优化前：29 个不同的值被读了 928 次

persist 模式下 `_moe_gemm2_then_body` 被 `range_constexpr(32)` 展开成 32 份直线代码。
其中**三处**读的值只依赖 `bx_m` 和 lane，与 N-tile 无关：

| | 每 N-tile | ×32 tile | 实际不同的值 |
|---|---|---|---|
| 路由权重 `tw_pf`（`sorted_weights[bx_m + row_in_tile]`） | 16 | **512** | 16 |
| fast-valid guard（`sorted_token_ids[bx_m + tile_m - 1]`） | 1 | **32** | 1 |
| **X 的 LDS 读**（`lds_load_packs_k64`） | 12 | **384** | 12 |

前两项是全局访存，占 f2 全部 `buffer_load_dword`（675 条）的 **81%**；第三项是 LDS 读，
占 f2 全部 LDS 指令（1,555 条/wave）的 **25%**。

**为什么这些天然是不变量**：一个 workgroup 负责固定的 64 个 sorted 行 × 全部 4096 个通道，
而 N-tile 切的是**通道**维度。行集合从头到尾不变，变的只是列。所以按行索引的东西
（路由权重、guard、X 本身）全是 N 循环的不变量，按列索引的（`sw_pf` 的权重 scale）天然不是。

X 那一项尤其反直觉：persist 的设计初衷就是"X 装进 LDS 一次、跨 N-tile 复用"，装载确实只做了
一次（`is_first_ntile`），但**读回**每个 N-tile 都重做，而三个地址参数
`row_a_lds` / `col_offset_base_bytes` / k-tile 的 LDS 基址全都不含 N-tile 分量：

```4106:4114:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        _lbk = arith.index(_kk * _lds_tile_elems_py)
                        # Same N-tile invariance as the reads inside compute_tile.
                        _a0p = _pf_hoist.get(("xpf0", _kk)) if _hoist_x else None
                        if _a0p is None:
                            _a0p = lds_load_packs_k64(
                                row_a_lds, col_offset_base_bytes, _lbk
                            )
                            if _hoist_x:
                                _pf_hoist[("xpf0", _kk)] = _a0p
```

（这是**加上外提之后**的样子——`_hoist_x` 那两行就是 4.2 讲的改动。优化前只有中间那个
`lds_load_packs_k64` 调用，每个 N-tile 都重新发一次。）注意三个地址参数
`row_a_lds` / `col_offset_base_bytes` / `_lbk` 里没有一个含 N-tile 分量。

编译器合并不掉这三处：中间隔着 `s_barrier` 和 `buffer_store`，没有别名信息就不能跨屏障 CSE。

### 4.2 优化后：emit 时缓存，代码不用挪

关键观察：`range_constexpr` 是 **Python 层的展开，不产生 scf region**，所以 32 份 body
全部落在同一个 MLIR block 里，**第一份里定义的 SSA value 支配后面所有份**。于是根本不需要
把代码搬出循环，只要在 emit 时把值缓存下来复用。三处共用同一个字典 `_pf_hoist`：

```python
# (a) 路由权重
tw_pf = _pf_hoist.get("tw") if _hoist_pf else None
if doweight_stage2 and tw_pf is None:
    ...                                    # 原来那 16 次 buffer_load
    if _hoist_pf:
        _pf_hoist["tw"] = tw_pf

# (b) X 的 LDS 读；lds_key 标识 K-tile，只在 persist 路径传入
def _x_packs(row_lds, col_base, sub_key):
    if lds_key is None or not _hoist_x:
        return lds_load_packs_k64(row_lds, col_base, lds_base)
    k = ("x", lds_key) + sub_key
    got = _pf_hoist.get(k)
    if got is None:
        got = lds_load_packs_k64(row_lds, col_base, lds_base)
        _pf_hoist[k] = got
    return got
```

guard 那一处缓存的是算完的 `blk_all_valid`（i1），连那三次比较也一起省了。

这和 3.2(b) 的 `_vec_scale_cache` 是同一个手法，区别只在作用域：那个跨 `ii`，这个跨 N-tile。

**输出宽度**是另一回事，纯粹是修一个默认值的疏漏：

```2819:2820:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
        _e_vec = 8 if int(tile_n) % (_cshuffle_nlane * 8) == 0 else 2
        # Experiment knob: force the CShuffle read-back / store width (elements per
```

真正的约束是再下一行的 `tile_n % (32 × e_vec) == 0`——`tile_n=128` 下 **e_vec=4 完全合法**，
但这个三元表达式只在 8 和 2 之间二选一，直接掉到 2。用现成的 `FLYDSL_MOE_STAGE2_EVEC` 覆盖即可
（该修默认值，见附二 T4）。

### 4.3 ISA 验证：外提确实生效了，而且没涨寄存器

外提有没有生效，看 ISA 比看时间直接。两处分别验证：

**(a) 路由权重 + guard**（`HOIST_PF`，相对 f2 的静态计数）：

| 类别 | f2 | +`HOIST_PF` | 差 |
|---|---|---|---|
| MFMA | 1536 | 1536 | 0 |
| VALU | 8073 | 7919 | −154 |
| SALU | 2759 | 2534 | −225 |
| LDS | 3235 | 3235 | 0 |
| **VMEM 读** | **870** | **343** | **−527** |
| VMEM 写 | 1023 | 1023 | 0 |
| SYNC | 1384 | 1302 | −82 |
| **合计** | **18881** | **17893** | **−988（−5.2%）** |

`buffer_load_dword` 从 675 降到 148，正好等于预测的 512 + 32 少掉的量。

**(b) X 的 LDS 读**（`HOIST_X`，叠在上面之后）：

| | `HOIST_X=0` | `HOIST_X=1` |
|---|---|---|
| `ds_read_b128` | 384 | **12** |
| LDS 指令/wave（动态） | 1,555 | **1,190** |
| ISA 行数 | 16,599 | 15,981 |

`ds_read_b128` 正好只剩一个 N-tile 的量。

**两处都没有涨寄存器压力**，这是事前最担心的一点：

| | VGPR | `accum_offset` |
|---|---|---|
| f2 | 169 | 168 |
| +`HOIST_PF` | **169** | 156 |
| +`HOIST_X` | **172** | 172 |

`HOIST_X` 按 16×16 B 估算本该要 +64 个 VGPR，实测只涨 3 个——编译器把这些常驻值放进了
**本来就空着的 AGPR**（`accum_offset` 反向移动就是证据）。旧内核 169 个 VGPR 里累加器只占 13 个
（见 12.7(2)），AGPR 侧有大量余量，正好接住了外提出来的值。

### 4.4 析因：三个 knob 两两基本可加

**`HOIST_PF` × `EVEC`**（GPU 4，各 3 次，叠在 f2 之上）：

| | 三次 e2e | 中位 | vs f2 |
|---|---|---|---|
| f2 基线 | 6780.4 / 6794.6 / 6795.9 | 6794.6 | — |
| +`HOIST_PF` | 6631.2 / 6631.7 / 6637.2 | 6631.7 | **−162.9** |
| +`EVEC=4` | 6749.8 / 6760.1 / 6768.1 | 6760.1 | −34.5 |
| 两个都开 | 6596.1 / 6599.4 / 6600.5 | 6599.4 | **−195.2** |

单独之和 −197.4，实际联合 −195.2，**交互项 +2.2，实质为零**。三组区间互不重叠
（组内全距最大 15.5 us）。

**`HOIST_PF` × `HOIST_X`**（各 5 次中位，叠在 f2 + `EVEC=4` 之上）：

| | 中位 | vs 都不开 |
|---|---|---|
| 都不开 | 6742.3 | — |
| 只开 `HOIST_PF` | 6569.4 | −172.9 |
| 只开 `HOIST_X` | 6660.8 | −81.5 |
| 都开 | **6471.2** | **−271.1** |

单独之和 −254.4，实际联合 −271.1，**交互项 −16.7**（6% 量级的轻微协同）。

所以三个 knob 之间没有强耦合。**这和前两个 feature 正好相反**：f1 和 f2 内部的 knob 是强协同的
（都在拆同一条依赖链，交互项比单独效应之和还大，拆开做任何一个另外的收益还埋着），
f3 则是各做各的。把它们放进同一个 feature 的理由是**前两个机制同源**——同一个 `_pf_hoist`
字典、同一个"展开体在同一 block 里、第一份支配后面"的论证——而不是因为耦合。

> **收益的来源和预期不一样。** `HOIST_PF` 省的 988 条里有 527 条是访存指令本身；
> 当初还以为能顺带省约 990 条地址 VALU，实测只省了 154 条（`v_or_b32` 1496、
> `v_lshlrev_b32` 952 一条没少，那些根本不是 `tw_pf` 的地址运算）。
> `HOIST_X` 同理：VALU 只降 46 条，**364 条全在 LDS 上**。
> 两者都印证了 12.2 那条"斜率不是常数"——**砍什么指令比砍多少条更重要**。

> **被证伪的：`tile_n=256` + `e_vec=8`。** 第十二章 12.5 那条留了个尾巴——`e_vec=8` 需要 `tile_n=256`，
> 而 VMEM 写还差 120 条。实测（各 5 次中位，叠在 f3 之上）：`tile_n=128` 6473.0 vs
> `tile_n=256` 6775.7，**慢 302.7 us**；不叠 `HOIST_X` 也一样（6821.2）。
> `lds_out` 翻倍到 32768、`num_acc_n` 变 4 的代价远超宽存储的收益。这个方向可以划掉了。

> **一个 env 陷阱。** 采这批数时写成 `env $VARS ...` 传参，而 **zsh 不对未加引号的变量做词分割**，
> 整串被当成一个 `VAR=VALUE`，于是 f2 的五个 knob 全部没生效，"f2 基线"测出来正好等于 base
> （7839.7 vs base 7835.9）。现象是数字看着合理、`cos` 也对，只有和已知基线对照才能发现。
> `run.sh` 用 `#!/usr/bin/env bash` 不受影响；手工跑要么用 bash，要么把 env 逐个写成命令前缀。

### 4.5 效果

同一 session（`20260807-093220`，GPU 4，3 次中位）：

| | f2 | f3 | 差 |
|---|---|---|---|
| `moe_gemm2_0`（stage2 GEMM） | 2571.3 | **2229.1** | **−342.2** |
| e2e | 6802.9 | **6493.7** | −309.3 |

三个 knob 的拆分（按 5.4 两组析因）：`HOIST_PF` 约 −173、`HOIST_X` 约 −82、`EVEC=4` 约 −35，
加上轻微协同，合计约 −310。

每 wave 指令数：

| | f2 | f3 | 差 | target |
|---|---|---|---|---|
| VALU | 6,731 | 5,534 | −1,197 | 3,886 |
| **LDS** | 1,645 | **1,190** | **−455** | 391 |
| SALU | 703 | 588 | −115 | 136 |
| **VMEM 读** | 828 | **312** | **−516** | 198 |
| **VMEM 写** | 492 | **246** | **−246** | 125 |
| MFMA | 1,504 | 1,504 | 0 | 1,504 |
| **合计** | **11,902** | **9,374** | **−2,528** | **6,240** |

**VMEM 读和写这两项到这里基本收干净了**（缺口从 630/366 收到 114/121），
剩下的差距集中到 VALU 和 LDS——也就是 12.5 那 1024 条 `ds_write_b16` 和配套的逐元素打包。


## 五、Feature 4：删掉掩码 epilogue

**e2e −44.1 us（累计补上总差距的 86.4%）**

```bash
FLYDSL_MOE_STAGE2_NO_MASK=1
```

### 5.1 为什么它是死代码

`FASTVALID`（f1 的一半）把掩码版和快速版**两条 epilogue 都编进二进制**，运行时按
"这个 block 的最后一行是不是真实 (token, slot)"选一条。实测命中率 94.10%，
掩码路径只有 5.9% 的块会走。

但在 **sorted-row 布局下，掩码路径根本不需要存在**：

- 存储地址是 `row * model_dim * out_elem_bytes`，`row = bx_m + row_in_tile` 必然落在
  `[bx_m, bx_m + tile_m)`，而 partial 缓冲按 `sorted_token_ids.numel() * model_dim` 开，
  **一定在界内**；
- 归约走 `build_sorted_partial_index` 建的反查表，只扫 `[0, num_valid)` 并把真实的
  (token, slot) 映射到 sorted 行，**padding 行永远不会被读回**。

所以 padding 行写进去的那点垃圾无人问津。改法就是在 emit 时把 `masked` 钉死成 `False`，
只生成一条 epilogue——原来那条 `blk_all_valid` 探测和它周围的分支整个不再发出：

```5004:5014:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                    if _no_mask:
                        # Sorted-row partial layout: a padding row's store lands at
                        # `row * model_dim` inside the padded buffer, and the reduce
                        # gathers through an inverted index built only from the valid
                        # sorted ids -- so nothing ever reads those rows back and writing
                        # them is harmless.  That makes the whole masked epilogue dead
                        # code: emit only the unmasked one, which also drops the
                        # `blk_all_valid` probe and the branch around it.
                        _epi["masked"] = False
                        _call_epilog()
                    elif _fast_valid_block:
```

`_epi["masked"]` 是个**发射期**的状态字典，epilogue 的三个回调都读它。置 False 之后
2.2(b) 那段谓词、以及 `precompute_row` 里的哨兵解码，在生成阶段就不会被写进 IR：

```4862:4867:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        if _epi["masked"]:
                            row_i32 = arith.index_cast(T.i32, row)
                            row_valid0 = arith.cmpi(
                                arith.CmpIPredicate.ult, row_i32, num_valid_i32
                            )
                            t_ok = arith.cmpi(
```

顺带 `blk_all_valid` 那次探测读和它的三次比较也没了。

**这一条依赖 f1 而不是 f3**——使能它的是 sorted 行布局，不是 persist 展开。这也是把它单列成
一个 feature 而不是并进 f3 的理由（另见 5.4）。

### 5.2 效果

正确性是这个 feature 最该先看的一栏：**`max_delta` 0.0112、`cos` 0.999995 与开关无关**，
逐位对得上。这直接验证了 5.1 的推理。

| 每 wave | `NO_MASK=0` | `NO_MASK=1` | 差 | target |
|---|---|---|---|---|
| **VALU** | 5,534 | **4,749** | **−785** | 3,886 |
| **SALU** | 588 | **228** | **−360** | 136 |
| LDS | 1,190 | 1,150 | −40 | 391 |
| VMEM 读 | 312 | 311 | −1 | 198 |
| VMEM 写 | 246 | 251 | +5 | 125 |
| MFMA | 1,504 | 1,504 | 0 | 1,504 |
| **合计** | **9,374** | **8,193** | **−1,181** | **6,240** |

```
ISA 行数   15,981 → 7,842   （腰斩）
s_cbranch_execz  320 → 32
VGPR          172 → 196
```

**SALU 一步补到 228，离 target 的 136 只差 92**——原来 452 的缺口去掉了 80%。

### 5.3 一条重要的方法论修正：指令数只对**执行到的**指令成立

这个 feature 的斜率和前面所有 feature 都不一样：

| | 指令变化 | e2e 变化 | 斜率 |
|---|---|---|---|
| f3 的 `HOIST_PF` | −8.3% | −9.5%（GEMM） | 1.14 |
| **f4** | **−12.6%** | **−0.67%** | **0.05** |

砍掉的指令是前者的 1.5 倍，时间收益却只有二十分之一。

原因是**这些指令本来就没在执行**：掩码路径只有 5.9% 的块会走，删掉它主要是砍静态体积。
而 `SQ_INSTS_*` 是按 dispatch 平均的，**把没执行的分支也算进了分母**，所以那"−1,181 条"
严重高估了真实的动态削减。

> 这给第十二章"时间跟每 wave 指令总数走"这条主线加了一个必要的限定：
> **只对实际执行的指令成立**。计数器里混有未执行分支时会失真——判断一个 feature 值不值，
> 不能只看 `SQ_INSTS_*` 的降幅，要先问"这些指令原本执行吗"。

### 5.4 与 f3 的重叠

`HOIST_X` × `NO_MASK` 的 2×2（各 5 次中位，叠在 f3 的 `HOIST_PF` + `EVEC=4` 之上）：

| | 中位 | vs 基线 |
|---|---|---|
| 都不开 | 6570.7 | — |
| 只开 `HOIST_X` | 6473.0 | −97.7 |
| 只开 `NO_MASK` | 6448.1 | −122.6 |
| 都开 | **6421.8** | **−148.9** |

```
单独之和  −220.3
实际联合  −148.9
交互项    +71.4      ← 次可加，两者重叠
```

**这是本文第一个次可加的组合。** f1、f2 内部是强协同（交互比单独之和还大），f3 内部严格可加，
f4 与 f3 则是**部分冗余**——掩码 epilogue 本身也是 X 读的一份副本，`HOIST_X` 已经削掉了一部分，
`NO_MASK` 再把整条路径删掉时就没那么多可省了。

实践含义：**这两个 feature 的收益不能相加，阶梯必须按固定顺序读**。第十章的阶梯里那两个数
是"f3 在 f2 之上"和"f4 在 f3 之上"，换个顺序数字会变（`NO_MASK` 先做的话是 −122.6）。

### 5.5 正确性防护

和 `SCALAR_ASCALE` 一样，这是个**运行时无法自检**的假设，所以 host 侧加了断言
（`moe_kernels.py` 的 `flydsl_moe_stage2`）：

```python
if os.environ.get("FLYDSL_MOE_STAGE2_NO_MASK", "0") in (...):
    if accumulate or not _stage2_sorted_partial():
        raise ValueError(
            "FLYDSL_MOE_STAGE2_NO_MASK requires the sorted-row partial layout ..."
        )
```

内核侧的 `_no_mask` 本来就 and 了 `not accumulate and _SORTED_PARTIAL`，所以配错不会算错、
只会**静默不生效**——而静默不生效同样是个坑（你以为开了、其实没开）。断言把它变成显式失败。

三种组合实测：

| | 结果 |
|---|---|
| `NO_MASK=1` + sorted 布局 | 6415.6 us，pass=True |
| `NO_MASK=1`，`SORTED_PARTIAL` 关 | **ValueError** |
| `NO_MASK=1`，atomic 模式 | **ValueError** |


---

## 六、Feature 5：CShuffle 两端一起加宽

**e2e −90.8 us（累计补上总差距的 93.7%）；stage2 GEMM −119.1 us**

```bash
FLYDSL_MOE_STAGE2_BFIRST=1       # Step 1：累加器改 B-first，一个 lane 拿 4 个连续通道
FLYDSL_MOE_STAGE2_NLANE_FIT=1    # Step 2：cshuffle_nlane 随 e_vec 收窄
FLYDSL_MOE_STAGE2_EVEC=8         #         于是 e_vec 能上到 8，存储变 dwordx4
FLYDSL_MOE_STAGE2_LDSPAD=4       # 解 B-first 引入的 LDS bank 冲突，必须配
```

**四个 knob 缺一不可**，而且前两个单独做都接近 0——析因见 6.4。

### 6.1 Step 1：B-first（12.4 的候选 C）

12.4 已经把根因讲清楚了：旧内核激活当 A、权重当 B，累加器是 `(token, channel)`，一个 lane 的
4 个累加值沿 **token** 排列，落到 LDS 是 4 个不同的行，只能一次写 2 个字节。

翻转成 B-first（权重当 A、激活当 B）之后，一个 lane 的 4 个值是 **4 个连续通道**，
在 `[行][列]` 的 LDS 里天然连续，一条 `ds_write2_b64` 写完：

```
ds_write_b16_d16_hi   1024  →  0
ds_write2_b64            0  →  128
```

**但它引入了一个新的 bank 冲突。** B-first 下相邻 lane 写的是 16 个**不同的行**，
而行跨步 `tile_n = 128` bf16 = 256 B = 64 个 bank ≡ 0 (mod 32)——**16 个 lane 全撞同一组 bank**。
不加 padding 时这一项就让 e2e **倒亏 977 us**，比 B-first 省下的多得多。

`LDSPAD=4` 把行跨步改成 264 B，`264/4 = 66 ≡ 2 (mod 32)`，相邻行错开 2 个 bank，
16 行正好铺满 32 个——冲突归零。实测 `LDSBankConflict` 从 f4 的 12.73% 降到 **6.87%**
（target 是 27.84%）。

pad 的值不能乱选，**只有 4 是对的**：

| `LDSPAD` | e2e（叠在 C0 之上） |
|---|---|
| 0 | 6362.5 |
| **4** | **6311.4** |
| 8 | 6382.8 |

### 6.2 Step 2：`cshuffle_nlane` 是个从没被传过的形参

12.5 说"`e_vec=8` 需要 `tile_n=256`，实测慢 302.7 us"——**这句话是错的**，当时只想到了一条路。

真正的约束在 `mfma_epilogues.py`：

```147:150:aiter/ops/flydsl/kernels/mfma_epilogues.py
    if (int(tile_n) % (int(cshuffle_nlane) * int(e_vec))) != 0:
        raise ValueError(
            f"tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_vec}, got tile_n={tile_n}"
        )
```

Step 2 一轮覆盖 `cshuffle_mlane 行 × (cshuffle_nlane × e_vec) 列`，要求列跨度整除 `tile_n`。
`nlane=32、e_vec=8` 时列跨度是 256，比 `tile_n=128` 还宽，放不进去——**但这 2048 个元素
换个形状就行**：

| `nlane` | `mlane = 256/nlane` | 一轮形状 | 塞得进 64×128 吗 |
|---|---|---|---|
| 32 | 8 | 8 × **256** | 否 |
| **16** | **16** | **16 × 128** | **是** ✓ |

而 `_cshuffle_nlane` 在旧内核里是个**写死的字面量**，`c_shuffle_epilog` 那边它其实是**带默认值
的形参，从来没被传过**：

```95:95:aiter/ops/flydsl/kernels/mfma_epilogues.py
    cshuffle_nlane: int = 32,
```

改成随 `e_vec` 收窄（这个写法直接抄自 `mixed_moe_gemm_2stage.py:1888`，那边早就这么做了）：

```python
if os.environ.get("FLYDSL_MOE_STAGE2_NLANE_FIT", "0") in (...):
    _cshuffle_nlane = min(32, int(tile_n) // _e_vec)
```

再把它真正传给 epilogue，`e_vec=8` 就通了，存储从 `buffer_store_dwordx2` 变
`buffer_store_dwordx4`：

```
buffer_store_dwordx2   256  →  0
buffer_store_dwordx4     0  →  128
```

### 6.3 实现：翻转朝向只有一行，连锁的是它下游的六处

早先评估 C2 时估的是"要重写两个操作数加载器、X 的 LDS 布局、累加器下标语义……等于重写"。
**这个判断错了**，而且错在一个可以事先想清楚的地方。

#### (a) 核心：交换 MFMA 的两个源操作数就够了

```3916:3931:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                                    if _bfirst:
                                        acc_list[acc_idx] = mfma_k64(
                                            acc_list[acc_idx],
                                            b_packs0[ni],
                                            b_packs1[ni],
                                            a0,
                                            a1,
                                        )
                                    else:
                                        acc_list[acc_idx] = mfma_k64(
                                            acc_list[acc_idx],
                                            a0,
                                            a1,
                                            b_packs0[ni],
                                            b_packs1[ni],
                                        )
```

**为什么加载器不用动**：MFMA 的两个源操作数**每 lane 的布局是完全相同的**——
`lane%16` 选 16 维那一侧、`lane/16` 选 K 切片。所以把 W 喂给 src0、X 喂给 src1，
得到的就是转置过的 16×16 结果，**两个 fragment 的加载代码一个字都不用改**。

这是 A×B 和 Bᵀ×Aᵀ = (A×B)ᵀ 在 MFMA 这个对称布局下的直接体现。早先的评估把
"操作数换位置"和"操作数换布局"混为一谈了。

同理 `mfma_scale_f32_16x16x128_f8f6f4` 那条路径也只是换个位置：

```3880:3884:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                                    # B-first transposes the result; both scale
                                    # operands are identical so they need no swap.
                                    src0, src1 = (
                                        (b_128, a_128) if _bfirst else (a_128, b_128)
                                    )
```

#### (b) 下游要跟着改的六处

累加器转置之后，**凡是"这个 lane 的第几个值对应谁"的地方都要重算**：

| # | 位置 | A-first | B-first |
|---|---|---|---|
| 1 | epilogue 的列坐标 | `lane%16 + ni*16` | `lane/16*4 + ni*16` |
| 2 | epilogue 的行坐标 | `mi*16 + lane/16*4 + ii` | `mi*16 + lane%16` |
| 3 | 权重量化 scale `sw` | 每 `ni` 一个标量 | 每 `ni` 一个 **4 宽向量**（4 个连续通道） |
| 4 | 路由权重 `tw` | 每 `mi` 取 4 个（4 个 token） | 每 `mi` 取 **1 个**（一个 token，广播） |
| 5 | LDS 写 | 4 条 `ds_write_b16` | 1 条 `ds_write_b64` |
| 6 | Step 1 的遍历 | `mi × ii` 两层 | 只剩 `mi` 一层 |

第 1 处在 `col_g_bf_list` 里单独算一份——注意 **B 操作数本身的寻址不能跟着变**：

```3439:3442:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                # Under B-first the epilogue's channel mapping moves from lane%16 to
                # lane/16*4 (+0..3), but the B *operand* layout is unchanged, so
                # col_g_list must stay as-is for n_blk/n_intra addressing.
                col_g_bf_list = []
```

第 3、4 处是**一对反向的变化**，也是 f5 的 VALU 涨跌互见的原因：

```python
# sw：A-first 是标量、B-first 变 4 宽向量（4 个连续通道各有各的 scale）
if _bfirst:
    row_w_idx = expert_off + col_g_bf_list[ni]
    sw_vals.append(buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=4, dtype=T.f32))
```

```python
# tw：A-first 要 gather 4 个 token 的权重、B-first 只需 1 个标量广播
if _bfirst:
    tw = tw_pf[mi]
    tw_vec = vector.from_elements(T.vec(4, T.f32), [tw] * 4)
else:
    tws = [tw_pf[(mi * 4) + jj] for jj in range_constexpr(4)]   # 4 次
    tw_vec = vector.from_elements(T.vec(4, T.f32), tws)
```

`tw_pf` 的预取量因此从 `m_repeat × 4` 降到 `m_repeat`（16 → 4 条 load）。

第 5、6 处合在一起就是收益的来源——整个 f32x4 一次落进 LDS：

```4716:4740:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        if _bfirst:
                            # The f32x4 holds 4 consecutive channels of one token, so
                            # the whole accumulator lands in 4 adjacent lds_out slots:
                            # one ds_write_b64 replaces four ds_write_b16.
                            for ni in range_constexpr(num_acc_n):
                                col_local = col_base_local + (ni * 16)
                                sv = _scaled_acc(mi, ni, 0, row)
                                outs = [
                                    _cvt_out(
                                        vector.extract(
                                            sv,
                                            static_position=[jj],
                                            dynamic_position=[],
                                        )
                                    )
                                    for jj in range_constexpr(4)
                                ]
                                v4 = vector.from_elements(T.vec(4, out_elem()), outs)
                                vector.store(
                                    v4,
                                    lds_out,
                                    [row_base_lds + col_local],
                                    alignment=8,
                                )
                            return
```

那 4 次 `vector.extract` + `_cvt_out` + `from_elements` 就是 6.5 里 `v_perm_b32` 那 512 条的
来源——**A-first 下这一步是靠 `ds_write_b16_d16_hi` 白拿的**。

第 6 处在共享的 `mfma_epilogues.py` 里，因为 `ii` 那一层没有了：

```178:189:aiter/ops/flydsl/kernels/mfma_epilogues.py
    def _step1(mi: int):
        # One call per `mi`: the callback emits all 4 channels as a single store,
        # so there is no `ii` axis to iterate here.
        row_in_tile = arith.constant(mi * 16, index=True) + lane_mod_16
        _write_row(
            mi,
            0,
            row_in_tile,
            bx_m + row_in_tile,
            lds_row=lane_mod_16 if chunk_m is not None else None,
        )
```

（f5 当时这里是直接写在 `if bfirst:` 分支里的一个循环；第九章把循环体提成了
`_step1` 好按块调用，`lds_row` 那个参数也是那时加的。B-first 这部分的逻辑没变。）

#### (c) Step 2 完全不用动

**这是 B-first 改动量能控制住的另一半原因。** CShuffle 的 Step 2 用的是它自己的映射
（`m_lane = tx / nlane`、`n_lane = tx % nlane`），和 MFMA 的 lane 布局本来就无关——
它只认"`lds_out` 里 `[行][列]` 摆着一个 `tile_m × tile_n` 的 tile"。朝向翻转只改了
**谁把值写进哪个格子**，格子本身没动，所以读回和存储那一侧一行都不用改。

`bfirst` 因此只作用于 Step 1：

```99:103:aiter/ops/flydsl/kernels/mfma_epilogues.py
    # B-first accumulator orientation: only Step 1's thread->element mapping
    # changes; Step 2 reads lds_out through its own mapping and is unaffected.
    bfirst: bool = False,
    # Row stride of lds_out in elements; defaults to tile_n (unpadded).
    lds_out_stride: int | None = None,
```

#### (d) `LDSPAD` 与 `NLANE_FIT`：各三行

padding 只是把 `lds_out` 的行跨步和分配大小换个数：

```python
_lds_out_stride = int(tile_n) + _lds_pad
lds_out_bytes = 2 * int(tile_m) * _lds_out_stride
```

`c_shuffle_epilog` 那边把写死的 `tile_n` 换成传进来的跨步（列仍在 `[0, tile_n)` 内）：

```152:155:aiter/ops/flydsl/kernels/mfma_epilogues.py
    # ---------------- Step 1: write C tile to LDS (row-major, fp16) ----------------
    # Row stride may exceed tile_n when lds_out is padded to break bank aliasing;
    # columns still live in [0, tile_n).
    tile_n_idx = arith.constant(int(lds_out_stride or tile_n), index=True)
```

`NLANE_FIT` 见 6.2，加上把 `cshuffle_nlane` 真正传给 epilogue，一共三行。

#### (e) 适用范围

B-first 目前**只接了 per-tensor 激活 scale + 向量化缩放这一条路**，其它 epilogue 变体
（原子累加、f32 输出、split-K）仍是 A-first，入口处直接拒绝：

```3064:3069:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    if _bfirst and not (_use_cshuffle_epilog and _vec_scale and not out_is_f32):
        raise ValueError(
            "FLYDSL_MOE_STAGE2_BFIRST currently requires the CShuffle epilogue with "
            "FLYDSL_MOE_STAGE2_SCALAR_ASCALE=1 and FLYDSL_MOE_STAGE2_VEC_SCALE=1 "
            "(f16/bf16 output). Other epilogue variants still assume A-first."
        )
```

**改动量实测：内核 +264 行、共享 epilogue +38 行**，其中一半是 `if _bfirst: ... else: ...`
的两条并存路径。早先估的"等于重写"高估了一个数量级——根源是误以为操作数换位要跟着换布局。

### 6.4 析因：两半都得做，缺一个就接近 0

GPU 0，各 3 次取中位，全部叠在 f4 之上：

| 配置 | 中位 e2e | vs f4 |
|---|---|---|
| f4（`e_vec=4`, `nlane=32`） | 6385.0 | — |
| 只做 Step 2（C0：`e_vec=8` + `nlane=16`） | 6362.5 | −22.5 |
| 只做 Step 2 + pad4 | 6311.4 | −73.6 |
| **两半都做（f5）** | **6275.4** | **−109.6** |
| 参考：只做 Step 1（B-first + pad4，早先单测） | — | **−13.7** |

**Step 1 单独 −13.7、Step 2 单独 −73.6，加起来 −87.3；实际联合 −109.6，协同 −22.3。**

原因很直接：**Step 1 管写、Step 2 管读和存，两端都变宽才吃得满**。只加宽写，读回还是窄的，
数据卡在 LDS 出不去；只加宽读存，写进去的还是 1024 条窄写。

这和 f1（`SORTED_PARTIAL` + `FASTVALID`）、f2（三个 knob）是同一种形态——
**都是一条链上的多个消费者，留任何一个，另外几个的收益就还埋着。**

### 6.5 ISA 与计数器

opcode（旧内核全展开、新内核是循环体，绝对数不可直接比，看的是形态）：

| opcode | f4 | **f5** | target |
|---|---|---|---|
| `ds_write_b16_d16_hi` | **1024** | **0** | 0 |
| `ds_write2_b64` | 0 | **128** | 0 |
| `ds_write_b64` | 0 | 0 | 16 |
| `ds_read2st64_b64` | 128 | 0 | 0 |
| `ds_read_b128` | 12 | 12 | 20 |
| `buffer_store_dwordx2` | **256** | **0** | 0 |
| `buffer_store_dwordx4` | 0 | **128** | 8 |
| `v_perm_b32`（bf16 打包） | 0 | **512** | 32 |

每 MFMA 归一的动态计数器：

| | f4 | **f5** | target |
|---|---|---|---|
| VALU | 3.158 | **3.632** | 2.584 |
| **LDS** | 0.765 | **0.182** | 0.260 |
| **VMEM 写** | 0.167 | **0.083** | **0.083** |
| VMEM 读 | 0.207 | 0.199 | 0.132 |
| SALU | 0.152 | 0.160 | 0.090 |
| `LDSBankConflict` | 12.73% | **6.87%** | 27.84% |
| `MeanOccupancyPerCU` | 7.76 | 7.75 | 15.35 |

**LDS 从 0.765 打到 0.182，已经低于 target 的 0.260；VMEM 写和 target 精确相等。**

代价是 **VALU 从 3.158 涨到 3.632**，每 wave +712 条。大头是 `v_perm_b32` 那 512 条：
A-first 用 `ds_write_b16_d16_hi` 直接写 VGPR 的**高 16 位**，bf16 的截断被折进 store、
**一条指令都不花**；B-first 要宽写就必须先把 2 个 bf16 打包成 dword，这笔就得实打实地付。

**这不是 f5 实现得差——按 MFMA 归一，f5 和 target 的打包成本精确相等，都是 0.333 条
`v_perm_b32` 每条 MFMA。** 打包是宽写的固有成本，target 一分不少地也在付。

VGPR 196 → 198，LDS 28928 → 29440 B。

### 6.6 附带发现：后端漏掉三分之一的打包机会（已放弃，代码已删）

> **结论先行**：这一节记录的 `FLYDSL_MOE_STAGE2_PK2` 实验最终被否掉，代码已从
> `moe_gemm_2stage.py` 删除。它在 f5 上确实有 −13.7~−23.8 us，但叠到 f6（第七章）
> 之后翻转成 **+46.0 us 的净拖累**——f6 已经把打包率提上去了，
> PK2 再插一手反而干扰后端。6.6.4 记录了这个反转，6.6.3 记录了当时那个
> splat 猜测是怎么被证伪的。保留全文是因为**过程结论比结果有用**：它划清了
> 「哪些东西从 MLIR 这一侧够得着、哪些够不着」的边界。

#### 6.6.1 现象

查 f5 剩余 VALU 时发现的，独立于上面四个 knob，曾用 `FLYDSL_MOE_STAGE2_PK2` 门控。

**IR 层是干净的**：f5 的 LLVM IR 里有 **576 条 `<4 x float>` fmul，一条标量都没有**。
但落到 ISA：

```
750 条 v_pk_mul_f32 + 804 条 v_mul_f32（标量）
750 × 2 + 804 = 2304 = 576 × 4     ← 精确对账
```

**576 条向量乘里有 201 条（35%）被后端整条拆成了标量。** 同一段代码里两种结果并存：

```asm
v_pk_mul_f32 v[182:183], v[4:5], v[174:175]     ← 打包了
v_mul_f32_e64 v186, v4, v180                    ┐ 操作数形状完全一样
v_mul_f32_e64 v187, v5, v181                    ┘ 却没打包
```

**先排除了寄存器压力**——打包率和 VGPR 不相关，f4 的 VGPR 更高反而打包得最好：

| 配置 | VGPR | vec4 fmul | pk_mul | 标量 mul | 打包率 |
|---|---|---|---|---|---|
| f2 | 169 | 1024 | 1065 | 494 | 52.0% |
| f3 | 172 | 1024 | 1053 | 518 | 51.4% |
| f4 | 196 | 512 | 714 | 684 | **69.7%** |
| f5 | 198 | 576 | 750 | 804 | 65.1% |

#### 6.6.2 试过的办法：源码直接发 2 宽

`v_pk_mul_f32` 天然是 2 宽的，喂它 4 宽的 `arith.mulf` 等于让后端自己拆再配对。
`_vmul4` 改成发两条 `vector<2xf32>`，1:1 映射：

```python
def _vmul4(a, b):
    if not _pk2:
        return a * b
    halves = []
    for h in range_constexpr(2):
        av = vector.from_elements(T.vec(2, T.f32), [...])   # a 的第 h 个半区
        bv = vector.from_elements(T.vec(2, T.f32), [...])
        halves.append(av * bv)
    return vector.from_elements(T.vec(4, T.f32), [...])     # 重新拼回
```

**结果：有效，但只解决了一部分。**

| | f5 | f5 + PK2 |
|---|---|---|
| IR | 576 × vec4 | 1152 × vec2 |
| `v_pk_mul_f32` | 750 | **814** |
| 标量 `v_mul_f32` | 804 | **676** |
| `v_mov_b32` | 197 | **37** |
| 总指令 | 7407 | **7255**（−152） |
| VGPR | 198 | **194** |

e2e（四次交错测量，控制机器漂移）：

| 轮次 | f5 | f5 + PK2 | 差 |
|---|---|---|---|
| 第一轮 | 6321.3 | 6307.6 | **−13.7** |
| 第二轮 | 6346.8 | 6323.0 | **−23.8** |

两轮区间都不重叠，**−15~24 us 是真的**。但预期是 −50~70，差距来自：
**换成原生 2 宽之后后端仍然拆掉 29%**（1152 → 814 打包 + 338 拆散），
只省了 152 条而不是 402 条。

**所以"宽度不匹配"只是部分原因，还有别的东西在阻止打包。**

`v_mov_b32` 从 197 掉到 37 是个附带好处：那些 mov 原本是为拆散的标量乘凑寄存器用的。

#### 6.6.3 追下去：两个假设都被证伪，够不着

**假设一：splat 操作数**。`_sw_x` 和 `svec` 那两处都有一个操作数是标量广播，
猜测 LLVM 因为"知道它是 uniform"而倾向标量化；target 走的是
`v_pk_mul_f32 ... op_sel_hi:[0,1]`，用硬件广播位而不是真的 splat 成向量。

**证伪**：若成立，被拆散的条数应等于某个乘法点的整数倍。实测 338 条，对不上任何一个：

| 假设来源 | 预期 vec2 条数 | 实测 338 |
|---|---|---|
| 只有 `_sw_x`（splat×向量） | 128 | 不符 |
| 只有 `svec`（splat×向量） | 512 | 不符 |
| 只有 `got`（向量×向量） | 512 | 不符 |
| 两个 splat 点合计 | 640 | 不符 |

而且 flydsl 没有 `rocdl.pk_mul_f32`（只有 cvt 类 pk 转换），就算成立也发不出带
`op_sel_hi` 的指令。更要命的是把残留的标量两两配对检查后发现——**20/20 采样组的
三个操作数全部偶对齐且连续，本来就完全满足 `v_pk_mul_f32` 的硬性要求**。同一段
ISA 里形状完全一样的两条，一条打包了一条没有：

```asm
v_mul_f32_e64 v172, v6, v164                    ┐ 没打包
v_mul_f32_e64 v173, v7, v165                    ┘
v_pk_mul_f32  v[176:177], v[6:7], v[170:171]      ← 形状一样，打包了
```

所以不是"做不到"，是后端"没做"。

**假设二：逐元素 extract 触发中端标量化**。`_cvt_out` 是标量的，每个 vec 结果都得
先 `vector.extract` 拆开才能喂进去；猜测 InstCombine 看到"vec2 fmul 的全部使用者
都是逐元素 extract"就把它改写成两条标量 fmul，而 `v_pk_mul_f32` 只在 ISel 阶段
从 vec2 fmul 生成，一旦提前拆掉就再也回不来。

**证伪**：照这个思路把 `_cvt_out` 改成向量原生（`vector<4xf32>` → bitcast → 移位 →
trunci → bitcast 到 `vector<4xbf16>`）。IR 层面完全按预期生效（标量 `arith.bitcast`
2048→0，标量 `arith.trunci` 1024→256 条向量形式），而且一路活到最终 LLVM IR。
但**最终 ISA 逐字节相同**。原因是前提就不成立：查交给后端的 LLVM IR，
**两个版本都是 576 条 `fmul <4 x float>`，一条标量 fmul 都没有**——中端从没标量化过，
根本没有需要保护的对象。

**结论**：拆散完全发生在 AMDGPU 后端 ISel 内部（576 条 vec4 → 750 打包 + 804 标量，
750×2+804 = 2304 ✓），从 MLIR 这一侧改任何 IR 形状都够不着。这条线到此为止。

#### 6.6.4 反转：f6 之后 PK2 变成负收益，代码已删

f6（第七章，去掉 B 下标拆分里的恒等取模）落地后，打包率**自己**从 65.1% 涨到 73.4%
（`v_mul_f32` 804→612，`v_pk_mul_f32` 750→846）——挪走无关的除法减轻了寄存器压力，
后端自己就多打包了。这反过来印证了 6.6.3：打包决策受寄存器压力和调度影响，
只是从 IR 形状那一侧推不动。

于是重测 PK2 是否还有价值（PTL 已确认开启，GPU 空闲，四轮交错）：

| 轮次 | f6 | f6 + PK2 | 差 |
|---|---|---|---|
| rep0 | 6293.4 | 6341.8 | +48.4 |
| rep1 | 6315.1 | 6351.4 | +36.3 |
| rep2 | 6294.1 | 6333.7 | +39.6 |
| rep3 | 6291.1 | 6337.7 | +46.6 |
| **中位数** | **6293.8** | **6339.8** | **+46.0** |

四轮全部同向，两组区间完全不重叠（f6 最慢 6315.1 < f6+PK2 最快 6333.7）。
PK2 从 −13.7~−23.8 us 翻转成 **+46.0 us**，`_vmul4` 与 `FLYDSL_MOE_STAGE2_PK2`
已从 `moe_gemm_2stage.py` 删除（删除后 ISA 与开关关闭时逐条相同，确认是死代码）。

**教训**：一个针对后端启发式的 codegen 提示，其收益取决于当时的寄存器压力和调度
状态，不是代码的固有属性。上游改动动了这些状态之后必须重测，不能默认它还成立。

### 6.7 效果与修正后的判断

| | f4 | f5 | 差 |
|---|---|---|---|
| `moe_gemm2_0` | 2166.2 | **2037.0** | **−129.2** |
| e2e | 6442.7 | **6347.8** | −94.9 |

stage2 GEMM 到 target 的剩余差距从 400.7 收到 **271.5 us（1.154×）**。

（这两行取自 10.1/10.2 那一批九档同 session 的数据，随阶梯延长重测过。早先只到 f5 时
测的是 `moe_gemm2_0` 2140.8 → 2021.7、e2e 6372.0 → 6281.2，卡不同、绝对值差 1~2%，
结论一样。**本文所有 feature 的收益都以 10.1/10.2 最新那一批为准**，各章的表跟着它走。）

**12.8 那个判断要改。** 它说"剩下的 399.8 us 全部压在累加器朝向这一个根因上"，
并暗示只有重写才能拿到。实际是：

1. **朝向确实是根因，但只解朝向不够**——B-first 单独只值 −13.7 us，
   因为 Step 2 那一半还卡着，而那一半根本不需要动朝向，只是一个从没被传过的形参。
2. **改动量也没那么大**：`NLANE_FIT` 是三行，`LDSPAD` 是给 `lds_out` 加个行 padding。
   真正大的只有 B-first 本身。
3. **收益的构成完全变了**：剩余 1,666 条指令缺口里 VALU 独占 1,576（95%），
   LDS 已经反超 target、VMEM 写已经打平。**访存和 LDS 这两条线到此为止。**

下一步只能从 VALU 下手，而 VALU 的大头是 occupancy 之外的东西——旧内核 VGPR 198 / LDS 29440
对 target 的 118 / 16384，occupancy 仍是 7.75 对 15.35 的 2 倍差距（候选 D：X 进寄存器）。
12.7(1)(2) 证伪 occupancy 那两个实验是在 f1 时代做的，**epilogue 已经榨干、结构差异占比大了
很多，值得重新验一次**再决定。

**第七章就是从这里接着往下走的**：把 f5 剩余的 VALU 按用途拆开后，最大的一块既不是
乘法也不是 bf16 打包，而是整数地址运算（41%），其中一处恒等取模可以整段删掉。

## 七、Feature 6：删掉一个恒等的取模

f5 之后 stage2 GEMM 离 target 还有 260.2 us。这一章是把那一段里**最便宜的一块**拿下来：
一处编译器无法自行证明、因而老老实实发了指令的冗余取模。

### 7.1 定位：剩下的差距已经全是 VALU

先按每 wave 指令数把 f5 和 target 摆开。波数（19,204）、MFMA 数（1,504/wave）、dispatch
数三项完全一致，可以直接比：

| counter | f5/wave | target/wave | f5 多出 |
|---|---|---|---|
| MFMA | 1,504 | 1,504 | 0 |
| **VALU** | **5,462** | **3,886** | **+1,576（+40.5%）** |
| SALU | 241 | 135 | +106 |
| LDS | 274 | 391 | **−117（f5 更少）** |
| VMEM 读 | 299 | 198 | +101 |
| VMEM 写 | 125 | 125 | 0 |
| **合计** | **7,905** | **6,239** | **+1,666** |

**VALU 独占这 1,666 条里的 95%。** 而且两边的 `VALUBusy` 接近（37.6% 对 34.0%），
说明**都是 VALU 发射受限**——f5 不是效率低，是单纯多干了 40% 的 VALU 活。
这一章只有一个战场。

> **一条采集陷阱，记在这里免得再踩。** `results/counters.csv` 里有一次 f5 的
> `SQ_INSTS_VMEM_WR` 是 target 的 **2.00 倍**。这个"恰好 2 倍"极具误导性，很容易顺着
> 推断成"f5 把输出写了两遍"或"f5 的 partial 是 f32、target 是 bf16"——后者甚至能和
> 字节数对上（294912×4096×4 B = 4.83 GB）。实际用同一条命令、同样 6 次 dispatch 重采，
> 两边**完全相同**（都是 2,406,400 条），`TCC_EA0_WRREQ` 也都是 2.46 GB，正对应 bf16
> 部分和。**存档的计数器在拿来做归因前必须重采确认，越是"恰好整数倍"的漂亮数字越要
> 怀疑。** 这次差点为它去改 partial 的 dtype。

把 f5 的静态 VALU 按用途拆开（`v_mfma` 除外）：

| 用途 | 静态条数 | 占 VALU |
|---|---|---|
| **整数地址/下标运算** | **1,663** | **41%** |
| 浮点乘法（标量 804 + 打包 750） | 1,554 | 38% |
| bf16 打包（`v_perm_b32` 512 + 移位 132） | 644 | 16% |

乘法那 38% 已经确认够不着（见 6.6，拆散发生在后端 ISel 内部）；bf16 打包是宽写的固有
成本，6.5 已经证明 target 每 MFMA 付得一分不少。**剩下能动的就是地址运算这 41%，
而且它比乘法还大。**

### 7.2 优化前：`idx2crd` 对非 2 的幂会发 magic-number 取模

在地址运算里翻，发现一段出现了 **64 次**的序列：

```asm
s_mov_b32     s36, 0x15390949
...
v_mul_hi_i32  v10, v4, s36
v_lshrrev_b32 v11, 31, v10
v_ashrrev_i32 v10, 12, v10
v_add_u32     v10, v10, v11
```

这是**按常数取模**的标准展开（乘 magic number 取高位、移位、补符号，再乘回来相减），
反推除数是 **49408**。LLVM IR 里对应的正是 64 条 `srem i32 %x, 49408`。

49408 是哪来的？源码里：

```python
c_n0_static = experts * model_dim // 16          # 193 * 4096 / 16 = 49408
layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
...
row_w = expert_off_idx + col_g
coord_w = fx.idx2crd(row_w, layout_n_blk_intra)
n_blk_list.append(fx.get(coord_w, 0))
n_intra_list.append(fx.get(coord_w, 1))
```

`idx2crd` 把线性下标按 layout 拆成二维坐标，展开就是：

- `coord 0 = (row_w / 16) % 49408` ← 49408 不是 2 的幂，6 条 VALU
- `coord 1 = row_w % 16` ← 是 2 的幂，降成一条掩码

**关键在于这个取模是恒等的**：`row_w = expert_off_idx + col_g`，按构造恒小于
`experts * model_dim = 790,528`，所以商恒小于 49408，取模什么也没做。编译器看不出这个
上界，只能把指令发出来。6 条里有 5 条是白干的。

64 这个数也对得上：`_compute_nidx_for`（persist 的跨 N-tile W2 预取）每个 N-tile 调用
`num_acc_n = 2` 次，32 × 2 = 64。

顺带一提，同一个内核里另外几个 `idx2crd` 的 layout 形状——`(?,?,4,16,16)`、`(64,16)`、
`(4,64)`、`(4,16)`、`(64,64)`——维度全是 2 的幂，早就降成移位加掩码了，没有油水。
**只有 `(49408,16)` 这一个因为首维不是 2 的幂而付了 6 倍代价。**

### 7.3 优化后：换成显式的移位和掩码

既然只需要 `row / 16` 和 `row % 16`，直接写出来即可，16 是 2 的幂：

```3444:3459:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                def _n_blk_intra(row):
                    """Split `row` into (row / 16, row % 16) for the B layout.

                    `idx2crd` lowers the leading coordinate to
                    `(row / 16) % c_n0_static`.  c_n0_static is
                    experts*model_dim/16, not a power of two, so that modulo
                    becomes a magic-number sequence (mul_hi/shift/mul/sub, ~6
                    VALU) instead of the single shift the division needs.  The
                    modulo is a no-op: row is expert_off_idx + col_g, which is
                    bounded by experts*model_dim, so the quotient never reaches
                    c_n0_static -- the compiler just cannot see that bound.
                    """
                    if _fastidx:
                        return row // fx.Index(16), row % fx.Index(16)
                    coord = fx.idx2crd(row, layout_n_blk_intra)
                    return fx.get(coord, 0), fx.get(coord, 1)
```

两个调用点都改走这个 helper——主路径的 `n_blk_list` 构建，以及 persist 跨 N-tile 预取
用的 `_compute_nidx_for`（就是它把这段代码乘了 32 倍）：

```3470:3482:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                    n_blk, n_intra = _n_blk_intra(expert_off_idx + col_g)
```

由 `FLYDSL_MOE_STAGE2_FASTIDX` 门控，默认关。

### 7.4 ISA 验证

64 条 `srem` 全部消失，整条 magic-number 序列连根拔掉：

| opcode | f5 | **f6** | 差 |
|---|---|---|---|
| `srem`（LLVM IR 层） | 64 | **0** | −64 |
| `v_mul_hi_i32` | 64 | **0** | −64 |
| `v_mul_i32_i24` | 64 | **0** | −64 |
| `v_sub_u32` | 128 | **0** | −128 |
| `v_ashrrev_i32` | 223 | **31** | −192 |
| `v_lshrrev_b32` | 132 | **4** | −128 |
| `v_add_u32` | 257 | **128** | −129 |
| **整数地址运算合计** | **1,663** | **1,081** | **−582** |
| **静态 VALU 合计** | **4,067** | **3,261** | **−806** |
| ISA 指令总数 | 7,402 | **6,633** | −769 |

地址运算占 VALU 的比例从 41% 降到 33%。

### 7.5 效果

| | f5 | **f6** | 差 |
|---|---|---|---|
| `moe_gemm2_0` | 2037.0 | **1977.6** | **−59.4** |
| e2e | 6347.8 | **6298.0** | −49.8 |

stage2 GEMM 到 target 的剩余差距从 271.5 收到 **212.1 us（1.120×）**。

每 wave 指令数：

| | f5 | **f6** | target | f6 剩余缺口 |
|---|---|---|---|---|
| VALU | 5,462 | **4,673** | 3,886 | **787** |
| SALU | 241 | 226 | 135 | 91 |
| LDS | 274 | 274 | 391 | −117 |
| VMEM 读 | 299 | 299 | 198 | 101 |
| VMEM 写 | 125 | 125 | 125 | 0 |
| MFMA | 1,504 | 1,504 | 1,504 | 0 |
| **合计** | **7,905** | **7,101** | **6,239** | **862** |

**VALU 降了 789 条/wave，正好是 f5 那 1,576 条缺口的一半。** 其余五项一条没动——
这次改的是纯粹的地址运算，不碰数据通路。

一个值得记的旁证：`VALUBusy` 从 **37.6% 降到 34.0%，与 target 的 34.0% 相同**。
f5 时旧内核的 VALU 发射比 target 更满（因为要挤更多指令进去），f6 之后这个过载消失了。
`MfmaUtil` 41.4% → 43.7%（target 52.6%）。occupancy 一如既往地没动（7.73 对 15.28）——
又一次印证 5.3/12.7 的结论：这个内核是发射受限，不是 occupancy 受限。

### 7.6 附带发现：乘法打包率自己涨了

没预料到的一项——**浮点乘法的打包率跟着涨了**：

| | f5 | **f6** |
|---|---|---|
| 标量 `v_mul_f32` | 804 | **612** |
| 打包 `v_pk_mul_f32` | 750 | **846** |
| 打包率 | 65.1% | **73.4%** |

乘法的**数量和写法一个字都没改**（两边都是 2,304 次 f32 乘法：`750×2+804 = 846×2+612 = 2304`），
纯粹是挪走了无关的除法之后，后端自己多打包了。

这件事有两重意义。一是它证明了**后端的打包决策确实受寄存器压力和调度状态影响**——
6.6 里为了推动它做的两次尝试（改乘法宽度、改转换写法）都失败了，因为从 IR 形状那一侧
够不着，而减轻无关的寄存器压力反倒推动了它。二是它直接导致 PK2 那个 knob 被否掉：
f6 已经把打包率提上去，PK2 再插一手从 −13.7~−23.8 us 翻转成 **+46.0 us 的净拖累**，
代码已删，详见 6.6。

### 7.7 一条方法论教训：静态占比不能线性外推成时间

动手前我按"地址运算占静态 VALU 的 41%、这次能砍掉其中约三分之一"估了 **−190 us**，
实际 GEMM 口径只有 **−59.4**。差了三倍，原因很简单：

**那 64 处取模位于 `_compute_nidx_for`——跨 N-tile 的预取路径，每个 N-tile 执行一次；
而内层的 MFMA 循环每个 N-tile 要跑很多轮。** 静态指令数里它们一条算一条，动态执行频率
却差着量级。5.3 已经记过一次"指令数只对执行到的指令成立"，这次是同一个陷阱的另一副
面孔：**执行到了，但执行的次数远少于它在静态清单里的占比。**

所以估收益之前先确认这段代码在循环的哪一层。**静态 ISA 适合用来定位**（它让这 64 条
取模一眼可见，动态计数器只会告诉你"VALU 多了 1,576 条"），**但不适合用来定价**——
定价必须回到每 wave 的动态计数和实测时间。

## 八、Feature 7：per-tensor 权重 scale 塌缩成标量

f6 之后 stage2 GEMM 离 target 还有 212.1 us，而且 7.1 那张表里 VALU 仍占缺口的 95%。
这一章把那 787 条 VALU 里的绝大部分一次性拿掉——办法不是优化算法，是**发现旧内核在
per-tensor 量化下一直按 per-channel 在算权重 scale**。

### 8.1 定位：差距全在浮点乘法上

先把 f6 的动态 VALU 按用途拆开。跨内核比静态绝对数是无效的（旧内核全展开、新内核有真
循环），所以用**各自实测的每 wave 动态 VALU，再按各自的静态构成比例拆分**——这只假设
构成在实际执行到的代码上是均匀的：

| 用途 | f6/wave | target/wave | f6 多出 |
|---|---|---|---|
| 整数地址/下标 | 1,561 | 1,917 | **−357（f6 已反超）** |
| **浮点乘法** | **2,089** | **952** | **+1,137** |
| 浮点加法/FMA | 0 | 386 | −386 |
| bf16 打包/转换 | 734 | 412 | +322 |
| 数据搬运 mov | 282 | 206 | +76 |
| **VALU 合计** | **4,673** | **3,886** | **+787** |

两件事一眼可见：**第七章之后地址运算这条线已经反超 target**，以及**剩下的差距压倒性
地在浮点乘法——f6 是 target 的 2.2 倍**，这一项单独就超过了总缺口，被其他项部分抵消。

而且不是打包率的问题。按 f6 的打包率反推，f6 每 wave 做约 3,300 次 f32 乘法，target 约
1,790 次——**工作量本身多了 84%**。

### 8.2 优化前：一个标量被当成 790,528 个数在用

旧内核 epilogue 每个 N-tile 发 18 个 vec4 乘法：

| 步骤 | vec4/N-tile | 说明 |
|---|---|---|
| `sw_x[ni] = sw_vec[ni] × sx` | 2 | 向量 × 广播 |
| `svec[mi][ni] = tw[mi] × sw_x[ni]` | 8 | 向量 × 广播 |
| `got = a × svec` | 8 | 真正把 scale 应用到累加器 |

**只有最后 8 个是必需的，前面 10 个都在算 scale 本身。** 之所以要算，是因为
`sw_vals[ni]` 被当成 per-channel 的向量：

```python
if _bfirst:
    # 4 consecutive channels per acc -> one 4-wide load.
    row_w_idx = expert_off + col_g_bf_list[ni]
    sw_vals.append(buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=4, dtype=T.f32))
```

但 host 侧喂进来的是什么？`moe_kernels.py` 里：

```python
flat_w_scale = _expand_per_tensor_scale(w2_scale, E * model_dim, model_dim)
```

而 `_expand_per_tensor_scale` 在 `numel()==1` 时做的是 `flat.expand(rows).contiguous()`
——**把一个标量物化成 193 × 4096 = 790,528 个完全相同的 f32**。内核每个 N-tile 读两个
4 宽向量，读回来的 8 个数全是同一个值，然后用它们做向量乘法。

这和 Feature 2 的 `SCALAR_ASCALE` 是同一个形态：host 为了统一接口把 per-tensor 标量广播
成了满缓冲，内核看到的是一个"看起来 per-channel"的东西，于是照着 per-channel 干活。
第三章解决了激活 scale 这一半，权重 scale 这一半一直留着。

**新内核根本不支持 per-channel**：`moe_stage2_pr1x4.py` 只接受 numel 为 1 或 E 的权重
scale，超出就抛错。它的整条 scale 链因此天然是标量的——这就是那 2.2 倍的来源。

### 8.3 优化后：整条链塌缩成每行一个标量

只要 scale 在一个 expert 的所有通道上是常数（per-tensor 或 per-expert 都满足），链条就是：

```
s      = sx × sw          标量，整个内核算一次（跨 32 个 N-tile 不变，用 _pf_hoist 外提）
s[mi]  = s × tw[mi]       标量，每行一次
got    = a × splat(s[mi]) 向量，每个输出一次
```

18 个 vec4 降到 8 个，**epilogue 每个输出元素从两次向量乘变成一次**。实现上三处改动，
由 `FLYDSL_MOE_STAGE2_SCALAR_WSCALE` 门控、默认关。

**(1)(2) 权重 scale 收成一个标量，并用 f3 的 `_pf_hoist` 外提到整个内核一次。**
原来是每个 N-tile 两次 `vec_width=4` 的加载（那 64 条 `buffer_load_dwordx4` 就是它），
现在只在 `expert_off` 读一个标量，再和 `sx` 乘一次：

```4588:4603:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                    _sx_sw = _pf_hoist.get("sx_sw") if (_wscalar and _hoist_pf) else None
                    if _wscalar and _sx_sw is None:
                        _sw_scalar = (
                            fx.Float32(1.0)
                            if not needs_scale_w
                            else buffer_ops.buffer_load(
                                sw_rsrc, expert_off, vec_width=1, dtype=T.f32
                            )
                        )
                        _sx_sw = (
                            sx_scalar * _sw_scalar
                            if sx_scalar is not None
                            else _sw_scalar
                        )
                        if _hoist_pf:
                            _pf_hoist["sx_sw"] = _sx_sw
```

**(3) `_scaled_acc` 走标量分支。** 关键是缓存的键：scale 现在只随 `mi`（行）变、
**不随 `ni`（通道块）变**，所以 `svec` 按 `mi` 缓存，`num_acc_n` 个通道块共用一份，
路由权重 `tw` 也只乘一次：

```4635:4655:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                            if _wscalar:
                                # Scale is one scalar per row; ni does not enter it.
                                skey = (bool(_epi["masked"]), mi)
                                svec = _svec_cache.get(skey)
                                if svec is None:
                                    s = _sx_sw
                                    if doweight_stage2:
                                        tw = (
                                            tw_pf[mi]
                                            if tw_pf is not None
                                            else buffer_ops.buffer_load(
                                                sorted_w_rsrc,
                                                row,
                                                vec_width=1,
                                                dtype=T.f32,
                                            )
                                        )
                                        s = s * tw
                                    svec = vector.from_elements(
                                        T.vec(4, T.f32), [s] * 4
                                    )
```

外层那两处 `range_constexpr(0 if _wscalar else num_acc_n)`
（`moe_gemm_2stage.py:3745`、`4377`）则是把原来逐通道块的 scale 加载循环整个跳过。

**正确性防护和 `SCALAR_ASCALE` 同款**：内核分辨不出真 per-channel 和被广播的 per-tensor，
喂错了不报错、只是静静算错。所以断言加在 host 侧还能看到未展开张量的地方：

```python
_n = 0 if w2_scale is None else w2_scale.numel()
if _n not in (1, E):
    raise ValueError("FLYDSL_MOE_STAGE2_SCALAR_WSCALE assumes a per-tensor or per-expert ...")
```

### 8.4 ISA 验证

| opcode | f6 | **f7** | 差 |
|---|---|---|---|
| `v_pk_mul_f32` | 846 | **364** | −482 |
| `v_mul_f32` | 612 | **301** | −311 |
| `v_mov_b32` | 197 | **43** | −154 |
| `v_or_b32` | 452 | 325 | −127 |
| `buffer_load_dwordx4` | 259 | **195** | **−64** |
| `v_lshlrev_b32` | 235 | 171 | −64 |
| **静态 VALU** | **3,261** | **2,060** | **−1,201** |
| **总指令** | **6,633** | **5,247** | **−1,386** |

f32 乘法总数 **2,304 → 1,029**（−55%），和"18 个 vec4 降到 8 个"的预期一致。
`buffer_load_dwordx4` 正好少 64 个——就是那 32 个 N-tile × 2 次的 per-channel scale 加载。
`v_mov_b32` 掉了 154 条是附带的：那些 mov 原本是为了给向量 scale 链凑寄存器。

### 8.5 效果

| | f6 | **f7** | 本档收益 | target | 剩余差距 |
|---|---|---|---|---|---|
| `moe_gemm2_0` | 1977.6 | **1827.5** | **−150.1** | 1765.5 | **62.0** |
| e2e | 6298.0 | **6170.9** | −127.1 | 6228.5 | **−57.6** |

**stage2 GEMM 的剩余差距从 212.1 收到 62.0 us（1.120× → 1.035×），一次吃掉 71%。**
e2e 上旧内核已经反超 target 57.6 us——但这不代表 GEMM 追平了，只是 target 那一档的
stage1 慢 51 us、归约慢 45 us（原因见 10.3），把 GEMM 剩下的 62 us 盖过去了。**归因仍然
看 GEMM 那一行。**

每 wave 指令数：

| | f6 | **f7** | target | f7 vs target |
|---|---|---|---|---|
| VALU | 4,673 | **3,498** | 3,886 | **−388** |
| SALU | 226 | 176 | 135 | +41 |
| LDS | 274 | 274 | 391 | −117 |
| VMEM 读 | 299 | 238 | 198 | +40 |
| VMEM 写 | 125 | 125 | 125 | 0 |
| MFMA | 1,504 | 1,504 | 1,504 | 0 |
| **合计** | **7,101** | **5,815** | **6,239** | **−424** |

**f7 的总指令数和 VALU 都已经低于 target。**

### 8.6 瓶颈换了性质，下一章要换个问法

到这里指令数这条线走到头了，因为 f7 已经比 target 少发 424 条指令却仍慢 62 us。看比率：

| | f6 | **f7** | target |
|---|---|---|---|
| `VALUBusy` | 34.2% | **28.5%** | 33.6% |
| `MfmaUtil` | 44.0% | **49.0%** | 52.0% |
| `MeanOccupancyPerCU` | 7.71 | **7.68** | 15.26 |
| `SQ_WAIT_ANY` | 209.1M | **181.3M** | 190.5M |
| `GRBM_GUI_ACTIVE` | 13.11M | **11.82M** | 10.83M |

f6 时旧内核和 target 的 `VALUBusy` 都在 34% 附近，两边都是**发射受限**，所以砍指令直接
换来时间。f7 把 `VALUBusy` 打到 28.5%——**发射不再是瓶颈了**，而且 `SQ_WAIT_ANY` 已经比
target 还低。这时候再砍指令的边际收益会迅速衰减。

剩下唯一还差着量级的结构性指标是 **occupancy：7.68 对 15.26，整整两倍**，对应
VGPR 198 对 118、LDS 29440 对 16384 B。第十二章 12.7 曾用 f1 时代的实验证伪过"occupancy 是
瓶颈"，但那时 epilogue 还没榨干、指令数差着 40%，occupancy 的影响被淹没了。**现在指令数
已经反超，那个证伪的前提不再成立，值得重新验一次**——这是 f8 的第一候选（12.6 的候选 D：
把 X 放进寄存器以省 LDS）。

## 九、Feature 8：CShuffle 分块暂存，把 occupancy 从 8 提到 12

f7 之后出现了一个新局面：**旧内核的指令数已经比 target 少 424 条，GEMM 却还慢 62 us**。
8.6 判断瓶颈已经不是发射，唯一还差着量级的是 occupancy（7.68 对 15.26）。这一章验证
那个判断并把它兑现。

### 9.1 先证明 occupancy 现在真的是瓶颈

第十二章 12.7 曾用两个实验证伪过"occupancy 是瓶颈"，但那是 f1 时代做的——当时指令数还差
40%，occupancy 的影响被淹没了。**前提变了就得重验。**

先算清楚谁在卡。MI308X（gfx942）每 CU：LDS 65536 B、每 SIMD 512 VGPR、4 SIMD，
workgroup 256 线程 = 4 wave：

| | VGPR | 允许 | LDS | 允许 | 实际上限 | 卡在 |
|---|---|---|---|---|---|---|
| f7 | 164 → 168 | 12 wave/CU | 29,440 | 2 wg = **8 wave/CU** | 8 | **LDS** |
| target | 118 → 120 | 16 wave/CU | 16,384 | 4 wg = 16 wave/CU | 16 | VGPR |

实测 `MeanOccupancyPerCU` 7.67 和 15.23，与算出来的 8 和 16 吻合。

然后做一个**只动 occupancy、不动别的**的实验：用 `LDSPAD` 把 LDS 撑大到每 CU 只剩 1 个
workgroup，同时保持 bank 行为完全一致。条件是 `stride_elems % 64 == 4`（这样
`stride_bytes/4 mod 32` 仍是 2，与 6.3 里 pad=4 的推导相同）：

| pad | stride | lds_out | 总 LDS | wg/CU | wave/CU | bank 步长 |
|---|---|---|---|---|---|---|
| 4 | 132 | 16,896 | 29,440 | 2 | 8 | 2 |
| 68 | 196 | 25,088 | 37,632 | 1 | 4 | 2 |

三次交错测量：

| 轮次 | pad=4（8 wave） | pad=68（4 wave） | 差 |
|---|---|---|---|
| rep0 | 6131.3 | 6674.7 | +543.4 |
| rep1 | 6143.0 | 6680.2 | +537.2 |
| rep2 | 6148.4 | 6684.1 | +535.7 |

**occupancy 减半代价 +536 us。** 三次同向、区间不重叠。12.7 那个证伪到此正式作废——
**在 f7 的指令数水平上，occupancy 是真瓶颈**。

### 9.2 LDS 花在哪，以及为什么 persist 不能别名

| 区域 | 字节 | 说明 |
|---|---|---|
| X（persist 存全部 3 个 K-tile） | 12,288 | `3 × 64 行 × 64 stride`，跨 32 个 N-tile 复用 |
| `lds_out`（CShuffle 暂存） | 16,896 | `2 × 64 行 × (128+4)` |
| `lds_tid` | 256 | |
| **合计** | **29,440** | |

非 persist 路径里 X 和 `lds_out` 是**别名共用**同一块（`max` 而非相加），因为 X 在
epilogue 之前就消费完了。但 persist 下 epilogue 每个 N-tile 跑一次、而 X 必须一直活着，
两者被迫分开——这正是 persist 这个设计付出的隐藏代价，之前没人算过这笔账。

X 的 12,288 动不了（persist 的立身之本），`lds_tid` 只有 256 B，**唯一够量的是
`lds_out` 那 16,896**。

### 9.3 关键观察：Step 1 和 Step 2 走的是同一批行

`lds_out` 之所以按整个 tile 分配，是因为 CShuffle 的写和读之间隔着一个 barrier，
看起来需要全部数据就位。但把两边的行索引摊开看：

```python
# Step 1（B-first）：mi 块写的是
row_in_tile = mi * 16 + lane_mod_16          # 行 [mi*16, mi*16+16)

# Step 2：mr 轮读的是
row_local   = mr * CShuffleMLane + m_lane    # 行 [mr*16, mr*16+16)
```

在 `CShuffleMLane == 16` 时（`NLANE_FIT` 在 tile_n=128 / e_vec=8 下正好给出这个值），
**两者的行区间逐块重合**。也就是说第 c 块写完就可以立刻读回，根本不必等其余三块——
`lds_out` 只需要容纳**一块 16 行**，而不是整个 64 行的 tile。

改法就是把「全写完，再全读」换成「逐块写读」：

```
for c in 0..m_repeat-1:
    barrier
    Step 1(c)      # 写 16 行，LDS 行号取块内偏移 lane_mod_16
    barrier
    Step 2(c)      # 读同样 16 行，LDS 行号取 m_lane
```

代价是每个 N-tile 多 3 对 barrier。由 `FLYDSL_MOE_STAGE2_LDSCHUNK` 门控、默认关，
并在 `cshuffle_mlane != 16` 时直接抛错——那种配置下两边行区间对不上，静默算错。

### 9.4 实现：难点在"把顺序变成可交错"，不在分块本身

分块这件事本身只要改行下标，真正花功夫的是**原来的代码把"写"和"读"写死成了两段顺序执行
的直线代码**，barrier 就横在中间。要能按块交错，先得把这两段变成可以按索引调用的东西。

#### (a) 把两段直线代码提成函数

原来的结构是：barrier、循环写全部 `mi`、barrier、循环读全部 `mr`。现在把循环体各自提成
`_step1(mi)` 和 `_step2(mr)`，barrier 从函数里挪到调用处：

```266:297:aiter/ops/flydsl/kernels/mfma_epilogues.py
    if chunk_m is None:
        # Ensure all LDS reads finished before the lds write.
        gpu.barrier()
        if bfirst:
            for mi in range_constexpr(m_repeat):
                _step1(mi)
        else:
            default_epilog(...)
        # Ensure all LDS writes are visible before the shuffle-read.
        gpu.barrier()
        for mr in range_constexpr(m_reps_shuffle):
            _step2(mr)
    else:
        ...
        for c in range_constexpr(m_repeat):
            gpu.barrier()
            _step1(c)
            gpu.barrier()
            _step2(c)
```

`chunk_m is None` 那条分支和改动前**逐字等价**——这也是为什么开关关掉时 ISA 不变。
分块分支就是把同样两个函数按块交替调用，多出的只有 barrier。

#### (b) 行下标：两处各改一行

`lds_out` 只剩一块的高度，所以写和读都要把全局行号换成块内行号。两边的块内偏移正好是
现成的量——写侧是 `lane_mod_16`，读侧是 `m_lane`：

```163:166:aiter/ops/flydsl/kernels/mfma_epilogues.py
    def _write_row(mi: int, ii: int, row_in_tile, row, lds_row=None):
        # row_base_lds = row_in_tile * tile_n; chunked keeps only one chunk of
        # rows resident, so the row index is taken modulo the chunk.
        row_base_lds = (row_in_tile if lds_row is None else lds_row) * tile_n_idx
```

```213:213:aiter/ops/flydsl/kernels/mfma_epilogues.py
        lds_row = m_lane if chunk_m is not None else row_local
```

注意 `row_in_tile = mi*16 + lane_mod_16` 而 `row_local = mr*16 + m_lane`——**块内偏移
就是把那个 `mi*16` / `mr*16` 的基址去掉**，不需要取模运算，也不需要额外寄存器。
这是 9.3 那个"行区间逐块重合"的观察在代码上的兑现：如果两边的块划分不一致，这里就得
真的做除法和取模了。

#### (c) LDS 尺寸跟着块高走

内核侧只需把 `lds_out` 的行数从 `tile_m` 换成一块的高度：

```2907:2908:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    _cshuffle_mlane = int(total_threads) // int(_cshuffle_nlane)
    _lds_out_rows = _cshuffle_mlane if _ldschunk else int(tile_m)
```

`_cshuffle_mlane = total_threads / cshuffle_nlane = 256/16 = 16`。这个 16 必须等于 Step 1
的块高（MFMA 的 16 行），否则两边块划分对不上。

#### (d) 防护：对不上就直接拒绝

这是整个改动唯一有正确性风险的地方。如果 `cshuffle_mlane != 16`，Step 2 的块和 Step 1 的
块跨度不同，块内偏移就不再等价于"去掉基址"，**读到的会是别的块的数据，而且不报错**。
所以两侧都拦一道：

```2909:2915:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    if _ldschunk and (int(tile_m) // 16) != (int(tile_m) // _cshuffle_mlane):
        raise ValueError(
            "FLYDSL_MOE_STAGE2_LDSCHUNK needs Step 1 and Step 2 to walk the same "
            f"row spans, i.e. cshuffle_mlane == 16, got {_cshuffle_mlane} "
            f"(cshuffle_nlane={_cshuffle_nlane}, total_threads={total_threads}). "
            "FLYDSL_MOE_STAGE2_NLANE_FIT=1 gives that at tile_n=128/e_vec=8."
        )
```

```287:292:aiter/ops/flydsl/kernels/mfma_epilogues.py
            raise ValueError("chunk_m requires the B-first Step-1 mapping")
        if m_repeat != m_reps_shuffle:
            raise ValueError(
                f"chunk_m needs Step 1 and Step 2 to walk the same row spans, but "
                f"m_repeat={m_repeat} and m_reps_shuffle={m_reps_shuffle}"
            )
```

注意 `cshuffle_mlane == 16` 不是巧合也不是白来的：它是 **f5 的 `NLANE_FIT` 把
`cshuffle_nlane` 从 32 收到 16 的副产物**（`256/16 = 16`）。在 f5 之前 `nlane=32` →
`mlane=8`，Step 2 一轮只走 8 行、Step 1 一块写 16 行，两边对不上，这个分块**根本做不了**。
f5 当时是为了让 Step 2 能存 `dwordx4` 才收窄 nlane 的，顺手为 f8 铺了路——事后看是运气，
但也说明**同一个参数往往同时约束着好几件事**，改之前值得把它牵动的东西列一遍。

#### (e) 改动量

`c_shuffle_epilog` 净增 54 行（其中一半是把原有循环体提成函数的机械改动），内核侧 25 行
（开关、尺寸、防护、传参各几行）。**没有新增任何算术**——分块前后发的 LDS 读写指令是同一
批，只是发生的顺序和落在 `lds_out` 的哪一行变了。

### 9.5 效果

`lds_out` 16,896 → 4,224（`2 × 16 行 × 132`），总 LDS **29,440 → 16,768**，
每 CU 从 2 个 workgroup 变 3 个：

| | f7 | **f8** | target |
|---|---|---|---|
| LDS (B) | 29,440 | **16,768** | 16,384 |
| VGPR | 164 | 159 | 118 |
| `MeanOccupancyPerCU` | 7.67 | **11.44** | 15.23 |
| `MfmaUtil` | 48.5% | **56.6%** | 52.6% |
| `VALUBusy` | 28.2% | 32.4% | 34.0% |
| `GRBM_GUI_ACTIVE` | 11.68M | **10.10M** | 10.88M |

时间：

| | f7 | **f8** | 本档收益 | target | vs target |
|---|---|---|---|---|---|
| `moe_gemm2_0` | 1824.5 | **1542.2** | **−282.3** | 1763.1 | **−220.9（0.875×）** |
| e2e | 6158.7 | **5979.3** | −179.4 | 6239.3 | **−260.0** |

**旧内核在 stage2 GEMM 本体上反超新内核 220.9 us（快 12.5%），e2e 快 260.0 us。**
`MfmaUtil` 56.6% 也已经高于 target 的 52.6%，`GRBM_GUI_ACTIVE` 低于 target——
不是靠别的算子的差异，是 GEMM 本身更快了。

每 wave 指令数几乎没动（5,815 → 5,879，LDS 那一项因为分块多了 120 条），
**这一档的收益完全来自 occupancy，不来自指令数**——和前七个 feature 的性质都不同。

### 9.6 还剩一档没拿到

f8 是 12 wave/CU，target 是 16。要进 4 个 workgroup 需要 LDS ≤ 16,384，现在是 16,768，
**只差 384 B**。拆开看：

| 项 | 字节 | 能不能省 |
|---|---|---|
| X | 12,288 | 不能，persist 的核心 |
| `lds_out` | 4,224 | pad 占 128 B（`2 × 16 × 4`） |
| `lds_tid` | 256 | **能**：`_bufstore` 路径不读 `t`/`s`，`NO_MASK` 又把掩码路径编译掉了，它已经是死的 |

去掉 `lds_tid` 得 16,512，还差 128；再去掉 pad 正好 16,384。但 pad 不能白去——6.3 记过，
没有它 16 个 lane 会全部落在同一个 bank，f5 时代实测代价是 977 us。

替代方案是 **XOR swizzle**：把列偏移按行号异或（`e' = e XOR ((r & 7) * 8)`），不额外占存储。
但异或的粒度必须 ≥ 读的粒度（Step 2 一次读 8 个元素 = 16 B，占 bit 3 以上），所以只能异或
bit 3~5；16 行落到 8 个不同的 bank，**2 路冲突，而 pad 是 0 路**。要做到 0 路就得把异或
下放到 4 元素（8 B）粒度，那样 Step 2 的一次 `ds_read_b128` 得拆成两次 `ds_read_b64`。

**这条路大概率不划算，先测了冲突的价码**。把 pad 去掉但保持分块（LDS 16,640，仍是
3 workgroup，占用不变），量到的就是纯冲突成本：

| 轮次 | pad=4（无冲突） | pad=0（16 路冲突） | 差 |
|---|---|---|---|
| rep0 | 5966.2 | 6984.1 | +1017.9 |
| rep1 | 5975.2 | 6973.6 | +998.4 |
| rep2 | 5965.4 | 6972.7 | +1007.3 |

**约 1000 us**——即使 tile 已经缩到 16 行，bank 冲突仍是这个内核里最贵的单项，比 f5 时代
量到的 977 us 还略高。而 occupancy 从 8 提到 12 只值 282 us（GEMM 口径）。

按 16 路冲突约 1000 us 粗估，2 路冲突大概在 100~150 us 量级，而 12→16 wave 的收益按
8→12 那一档外推最多也就 150~200 us（且有递减）。**两者同量级，风险大于收益**，除非能做到
真正 0 冲突——那要付两倍的 LDS 读指令。f9 若要做，应该先量 2 路 swizzle 的实际冲突率，
而不是直接实现。

## 十、性能汇总

### 10.1 e2e 阶梯

5 次取中位，组内全距 0.1~0.3%，全部 `pass=True` / `cos=0.999995`。
**十档同一 session**（`20260812-233151`，**GPU 4**，PTL 开——PTL 是什么、为什么必须确认，见 1.2）。

| stage | e2e (us) | 本 feature | 累计 | 已补上差距 | 说明 |
|---|---|---|---|---|---|
| `base` | **7834.5** | — | — | 0% | 旧内核 reduce，未改动 |
| `f1` | **7064.0** | −770.6 | −770.6 | **48.3%** | partial 存储从 (token, slot) 改成 sorted 行序 |
| `f2` | **6715.0** | −349.0 | −1119.5 | **70.2%** | epilogue 的 scale 与地址从逐元素重算改成提前算一次 |
| `f3` | **6470.1** | −244.9 | −1364.4 | **85.5%** | 循环不变量外提 + 输出宽度 |
| `f4` | **6443.1** | −27.0 | −1391.4 | **87.2%** | 删掉掩码 epilogue |
| `f5` | **6340.4** | −102.7 | −1494.1 | **93.7%** | CShuffle 两端一起加宽（第六章） |
| `f6` | **6299.5** | −40.9 | −1535.0 | **96.2%** | 删掉一个恒等的取模（第七章） |
| `f7` | **6158.7** | −140.8 | −1675.8 | **105.1%** | 权重 scale 塌缩成标量（第八章） |
| `f8` | **5979.3** | −179.4 | −1855.2 | **116.3%** | CShuffle 分块暂存，occupancy 8→12（第九章） |
| *(下一个 feature)* | | | | | |
| `target` | **6239.3** | — | −1595.2 | 100% | 新内核 pr1x4 + Triton 归约 |

**f8 在 e2e 上比 target 快 260.0 us，在 stage2 GEMM 本体上快 220.9 us（0.875×）。**
两个口径这次方向一致——不像 f7 那样只是被别的算子的差异盖过去（见 10.3）。

> 跑在哪张卡上取决于当天谁在用，绝对值因此有 1~3% 的卡间差异（见 12.1），比例不受影响。
> `run.sh` 会自己拦被占用的卡。
>
> **这一批数据换了三次卡才拿到。** 中途节点上有别人的大作业，制造出量级完全不同的干扰：
> 单次 e2e 从正常的 6300 us 跳到 24,677、95,636、甚至 330,076 us。3 次重复取中位挡不住
> 这种——只要有两次被打中，中位数就跟着走了（f2 有一次中位数报成 24,677）。**教训有两条**：
> 重复次数要够（现在用 5 次），以及**不要在别的卡上并发跑别的东西**——一次在 GPU 6 采
> 计数器、同时在 GPU 4 跑阶梯，GPU 4 的前三档被主机侧争抢整体抬高了 20%，而且抬得很均匀、
> 没有离群值的样子，光看数据发现不了。附二 T5 那条离群值告警该做了。

> **f3 和 f4 的收益不能相加，阶梯必须按这个顺序读。** 两者次可加（交互 +71.4，见 5.4）：
> 换成先做 f4 的话，f4 是 −122.6 而 f3 只剩 −97.7。表里的数都是"在前一档之上"。

按每 wave 指令数（这才是和时间同向的量，但只对**执行到的**指令成立，见 5.3）：

| | base | f2 | f3 | f4 | f5 | f6 | f7 | **f8** | target |
|---|---|---|---|---|---|---|---|---|---|
| VALU | 12,867 | 6,731 | 5,534 | 4,749 | 5,462 | 4,673 | 3,498 | **3,441** | 3,886 |
| LDS | 2,011 | 1,645 | 1,190 | 1,150 | 274 | 274 | 274 | **394** | 391 |
| SALU | 2,123 | 703 | 588 | 228 | 241 | 226 | 176 | **177** | 135 |
| VMEM 读 | 1,297 | 828 | 312 | 311 | 299 | 299 | 238 | **238** | 198 |
| VMEM 写 | 492 | 492 | 246 | 251 | 125 | 125 | 125 | **125** | 125 |
| MFMA | 1,504 | 1,504 | 1,504 | 1,504 | 1,504 | 1,504 | 1,504 | 1,504 | 1,504 |
| **合计** | **20,293** | **11,902** | **9,374** | **8,193** | **7,905** | **7,101** | **5,815** | **5,879** | **6,239** |

**指令数这条线在 f7 就到头了**：f7 已经比 target 少 424 条却仍慢 62 us（第八章 8.6）。
**f8 的指令数几乎没动（+64，分块多出的 LDS 操作），时间却又降了 282 us——它买的是
occupancy，不是指令数**（第九章）。这是本文第一个不靠减指令取胜的 feature。

（指令计数不像时间那样受机器状态影响，跑几次都一样，所以这张表允许跨 session 拼——
f5 在两个 session 分别测得 5,461 和 5,462。时间表就不行，见 10.1 的注。）

f4 那一列的两个大缺口（LDS 759、VALU 863）**f5 全动了**，但方向相反：

- **LDS 1,150 → 274，已经低于 target 的 391**（−117）
- **VMEM 写 251 → 125，与 target 精确相等**
- **VALU 4,749 → 5,462，反而涨了 713**

f5 净指令 −288，时间 −151 us（GEMM 口径）。代价是**剩余缺口的构成彻底变了**：
1,666 条里 VALU 独占 **1,576（95%）**，其余四项加起来只剩 90 条。访存和 LDS 这两条线
已经走到头（第六章解释了 VALU 为什么涨、以及它是不是还能降）。

**f6 接着只打 VALU 这一项**：5,462 → 4,673（−789），其余五项一条没动，因为它改的是纯
地址运算、不碰数据通路。这 789 条正好是 f5 那 1,576 条缺口的一半。

### 10.2 逐算子阶梯

`AITER_LOG_MORE=1` 的 `device_time_avg`，us/次，与 10.1 同一 session。`-` 表示该 stage 不跑这个 kernel。

| kernel | base | f1 | f2 | f3 | f4 | f5 | f6 | f7 | **f8** | target |
|---|---|---|---|---|---|---|---|---|---|---|
| `moe_gemm2_0`（旧 stage2 GEMM） | **3802.8** | **2861.7** | **2474.2** | **2211.4** | **2169.4** | **2037.6** | **1975.2** | **1824.5** | **1542.2** | — |
| `moe_2stage_down_prefill_1x4_0`（新 stage2 GEMM） | — | — | — | — | — | — | — | — | — | **1763.1** |
| `ck::kernel_moe_gemm`（stage1） | 2421.8 | 2493.6 | 2532.4 | 2555.8 | 2563.5 | 2580.7 | 2596.1 | 2607.5 | 2680.4 | 2661.3 |
| `_topk_sum_kernel`（归约，slab 布局） | 727.7 | — | — | — | — | — | — | — | — | — |
| `_topk_sum_gather_kernel`（归约，sorted 布局） | — | 789.4 | 769.1 | 769.8 | 770.6 | 759.3 | 759.3 | 771.1 | 772.5 | 815.5 |

只看 stage2 GEMM 这一行：**3802.8 → 2861.7 → 2474.2 → 2211.4 → 2169.4 → 2037.6 → 1975.2 → 1824.5 → 1542.2**，
对 target 的 **1763.1**，即 f1 **−941.1**、f2 **−387.5**、f3 **−262.8**、f4 **−42.0**、
f5 **−131.8**、f6 **−62.4**、f7 **−150.7**、f8 **−282.3**，
**最终比 target 快 220.9 us（0.875×）**。

f7 那一档曾出现两个口径方向相反的情况（e2e 已反超、GEMM 仍慢 62 us），到 f8 两边一致了。
但**归因始终看这张表**：stage1 在各档之间有 8~10% 的漂移（见 10.3），e2e 会把它算进来。

这个口径也稳得多。同一个 f6 在三个 session 的 e2e 增量是 −27.6 / −53.6 / −68.7 us（差 2.5 倍），
而 GEMM 增量是 −64.7 / −58.9 / −65.6 us（差 11%）。**e2e 上 f5 和 f6 的分配尤其不可靠**，
但两者之和很稳（−144 / −138 / −140），因为噪声来自它们之外的算子。

### 10.3 三条需要留意的读数

**stage1 在十档下是 2421.8 / 2493.6 / 2532.4 / 2555.8 / 2563.5 / 2580.7 / 2596.1 / 2607.5 / 2680.4 / 2661.3**——
同一个 CK kernel、同一组参数、同一 session，却整体抬高了 10.7%。`target` 那一档多分配了
2.34 GB 的 padded 中间缓冲，怀疑是 HBM 页面放置的副作用。这意味着 e2e 层面有约 240 us
**不能算到 stage2 头上**；做 feature 归因时以 10.2 的 stage2 GEMM 那一行为准，e2e 只作为总账。

**f7 那一档这个偏差足以翻转结论**：e2e 上 f7 比 target 快 57.6 us，GEMM 上却仍慢 62.0 us
——只看 e2e 会以为已经赢了。到 f8 两个口径才真正一致（GEMM −220.9、e2e −260.0），
但那是结果碰巧一致，不是方法可以放松。

**归约在 sorted 布局下比 slab 布局贵 61.7 us**（727.7 → 789.4）。这是 Feature 1 的固有
代价，已经算进那 −941.1 里了。新内核付的是同一笔，而且更贵（815.5）。

**逐算子合计和 e2e 差 −8 ~ +75 us**。这不是矛盾：逐算子那一遍开着 `AITER_LOG_MORE`，
ROCTracer 的开销把每个 kernel 都抬高约 0.5%，而 e2e 那一遍是干净的（见 1.2）。
所以**绝对值看 10.1，构成看 10.2**，两张表之间不要做减法。

---

## 十一、怎么跑

### 11.1 驱动脚本

```bash
cd /data/aiter/moe_stage2_opt

./run.sh --list                  # 看有哪些 stage、各自开什么 env
./run.sh                         # 全部 stage，各 3 次
./run.sh base f1                 # 只跑指定的
./run.sh --repeats 5             # 改重复次数
./run.sh --kernels               # 额外做一遍 AITER_LOG_MORE，出逐算子表
./run.sh --counters              # 额外采一遍硬件计数器
GPU=3 ./run.sh                   # 换卡
./run.sh --no-ptl-check          # 明知 PTL 关着也要跑（数字不可用，仅调试用）
```

输出追加到 `results/e2e.csv` 和 `results/kernels.csv`，每次调用一个 session id，
所以只重跑其中几个 stage 不会丢掉其余的数据。

脚本会做三件容易被忽略但很关键的事：

1. 每次运行前 `unset` 所有 knob 再按 stage 重设，避免父 shell 里 export 过的值串进来；
2. 核对实际派发的 `kernelName2` 与配置是否一致，名字写错时 fused_moe 会静默回落；
3. e2e 和 `AITER_LOG_MORE` 分两遍跑，tracer 的开销不会污染 headline 数字。

### 11.2 加一个 feature

在 `run.sh` 的 `STAGES` 数组里追加一行，格式 `id|config|说明|env`：

```bash
STAGES=(
  "base|old|旧内核 reduce，未做任何改动（起点）|"
  "f1|old|partial 存储从 (token, slot) 改成 sorted 行序 + 去掉哨兵掩码|AITER_FLYDSL_STAGE2_SORTED_PARTIAL=1 FLYDSL_MOE_STAGE2_FASTVALID=1"
  "f2|old|epilogue 的 scale 与地址从逐元素重算改成提前算一次：per-tensor scale 提到入口 + 向量化缩放 + per-block buffer 存储|FLYDSL_MOE_STAGE2_SCALAR_ASCALE=1 FLYDSL_MOE_STAGE2_VEC_SCALE=1 FLYDSL_MOE_STAGE2_BUFSTORE=1"
  "f3|old|<下一个 feature>|<它自己的 knob>"          # <-- 加在这里
  "target|new|新内核 pr1x4 + Triton 归约（目标）|!AITER_PR1X4_TRITON_REDUCE=1"
)
```

env 是**累积**的：`f3` 会带着 `f1`、`f2` 的 knob 一起跑，所以每行只写自己新增的。
`!` 前缀表示这一档不继承（`target` 是另一个内核，从头来）。

如果这个 feature 是**改代码**而不是加 env，仍然给它一行、env 留空，但**在代码里用一个
`AITER_*` 变量把它门控住**——否则代码一落地，`base` 就不再是 base 了，整条阶梯失去意义。
`moe_stage2_pr1x4.py` 里的 `AITER_PR1X4_TRITON_REDUCE` 就是这个模式。

### 11.3 手工跑单条

```bash
cd /data/aiter

# base
AITER_CONFIG_FMOE=moe_stage2_opt/configs/old.csv HIP_VISIBLE_DEVICES=0 \
python test_qmoe_multi.py --token 32768 --model-dim 4096 --inter-dim 192 \
  --expert 193 --topk 9 --activation silu --dtype bf16 --use-g1u1 1 \
  --doweight-stage1 0 --quant fp8 --quant-type per_tensor

# f1：再加两个 env
AITER_FLYDSL_STAGE2_SORTED_PARTIAL=1 FLYDSL_MOE_STAGE2_FASTVALID=1 \
AITER_CONFIG_FMOE=moe_stage2_opt/configs/old.csv HIP_VISIBLE_DEVICES=0 \
python test_qmoe_multi.py ...同上

# target
AITER_PR1X4_TRITON_REDUCE=1 \
AITER_CONFIG_FMOE=moe_stage2_opt/configs/new.csv HIP_VISIBLE_DEVICES=0 \
python test_qmoe_multi.py ...同上
```

`AITER_CONFIG_FMOE` 不能省：不设的话会去 glob `aiter/configs/model_configs/*tuned_fmoe*`
把所有匹配文件合并、按最低 us 选，两边都会跑到 pr1x4 上。

env 必须写成命令前缀而不是 `export`，否则会串到下一次运行。

---

## 十二、优化方法论：怎么找下一个瓶颈

前面十一章讲的是**做了什么**，这一章讲**怎么知道该做什么**——它和任何一个具体 feature 都无关，
是可以直接套用到下一轮优化的流程和陷阱。**要开新一轮优化，先读这一章。**

正文是 f4 之后那次定位的完整实录：四个 feature 把 stage2 GEMM 从 3794.2 us 打到 2161.0 us，
离 target 的 1761.2 还差 **399.8 us（1.227×）**，这 399.8 us 是怎么一步步拆开的。
**它直接产出了 f5**（12.6 的候选 C 就是第六章）。

后来的实测修正了它当时的几处判断，正文都用引用块标了出来。保留这些错误是有意的——
**判断错在哪比结论本身更有参考价值**，它们恰好演示了下面这几条陷阱。

### 12.0 定位流程与已知陷阱

流程是四步，正文按这个顺序展开：

| 步 | 做什么 | 见 |
|---|---|---|
| 1 | 先把测量口径钉死（报比例、注明卡号，否则卡间差异比收益还大） | 12.1 |
| 2 | 按指令类别做每 wave 归一化对比，找出缺口集中在哪一类 | 12.2 |
| 3 | 到 ISA 里定位这一类指令由哪段代码发出 | 12.3、12.4 |
| 4 | 列候选、逐条估收益与改动量，并**主动证伪** | 12.6、12.7 |

贯穿全文的主线索是：**时间跟每 wave 要发射的指令总数走**。这条线索很有效，但有三个边界，
每一个都是踩出来的，分散在各 feature 章里：

| 陷阱 | 一句话 | 见 |
|---|---|---|
| 指令数只对**执行到的**指令成立 | 静态 ISA 里的两条分支只有一条会跑，静态计数会高估 | 5.3 |
| 静态占比**不能线性外推成时间** | f6 削掉 12% 静态 VALU 只换来 0.67% 时间 | 7.7 |
| 这条线索**本身会到头** | f7 已经比 target 少 424 条指令却仍慢 62 us，之后瓶颈换成 occupancy | 8.6、9.1 |
| 够不着的东西要及时收手 | 后端的打包决策改不动，两次尝试都失败，代码已删 | 6.6 |

再加一条不属于分析、但比所有噪声都大的环境陷阱：**PTL 没开会让整机慢一个数量级**，
见 1.2。`run.sh` 现在会自动检查，关着直接拒绝跑。

### 12.1 先说测量：报比例，绝对值必须注明卡号

GPU 4~7 并行各跑 5 次取中位。并行不污染测量——GPU 4 这一列与串行基准
（`20260806-151608`：7828.8 / 7005.8 / 6788.6 / 6194.3）差都在 0.2% 以内：

| GPU | base | f1 | f2 | target | f2−target | f2 补上 |
|---|---|---|---|---|---|---|
| 4 | 7835.9 | 7020.4 | 6778.2 | 6205.8 | 572.4 | 64.9% |
| 5 | 7753.3 | 6932.9 | 6691.9 | 6090.6 | 601.4 | 63.8% |
| 6 | 7736.5 | 6845.3 | 6615.6 | 6035.4 | 580.2 | 65.9% |
| 7 | 7782.4 | 6970.9 | 6739.5 | 6140.2 | 599.3 | 63.5% |
| **四卡全距** | 99.3 | 175.1 | 162.6 | 170.4 | **29.0** | **2.4 pt** |

同一张卡的组内全距 <0.3%，所以这 1.3~2.8% 是**真实的卡间差异**，不是噪声。GPU 6 一贯最快、
GPU 4 一贯最慢。

但注意最后两列：**绝对值跨卡差最多 175 us，剩余差距只差 29 us、补上比例只差 2.4 个百分点。**
比例把卡间差异约掉了。

> 方法论上的结论：**报比例，绝对值必须注明卡号和 session**。第十章的绝对数都来自 GPU 4。

### 12.2 差距落在哪一类指令

每 wave 指令数（`SQ_INSTS_* / SQ_WAVES`）：

| | f4 | target | f4/tgt | 缺口 | 占缺口 |
|---|---|---|---|---|---|
| **VALU** | 4,749 | 3,886 | 1.22 | **863** | **44.2%** |
| **LDS** | 1,150 | 391 | 2.94 | **759** | **38.9%** |
| VMEM 写 | 251 | 125 | 2.01 | 126 | 6.5% |
| VMEM 读 | 311 | 198 | 1.57 | 113 | 5.8% |
| SALU | 228 | 136 | 1.68 | 92 | 4.7% |
| MFMA | 1,504 | 1,504 | 1.00 | 0 | — |
| **合计** | **8,193** | **6,240** | **1.31** | **1,953** | |

和四个 feature 之前相比，结构变了：

| | f2 时的缺口 | f4 时的缺口 | 收掉了 |
|---|---|---|---|
| VALU | 2,845 | 863 | 70% |
| LDS | 1,254 | 759 | 39% |
| VMEM 读 | 630 | 113 | **82%** |
| SALU | 567 | 92 | **84%** |
| VMEM 写 | 366 | 126 | 66% |

**VMEM 读和 SALU 基本收干净了**，剩下的 1,953 条有 **83% 集中在 VALU 和 LDS**，
而且下面会看到这两项是同一个根因的两面。

工作量侧仍然完全相同：`SQ_INSTS_MFMA` 28,876,800、`SQ_WAVES` 19,204 在所有档位上逐位相等。
**四个 feature 一条 MFMA 都没省，差距全在别处。**

### 12.3 ISA 定位：GEMM 内循环依然不是问题所在

把两个内核的 `17_final_isa.s` 按基本块拆开。**静态计数不能跨内核直接比**：旧内核是全展开的
（32 个 N-tile 各一份直线代码），新内核保留了真实的 N 循环（静态只是循环体）。
只有**按每条 MFMA 归一**才有意义。

| | 块数 | ISA 行数 | 每 MFMA 的 VALU | 每 MFMA 的 LDS |
|---|---|---|---|---|
| f4 | 34 | 7,842 | **2.17** | 0.78 |
| target | 3 | 726 | **3.15** | 0.41 |

**旧内核的 GEMM 内循环每条 MFMA 只用 2.17 条 VALU，比 target 的 3.15 还少 31%**——
计算部分不但不差，还更省。f4 之后这个结论比 f2 时更强（那时是 2.55 vs 3.15）。

顺带一个 f4 带来的结构变化：掩码路径删掉之后，**epilogue 不再有独立的基本块**，
整个 kernel 只剩 34 个块（f2 时是 353 个），LLVM 把 epilogue 直接调度进了 MFMA 所在的块。
ISA 也从 16,599 行降到 7,842 行。

**所以不用去动内循环。剩下的 1,953 条几乎全在 epilogue 的写出路径上。**

### 12.4 唯一的热点：epilogue 一次只写 2 个字节

opcode 直方图把问题摊得很平：

| | f4（全展开，32 个 N-tile） | target（循环体，2 个 N-tile） |
|---|---|---|
| `ds_write_b16_d16_hi` | **1024** | 0 |
| `ds_write_b64` / `ds_write_b128` | 0 / 3 | **16 / 3** |
| `ds_read2st64_b64` | 128 | 0 |
| `ds_read_b128` | 12 | **20** |
| `buffer_store_dwordx2` | **256** | 0 |
| `buffer_store_dwordx4` | 0 | **8** |
| `v_perm_b32`（bf16 打包） | 0 | **32** |

每 wave 实际执行 **1024 条 16 位 LDS 写**（`tile_m × tile_n / 256 = 32` 元素/线程/N-tile ×
32 个 N-tile），**占 f4 全部 LDS 指令 1,150 的 89%**。搬运的字节数和 target 差不多，
纯粹是发射条数的差别。

根因是**累加器朝向**：旧内核把激活当 A、权重当 B，累加器是 `(token, channel)`，
**一个 lane 的 4 个累加值沿 token 排列**——落到 LDS 是 4 个不同的行；而沿 n 方向
`col_local` 的步长是 16，也不连续。**两个方向都不连续，所以只能一次写一个 bf16。**

新内核反过来（权重当 A、激活当 B），累加器第一模是输出通道，一个 lane 的 4 个值是 4 个连续
通道，`v_perm_b32` 打包成 2 个 dword、一条 `ds_write_b64` 写完
（`moe_gemm_2stage_gfx942.py:2279-2286`）。

**这一条同时解释了 VALU 和 LDS 两项缺口**：1024 条 LDS 写占 LDS 缺口的绝大部分；
配套的逐元素 bf16 转换（f4 用 bitcast + shift + trunc，target 用一条 `v_perm_b32` 打包两个）
则占 VALU 863 条缺口里的约一半。**12.2 那 83% 是同一个根因。**

### 12.5 剩下的零头

三项加起来 331 条，占缺口 17%，都没有明显的下手点：

- **VMEM 写 126**：f3 的 `EVEC=4` 已经把宽度从 4 B 提到 8 B（`buffer_store_dwordx2`），
  再往上要 `e_vec=8`，而那需要 `tile_n=256`——实测慢 302.7 us，见 4.4 里那条证伪记录。

  > **这一条被 f5 推翻了。** `e_vec=8` 不是非要 `tile_n=256` 不可——约束是
  > `tile_n % (cshuffle_nlane × e_vec) == 0`，把**写死的 `cshuffle_nlane` 从 32 降到 16**
  > 同样能满足，而且不用动 tile。当时只想到了抬 `tile_n` 这一条路。详见 6.2。
- **VMEM 读 113**：f3 的外提之后只剩真正每 N-tile 都不同的量（`sw_pf` 的逐列权重 scale）。
- **SALU 92**：f4 删掉掩码路径后从 588 掉到 228，剩下的是地址基址和循环控制。

### 12.6 候选：只剩一条，而且它就是架构分界线

> **本节的判断被 f5 部分推翻了**：候选 C 做出来了（第六章），LDS 缺口不但补上还反超 target；
> 但"改动量大"是对的，而且它单独做**基本打平**，必须配合另一个当时没看见的旋钮
> （`cshuffle_nlane`）才有收益。原文保留如下。

| 候选 | 能省的指令（每 wave） | 改动量 | 状态 |
|---|---|---|---|
| **C. LDS 写宽度 16 → 64 位** | ~768 LDS + ~430 VALU ≈ **−15%** | **大** | 唯一剩下的 |
| ~~循环不变量外提~~ | — | — | 已做（f3） |
| ~~输出宽度~~ | — | — | 已做（f3，`e_vec=4`；`e_vec=8` 已证伪） |
| ~~掩码 epilogue~~ | — | — | 已做（f4） |
| ~~occupancy / LDS / VGPR / bank 冲突~~ | — | — | 已证伪（12.7） |

**C 有两条路，代价差很远**：

- **C1 只转置 `lds_out` 布局**——让一个 lane 的 4 个 token 行在 LDS 里连续，写就能宽。
  但读回时同一行的相邻列会变成跨 `tile_m` 的跨步，读反而变窄。**改动局部，收益不确定**，
  是唯一还没试过的局部手段。新内核证明这条路走得通（同一块 LDS 的两个转置视图，
  `moe_gemm_2stage_gfx942.py:2177-2185`），只是它的累加器朝向让写和读回能**同时**连续。
- **C2 翻转 MFMA 操作数顺序（B-first）**——根因的正解，但这正是新旧内核的架构分界线，
  改完基本等于把旧内核重写成新内核。

### 12.7 已经证伪的方向

这几条都实测过，**别再往这些方向花时间**。

**(1) occupancy 不是杠杆。** 这份工作早先的核心论点是"瓶颈是 occupancy，要把它从 8 顶到
16 waves/CU"。`n_per_wave=16` 直接证伪：`MeanOccupancyPerCU` 从 7.62 精确翻倍到 15.31
（与 target 的 15.22 一致），`MfmaUtil` 纹丝不动（28.31% → 28.13%），**e2e 反而慢 186 us**。

f2 从反方向又印证了一次：

| | f1 | f2 | 变化 |
|---|---|---|---|
| 每 wave 指令合计 | 14,759 | 11,902 | −19.4% |
| `GRBM_GUI_ACTIVE`（GPU 忙周期） | 20,227,637 | 18,134,001 | **−10.3%** |
| `MeanOccupancyPerCU` | 7.613 | 7.690 | **+1.0%** |

**一个几乎不碰 occupancy 的 feature 拿到了实打实的加速。**

`MfmaUtil` 也不是独立变量：两边 MFMA 忙碌周期相同（每 CU 5.775M），所以
`MfmaUtil = 5.775M / GPU 总周期`，它就是时间的倒数——"提升 MfmaUtil"是循环论证。
`MeanOccupancyPerCU` 同理，按时间加权的驻留 wave 数也跟着时间走。**这两个数描述差距，
不解释差距。**

**(2) 只降 LDS 或只降 VGPR，连 occupancy 都抬不动。** `tile_n` 只影响 `lds_out`，是个干净的旋钮：

| tile_n | LDS | VGPR | 理论允许 | **实测 occ** | e2e |
|---|---|---|---|---|---|
| 64 | 20736 | 169 | LDS 12 / VGPR 8 waves | **5.56** | 7651.9 |
| 128（当前） | 28928 | 169 | LDS 8 / VGPR 8 | **7.65** | **7079.5** |
| 256 | 45312 | 169 | LDS 4 / VGPR 8 | **7.80** | 7185.8 |

LDS 从 28928 降到 20736、理论允许从 8 涨到 12 waves，**实测 occupancy 不升反降**，e2e 慢 8%。
另外两条附带读数：`tile_n` 从 64 到 256 时 VGPR 恒为 169（输出 tile 大小根本不吃 VGPR，
旧内核 169 个 VGPR 里累加器只占 13 个）；tile_n=256 理论只允许 4 waves 实测却有 7.80，
说明 **`MeanOccupancyPerCU` 不能拿来反推资源限制**，只能横向比。

**(3) LDS bank 冲突和 L2 命中都不是关键路径。** 新内核在这几项上明显更差，却仍然快 1.85 倍：

| counter | 旧内核 | target 新内核 | 新/旧 |
|---|---|---|---|
| `SQ_LDS_BANK_CONFLICT` | 38,502,400 | 60,826,240 | **1.580** |
| `LDSBankConflict` | 9.44% | 27.72% | **2.937** |
| `MemUnitStalled` | 0.73% | 3.65% | **5.011** |
| `TCC_HIT_sum` | 58,793,409 | 40,875,892 | 0.695 |
| `TCC_MISS_sum` | 24,688,790 | 27,132,451 | 1.099 |

**(4) 64 位裸指针寻址——f2 已经解决。** 早先测到旧内核在 VGPR 里做了 1263 条 64 位地址运算
（`v_lshl_add_u64` 944 + `v_lshlrev_b64` 319），而新内核只有 18 条标量。f2 的 `BUFSTORE`
用 per-block 描述符把这块拿掉了（实现见 3.2c）。但当时**对收益机制的判断是错的**：
以为是省 VGPR 抬 occupancy，实际省的是指令。

### 12.8 一个需要正视的判断

> **这一节写于 f4 之后，f5 已经把它的核心判断推翻了一半。** 保留原文，修正见第六章末尾。

四个 feature 拿到了 **1,626 us 里的 1,361 us（83.7%）**，stage2 GEMM 上是
3794.2 → 2161.0（**1.76×**）。剩下的 **399.8 us（GEMM 口径）全部压在一个根因上**：
累加器朝向决定了 epilogue 只能一次写 2 个字节，而它同时拖着 LDS 和 VALU 两项缺口。

C1 那条"只转置 LDS 布局"是唯一还没试过的局部手段，值得做个实验版，但它是拿读回宽度换写入宽度，
收益不确定；C2 则等于重写。

所以到这一步应该重新问一次：继续改旧内核，还是直接用新内核。这份追赶工作真正的产出
可能不是"追平"，而是路上拿到的这些**可以移植回任何 FlyDSL epilogue 的通用手法**——
sorted-row 输出布局、per-tensor scale 提到入口、缩放按 f32x4 向量化、per-block 描述符、
循环不变量的 emit 时缓存、以及"布局一变，整条掩码路径就成了死代码"这个观察。

---

## 附：本文不沿用的旧数据

`docs/moe_stage2_pr1x4_vs_atomic_32k.md` 的基线在今天不复现：同一份调优配置
（`t32x128x64_atomic_persist`，`block_m=128/block_m2=32`）它记的是 e2e 8574.2 us /
`moe_gemm2_0` 4185.1 us，实测是 e2e 7009 us / `moe_gemm2_0` 3430.8 us，差 19% / 22%，
且不是均匀的机器状态偏移（新内核那一侧只差 8%）。那份文档 4.2 节把"输出路径 32 位原子 →
128 位普通存储"列为首要收益机制的结论也不成立——同 tile 下 atomic 与 reduce 的 stage2
总代价差在 ±100 us 以内。本文所有数字都在同一天、同一台机器、同一套方法下重采。

---

## 附二、达标之后要回来做的事（TODO）

这些都是"当前为了快速探索而故意留的技术债"，等旧内核追平 target、feature 集合稳定之后再收。

### T1. `SCALAR_ASCALE` / `VEC_SCALE` 改成按量化类型自动启用

**现状**：靠 env knob 手工开，host 侧只有一道断言兜底
（`moe_kernels.py` 的 `flydsl_moe_stage2`）：

```python
if os.environ.get("FLYDSL_MOE_STAGE2_SCALAR_ASCALE", "0") in (...):
    _n = 0 if a2_scale is None else a2_scale.numel()
    if _n != 1:
        raise ValueError(
            "FLYDSL_MOE_STAGE2_SCALAR_ASCALE assumes a per-tensor activation "
            f"scale, but a2_scale has {_n} elements. ..."
        )
```

断言是必要的，因为**不加就是静默算错**。实测 per-token 下开着这个 knob：

| per_token | pass | max_delta | cos | e2e |
|---|---|---|---|---|
| knob 关 | True | 0.0112 | 0.999995 | 6866.5 |
| knob 开（加断言前） | **False** | **0.7324** | **0.955568** | 6639.5 |

没有异常、没有 NaN、还更快（少读了 294912 次 scale），只有对着参考实现比才看得出来。
cos 0.956 这个量级在生产里可能要到下游指标退化才会暴露。

**目标形态**：这个优化在 per-tensor 下是**无条件正确且免费**的，不该让调用方手工开。
`flydsl_moe_stage2` 手上有未广播的 `a2_scale`，能直接判断；把"是不是 per-tensor"
作为编译参数传给 `compile_moe_gemm2`，内核据此自动启用，env 退化成 kill-switch。

**为什么现在不做**：会改 `compile_moe_gemm2` 的签名和 module 名（影响编译缓存 key），
在 feature 集合还在变的阶段做这个会让每次实验都重编。等阶梯收敛再一次性改。

**做完的收益**：f2 的 −256 us 从"实验旋钮"变成 HY3 per_Tensor 路径的默认收益。

### T2. `BUFSTORE` 的 4 GiB 前提写进代码

per-block 描述符只覆盖 `tile_m × model_dim × out_elem_bytes = 512 KB`，与 token 数无关，
所以比原来的全局 64 位寻址更安全。但这个推理只写在注释里，没有断言。
真正的约束是 **`_SORTED_PARTIAL` 必须为真**（ts 布局下一个 workgroup 的行散布全缓冲，
没有小窗口能覆盖）——现在靠 `_bufstore` 的 and 条件保证，值得补一条显式的失败路径。

### ~~T3. trace 章的 occupancy 论证需要重写~~ —— 已完成

trace 章后来又整章按 f4 对 target 重写（中间经过 f1、f2 两版）。occupancy 那条因果链改挂到
"每 wave 指令数"上，并补了 f2 的计数器作为正面证据（砍 19.4% 指令、时间降 10.3%，
occupancy 只动 1.0%）。被证伪的三条结论（occupancy、只降一边、bank 冲突/L2）压进 4.7 保留，
因为它们能替后来人省时间；f1 时代的 VGPR 追查结论并进 4.7 第 4 条，因为 f2 已经解决了它。

### T4. `_e_vec` 的默认值跳过了 4，`HOIST_PF` 应该无条件开

```python
_e_vec = 8 if int(tile_n) % (_cshuffle_nlane * 8) == 0 else 2
```

真正的约束是下一行的 `tile_n % (32 * e_vec) == 0`，`tile_n=128` 下 **e_vec=4 合法**，
但这个三元表达式只在 8 和 2 之间二选一，直接掉到 2。实测改成 4 值 −34.5 us（5.3）。
应该改成取满足约束的最大值，`FLYDSL_MOE_STAGE2_EVEC` 退化成 kill-switch。

> **f5 之后这一条要往前再推一步。** 约束里的 `32` 本身就是可调的（`cshuffle_nlane`），
> 把它随 `e_vec` 收窄之后 `e_vec=8` 也合法（6.2）。所以正确的默认值不是"满足
> `tile_n % (32*e_vec)==0` 的最大 e_vec"，而是**在 `(nlane, e_vec)` 这个二维空间里取
> `e_vec` 最大的可行解**：`e_vec = min(8, tile_n*32//total_threads ...)` 之类，
> 再反解 `nlane = min(32, tile_n // e_vec)`。`NLANE_FIT` / `LDSPAD` / `BFIRST`
> 目前都还是 opt-in。

`FLYDSL_MOE_STAGE2_HOIST_PF` 同理。它是纯粹的循环不变量外提，**没有正确性前提、
没有寄存器代价**（实测 VGPR 169 → 169），也不依赖任何其他 knob——留成 opt-in 只是为了
在阶梯里能单独计量。默认值应该是开，env 留作 kill-switch。

### T5. `run.sh` 的 e2e 离群值告警

PTL 前置检查已经加了（`check_ptl`，关着就 exit 3，`--no-ptl-check` 可以跳过）。
还差一个离群值告警：`results/e2e.csv` 里 `20260731-153542` 那次 base 有一个 9666.754，
比同组中位数高 23%，当时被当成噪声记下来了——现在回看，**那正是 PTL 掉下去的那一刻**。

组内单点偏离中位数超过 5% 就该打警告。这类信号出现时往往不是噪声，而是机器状态在变。

### T6. `FASTIDX` 改成按专家数自动启用

**现状**：`FLYDSL_MOE_STAGE2_FASTIDX` 手工开、默认关，host 侧没有任何判断。

和 T1 不同，**这个 knob 不存在算错的风险**——被删掉的那个取模在任何 shape 下都是恒等的：

```
row = expert_off_idx + col_g = expert_idx * model_dim + col_g
      expert_idx < experts，col_g < model_dim
⟹  row < experts * model_dim
⟹  row / 16 < experts * model_dim / 16 = c_n0_static
⟹  (row / 16) % c_n0_static 恒等
```

推导只用到 `expert_idx < experts` 和 `col_g < model_dim`，与 tile 尺寸、token 数、topk
都无关（唯一的隐含前提 `model_dim % 16 == 0` 本来就被 B 的 layout 要求了）。
**所以它永远安全，问题只在于有没有收益。**

**收益判据**：`c_n0_static = experts × model_dim / 16`。`model_dim` 实际总是 2 的幂、
`/16` 也是，所以

> `c_n0_static` 是不是 2 的幂 ⟺ **`experts` 是不是 2 的幂**。

| 专家数 | 2 的幂？ | 有收益？ |
|---|---|---|
| 8（Mixtral）、64、128、256 | 是 | ✗ 编译器本来就降成掩码 |
| 160（DeepSeek-V2 路由专家） | 否 | ✓ |
| 60（Qwen MoE） | 否 | ✓ |
| **193（本文）** | 否 | ✓ **f6 的 −806 静态 VALU 全靠这个 193** |

**目标形态**：host 侧一行判断即可，`experts & (experts - 1) != 0` 就自动开，
env 退化成 kill-switch。`experts` 在 `compile_moe_gemm2` 的参数里现成就有，
不像 T1 那样需要新增编译参数。

**注意收益随展开度变**：那 64 条 `srem` 来自 `_compute_nidx_for` 被调
`N-tile 数 × num_acc_n` 次。N-tile 数 = `model_dim / tile_n`，所以 `model_dim` 越大、
`tile_n` 越小，重复份数越多、收益越大；非 persist 路径没有这个放大效应，收益会小很多。

**做完的收益**：f6 的 −59.4 us（stage2 GEMM，见 7.5）对所有专家数非 2 的幂的模型
自动生效，且零风险。
