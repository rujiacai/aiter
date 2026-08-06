# 旧 stage2 内核追赶 pr1x4：逐 feature 优化记录（token=32768）

> shape: token=32768, model_dim=4096, inter_dim=192, expert=193, topk=9, bf16 输出，
> fp8(e4m3fnuz) per_tensor 权重与激活
> 硬件: MI308X（gfx942:sramecc+:xnack-，80 CU），HIP 7.2.53211，torch 2.9.1+rocm7.2.3
> 软件: aiter `moe_opt_0727` @ 6b1cb649；flydsl 0.1.2
> 驱动: `moe_stage2_opt/run.sh`，数据落在 `moe_stage2_opt/results/`

本文是一份**进行中的优化记录**，不是一次性的对比报告。结构是：

- 第一章说明测试怎么做、起点和目标各是多少；
- 之后**每章一个 feature**，记录它做了什么、值多少、为什么；
- 倒数第二章是汇总表，e2e 和逐算子两个口径，每加一个 feature 一行；
- 最后一章是怎么复现。

目标是在**旧内核**上一个 feature 一个 feature 地加，直到追平新内核。

---

## 一、测试情况与基线

### 1.1 起点和目标

两端都是 `fused_moe` 的完整 e2e，stage1 和 moe_sorting 完全相同
（CK `256x64x64x128`、`block_m = block_m2 = 64`），只有 stage2 不同：

| | stage2 内核 | e2e |
|---|---|---|
| **起点 `base`** | 旧 `flydsl_moe2_..._t64x128x64_reduce_persist_bnt0`，未做任何改动 | **7828.8 us** |
| **目标 `target`** | 新 `flydsl_moe2_..._t64x128x64_pr1x4_bnt0` + Triton 归约 | **6194.3 us** |
| | | **差距 1634.5 us（1.264×）** |

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
| `moe_gemm2_0`（stage2 GEMM） | **3801.5** | 48.3% |
| `ck::kernel_moe_gemm`（stage1） | 2426.3 | 30.8% |
| `_topk_sum_kernel`（归约） | 705.8 | 9.0% |
| `_quant_from_per_tensor_amax_kernel` | 464.5 | 5.9% |
| 其余（量化、sorting、elementwise） | 468.9 | 6.0% |
| **合计** | **7867.0** | |

stage2 GEMM 一个就占了一半，所以优化都围绕它。

---

## 二、Feature 1：sorted-row 输出路径

**e2e −823.0 us（补上总差距的 50.4%）；stage2 GEMM −966.1 us**

### 2.1 优化前：每一行输出都被同一个 LDS 读卡住

旧 epilogue 里，每个输出行要先从 LDS 取回 `moe_sorting` 打包的 sorted id，
再从里面解出 token 号 `t` 和 topk 槽号 `s`：

```4179:4182:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                    def precompute_row(*, row_local, row):
                        fused2 = memref.load(lds_tid, [row_local])
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
```

解出来的 `(t, s)` 有**两个下游消费者**，这一点是这个 feature 的全部关键：

**消费者一：算存储地址。** partial 缓冲的布局是 `(token, topk, model_dim)`，
所以行号必须是 `t*topk + s`：

```4223:4226:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                            else:
                                row_byte_base = out_base_idx + ts_idx * fx.Index(
                                    model_dim * out_elem_bytes
                                )
```

**消费者二：算存储谓词。** `moe_sorting` 把每个专家补齐到 `block_m` 的倍数，
补出来的行填的是哨兵值，写出去会污染真实数据，所以每行要查三个条件：

```4183:4197:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
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

```4215:4222:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
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

```2123:2140:aiter/ops/flydsl/moe_kernels.py
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

```2220:2232:aiter/ops/flydsl/moe_kernels.py
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

```4195:4197:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        else:
                            # fast-valid block: every row stores unconditionally.
                            row_valid = None
```

这不是精度换性能。编译期把两条 epilogue 都编进去，运行时选一条：

```4321:4354:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                    if _fast_valid_block:
                        # blk_all_valid is uniform across the workgroup (depends only on
                        # bx_m). moe_sorting pads each expert to a tile_m multiple, so
                        # sentinel padding only ever occupies a block's tail rows. Hence
                        # if the block's LAST row is a real (token, slot) pair, every row
                        # in the block is real and we can run a masking-free epilogue.
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

每格 5~6 次取中位数。**这四格是一批独立的手工运行**（D 格等价于 f1，7085.5 与第五章那一
session 的 7005.8 不同批），只在本节内部可比：

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

`GRBM_GUI_ACTIVE` 降到 0.724，与逐算子表里 `moe_gemm2_0` 的 2835.4/3801.5 = 0.746 吻合，
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
| `moe_gemm2_0` | 3801.5 | **2835.4** | **−966.1** |
| 归约 | 705.8（`_topk_sum_kernel`） | 768.6（`_topk_sum_gather_kernel`）+ 9.8（invert） | +72.6 |
| e2e | 7828.8 | **7005.8** | **−823.0** |

GEMM 本体省 966 us，归约那边因为要 gather 多花 73 us，净赚 823 us。

---

## 三、Feature 2：拆掉 epilogue 的逐行标量链

**e2e −217.2 us（累计补上总差距的 63.6%）；stage2 GEMM −269.2 us**

三个 env knob，和 Feature 1 一样**必须一起开**：

```bash
FLYDSL_MOE_STAGE2_SCALAR_ASCALE=1   # 激活 scale 只读一次
FLYDSL_MOE_STAGE2_VEC_SCALE=1       # 缩放按 f32x4 整体做
FLYDSL_MOE_STAGE2_BUFSTORE=1        # 输出走 per-block buffer 描述符
```

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

**(b) 逐元素取累加器。** `acc[acc_idx]` 是一个 f32x4，但代码一次只取一个：

```python
v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
v = v * (sx_row * sw)
```

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

#### (a) 激活 scale 提到入口

per-tensor 时那个数组里每个元素都一样，读一次就够：

```2785:2790:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
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

关键观察：`write_row_to_lds` 对固定的 `mi` 会被调用 `ii = 0..3` 四次，而
`acc[mi*num_acc_n + ni]` 这个 f32x4 装的**正好就是那四行**。所以缩放可以一次做完，
缓存起来，后三次直接 extract：

```python
def _scaled_acc(mi, ni, ii, row):
    """acc[mi,ni] * scale, as an f32x4 covering ii = 0..3."""
    key = (bool(_epi["masked"]), mi, ni)
    got = _vec_scale_cache.get(key)
    if got is None:
        if doweight_stage2:
            tws = [tw_pf[(mi * 4) + jj] for jj in range_constexpr(4)]
            tw_vec = vector.from_elements(T.vec(4, T.f32), tws)
            svec = tw_vec * _sw_x[ni]
        else:
            svec = _sw_x[ni]
        got = acc[mi * num_acc_n + ni] * svec
        _vec_scale_cache[key] = got
    return got
```

`_sw_x[ni]` 是 `sx * sw_vals[ni]` 提前折好并 splat 成向量的结果——**MLIR 的 `arith.mulf`
不做 vector×scalar 广播**，标量必须先 splat，否则报
`'arith.mulf' op requires the same type for all operands and results`。

**LDS 写仍然逐行**（四行落在四个不同的 LDS 行），这里只向量化了算术。

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

**这一节的数据来自一批独立的手工运行**（f1 基线是 7066.0，与第五章那一 session 的 7005.8
不是同一批），只在本节内部可比，**不要和第五章的绝对值混着读**。5 次取中位，全部叠在 f1 之上：

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

依赖结构（这一条不依赖上面的数字，是代码层面的硬约束）：

```
SCALAR_ASCALE  ──┬──> BUFSTORE      需要 sorted 布局才有小窗口可收（4.7）
   (使能者)      └──> VEC_SCALE     `_vec_scale = _scalar_ascale and env(...)`，直接 and 掉
```

`SCALAR_ASCALE` 把 `ts2 = t*topk + s` 那条链连根拔掉，于是 epilogue 不再需要逐行的标量
上下文，`VEC_SCALE` 才可能整块向量化——逐行的 `sx` 要进 f32x4 就得凑齐 4 行，
而凑齐 4 行正好要把那条链请回来。这是代码里 `and` 出来的，不是测出来的。

这和 Feature 1 是同一种形态——两个 feature 都在拆同一条依赖链
（`sorted id → (t,s) → 地址/谓词/scale 索引`），链上留任何一个消费者，标量化就还在。

### 3.4 效果

时间（与第五章同一 session）：

| | f1 | f2 | 差 |
|---|---|---|---|
| `moe_gemm2_0` | 2835.4 | **2566.2** | **−269.2** |
| e2e | 7005.8 | **6788.6** | −217.2 |

**归因看 GEMM 那一行。** e2e 的 −217.2 被 stage1 在 f2 那一档 +21.7 us 的漂移压低了，
真实收益是 stage2 GEMM 的 −269.2；理由见 5.3。`cos` 全程 0.999995，与 base 一致。

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

f2 自己的计数器还给了第四章那条修正一个直接证据：

| | f1 | f2 | 变化 |
|---|---|---|---|
| 每 wave 指令合计 | 14,759 | 11,902 | **−19.4%** |
| `GRBM_GUI_ACTIVE`（GPU 忙周期） | 20,227,637 | 18,134,001 | **−10.3%** |
| `MeanOccupancyPerCU` | 7.613 | 7.690 | **+1.0%** |

**f2 砍掉近两成指令、时间降了一成，而 occupancy 纹丝不动。** 斜率正好是"砍 2% 指令 ≈ 快 1%"，
与第四章开头那条修正吻合。反过来说：如果 occupancy 真是杠杆，f2 这个 feature 就不该有收益。

> 采集这批数时踩过一次坑：某一趟的 `SQ_INSTS_MFMA` 采成了 34,698,273 而不是 28,876,800。
> 四个 stage 跑的是同一个 GEMM，MFMA 数必须相同，不同就说明那趟被扰动了、所有每-MFMA 的
> 派生值都不能用。`run.sh` 现在会自动检查并告警。


## 四、Trace 分析：剩余差距在哪

f1 之后，两边的输出路径已经完全一致（同一个 stage1、`block_m=block_m2=64`、sorted 行
partial 缓冲、f32 累加、同一个 Triton `_topk_sum_gather_kernel`、无原子、无哨兵掩码），
剩下的差距全在 GEMM 本体。这一章用硬件计数器去定位它。

> **读之前先看这两条。**
>
> **(1) 本章的计数器全部采于 f1，对照的是 target。** 下面每一处"旧内核"都要读成"f1 的旧内核"。
> f2 已经吃掉了其中一部分——4.5 定位到的 64 位裸指针正是 f2 的 `BUFSTORE` 干掉的那一块，
> 事后对账见 4.7。当前的实际差距以 5.1 为准。
>
> **(2) 本章原先的结论是"瓶颈是 occupancy"，这条已经被后续实验证伪。**
> `n_per_wave=16` 把 `MeanOccupancyPerCU` 从 7.62 精确翻倍到 15.31（与 target 的 15.22 一致），
> 而 `MfmaUtil` 纹丝不动（28.31% → 28.13%），e2e 反而慢 186 us。
>
> 正确的结论是：**这个内核发射受限，时间 ≈ 每 wave 要发射的指令总数**，斜率约"砍 2% 指令 ≈ 快 1%"。
> 三条证据——吞吐下界（VALU 1.87M 周期、MFMA 1.44M 周期）只占实际 19.5M 周期的 10%；
> 时间比跟的是 `SQ_ACTIVE_INST_ANY`（2.14×）而不是 `SQ_WAIT_ANY`（1.44×）；
> target 的 `SQ_WAIT_INST_ANY` 反而是 f2 的 2.1 倍，**等得更多却更快**。
>
> `MfmaUtil` 也不是独立变量：两边 MFMA 忙碌周期相同（每 CU 5.775M），所以
> `MfmaUtil = 5.775M / GPU 总周期`，它就是时间的倒数——"提升 MfmaUtil"是循环论证。
>
> **4.1~4.6 的测量数据全部有效，要换的只是它们挂在哪条因果线上。**

### 4.1 两个 GEMM 的计数器对比

`rocprofv3 --pmc`，分别只采 `moe_gemm2_0`（旧内核）和 `moe_2stage_down_prefill_1x4_0`（target），
计数器分组分趟采，每次 dispatch 的平均值。**下表已补上 f2 一列**，与第五章同一批
（`20260806-152049`，`./run.sh --counters`）；标 † 的三项不在 `run.sh` 的 counter 组里，
沿用更早一批采集，它们都是工作量类计数器，不随机器状态变。

> 采集时务必确认没有残留进程占着同一张卡。本文这批数据第一次采集时，一个崩溃的
> rocprofv3 子进程（signal 6 后没退干净）还在后台空转，把 e2e 从 7060 抬到了 8290。
> 硬件计数器本身不受影响，但墙钟时间会被污染。`ps aux | grep test_qmoe` 查一下再采。

**工作量完全相同**：

| counter | f1 旧内核 | **f2 旧内核** | target 新内核 | target/f1 |
|---|---|---|---|---|
| `SQ_INSTS_MFMA` | 28,876,800 | 28,876,800 | 28,876,800 | **1.000** |
| `SQ_WAVES` | 19,204 | 19,204 | 19,204 | **1.000** |
| `SQ_LDS_BANK_CONFLICT` | 38,502,400 | 38,502,400 | 60,826,240 | 1.580 |
| `SQ_VALU_MFMA_BUSY_CYCLES` † | 462,028,800 | — | 462,028,800 | **1.000** |
| `TCP_TCC_WRITE_REQ_sum` † | 37,748,736 | — | 38,502,400 | 1.020 |
| `TCC_EA0_WRREQ_64B_sum` † | 37,748,736 | — | 38,502,400 | 1.020 |

同样多的 MFMA、同样多的 wave、同样多的写请求（而且四舍五入 100% 是 64B 满请求）。
**新内核不是少算了什么，也不是少写了什么。** f2 也一样——它一条 MFMA 都没省，
连 LDS bank 冲突都和 f1 逐位相同。

**时间差在这里**：

| | f1 旧内核 | **f2 旧内核** | target 新内核 | target/f1 |
|---|---|---|---|---|
| `MeanOccupancyPerCU` | **7.61** | **7.69** | **15.19** | **1.995** |
| `MfmaUtil` | 28.23% | 31.35% | 52.40% | 1.856 |
| `GRBM_GUI_ACTIVE`（GPU 忙周期） | 20,227,637 | 18,134,001 | 10,754,719 | **0.532** |
| `SQ_BUSY_CYCLES` | 80,220,620 | 72,025,500 | 42,547,362 | 0.530 |
| `SQ_WAIT_ANY` | 257,237,374 | 215,928,872 | 190,130,941 | 0.739 |

**f2 这一列是关键**：它把忙周期降了 10.3%，而 `MeanOccupancyPerCU` 只从 7.61 动到 7.69。
一个不碰 occupancy 的 feature 拿到了实打实的加速——这是本章开头那条修正最干净的证据。

occupancy 正好 2 倍、MFMA 利用率 1.85 倍、时间 0.54，三个数确实互相自洽——
但**它们是同一件事的三种写法，不是一条因果链**。MFMA 忙碌周期两边完全相同，
所以 `MfmaUtil` 就是时间的倒数；`MeanOccupancyPerCU` 是按时间加权的驻留 wave 数，
同样跟着时间走。**这张表描述了差距，没有解释差距**——本文早先把 occupancy 当成原因，
后来被 `n_per_wave=16` 那个实验直接证伪（见本章开头）。

0.532 与逐算子表里 `moe_gemm2_0` → `down_prefill_1x4` 的 1761.3/2835.4 = 0.62
方向一致（后者含归约以外的差异）。

**指令数——这才是和时间同向的量**：

| counter | f1 旧内核 | **f2 旧内核** | target 新内核 | target/f1 | f2/f1 |
|---|---|---|---|---|---|
| `SQ_INSTS_VALU` | 162,007,504 | 129,257,488 | 74,631,528 | 0.461 | 0.798 |
| `SQ_INSTS_SALU` | 23,591,212 | 13,502,116 | 2,604,904 | **0.110** | 0.572 |
| `SQ_INSTS_LDS` | 33,995,312 | 31,588,912 | 7,507,260 | **0.221** | 0.929 |
| `SQ_INSTS_VMEM_RD` | 25,512,004 | 15,905,204 | 3,800,024 | **0.149** | 0.623 |
| `SQ_INSTS_VMEM_WR` | 9,440,448 | 9,440,448 | 2,406,400 | **0.255** | 1.000 |

`SQ_INSTS_VMEM_WR` 是 0.255 而写字节是 1.020——**同样的数据用 1/4 的指令写完**，
就是 `buffer_store_dwordx4`（16 B）对 `global_store_dword`（4 B）。LDS 侧同理。

### 4.2 occupancy 卡在哪：LDS 和 VGPR 同时顶格

gfx942 每 CU 有 64 KB LDS、每 SIMD 512 个 VGPR（wave64）、每 CU 4 个 SIMD；
两个内核都是每 workgroup 256 线程 = 4 waves。从 kernel descriptor 取资源用量：

| | LDS | VGPR（分配后） | LDS 允许 | VGPR 允许 | 实测 occ |
|---|---|---|---|---|---|
| f1 旧 | 28928 B | 169（176） | 2 WG/CU = **8 waves/CU** | 2 waves/SIMD = **8 waves/CU** | 7.64 |
| target 新 | 16384 B | 118（120） | 4 WG/CU = **16 waves/CU** | 4 waves/SIMD = **16 waves/CU** | 15.29 |

**两边都是 LDS 和 VGPR 同时顶到同一个数**，实测值也贴着理论上限。

- 只把 LDS 从 28928 降到 16384 → LDS 允许 16 waves，VGPR 仍然只允许 8 → **min 还是 8**。
- 只把 VGPR 从 169 降到 128 → VGPR 允许 16，LDS 仍然只允许 8 → **min 还是 8**。

要抬 occupancy 就**必须两个一起降**，且要降到 LDS ≤ 16384 且 VGPR ≤ 128。

> 本文早先由此推出"下一个 feature 就该这么做"。**那个推论作废了**——occupancy 不是杠杆
> （见本章开头）。这一节现在的作用只剩两条：解释 occupancy 为什么被钉死在 8，
> 以及说明为什么单独降一边连 occupancy 都抬不动（4.3 有实测）。
> 但 4.4/4.5 顺着这条线查出来的 VGPR 去向仍然有价值，因为**减少寻址指令本身就是收益**，
> 与它能不能抬 occupancy 无关——f2 的 `BUFSTORE` 就是这么拿到的。

旧内核的 LDS 构成（`moe_gemm_2stage.py:2555-2563`，persist 分支）：

| 项 | 字节 | 说明 |
|---|---|---|
| `lds_x_bytes` | 12288 | X tile 的全部 3 个 K-tile，persist 下要跨整个 N 循环常驻 |
| `lds_out_bytes` | 16384 | CShuffle 暂存区 `2 × tile_m × tile_n` |
| `lds_tid_bytes` | 256 | 每行的 sorted id 暂存 |
| **合计** | **28928** | |

代码注释把这件事说得很清楚：persist 模式下 X 和 `lds_out` **不能复用同一块**，
因为 epilogue 每个 N-tile 跑一次而 X 要一直活着。新内核绕开这一点的办法是把 X
在进 N 循环前整块搬进寄存器（`moe_gemm_2stage_gfx942.py:2289-2290`），LDS 随即让给 C。
12288 + 256 正好是要砍掉的量，砍完是 16384。

### 4.3 实验证伪：只降 LDS 确实没有用

上面那条"只降一个白做"不是推理，直接测了。`tile_n` 只影响 `lds_out`，不影响别的：

| tile_n | LDS | VGPR | LDS 允许 | VGPR 允许 | **实测 occ** | `MfmaUtil` | e2e |
|---|---|---|---|---|---|---|---|
| 64 | **20736** | 169 | 12 waves | 8 waves | **5.56** | 21.80% | 7651.9 |
| 128（当前） | 28928 | 169 | 8 waves | 8 waves | **7.65** | 28.36% | **7079.5** |
| 256 | 45312 | 169 | 4 waves | 8 waves | **7.80** | 27.23% | 7185.8 |

三条读数：

1. **LDS 从 28928 降到 20736（理论允许 12 waves），实测 occupancy 不升反降**（7.65 → 5.56），
   e2e 慢 8%。"只降 LDS"被直接证伪。
2. **`tile_n` 从 64 变到 256，VGPR 恒为 169。** 输出 tile 大小根本不影响 VGPR 用量。
3. 理论上限只是上限：tile_n=256 的 LDS 只允许 4 waves，实测却有 7.80。
   `MeanOccupancyPerCU` 是按时间加权的驻留 wave 数，和静态上限不是一回事，
   **不能拿它反推资源限制**，只能横向比。

### 4.4 VGPR 花在哪：不是累加器

kernel descriptor 里的 `accum_offset` 把 VGPR 分成两段：

| | `next_free_vgpr` | `accum_offset` | 普通 VGPR | AGPR（MFMA 累加器） |
|---|---|---|---|---|
| f1 旧 | 169 | 156 | **156** | 13 |
| target 新 | 118 | 120 | **118** | 0 |

旧内核 169 个里有 **156 个是普通 VGPR**，累加器只占 13 个。这和 4.3 第 2 条对上了——
缩小输出 tile 省不到 VGPR，因为 VGPR 压根不是被累加器吃掉的。

按 tile 几何算一遍也印证这点——两边**每 wave 的累加器footprint 完全相同**：

| | 每 wave 累加器 | 缓冲 | 合计 |
|---|---|---|---|
| f1 旧 | `tile_m 64 × n_per_wave 32` = 2048 f32 / 64 lane = 16 VGPR × 2（f32x4 组） = 32 | 单缓冲 | **32** |
| target 新 | `BLOCK_N 64 × BLOCK_M 64` / 4 waves = 1024 f32 / 64 lane = 16 | 双缓冲 `fragC[0..1]` | **32** |

新内核输出 tile 小一半，但双缓冲正好把它抵消回来。所以差的 51 个 VGPR 在别处。

### 4.5 VGPR 的大头：输出寻址用了 64 位裸指针

统计两个内核的 64 位地址运算，按**标量 / 向量**分开：

| 指令 | f1 旧内核 | target 新内核 |
|---|---|---|
| `v_lshl_add_u64`（向量 64 位） | **944** | **0** |
| `v_lshlrev_b64`（向量 64 位） | **319** | **0** |
| `s_lshl_b64` + `s_add_u32` + `s_addc_u32`（标量 64 位） | 4 | **18** |

**新内核把全部 64 位地址运算搬进了标量寄存器，只做 18 条、在 kernel 入口做一次；
旧内核在向量寄存器里做了 1263 条。** 64 位地址在 VGPR 里要占**一对**寄存器，
而且每条 in-flight 的 store 都得占着它。

落到存储指令形态上：

```asm
; 旧：64 位裸指针放在 VGPR 对里
global_store_dword v[84:85], v98, off nt                  ×512

; 新：buffer 描述符在 SGPR，VGPR 里只有 32 位偏移
buffer_store_dwordx4 v[100:103], v44, s[28:31], 0 offen   ×8
```

#### 旧内核为什么被迫用裸指针

源码里写明了原因：

```3919:3923:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                    # Precompute the output base address when an i32 buffer offset is not enough.
                    # accumulate=False writes a temporary [tokens, topk, model_dim] buffer, which
                    # can exceed 4 GiB for large token counts.
                    out_base_idx = None
                    if (not bool(accumulate)) or _needs_global_atomic_bf16:
```

buffer 描述符的 `num_records` 字段是 **32 位**，最大寻址 4 GiB。而 reduce 模式的 partial
缓冲是 `token × topk × model_dim`，大 token 下会超，所以这条路径退回
`llvm.StoreOp` + `llvm.inttoptr` 的 64 位裸指针（`store_pair`，`moe_gemm_2stage.py:4262-4272`）。

注意旧内核**其实已经建了** `out_rsrc`（`moe_gemm_2stage.py:2773`），只是 reduce 路径没用它。

#### 新内核为什么不需要

它在 kernel 入口就把基址按 block 前移，描述符只覆盖本 block 那一小片：

```2107:2111:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
        arg_p_output = fxh.view_as_torch_tensor(
            _as_ptr(p_output, fx.BFloat16)
            + fx.Int64(e_idx) * (BLOCK_TILE_SIZE_M * N),
            (BLOCK_TILE_SIZE_M, N),
        )
```

`e_idx` 是 block id、**整个 workgroup 一致**，所以这次 64 位乘加编译器放进了 SGPR
（上表那 18 条标量指令）。之后描述符覆盖的窗口只有
`BLOCK_M × N × 2 B = 64 × 4096 × 2 = 512 KB`，**32 位偏移绰绰有余**。

### 4.6 新内核也不是全面更好

这几项新内核明显更差，恰好说明它们**不是**瓶颈：

| counter | f1 旧内核 | target 新内核 | 新/旧 |
|---|---|---|---|
| `SQ_LDS_BANK_CONFLICT` | 38,502,400 | 60,826,240 | **1.580** |
| `LDSBankConflict` | 9.44% | 27.72% | **2.937** |
| `MemUnitStalled` | 0.73% | 3.65% | **5.011** |
| `TCC_HIT_sum` | 58,793,409 | 40,875,892 | 0.695 |
| `TCC_MISS_sum` | 24,688,790 | 27,132,451 | 1.099 |
| `VALUBusy` | 39.95% | 34.01% | 0.851 |

新内核的 LDS bank 冲突率是旧内核的近 3 倍、访存单元阻塞 5 倍、L2 命中率还更低，
但它仍然快 1.85 倍。**所以 bank 冲突和 L2 命中在这个 shape 上都不是关键路径**，
往这两个方向优化旧内核是浪费时间。

### 4.7 事后对账：这几个候选后来怎么样了

本节原先列的是"f2 该做什么"。f2 已经落地，这里改成对账——当时的判断对了几条：

| 候选 | 当时的判断 | 实际结果 |
|---|---|---|
| **A. reduce 路径的输出改成 buffer store** | 4.5 定位到的最大一块，改动局部 | **做了，就是 f2 的 `BUFSTORE`**（实现见 3.2c）。方向对，但收益来源判断错了，见下 |
| B. X 进寄存器 + LDS 合并（28928 → 16384） | 单独做 0；和 A 一起才能把 occupancy 顶到 16 waves/CU | **前提作废**——occupancy 不是杠杆。仍然值得做，但理由要换成"少发指令" |
| C. 只压 VGPR 或只降 LDS | 0 或负 | 4.3 实测慢 8%，**结论不变** |
| D. 修 LDS bank 冲突 / 提 L2 命中 | 大概率 0 | 4.6 已证伪，**结论不变** |

**A 选对了，但预期的收益机制是错的。** 当时的理由是"砍掉 1263 条 64 位向量地址运算，
每条 in-flight store 的地址寄存器减半，VGPR 降下来再配合 B 抬 occupancy"。实际上
`BUFSTORE` 单独只值 −24.9 us（在噪声量级，见 3.3），真正的收益来自它和 `SCALAR_ASCALE`
的协同——**省下来的是指令，不是寄存器压力**。这正是本章开头那条修正的一个实例：
按 occupancy 估值会高估 A，按指令数估值才对得上。

A 的两个边界条件当时判断正确，代码也是照这个落地的：

1. **只在 sorted 布局下成立**。ts 布局（`t*topk+s`）的行散布全缓冲，没有小窗口能覆盖它们。
   所以 `_bufstore` 的使能条件里硬编了 `and _SORTED_PARTIAL`——这个 feature 依赖 f1，不能单独上。
2. **窗口大小与 token 数无关**：`tile_m × model_dim × 2 = 64 × 4096 × 2 = 512 KB`，
   任何 token 规模下都远小于 4 GiB。**所以这个改法比原来的全局 64 位寻址更安全，不是更危险**
   ——4.5 引用的那条 ">4 GiB" 注释针对的是整个 partial 缓冲，按 block 切片之后
   那个约束根本不存在。

改动确实局限在 `precompute_row` / `store_pair`（`moe_gemm_2stage.py:4179-4293`）和描述符构造
（`3939-3976`）三处，**没动 epilogue 的线程映射，也没动共享的 `mfma_epilogues.py`**。

**f3 的候选**：B 仍然是剩下最大的一块，但要按"每 wave 指令数"重新估值而不是按 occupancy。
5.1 那张表里 LDS 一项 f2 仍是 target 的 4.2 倍（1645 vs 391），是缺口最大的一类指令。


## 五、性能汇总

### 5.1 e2e 阶梯

5 次取中位，全距 <0.3%，全部 `pass=True` / `cos=0.999995`。
**四档同一 session**（`20260806-151608`，GPU 4，PTL 开——PTL 是什么、为什么必须确认，见 1.2）。

| stage | e2e (us) | 本 feature | 累计 | 已补上差距 | 说明 |
|---|---|---|---|---|---|
| `base` | **7828.8** | — | — | 0% | 旧内核 reduce，未改动 |
| `f1` | **7005.8** | −823.0 | −823.0 | **50.4%** | sorted-row 输出路径 |
| `f2` | **6788.6** | −217.2 | −1040.2 | **63.6%** | 拆掉 epilogue 的逐行标量链 |
| *(下一个 feature)* | | | | | |
| `target` | **6194.3** | — | −1634.5 | 100% | 新内核 pr1x4 + Triton 归约 |

**当前离目标还差 594.3 us（1.096×）。**

> **e2e 这个口径低估了 f2。** 逐算子表里 stage2 GEMM 降了 **269.2 us**，比 e2e 的 217.2 多 52 us，
> 差额被 stage1 在 f2 那一档 +21.7 us 的漂移和其余小算子的零碎波动吃掉了。
> 归因以 5.2 为准，理由见 5.3。

按每 wave 指令数（这才是和时间同向的量，见第四章）：

| | base | f1 | f2 | target | f1→target 差 | f2 补上 |
|---|---|---|---|---|---|---|
| VALU | 12,867 | 8,436 | 6,731 | 3,886 | 4,550 | 37% |
| LDS | 2,011 | 1,770 | 1,645 | 391 | 1,379 | 9% |
| SALU | 2,123 | 1,228 | 703 | 136 | 1,093 | 48% |
| VMEM 读 | 1,297 | 1,328 | 828 | 198 | 1,131 | 44% |
| VMEM 写 | 492 | 492 | 492 | 125 | 366 | 0% |
| MFMA | 1,504 | 1,504 | 1,504 | 1,504 | 0 | — |
| **合计** | **20,293** | **14,759** | **11,902** | **6,240** | **8,519** | **34%** |

### 5.2 逐算子阶梯

`AITER_LOG_MORE=1` 的 `device_time_avg`，us/次，与 5.1 同一 session。`-` 表示该 stage 不跑这个 kernel。

| kernel | base | f1 | **f2** | target |
|---|---|---|---|---|
| `moe_gemm2_0`（旧 stage2 GEMM） | **3801.5** | **2835.4** | **2566.2** | — |
| `moe_2stage_down_prefill_1x4_0`（新 stage2 GEMM） | — | — | — | **1761.3** |
| `ck::kernel_moe_gemm`（stage1） | 2426.3 | 2496.9 | 2518.6 | 2655.0 |
| `_topk_sum_kernel`（归约，slab 布局） | 705.8 | — | — | — |
| `_topk_sum_gather_kernel`（归约，sorted 布局） | — | 768.6 | 770.2 | 817.7 |
| `_invert_sorted_ids_kernel` / `invert_sorted_ids_kernel_0` | — | 9.8 | 9.8 | 10.3 |
| `_quant_from_per_tensor_amax_kernel` | 464.5 | 470.8 | 472.6 | 487.0 |
| `at::native::vectorized_elementwise_kernel`（各变体合计） | 142.1 | 147.7 | 148.4 | 143.2 |
| `aiter::scaled_quant_kernel` | 116.6 | 119.0 | 119.4 | 119.8 |
| `aiter::data_to_scale_kernel` | 91.1 | 91.8 | 92.1 | 92.3 |
| `_per_tensor_amax_kernel` | 53.2 | 54.0 | 54.4 | 55.4 |
| `scatter_kernel_0` | 33.2 | 34.9 | 35.6 | 38.1 |
| `count_kernel_0` | 17.3 | 17.7 | 17.8 | 18.7 |
| `_fused_init_kernel` | 5.9 | 5.9 | 5.9 | — |
| `aiter::initializeScale` | 4.9 | 5.1 | 5.0 | 5.2 |
| `cumsum_kernel_0` | 4.6 | 4.7 | 4.7 | 4.9 |
| **合计** | **7867.0** | **7062.3** | **6820.7** | **6208.9** |

只看 stage2 GEMM 这一行：**3801.5 → 2835.4 → 2566.2 → 1761.3**，
即 f1 **−966.1**、f2 **−269.2**，**剩余 804.9 us（1.457×）**。

注意这个口径下的剩余差距（804.9 us）比 e2e 口径（594.3 us）**大**，因为 stage1
在 target 那一档反而更慢，e2e 上把 GEMM 的真实差距抵掉了一部分。

### 5.3 三条需要留意的读数

**stage1 在四档下是 2426.3 / 2496.9 / 2518.6 / 2655.0**——同一个 CK kernel、同一组参数、
同一 session，却单调递增了 9.4%。`target` 那一档多分配了 2.34 GB 的 padded 中间缓冲，
怀疑是 HBM 页面放置的副作用。这意味着 e2e 层面有约 229 us **不能算到 stage2 头上**；
做 feature 归因时以 5.2 的 stage2 GEMM 那一行为准，e2e 只作为总账。

f2 就是现成的例子：e2e 只降 217.2，stage2 GEMM 实降 269.2。差的 52 us 里 21.7 是 stage1 的漂移，
其余是七八个小算子每个涨零点几 us 累出来的。**单看 e2e 会把这个 feature 低估两成。**

**归约在 sorted 布局下比 slab 布局贵 72.6 us**（705.8 → 768.6 + 9.8）。这是 Feature 1 的固有代价，
已经算进那 −966.1 里了。新内核付的是同一笔（817.7 + 10.3）。

**逐算子合计比 e2e 高 15~57 us**（四档都是）。这不是矛盾：逐算子那一遍开着 `AITER_LOG_MORE`，
ROCTracer 的开销把每个 kernel 都抬高了约 0.5%（见 1.2）。所以**绝对值看 5.1，构成看 5.2**，
两张表之间不要做减法。

---

## 六、怎么跑

### 6.1 驱动脚本

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

### 6.2 加一个 feature

在 `run.sh` 的 `STAGES` 数组里追加一行，格式 `id|config|说明|env`：

```bash
STAGES=(
  "base|old|旧内核 reduce，未做任何改动（起点）|"
  "f1|old|sorted-row 输出路径：...|AITER_FLYDSL_STAGE2_SORTED_PARTIAL=1 FLYDSL_MOE_STAGE2_FASTVALID=1"
  "f2|old|拆掉 epilogue 的逐行标量链：...|FLYDSL_MOE_STAGE2_SCALAR_ASCALE=1 FLYDSL_MOE_STAGE2_VEC_SCALE=1 FLYDSL_MOE_STAGE2_BUFSTORE=1"
  "f3|old|<下一个 feature>|<它自己的 knob>"          # <-- 加在这里
  "target|new|新内核 pr1x4 + Triton 归约（目标）|!AITER_PR1X4_TRITON_REDUCE=1"
)
```

env 是**累积**的：`f3` 会带着 `f1`、`f2` 的 knob 一起跑，所以每行只写自己新增的。
`!` 前缀表示这一档不继承（`target` 是另一个内核，从头来）。

如果这个 feature 是**改代码**而不是加 env，仍然给它一行、env 留空，但**在代码里用一个
`AITER_*` 变量把它门控住**——否则代码一落地，`base` 就不再是 base 了，整条阶梯失去意义。
`moe_stage2_pr1x4.py` 里的 `AITER_PR1X4_TRITON_REDUCE` 就是这个模式。

### 6.3 手工跑单条

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

### ~~T3. 第四章的 occupancy 论证需要重写~~ —— 已完成

occupancy 那条因果链已经从第四章开头、4.1、4.2、4.7 全部改挂到"每 wave 指令数"上，
并补了 f2 的计数器作为正面证据（f2 砍 19.4% 指令、时间降 10.3%，occupancy 只动 1.0%）。
4.2~4.6 的测量数据保持原样，只换了结论挂在哪条线上。

### T4. `run.sh` 的 e2e 离群值告警

PTL 前置检查已经加了（`check_ptl`，关着就 exit 3，`--no-ptl-check` 可以跳过）。
还差一个离群值告警：`results/e2e.csv` 里 `20260731-153542` 那次 base 有一个 9666.754，
比同组中位数高 23%，当时被当成噪声记下来了——现在回看，**那正是 PTL 掉下去的那一刻**。

组内单点偏离中位数超过 5% 就该打警告。这类信号出现时往往不是噪声，而是机器状态在变。
