# FlyDSL MoE stage2 的 CShuffle：新旧内核对照与改造评估

> 面向不熟悉 CShuffle 的读者。前置知识只要两条：MFMA 是 GPU 的矩阵乘指令，LDS 是片上共享内存。
>
> 对象：旧内核 `moe_gemm2_0`（`kernels/moe_gemm_2stage.py` + `kernels/mfma_epilogues.py`）
> 与新内核 `moe_2stage_down_prefill_1x4_0`（`kernels/moe_gemm_2stage_gfx942.py`）。
> shape 取 token=32768 / model_dim=4096 / inter_dim=192 / expert=193 / topk=9，fp8 输入、bf16 输出。
>
> 相关文档：`moe_stage2_reduce_parity_32k.md`（优化阶梯与实测数据）、
> `../docs/moe_stage2_pr1x4_vs_atomic_32k.md`（新旧内核首次对比）。

---

## 一、CShuffle 要解决什么问题

### 1.1 MFMA 输出的摆法是硬件定死的

`v_mfma_f32_16x16x32_fp8_fp8` 算 `D[16×16] = A[16×32] × B[32×16]`。输出 256 个值分给一个 wave 的
64 个 lane，每 lane 拿 4 个 f32，**分法由硬件规定，软件改不了**：

```
lane l:   N 方向坐标 = l % 16
          M 方向坐标 = (l / 16) * 4 + ii,   ii = 0..3
```

也就是**一个 lane 持有「M 方向 4 个连续值 × N 方向 1 个值」**。

唯一能改的是：**你把哪个张量当 A、哪个当 B**——M 是 A 的行，N 是 B 的列。

### 1.2 全局内存想要的摆法

输出缓冲是 `[行][model_dim]`，通道连续。要让写高效，需要两件事同时成立：

- **相邻 lane 写相邻地址**（合并成大事务）
- **每个 lane 一次写尽量多的字节**（`buffer_store_dwordx4` = 16 B，比 `dword` = 4 B 少 4 倍指令）

后者要求一个 lane 手里攥着**同一行的 8 个连续通道**。

### 1.3 两者对不上

旧内核里 M = token。于是：

```
MFMA 给你的                          全局写想要的
lane 0: 通道 c, token 0~3            lane 0: token t, 通道 0~7
lane 1: 通道 c+1, token 0~3          lane 1: token t, 通道 8~15
...                                  ...
```

一个 lane 手里的 4 个值分属 4 个不同的 token，在输出里相隔 `model_dim × 2 B = 8 KB`。直接写出去
只能发 4 条各 2 字节的独立存储。

**这两个布局之间差一次线程之间的数据交换**，而 GPU 上线程间交换数据的标准途径就是 LDS。
这就是 CShuffle。名字来自 CK（Composable Kernel）：C 指 GEMM 的输出矩阵，shuffle 指洗牌。

> **CShuffle 不是优化，是必须付的过路费。** 能优化的是这趟 LDS 往返走得多宽。

---

## 二、三步流程

不管新旧内核，骨架都是同一个：

```
Step 1   每个线程按 MFMA 给它的 (M, N) 坐标，把累加值写进 LDS 暂存区
  ↓
barrier  等所有线程写完
  ↓
Step 2   每个线程换一套【新的映射】去读——读的是"某一行的一段连续通道"
         然后一次性宽写到全局内存
```

关键在于 **Step 2 读的位置和 Step 1 写的位置不是同一个**。数据在 LDS 里换了主人。

旧内核里这三步被明确标了出来：

```145:145:aiter/ops/flydsl/kernels/mfma_epilogues.py
    # ---------------- Step 1: write C tile to LDS (row-major, fp16) ----------------
```

```188:192:aiter/ops/flydsl/kernels/mfma_epilogues.py
    # Ensure all LDS writes are visible before the shuffle-read.
    gpu.barrier()

    # ---------------- Step 2: shuffle mapping + half2 store/atomic ----------------
    CShuffleNLane = int(cshuffle_nlane)
```

---

## 三、旧内核的实现

参数：`tile_m=64`、`tile_n=128`、256 线程（4 wave）、`m_repeat=4`、`num_acc_n=2`、`e_vec=2`。

### 3.1 Step 1：一次写 2 个字节

外层遍历由 `default_epilog` 驱动，走的正是 MFMA 的原生映射：

```75:81:aiter/ops/flydsl/kernels/mfma_epilogues.py
    for mi in range_constexpr(m_repeat):
        mi_base = arith.constant(mi * 16, index=True)
        for ii in range_constexpr(4):
            row_off = lane_div_16_mul4 + ii_idx_list[ii]
            row_in_tile = mi_base + row_off
            row = bx_m_v + row_in_tile
            body_row(mi=mi, ii=ii, row_in_tile=row_in_tile, row=row)
```

`row_in_tile = mi*16 + lane_div_16*4 + ii` —— 就是 1.1 那条硬件规则。

每一行内部再循环 `ni`，落到 LDS：

```4402:4408:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                                for ni in range_constexpr(num_acc_n):
                                    col_local = col_base_local + (ni * 16)
                                    v = vector.extract(
                                        _scaled_acc(mi, ni, ii, row),
                                        static_position=[ii],
                                        dynamic_position=[],
                                    )
```

LDS 下标是 `row_base_lds + col_local`，其中 `row_base_lds = row_in_tile * tile_n`、
`col_local = n_tile_base + lane_mod_16 + ni*16`。

**两个方向都不连续**：

| 变量 | 相邻取值之间的 LDS 跨步 |
|---|---|
| `ii`（f32x4 的 4 个分量） | `tile_n` = 128 个元素 |
| `ni`（同一行的 2 个列块） | 16 个元素 |

所以只能一个元素一条指令，生成 `ds_write_b16`。每线程每 N-tile：
`m_repeat 4 × ii 4 × ni 2 = 32` 条。

### 3.2 Step 2：换映射读回

新映射把 256 个线程重切成 `8(M) × 32(N)` 的网格，和 MFMA 那套 `lane%16 / lane/16` 毫无关系：

```192:202:aiter/ops/flydsl/kernels/mfma_epilogues.py
    CShuffleNLane = int(cshuffle_nlane)
    CShuffleMLane = int(cshuffle_mlane)
    EVec = int(e_vec)

    m_reps_shuffle = int(tile_m) // CShuffleMLane
    n_reps_shuffle = int(tile_n) // (CShuffleNLane * EVec)

    c_nlane = fx.Index(CShuffleNLane)
    m_lane = tx // c_nlane
    n_lane = tx % c_nlane
    c_evec = fx.Index(EVec)
```

`cshuffle_mlane = 256 / 32 = 8`，所以 `m_reps_shuffle = 64/8 = 8`、
`n_reps_shuffle = 128/(32×2) = 2`。

读回和存储：

```241:257:aiter/ops/flydsl/kernels/mfma_epilogues.py
            for nr in range_constexpr(n_reps_shuffle):
                col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
                col_pair0 = col_base_nr + (n_lane * c_evec)  # even col within tile

                lds_idx_pair = row_base_lds + col_pair0
                frag = vector.load_op(vec_frag, lds_out, [lds_idx_pair])
                loaded.append((col_pair0, frag))

            for col_pair0, frag in loaded:
                store_pair(
                    row_local=row_local,
                    row=row,
                    row_ctx=row_ctx,
                    col_pair0=col_pair0,
                    col_g0=by_n_v + col_pair0,
                    frag=frag,
                )
```

一个线程负责 `m_reps_shuffle × n_reps_shuffle = 8 × 2 = 16` 个片段，每片段 `e_vec = 2` 个 bf16。
读是 `ds_read_b32`（4 B），写是 `buffer_store_dword`（4 B）。

注意 `n_lane * c_evec`：**相邻 lane 的地址差 `e_vec` 个元素**，所以全局写是合并的——
`e_vec=2` 时 32 lane × 4 B = 128 B 连续。合并度没问题，问题在**每条指令只搬 4 字节**。

### 3.3 每线程每 N-tile 的账

| 步骤 | 指令 | 条数 |
|---|---|---|
| Step 1 写 LDS | `ds_write_b16`（2 B） | **32** |
| Step 2 读 LDS | `ds_read_b32`（4 B） | 16 |
| Step 2 写全局 | `buffer_store_dword`（4 B） | 16 |
| **合计** | | **64 条 / 32 个元素 = 2.0 条/元素** |

`e_vec=4`（f3 里开的 `FLYDSL_MOE_STAGE2_EVEC=4`）只改 Step 2：读写各减半到 8 条，
合计 48 条 / 32 元素 = **1.5 条/元素**。Step 1 那 32 条一条不动。

---

## 四、新内核的实现

参数：`BLOCK_M=64`、`BLOCK_N=64`、256 线程。

### 4.1 根因：累加器朝向翻过来了（B-first）

激活被装进**片段 B**，权重装进片段 A：

```2289:2289:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
        frag_act = flyobj.load_tiled_mma_fragB(mm, ldsA, copy_atom_bits=128)
```

累加器也照这个朝向构造，shape 的第一个模（= MFMA 的 M 维）是 `BLOCK_N`（输出通道）：

```2279:2286:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
        c_fake_tensor = fx.make_view(
            fx.get_iter(arg_p_input),
            fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
        )
        fragC = [
            mm.make_fragment_C(c_fake_tensor),
            mm.make_fragment_C(c_fake_tensor),
        ]
```

于是：

| | 旧内核 | 新内核 |
|---|---|---|
| A 操作数 | 激活 | 权重 |
| MFMA 的 M 维 | token | **输出通道** |
| MFMA 的 N 维 | 输出通道 | token |
| **一个 lane 的 4 个累加值** | 4 个 token × 1 个通道 | **4 个通道 × 1 个 token** |

**这一条决定了后面所有差异。**

### 4.2 同一块 LDS 的两个转置视图

```2177:2185:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
        layoutC = fx.make_composed_layout(
            fx.static(swz),
            fx.make_ordered_layout((BLOCK_M, BLOCK_N, 2), (1, 0, 2)),
        )
        layoutCt = fx.make_composed_layout(
            fx.static(swz), fx.make_ordered_layout((BLOCK_N, BLOCK_M, 2), (0, 1, 2))
        )
        ldsC = lds.C.peek().view(layoutC)
        ldsCt = lds.C.peek().view(layoutCt)
```

`make_ordered_layout(shape, order)` 的语义是**order 值越小、stride 越小**。从 dump 的 MLIR
可以直接读出降级结果（`/tmp/isa_tgt/.../00_origin.mlir`）：

```
(64,64,2) order (1,0,2)  ->  layout (64,64,2):(64,1,4096)     ← ldsC  逻辑下标 (token, 通道)
(64,64,2) order (0,1,2)  ->  layout (64,64,2):(1,64,4096)     ← ldsCt 逻辑下标 (通道, token)
```

**两个视图的 shape 顺序和 order 都反过来，一正一反抵消，物理排布完全相同——通道的 stride 都是 1。**
变的只是"用哪两个下标去索引它"。末维的 `2` 是 C 的双缓冲。

这是理解新内核的关键：不是"两块 LDS"，是**一块 LDS 的两种索引方式**。

### 4.3 Step 1 / Step 2

```2438:2452:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
        def postprocess_store2lds(fragC, ldsc_idx):
            for fc, fsw in fxh.all_elements(fragC, frag_sorted_weight):
                fc.store(fc.load() * fsw.load())
            vec_f32 = fragC.load()
            fragC_bf16.store(f32_to_bf16(vec_f32))
            fx.copy(copy_atom_, fragC_bf16r, thrv_ldsCt[None, None, None, ldsc_idx])

        arg_p_output = fx.flat_divide(arg_p_output, (BLOCK_M, BLOCK_N))
        cp_atom_out_128b = flyobj.get_buffer_copy_atom(fx.BFloat16, 128)
        thrv_out = tcopyLDS.partition_D(arg_p_output)
        fragOut = fx.make_fragment_like(thrv_ldsC[None, None, None, 0])

        def postprocess_store2vmem(n, ldsc_idx):
            fx.copy(cp_ldsc, thrv_ldsC[None, None, None, ldsc_idx], fragOut)
            fx.copy(cp_atom_out_128b, fragOut, thrv_out[None, None, None, 0, n])
```

**Step 1 写走 `ldsCt`**，用的是 MMA 的 C 分区（`get_tiled_mma_copy(copy_atom_, mm, "C")`），
copy atom 是 **64 位**：

```2432:2436:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
        copy_atom_ = flyobj.get_universal_copy_atom(fragC_bf16.dtype, 64)
        tcopy = flyobj.get_tiled_mma_copy(copy_atom_, mm, "C")
        fragC_bf16r = flyobj.get_retile(tcopy, fragC_bf16)

        thrv_ldsCt = flyobj.get_partition_D(tcopy, ldsCt)
```

一个 lane 的 4 个连续通道 → 4 × bf16 = 8 字节 → 一条 `ds_write_b64`。

**Step 2 读走 `ldsC`**，映射由 `get_tiled_copy_coalesced_mn` 生成，copy atom 是 **128 位**：

```2426:2430:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
        tcopyLDS, cp_ldsc = flyobj.get_tiled_copy_coalesced_mn(
            ldsC[None, None, 0], copy_atom_bits=128, num_threads=256
        )

        thrv_ldsC = tcopyLDS.partition_S(ldsC)
```

这个 helper 的算法很直白：

```629:638:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942_utils.py
        shape = get_d1_shape(tensor)
        num_rows = shape[0]
        num_cols = shape[1]
        num_vals = copy_atom_bits // (tensor.dtype.width)
        assert num_cols >= num_vals, f"expect {num_cols} >= {num_vals}"
        assert (num_cols % num_vals) == 0, f"expect {num_cols} % {num_vals} == 0"
        thread_n = num_cols // num_vals
        thread_m = num_threads // thread_n
        tile_mn = (thread_m, thread_n * num_vals)
```

代入：`num_cols = 64`、`num_vals = 128/16 = 8` → `thread_n = 8`、`thread_m = 32`。
即线程被切成 `32(M) × 8(N)`，每线程一次读 8 个连续 bf16 = 16 字节 → `ds_read_b128`，
接着一条 `buffer_store_dwordx4`。`num_rows / thread_m = 64/32 = 2` 轮。

注意 `tcopyLDS` **同时**用来分区 LDS 源和全局目标：

```2447:2447:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
        thrv_out = tcopyLDS.partition_D(arg_p_output)
```

一套映射管两头，读出来的片段直接就是要写出去的形状，中间不需要任何重排。

### 4.4 每线程每 N-tile 的账

输出 tile `64×64 = 4096` 元素 / 256 线程 = 16 元素/线程。

| 步骤 | 指令 | 条数 |
|---|---|---|
| Step 1 写 LDS | `ds_write_b64`（8 B） | **4** |
| Step 2 读 LDS | `ds_read_b128`（16 B） | 2 |
| Step 2 写全局 | `buffer_store_dwordx4`（16 B） | 2 |
| **合计** | | **8 条 / 16 个元素 = 0.5 条/元素** |

### 4.5 顺带：软件流水

新内核把 Step 1 和 Step 2 拆到 barrier 两侧，和 GEMM、权重预取交织成一条流水线：

```2564:2580:aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942.py
            for n, state in range(0, nBN - 2, 2, init=[]):
                fxh.asm_mark("aaa")
                postprocess_store2vmem(n, 0)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 2, None], frag_weights[0]
                )
                if fx.const_expr(
                    arg_w_scale is not None and weight_quant_type != "per_tensor"
                ):
                    flyobj.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n + 2, 0], frag_pc_scales[0]
                    )
                gemm_compute(frag_weights[1], frag_pc_scales[1], fragC[1])
                postprocess_store2lds(fragC[1], 1)

                hot_loop_scheduler()
                fx.gpu.barrier()
```

一个基本块里同时放「tile n 的全局写 + n+2 的权重预取 + n+1 的 MFMA + n+1 的 LDS 写」，
靠 `fragC[0..1]` 和 `ldsC[..., 2]` 的双缓冲错开。旧内核的 epilogue 是内联在每个 N-tile 之后的，
没有这层流水。**这是独立于 CShuffle 宽度的另一项差异**，本文不展开。

---

## 五、全程追踪：一个 lane 的数据到底去了哪

前面讲的是机制，这一节把具体数字走一遍。**建议对着 3.1/3.2 和 4.3 看。**

设 workgroup 负责的 M 块基址是 `bx_m`、N 块基址是 `by_n`，输出缓冲是 `out[行][4096]`。

> **关于"lane 5"。** 一个 wave 是 64 个 lane，这个 workgroup 是 4 wave × 64 = 256 线程。
> 每个 lane 拿到的数据都不一样——MFMA 的输出布局就是一组关于 lane 号的公式，
> 不代进具体的号就看不出"跨步 256 字节"这类结论。
>
> **5 是随手挑的**，唯一的讲究是 `5 / 16 = 0`，行号从 0 开始好写。换任何 lane 结论都一样：
>
> | lane | `lane%16` | `lane/16` | 负责的列（跨 `ni`） | `acc[mi=0]` 的行 |
> |---|---|---|---|---|
> | 0 | 0 | 0 | 0, 16 | 0, 1, 2, 3 |
> | **5** | **5** | **0** | **5, 21** | **0, 1, 2, 3** |
> | 21 | 5 | 1 | 5, 21 | 4, 5, 6, 7 |
> | 63 | 15 | 3 | 15, 31 | 12, 13, 14, 15 |
>
> 最后一列只写了 `mi=0` 那一块；同一个 lane 在 `mi=1,2,3` 上还覆盖 16~19、32~35、48~51。
>
> lane 5/21/37/53 负责同样两列的不同四行，凑起来才覆盖完 16 行；lane 5 和 6 则是同一组行的相邻列。
>
> **还要注意 lane 号在下面出现在两个坐标系里**：第 0、1 站用的是 **wave 内的 lane 号**
> （MFMA 映射基于 `lane % 16`），第 2 站用的是**全局线程号 `tx`**（`m_lane = tx / 32`）。
> 这里特意取 **wave 0**，因为 wave 0 里 `tx == lane`，两站说的是**同一个物理线程**——
> 这样"它读回来的和它自己写进去的不是同一批"才能在同一个线程上直接看出来。

### 5.1 旧内核：lane 5 的 32 个值

#### 第 0 站：MFMA 算完，值在寄存器里

参数 `tile_m=64`、`tile_n=128`、4 个 wave，每 wave 负责 32 列
（`n_tile_base = wave_id × 32`）。取 **wave 0 的 lane 5**：

```
lane_mod_16 = 5 % 16 = 5      ← 决定【列】
lane_div_16 = 5 / 16 = 0      ← 决定【行组】
```

> **先厘清"16×16"和这里的关系。** 一条 MFMA 指令的输出**确实只有 16×16**（64 lane × 4 值 = 256）。
> 但 tile 是 `64 × 128`，比 16×16 大得多，所以要平铺三层：
>
> | 层 | 覆盖 | 怎么来的 |
> |---|---|---|
> | 一条 MFMA | 16 × 16 | 硬件规定，见 1.1 |
> | 一个 wave | 64 × 32 | `m_repeat=4` 块摞在行方向 × `num_acc_n=2` 块排在列方向 |
> | 4 个 wave | 64 × 128 | 每 wave 一个 32 列的切片 |
>
> 一条 MFMA 内部长这样（这一层才是硬件定的）：
>
> ```
>           col 0    col 1   ...  col 15
>   row  0  lane0    lane1        lane15    ← 各 lane 的 acc 分量 0
>   row  1  lane0    lane1        lane15    ← 分量 1
>   row  2  lane0    lane1        lane15    ← 分量 2
>   row  3  lane0    lane1        lane15    ← 分量 3
>   row  4  lane16   lane17       lane31    ← lane16 的分量 0
>   ...
>   row 15  lane48   ...          lane63    ← lane48 的分量 3
> ```
>
> **不管平铺多少块，每一块里一个 lane 拿到的都是「4 个连续行 × 1 列」**——这个性质才是后面
> 一切的根源。下面的表是把 8 块合起来看。

它手里有 `m_repeat × num_acc_n = 4 × 2 = 8` 个 f32x4，共 32 个值（= 8 块 × 每块 4 个）：

| 累加器 | 覆盖的行（token） | 覆盖的列（通道） |
|---|---|---|
| `acc[mi=0][ni=0]` | 0, 1, 2, 3 | 5 |
| `acc[mi=0][ni=1]` | 0, 1, 2, 3 | 21 |
| `acc[mi=1][ni=0]` | 16, 17, 18, 19 | 5 |
| `acc[mi=1][ni=1]` | 16, 17, 18, 19 | 21 |
| `acc[mi=2][*]` | 32~35 | 5 / 21 |
| `acc[mi=3][*]` | 48~51 | 5 / 21 |

通式：`行 = mi*16 + lane_div_16*4 + ii`，`列 = n_tile_base + lane_mod_16 + ni*16`。

**一个 f32x4 里的 4 个值是 4 个不同的 token、同一个通道。** 这就是 1.1 那条硬件规则。

#### 第 1 站：写进 LDS

`lds_out` 是 `[64][128]` 行主序，元素 `(r, c)` 的偏移是 `r*128 + c`。
lane 5 的 `acc[0][0]` 那 4 个值落在：

| `ii` | (行, 列) | LDS 元素偏移 | 距上一个 |
|---|---|---|---|
| 0 | (0, 5) | 5 | — |
| 1 | (1, 5) | 133 | **+128** |
| 2 | (2, 5) | 261 | **+128** |
| 3 | (3, 5) | 389 | **+128** |

**跨步 128 个元素 = 256 字节。** 另一个方向（`ni`：列 5 → 21）跨步 16 个元素。

两个方向都不连续 → 只能一个元素一条指令，**32 个值 = 32 条 `ds_write_b16`（每条 2 字节）**。

> **32 的来历**：发射这些写的是一个编译期全展开的三层循环，
> `mi`（4，`= tile_m/16`）× `ii`（4，f32x4 的分量）× `ni`（2，`= n_per_wave/16`）= **32**，
> 每次循环体末尾一条 `vector.store`。
> 换个数法也一样：`mi × ni = 8` 个 f32x4 × 每个 4 个值 = 32。
>
> 两处交叉验证：32 × 256 线程 = 8192 = `tile_m × tile_n` ✓；
> 32 × 32 个 N-tile = **1024** 条每 wave，与 ISA 实测的 2048 静态计数（`FASTVALID` 编了掩码和
> 快路径两份，每 wave 只跑一条）对得上 ✓

```
LDS  [64 行][128 列]，行主序
     列 →   0    5   21        74
   ┌──────────────────────────────────┐
行0│         ●    ●                   │  ← lane5 的 acc[0][0].ii=0 和 acc[0][1].ii=0
行1│         ●    ●                   │  ← ii=1        （相隔 256 字节）
行2│         ●    ●                   │  ← ii=2
行3│         ●    ●                   │  ← ii=3
   │                                  │
行8│              ...                 │
   └──────────────────────────────────┘
```

#### 第 2 站：barrier，然后换映射读回

**Step 2 的映射和 MFMA 那套完全无关**，它直接从全局线程号 `tx` 切出一个二维网格：

```192:202:aiter/ops/flydsl/kernels/mfma_epilogues.py
    CShuffleNLane = int(cshuffle_nlane)
    CShuffleMLane = int(cshuffle_mlane)
    EVec = int(e_vec)

    m_reps_shuffle = int(tile_m) // CShuffleMLane
    n_reps_shuffle = int(tile_n) // (CShuffleNLane * EVec)

    c_nlane = fx.Index(CShuffleNLane)
    m_lane = tx // c_nlane
    n_lane = tx % c_nlane
    c_evec = fx.Index(EVec)
```

代入 `nlane=32`、`mlane = 256/32 = 8`、`e_vec=2`、`tile_m=64`、`tile_n=128`：

| 量 | 值 | 含义 |
|---|---|---|
| `m_lane = tx / 32` | 0~7 | 这一轮负责第几行 |
| `n_lane = tx % 32` | 0~31 | 这一轮负责第几个列对 |
| 一轮覆盖 | 8 行 × 64 列 | `mlane × (nlane × e_vec)` |
| `m_reps = 64/8` | **8** | 要转 8 轮才盖满 64 行 |
| `n_reps = 128/(32×2)` | **2** | 要转 2 轮才盖满 128 列 |

**为什么是 8 行 × 64 列？** 因为 256 个线程一次只能覆盖 `256 × e_vec = 512` 个元素，
而 tile 有 8192 个——`8192 / 512 = 16 = m_reps × n_reps`。怎么把这 512 个元素摆成矩形是个
自由选择，选成 `8 × 64` 是为了让**相邻的 `tx` 落在同一行的相邻列上**，这样第 3 站的全局写才连续。

取 **thread 5**：`m_lane = 0`、`n_lane = 5`。它读 `8 × 2 = 16` 个片段：

| `mr` | `nr` | 行 = `mr*8 + m_lane` | 列 = `nr*64 + n_lane*2` | LDS 偏移 | 宽度 |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 10, 11 | 10 | 4 B |
| 0 | 1 | 0 | 74, 75 | 74 | 4 B |
| 1 | 0 | 8 | 10, 11 | 1034 | 4 B |
| 1 | 1 | 8 | 74, 75 | 1098 | 4 B |
| … | | 16, 24, 32, 40, 48, 56 | | | |

每片段 `e_vec = 2` 个连续 bf16 → 一条 `ds_read_b32`（4 字节）。

#### shuffle 到底"洗"在哪

**没有任何一条"洗牌指令"。** 整个 shuffle 就是一句话：

> **A 线程按公式甲算出地址往 LDS 写，B 线程按公式乙算出同一个地址去读。**
> 两个公式不同，值就从 A 换到了 B。LDS 只是个按 `(行, 列)` 索引的信箱。

把两组公式并排（都取 wave 0，此时 `tx == lane`）：

```
写侧（MFMA 映射）                    读侧（CShuffle 映射）
  行 = mi*16 + (lane/16)*4 + ii        行 = mr*8 + tx/32
  列 = (lane%16) + ni*16               列 = nr*64 + (tx%32)*2
```

拿具体格子对一下——**thread 5 读到的第一个片段 `(行 0, 列 10~11)` 是谁写的？**

反解写侧公式：行 0 → `mi=0`、`lane/16=0`、`ii=0`；列 10 → `lane%16=10`、`ni=0`。
所以 `lane = 0*16 + 10 = 10`。列 11 同理是 lane 11。

| thread 5 读的 | 反解写侧公式 | 实际由谁写的 |
|---|---|---|
| (行 0, 列 10) | `lane/16=0`、`lane%16=10` | **thread 10** |
| (行 0, 列 11) | `lane/16=0`、`lane%16=11` | **thread 11** |
| (行 8, 列 10) | `lane/16=2`、`lane%16=10` | **thread 42** |

反过来，**thread 5 自己写的那些值被谁读走了？**

| thread 5 写的 | 被谁读走 |
|---|---|
| (行 0, 列 5) | 列 5 落在列对 `[4,5]` → `n_lane=2`；行 0 → `m_lane=0` → **thread 2** |
| (行 0, 列 21) | 列对 `[20,21]` → `n_lane=10` → **thread 10** |
| (行 1, 列 5) | 行 1 → `m_lane=1` → **thread 34** |
| (行 2, 列 5) | **thread 66**（已经是 wave 1 的线程了） |

**注意最后一行：数据不但换了 lane，还跨了 wave。** 这正是必须用 LDS 而不能用
`ds_permute` / `ds_swizzle` 之类 wave 内洗牌指令的原因——那些指令只能在 64 个 lane 内部换。

#### 为什么先攒够再写

Step 2 的循环体是"**先把一行的所有 LDS 读全发出去，再统一发存储**"，不是读一个写一个：

```240:257:aiter/ops/flydsl/kernels/mfma_epilogues.py
            loaded = []
            for nr in range_constexpr(n_reps_shuffle):
                col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
                col_pair0 = col_base_nr + (n_lane * c_evec)  # even col within tile

                lds_idx_pair = row_base_lds + col_pair0
                frag = vector.load_op(vec_frag, lds_out, [lds_idx_pair])
                loaded.append((col_pair0, frag))

            for col_pair0, frag in loaded:
                store_pair(
                    row_local=row_local,
                    row=row,
                    row_ctx=row_ctx,
                    col_pair0=col_pair0,
                    col_g0=by_n_v + col_pair0,
                    frag=frag,
                )
```

源码注释写明了动机：这样后端能把这些 `ds_read` 压在**一次 `s_waitcnt lgkmcnt`** 下面，
然后把存储背靠背发出去；否则会退化成
`ds_read → s_waitcnt lgkmcnt(0) → store → ds_read → …` 的串行链，每条存储都要等一次完整的 LDS。

#### 第 3 站：写出全局

地址分两半算。**行的部分在 `precompute_row` 里算一次**（每行一次，不是每个片段一次）：

```4532:4540:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        if _bufstore:
                            # Offset inside this block's descriptor window, in output
                            # elements.  row_local is already the row within the tile,
                            # so neither `t` nor `s` is needed here -- which is also what
                            # lets the sentinel load die when the mask is off.
                            return (
                                (fused2, row_local * fx.Index(model_dim)),
                                row_valid,
                            )
```

**列的部分在 `store_pair` 里加上**：

```4570:4578:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                        if _bufstore:
                            # row_byte_base holds the row's *element* base in the window
                            # (see precompute_row); buffer_store scales it to bytes.
                            buffer_ops.buffer_store(
                                frag,
                                blk_out_rsrc,
                                row_byte_base + col_g0,
                                cache_modifier=_store_nt,
                            )
```

合起来（`col_g0 = by_n + col_pair0`）：

```
描述符 blk_out_rsrc 覆盖 [bx_m, bx_m + tile_m) 这 64 行
偏移（元素） = row_local * model_dim + by_n + col_pair0
```

thread 5 的第一个片段：`row_local=0`、`col_pair0=10` → 偏移 `by_n + 10`，写 4 字节。

**为什么拆成两半算**：`row_local * model_dim` 只依赖 `mr`，每行算一次就够；`col_pair0` 才是
逐片段变的。这是 f3 的 `BUFSTORE` 带来的——描述符按 block 切片之后，整个地址都在 32 位里，
不需要 64 位指针对（详见 `moe_stage2_reduce_parity_32k.md` 的 3.2c）。

合并度检查——线程 0~31 的 `n_lane` 是 0~31、`m_lane` 都是 0，落在同一行：

| thread | 列 | 字节地址（相对行首） |
|---|---|---|
| 0 | 0, 1 | +0 |
| 1 | 2, 3 | +4 |
| 2 | 4, 5 | +8 |
| … | | |
| 31 | 62, 63 | +124 |

**32 个 lane 拼出 128 字节连续。合并度没问题**，问题是**每条指令只搬 4 字节**。

线程 32~63（`m_lane=1`）落在下一行，和上一行相隔 `model_dim × 2 = 8 KB`。所以**一个 wave 的
一条存储指令实际打出两段各 128 字节的连续写**——依然是满的 64 B 请求，只是分两处。

（f3 开了 `e_vec=4` 之后：`n_reps` 变 1，线程 0~31 覆盖列 0~127，32 lane × 8 B = 256 B 连续，
指令数减半。**合并度反而更好。**）

#### 顺带：行谓词也挂在这一层

`precompute_row` 除了返回地址，还返回一个 i1 谓词，整个 N 循环被它包住：

```259:264:aiter/ops/flydsl/kernels/mfma_epilogues.py
        if row_pred is not None:
            _if_row = scf.IfOp(row_pred)
            with _if_then(_if_row, scf):
                _do_store_row()
        else:
            _do_store_row()
```

**一行一个判断，而不是每个片段一个**——`moe_sorting` 补出来的哨兵行整行跳过。
开了 `FASTVALID` 之后连这个判断都没有（`row_pred is None`），走的是下面那条直线分支。

### 5.2 新内核：lane 5 的 4 个值

#### 第 0 站：MFMA 算完

参数 `BLOCK_M=64`（token）、`BLOCK_N=64`（通道）。因为 A/B 换了位置，**M 维是通道、N 维是 token**：

```
token   = lane % 16          ← 决定【列】(N)
通道基址 = (lane / 16) * 4    ← 决定【行】(M)
```

**lane 5**：`token = 5`、`通道 = 0,1,2,3`。

| | 旧内核 lane 5 | 新内核 lane 5 |
|---|---|---|
| 一个 f32x4 装的 | token 0~3 的通道 5 | **token 5 的通道 0~3** |
| 4 个值之间差 | 一个 token | **一个通道** |

#### 第 1 站：写进 LDS

物理排布是**通道 stride 1、token stride 64**（`ldsCt` 视图，见 4.2 那两行 IR）。
元素 `(token, 通道)` 的偏移是 `token*64 + 通道`。lane 5 的 4 个值：

| 分量 | (token, 通道) | LDS 元素偏移 | 距上一个 |
|---|---|---|---|
| 0 | (5, 0) | 320 | — |
| 1 | (5, 1) | 321 | **+1** |
| 2 | (5, 2) | 322 | **+1** |
| 3 | (5, 3) | 323 | **+1** |

**连续。** 4 × bf16 = 8 字节 → **一条 `ds_write_b64`**。

```
LDS  物理上【通道】连续
     通道 →  0  1  2  3 ...                63
        ┌────────────────────────────────────┐
token 5 │  ●  ●  ●  ●                        │  ← lane5 的 4 个值，一条 b64 写完
token 6 │  ●  ●  ●  ●                        │  ← lane6（token 6，同样的通道）
        └────────────────────────────────────┘
```

#### 第 2 站：换映射读回

映射由 `get_tiled_copy_coalesced_mn` 算出（4.3）：`num_vals = 128/16 = 8`、
`thread_n = 64/8 = 8`、`thread_m = 256/8 = 32`。

```
n 下标 = tx % 8      → 负责哪 8 个通道
m 下标 = tx / 8      → 负责哪个 token
```

**thread 5**：`n = 5`、`m = 0` → 读 token 0 的通道 40~47。

| 轮次 | token | 通道 | LDS 偏移 | 宽度 |
|---|---|---|---|---|
| 0 | 0 | 40~47 | 40 | **16 B** |
| 1 | 32 | 40~47 | 2088 | **16 B** |

用 `ldsC` 视图（逻辑下标 `(token, 通道)`）读，但物理上还是那块内存——**通道连续，所以
8 个 bf16 是一条 `ds_read_b128`**。只需 `64 / 32 = 2` 轮。

#### 第 3 站：写出全局

同一个 `tcopyLDS` 既分区 LDS 源、又分区全局目标（4.3 结尾那句），读出来的片段形状直接就是
要写的形状，中间零重排：

| thread | token | 通道 | 字节地址（相对行首） |
|---|---|---|---|
| 0 | 0 | 0~7 | +0 |
| 1 | 0 | 8~15 | +16 |
| … | | | |
| 7 | 0 | 56~63 | +112 |
| 8 | 1 | 0~7 | 下一行 |

**8 个 lane 拼出 128 字节连续**，每 lane 一条 `buffer_store_dwordx4`。

### 5.3 并排看

同一件事（把 128 字节写到输出的一行）两边的做法：

| | 旧内核（e_vec=2） | 新内核 |
|---|---|---|
| 参与的 lane 数 | 32 | **8** |
| 每 lane 搬 | 4 B | **16 B** |
| 一行合计 | 128 B | 128 B |
| **合并度** | **相同** | **相同** |
| **指令数** | **32 条** | **8 条** |

**合并度两边一样好，差的纯粹是发射条数。** 这一点很容易搞错——看到 `buffer_store_dword` 就
以为是"写没合并"，其实合并得挺好，只是每条指令搬得太少。
`moe_stage2_reduce_parity_32k.md` 的 2.4 用硬件计数器验过：`TCC_EA0_WRREQ_64B_sum` 恒等于
`TCC_EA0_WRREQ_sum`，**四种配置下都是 100% 的 64 B 满请求**。

全程一张表（每线程每 N-tile）：

```
                旧内核                              新内核
              ─────────                          ─────────
MFMA 输出     8 个 f32x4                          4 个 f32x4
              每个 = 4 token × 1 通道              每个 = 4 通道 × 1 token
                  │                                   │
                  │ 32 × ds_write_b16 (2 B)           │ 4 × ds_write_b64 (8 B)
                  ▼                                   ▼
LDS           [token][通道] 行主序                  [通道] 连续（两个转置视图）
              4 个值跨步 256 B                      4 个值连续
                  │                                   │
              ────┼──── barrier ────                ──┼──── barrier ────
                  │                                   │
                  │ 16 × ds_read_b32 (4 B)            │ 2 × ds_read_b128 (16 B)
                  ▼                                   ▼
寄存器        16 个 bf16x2                         2 个 bf16x8
                  │                                   │
                  │ 16 × buffer_store_dword           │ 2 × buffer_store_dwordx4
                  ▼                                   ▼
全局          32 元素，64 条指令                    16 元素，8 条指令
              = 2.0 条/元素                        = 0.5 条/元素
```

**瓶颈在第一步。** Step 2 的宽度是自己挑的（`e_vec` / `cshuffle_nlane` 都是可调参数），
Step 1 的宽度却被 MFMA 的输出布局锁死——这就是第七章评估的出发点。

## 六、两边对照

每线程每 N-tile，归一到"每个输出元素要发几条 epilogue 指令"：

| | 旧（当前默认 e_vec=2） | 旧（f3 的 e_vec=4） | 新内核 |
|---|---|---|---|
| 输出 tile | 64×128 | 64×128 | 64×64 |
| 每线程元素 | 32 | 32 | 16 |
| Step 1 LDS 写 | 32 × `ds_write_b16` | 32 × `ds_write_b16` | **4 × `ds_write_b64`** |
| Step 2 LDS 读 | 16 × `ds_read_b32` | 8 × `ds_read_b64` | **2 × `ds_read_b128`** |
| Step 2 全局写 | 16 × `buffer_store_dword` | 8 × `dwordx2` | **2 × `dwordx4`** |
| 合计 | 64 条 | 48 条 | **8 条** |
| **每元素指令** | **2.0** | **1.5** | **0.5** |

搬运的字节数三者相同，差的全是**发射条数**。

整核 ISA 静态计数（旧内核全展开，静态≈每 wave 动态；新内核有真实 N 循环，静态是循环体）：

| opcode | 旧内核 | 新内核 |
|---|---|---|
| `ds_write_b16` + `ds_write_b16_d16_hi` | 2048（两条 epilogue 路径） | 0 |
| `ds_write_b64` / `ds_write_b128` | 0 / 0 | 16 / 3 |
| `buffer_store_dword` | 1023 | 0 |
| `buffer_store_dwordx4` | 0 | 8 |
| `v_perm_b32`（bf16 打包） | 0 | 32 |

最后一行也值得一提：新内核用 `v_perm_b32` 一条把两个 f32 的高 16 位拼成一个 dword 完成 bf16
打包；旧内核是 `>>16` + 逐元素的 `ds_write_b16_d16_hi`（移位被折进了 store 的 `d16_hi`）。

---

## 七、旧内核能不能改成新内核那样

先说结论：**有三条路，代价和收益差得很远。真正的正解等于重写。**

### 7.0 为什么"两个方向都连续"在旧内核里不可能

这是评估的前提，先讲清楚。

一块二维 LDS **只能有一个连续维**。新内核之所以写和读都连续，不是因为它有什么魔法布局，而是
因为**它的两个需求方向本来就是同一个**：

- 累加器给出的是 **4 个连续通道**（写方需求：通道连续）
- 全局写想要的是 **8 个连续通道**（读方需求：通道连续）

方向一致，一个布局同时满足。

旧内核的两个需求是**正交**的：

- 累加器给出的是 **4 个连续 token**（写方需求：token 连续）
- 全局写想要的是 **8 个连续通道**（读方需求：通道连续）

**所以旧内核的 LDS 布局无论怎么选，必然有一侧是跨步的。** 这不是实现没写好，是几何决定的。

### 7.1 C0：只调 CShuffle 的线程映射（改动最小）

`_cshuffle_nlane` 是个写死的字面量：

```2534:2534:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    _cshuffle_nlane = 32
```

而它在 `mfma_epilogues.py` 里其实是**带默认值的形参**，`_call_epilog()` 从来没传过它：

```95:95:aiter/ops/flydsl/kernels/mfma_epilogues.py
    cshuffle_nlane: int = 32,
```

约束是 `tile_n % (cshuffle_nlane * e_vec) == 0` 和 `tile_m % (block_size / cshuffle_nlane) == 0`。
`tile_n=128` 下：

| nlane | e_vec | `nlane × e_vec` | 合法？ | Step 2 读+写 |
|---|---|---|---|---|
| 32 | 2 | 64 | 是（当前默认） | 16 + 16 |
| 32 | 4 | 128 | 是（f3 用的） | 8 + 8 |
| 32 | 8 | 256 > 128 | **否** | — |
| **16** | **8** | **128** | **是** | **4 + 4** |

**把 nlane 降到 16、e_vec 提到 8，就能用上 `buffer_store_dwordx4`**，Step 2 的读写各再减半。

- **改动量**：给 `_call_epilog()` 多传一个 `cshuffle_nlane`，`_e_vec` 的默认表达式在 nlane=16
  下会自己算出 8，不用另外改。两行，不碰 MFMA、不碰累加器、不碰 Step 1。

#### C0 实测结果：指令确实省了，但**时间一点没变**

预测过 −15~20 us。实测是 **0**。这条路走不通，原因比结论本身更有价值。

> 下面的数据是用一个临时 knob（`FLYDSL_MOE_STAGE2_NLANE`，把 `_cshuffle_nlane` 从字面量改成
> 读环境变量，并传给 `_call_epilog()`）测出来的。既然确认无收益，**这个 knob 已经撤销**，
> 代码保持原样。要复现的话按上面"改动量"那两行改回去即可。

指令侧完全兑现了。整核 ISA 静态计数：

| opcode | nlane=32, e_vec=4 | nlane=16, e_vec=8 | 差 |
|---|---|---|---|
| `buffer_store_dwordx2` | 256 | 0 | −256 |
| `buffer_store_dwordx4` | 0 | 128 | +128 |
| `ds_read2st64_b64` | 128 | 0 | −128 |
| `ds_read_b128` | 12 | 140 | +128 |
| `v_lshlrev_b32` / `v_or_b32` | 426 / 499 | 299 / 367 | −127 / −132 |
| **总指令** | **7657** | **7280** | **−377（−4.9%）** |

注意 **DS 指令条数没变**（两边都是 1199）。原以为读能减半，其实 nlane=32 那边 LLVM 早就把成对的
8 B 读合并成了 `ds_read2st64_b64`（一条读两个 8 B，跨 64 dword），条数已经是 128 了；
换成 `ds_read_b128` 还是 128 条。**省的只有 store 和地址算术。**

动态计数器（per-wave，rocprofv3）：

| 计数器 | nlane=32 | nlane=16 | 差 |
|---|---|---|---|
| `SQ_INSTS_VALU` | 4749.0 | 4467.0 | **−282（−5.9%）** |
| `SQ_INSTS_VMEM_WR` | 250.6 | 125.3 | **−125（−50%）** |
| `SQ_INSTS_LDS` | 1150.3 | 1150.3 | 0 |
| `SQ_LDS_IDX_ACTIVE` | 5145.4 | 5145.4 | 0 |
| `SQ_LDS_BANK_CONFLICT` | 2004.9 | 2004.9 | 0 |
| 冲突率 | 38.96% | 38.96% | 0 |

**bank 冲突一位没涨**（原先担心从 8 路变 16 路）。`ds_read2st64_b64` 和 `ds_read_b128` 在 LDS
硬件里产生的访问周期完全一样，`SQ_LDS_IDX_ACTIVE` 逐位相同就是证据。

时间侧：kernel trace 630 次 dispatch 合计 199832 us → 199820 us；e2e 6428.19 us → 6428.38 us。
两个独立测量都是平的，cos 保持 0.999995。

#### 为什么省了指令却不快：内核已经从"发射受限"变成"停顿受限"

等待周期计数器给出了答案：

| 计数器（per-wave） | nlane=32 | nlane=16 | 差 |
|---|---|---|---|
| `SQ_ACTIVE_INST_VALU` | 9497.9 | 8934.1 | **−563.8** |
| `SQ_WAIT_ANY` | 20673.6 | 20883.8 | **+210.2** |
| `SQ_ACTIVE_INST_LDS` | 4391.1 | 4398.1 | +7.0 |

**省下来的 564 个 VALU 执行周期，原封不动变成了等待周期。** 每个 wave 花 20674 个周期在等，
只花 9498 个周期在跑 VALU——**等待是 VALU 的 3.4 倍**。在这个比例下，再削减发射条数不会缩短
关键路径，只是让 wave 更早地开始等。

这解释了 f1→f4 收益逐级递减的走势：那几步把发射受限的余量抽干了，现在剩下的是访存/LDS 延迟，
削指令碰不到它。**指令数不再是有效的优化指标了。**

#### 那 38.96% 的 bank 冲突不是抓手：它对任何布局改动都不响应

一度以为下一步该去修 `lds_out` 的行跨度（`tile_n × 2 = 256 B`，正好是 LDS 32 banks × 4 B
= 128 B 的整数倍，行与行精确重叠在同一组 bank 上）。实测把这条路也否掉了。

六种完全不同的 LDS 访问配置：

| 配置 | Step 2 几何 | `SQ_INSTS_LDS` | `SQ_LDS_IDX_ACTIVE` | `SQ_LDS_BANK_CONFLICT` | 冲突率 |
|---|---|---|---|---|---|
| nlane=32, e_vec=2 | 8 行 × 32 lane × 4 B | 146978400 | 592876800 | 231014400 | 38.96% |
| nlane=32, e_vec=4 | 8 行 × 32 lane × 8 B | 132540000 | 592876800 | 231014400 | 38.96% |
| nlane=16, e_vec=8 | 16 行 × 16 lane × 16 B | 132540000 | 592876800 | 231014400 | 38.96% |
| nlane=64, e_vec=2 | 4 行 × 64 lane × 4 B | 146978400 | 592876800 | 231014400 | 38.96% |
| `FLYDSL_CK_LDS128=1`（X 无 pad） | — | — | 592876800 | 231014400 | 38.96% |
| `FLYDSL_CK_LDS128=0`（X pad +8） | — | — | 595584000 | 233721600 | 39.24% |

`SQ_INSTS_LDS` 会随配置变，说明 knob 确实生效了；但 `IDX_ACTIVE` 和 `CONFLICT` 在前四种里
**逐位相同**（592876800 / 231014400）。最后一组尤其说明问题：`FLYDSL_CK_LDS128=0` 给 X 的
LDS 缓冲区实打实加了 padding（`lds_stride = tile_k + 8`），冲突不但没降，还**涨了**
（+1.17%），e2e 6446.25 → 6445.89 us，没动。

`IDX_ACTIVE` 和 `CONFLICT` 始终锁在 38.96% 这个比例上、且只随搬运字节数等比例变化，说明这个
计数器量的是**结构性的额外周期**（64 lane 的宽读必须拆成多拍），不是可以靠错开地址消除的
寻址碰撞。**这条计数器在这个内核上不是可优化项，不要再拿它当目标。**

#### 占用率也不是抓手：砍掉一半，时间没动

`waves_per_eu` 提供了一个免费的敏感度探针——它会主动把 LDS 撑大来压占用率
（`_min_lds = 65536 // (wpe+1) + 1`，`moe_gemm_2stage.py:2583`）。当前配置
`waves_per_eu = 0`（fp8 走 `moe_kernels.py:286` 的 else 分支），**没有**被人为限制，
28928 B 是真实占用：

```
lds_x   = 3 K-tile × 64 行 × 64 B = 12288 B
lds_out = 2 × 64 × 128            = 16384 B
lds_tid = 64 × 4                  =   256 B
                             合计 = 28928 B   → 65536/28928 = 2 WG/CU = 8 waves
```

在 kernel 名后加 `_w1` 把 LDS 撑到 32896 B（`32896 × 2 = 65792 > 65536`，只放得下 1 个
workgroup；VGPR 同时从 196 涨到 257，也限制到 1 wave/SIMD）：

| | LDS | VGPR | WG/CU | stage2 kernel 中位时间 | e2e |
|---|---|---|---|---|---|
| `wpe=0`（现状） | 28928 | 196 | 2 | 241.68 us | 6441.1 us |
| `wpe=1` | 32896 | 257 | **1** | **241.80 us（+0.05%）** | 6447.5 us |

**占用率砍掉一半，kernel 时间没动。** 所以"把 `lds_out` 减半冲 3 WG/CU"这条路也不用走了。

（注意 `MeanOccupancyPerCU` 这个派生指标两边都报 7.70，没反映出变化，不可信；
LDS/VGPR 的分配数字和 kernel 计时才是准的。）

#### 综合：这个内核对三类改动全都不敏感

| 改动 | 效果量 | 时间变化 |
|---|---|---|
| 指令数（C0：VALU −5.9%、store −50%） | 每 wave −282 VALU、−125 store | **0** |
| LDS 布局 / bank 冲突（6 种配置 + X padding） | 冲突率恒为 ~39%，推不动 | **0** |
| 占用率（8 waves/CU → 4） | 减半 | **+0.05%** |

三类都不敏感，说明限制既不在发射带宽、不在 LDS、也不在并行度不足。**减少 wave 数不变慢，
意味着 CU 根本没被抢占**——瓶颈在单个 wave 内部的串行依赖链（MFMA ↔ LDS ↔ 全局加载之间的
`s_waitcnt`），要动它等于重做软件流水，属于"重写"档次。

结论：**stage2 GEMM 在 f4 之后已经收敛，继续在这个内核上抠没有性价比。**
  不过 `moe_stage2_reduce_parity_32k.md` 的 4.7(3) 已经实测过 **bank 冲突不在关键路径上**
  （新内核冲突率是旧内核的近 3 倍却快 1.85 倍），所以这个风险可能可以接受——**但必须实测**。

**建议：值得做，成本低。** 但它只动 Step 2，动不了 Step 1 那 32 条 `ds_write_b16`，天花板有限。

### 7.2 C1：转置 `lds_out`（改动中等，但大概率不值）

把 `lds_out` 从 `[token][通道]` 转成 `[通道][token]`，让 Step 1 的 4 个 token 变连续。

**Step 1 收益**：一个 lane 的每个 f32x4 从 4 条 `ds_write_b16` 变成 1 条 `ds_write_b64`。
每线程每 N-tile 从 32 条降到 8 条，**省 24 条**。

**Step 2 代价**：读方要"某个 token 的 e_vec 个连续通道"，在 `[通道][token]` 布局下通道的跨步
变成 `tile_m = 64` 个元素，**不再连续**。`vector.load_op(vec<e_vec>)` 退化成 e_vec 条独立的
2 字节读。`e_vec=2` 时从 16 条变 32 条，**多 16 条**。

**净账**：每线程每 N-tile `−24 + 16 = −8` 条，每 wave 约 −256 条，占总指令的 2.2%。
按 4.2 的斜率区间估，大概 **−20~30 us**——和 C0 一个量级，但改动大得多（要动 Step 1 和
Step 2 两侧的下标计算）。

而且全局写那一侧还有个更麻烦的问题：读出来的 e_vec 个元素既然不连续，就没法直接构成一次宽写，
Step 2 的全局存储宽度也跟着退回去。**算上这一项，C1 很可能是负收益。**

**建议：不做。** 除非配合别的改动一起。

### 7.3 C2：翻转 MFMA 操作数顺序 —— 已实现并实测，结果是**基本打平**

> **本节的预测被实测推翻了，保留原文并在末尾给出实测数据。**
> 结论先说：B-first 做出来了、正确（cos 0.999995），但相对 f4 只值 **−13.7 us**，
> 而不是下面估的 −140~180 us。原因见 7.3.1。



这是唯一能同时解决 Step 1 和 Step 2 的办法：让 M 维变成输出通道。

要改的东西（按依赖顺序）：

1. **MFMA 调用的操作数位置**。当前是激活当 A、权重当 B：

   ```3583:3590:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                                       acc_list[acc_idx] = mfma_k64(
                                           acc_list[acc_idx],
                                           a0,
                                           a1,
                                           b_packs0[ni],
                                           b_packs1[ni],
                                       )
   ```

2. **累加器的下标语义翻转**。`acc[mi * num_acc_n + ni]` 每个 lane 现在拿的是
   "4 个连续 token × 1 个通道"，翻转后变成"4 个连续通道 × 1 个 token"。
   MFMA 的个数和分块都不变（还是 `m_repeat=4` 个 token 块 × `num_acc_n=2` 个通道块），
   只是每个 16x16 的结果转置了。

3. **`write_row_to_lds` 重写**（主要工作量）。索引从
   `token = mi*16 + lane_div_16*4 + ii` / `channel = ni*16 + lane_mod_16`
   变成 `token = mi*16 + lane_mod_16` / `channel = ni*16 + lane_div_16*4 + ii`。
   收益也在这里：4 个 `ii` 现在是 4 个**连续通道**，`lds_out` 里地址连续，
   32 条 `ds_write_b16` 可以合成 8 条 `ds_write_b64`。
   三个变体（`_vec_scale`、`sx_scalar`、通用路径）都要跟着改。

4. **`default_epilog` 要一个翻转版本**。现在写死了 `row = bx_m + mi*16 + lane_div_16*4 + ii`
   （`mfma_epilogues.py:75-81`），翻转后遍历的是通道。这是个 30 行的小函数，
   加一个并列版本即可，不用改原来的（原版 stage1 等还在用）。

5. **逐行的标量活会变少**。`tw_pf`（topk 权重）、sorted id 解码、`sx` 加载现在是每
   `(mi, ii)` 一次，翻转后一个 lane 只对应一个 token，变成每 `mi` 一次——**少 4 倍**。
   反过来 `sw_vals[ni]`（每通道权重 scale）从 1 个标量变成 4 个（per-tensor 量化下无影响）。

#### 不用改的（原文档在这几条上高估了）

- **两个操作数的加载器都不用动。** MFMA 的 A 片段和 B 片段在 CDNA 上每-lane 布局是**同构**的，
  都是 `lane%16` 选 16 维、`lane/16` 选 K 切片。代码里两边完全对称：

  ```3093:3097:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                  row_a_lds = lane_mod_16
                  # A-side kpack is always 16 bytes; kpack_bytes is B-side (may be 8 for int4).
                  a_kpack_elems = 16 // elem_bytes
                  col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
  ```

  ```3123:3130:aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                      col_g = by_n + n_tile_base + offset + lane_mod_16
                      col_g_list.append(col_g)

                      row_w = expert_off_idx + col_g
                      coord_w = fx.idx2crd(row_w, layout_n_blk_intra)
                      n_blk_list.append(fx.get(coord_w, 0))
                      n_intra_list.append(fx.get(coord_w, 1))
  ```

  两个操作数的寄存器内容本来就是对的，交换 `mfma` 的前两个参数只会让结果转置。
  `lds_load_packs_k64` 和 `load_b_tile` 一行都不用改。

- **X 在 LDS 里的布局不用换**，理由同上。
- **Step 2 完全不用动。** `c_shuffle_epilog` 在 barrier 之后按自己的 `(m_lane, n_lane, e_vec)`
  映射读 `lds_out`，和累加器朝向**解耦**——它只关心 `lds_out` 是 `[tile_m][tile_n]` 行主序。
- **`lds_out` 不用改成转置视图**，还是同一块 `[tile_m][tile_n]`，只有写入方的索引算法变。
- **`mfma_epilogues.py` 不用全线改**，加并列函数即可，stage1 那些调用方不受影响。

**改动量重估**：MFMA 交换约 5 行（两个调用点：`mfma_k64` 和 k128 的
`mfma_scale_f32_16x16x128_f8f6f4`，后者还要对调 scale 操作数）+ `default_epilog` 变体约 30 行
+ `write_row_to_lds` 三个变体约 150 行。**总共 200 行上下，集中在一个文件，不是"等于重写"。**

#### 但是：收益估算已经被证伪，建议仍然是不做

原来那个 −140~180 us 是**按指令条数推的**。7.1 的三组实验已经证明这个内核对指令数不敏感：
削掉每 wave 282 条 VALU、一半的 store，时间是 0；占用率砍一半，时间也是 0。

具体到 B-first 想省的那部分，LDS 周期数其实**根本不变**：

| | 指令数 | 每条搬运 | LDS 周期 |
|---|---|---|---|
| 现在：`ds_write_b16` | 32 条 | 64 lane × 2 B = 128 B = 1 个 bank 行 | 32 × 1 = **32** |
| B-first：`ds_write_b64` | 8 条 | 64 lane × 8 B = 512 B = 4 个 bank 行 | 8 × 4 = **32** |

和 C0 遇到的情况一模一样（`ds_read2st64_b64` 换成 `ds_read_b128`，`SQ_LDS_IDX_ACTIVE` 一位没变）。
Step 1 无论怎么写都是那 16 KB，字节数决定周期数。

**旁证是新内核本身**：它就是 B-first（见 4.1），朝向"正确"、且完全按这个朝向调优过。
f1~f4 用完全不同的手段（sorted 布局、去标量链、外提、删掩码）已经把大部分差距吃掉了，
说明 B-first 不是唯一通路。

> **勘误。** 本节先前写的是"f4 之后旧内核 6192、新内核 6191，**打平**"——**这是记错了**，
> 6192 是 `20260807-093353` 那一 session 里 **target** 的数字，同一 session 的 f4 是 6457.1。
> 差距一直存在，四个 session 都在 220~265 us 之间：
>
> | session | f4 | target | 差距 |
> |---|---|---|---|
> | 20260806-233148 | 6428.9 | 6208.4 | 220.5 |
> | 20260807-093353 | 6457.1 | 6192.0 | 265.1 |
> | 20260811-163350 | 6442.1 | 6198.3 | 243.8 |
>
> 结论不变，但支撑它的理由要换成 7.3.2 的直接实测。

**建议：不做。** 不是因为改不动，而是因为改完大概率是 0——下面直接测了。

#### 7.3.2 实测：预测对了，但机制不是原先想的那样

`FLYDSL_MOE_STAGE2_BFIRST=1` 已经实现（配套 `FLYDSL_MOE_STAGE2_LDSPAD`）。叠在 f4 之上，
GPU 4，每档 3 次取中位：

| 配置 | 中位 e2e | vs f4 |
|---|---|---|
| f4（B-first 关） | 6452.3 | — |
| BFIRST，`LDSPAD=0` | 7429.5 | **+977.2** |
| **BFIRST，`LDSPAD=4`** | **6438.6** | **−13.7** |
| BFIRST，`LDSPAD=8` | 6439.7 | −12.6 |
| BFIRST，`LDSPAD=16` | 6472.7 | +20.4 |

正确性没问题，`cos = 0.999995` 与 base 一致。**"大概率是 0"这个预测是对的**——
−13.7 us 虽然两组区间不重叠（是真的），但小到不值得为它背一套朝向。

但有两件事是预测里没有的。

**(1) bank 冲突极贵，不加 padding 会倒亏 977 us。**
B-first 下相邻 lane 写的是 16 个**不同的行**，行跨步 `tile_n=128` bf16 = 256 B = 64 个 bank
≡ 0 (mod 32)——16 个 lane 全撞同一组 bank。`pad=4` 把跨步变成 264 B，`264/4 = 66 ≡ 2 (mod 32)`，
相邻行错开 2 个 bank，16 行正好铺满 32 个。这也印证了 7.1 对 C0 的同一条担心：
**换线程映射之前先算 bank。**

**(2) LDS 那边分毫不差地兑现了，但 VALU 反噬更多。** ISA 静态计数：

| 类别 | f4 | f5（B-first + pad4） | 差 |
|---|---|---|---|
| MFMA | 1536 | 1536 | 0 |
| **LDS** | 1199 | 431 | **−768** |
| **VALU** | 3339 | 4180 | **+841** |
| SALU | 607 | 648 | +41 |
| SYNC | 379 | 527 | +148 |
| **合计** | **7659** | **7909** | **+250** |

`ds_write_b16_d16_hi` 1024 → 0，换成 128 条 `ds_write2_b64`——**省下的 768 条正好是 7.3 开头
预测的那个数**。但 VALU 涨了 841：

| opcode | f4 | f5 | 差 |
|---|---|---|---|
| `v_perm_b32` | 0 | 512 | **+512** |
| `v_mov_b32_e32` | 32 | 197 | +165 |
| `v_pk_mul_f32` | 714 | 832 | +118 |

**关键是 `v_perm_b32` 那 512 条。** A-first 写的是 `ds_write_b16_d16_hi`——这条指令直接写
VGPR 的**高 16 位**，于是 bf16 的截断（`>>16`）被折进 store，**一条指令都不花**。
B-first 要做宽写，就必须先把 2 个 bf16 打包进一个 dword，`v_perm_b32` 得实打实地发出来。

**这是 7.3 开头那个估算漏掉的一项：只算了省下的窄写，没算失去的免费转换。**
A-first 的逐元素窄写有个隐藏红利，换成宽写就得吐回去。新内核也付这笔——它循环体里有
32 条 `v_perm_b32`，**打包是宽写的固有成本**。

指令总数 +250 而时间降了 13.7 us，说明**这里一条 LDS 指令比一条 VALU 值钱**，两边几乎抵消。

代价：VGPR 196 → 210，LDS 28928 → 29440 B。

### 7.3.3 后续：C0 + C2 一起做才有收益

7.3.2 测出 B-first 单独只值 −13.7 us，7.1 估 C0 值 −15~20 us。**两个一起做是 −109.6 us。**

| 配置（都叠在 f4 之上，GPU 0，各 3 次中位） | e2e | vs f4 |
|---|---|---|
| f4 | 6385.0 | — |
| 只做 C0（`e_vec=8` + `nlane=16`） | 6362.5 | −22.5 |
| 只做 C0 + `LDSPAD=4` | 6311.4 | −73.6 |
| 只做 C2（B-first + pad4） | — | −13.7 |
| **C0 + C2 + pad4** | **6275.4** | **−109.6** |

单独效应之和 −87.3，实际联合 −109.6，**协同 −22.3**。

原因就是 7.0 那张图的两端：**C2 管 Step 1 的写，C0 管 Step 2 的读和存**。只加宽写，
数据卡在 LDS 出不去；只加宽读存，写进去的还是 1024 条窄写。**两端都宽才吃得满。**

这也说明 7.1 和 7.3 把它们**当成两个独立候选分别估值是错的**——它们是一条链上的两个消费者，
和 f1 的两个 knob、f2 的三个 knob 是同一种形态。

落地为阶梯上的 **f5**，详见 `moe_stage2_reduce_parity_32k.md` 第八章：

```
LDS 每 MFMA    0.765 → 0.182   （target 0.260，已反超）
VMEM 写        0.167 → 0.083   （与 target 精确相等）
VALU           3.158 → 3.632   （target 2.584，反而涨了）
stage2 GEMM    2140.8 → 2021.7 us
```

**访存和 LDS 这两条线到此为止**，剩余指令缺口里 VALU 占 95%。

### 7.4 汇总

| 方案 | 改什么 | 单独实测 | 改动量 | 结论 |
|---|---|---|---|---|
| **C0** | `cshuffle_nlane` 32→16、`e_vec` 4→8 | −22.5（不加 pad）/ **−73.6（+pad4）** | 三行 + 传参 | **做了** |
| C1 | 转置 `lds_out` | 未测 | 中等 | 不做，被 C2 取代 |
| **C2** | 翻转 MFMA 操作数朝向（B-first） | **−13.7**（必须配 pad4，否则 **+977**） | ~200 行 | **做了** |
| **C0 + C2 + pad4** | 两端一起加宽 | **−109.6** | | **落地为 f5** |

> **这张表先前记的是"C0 实测 0、C2 重估 ≈ 0，两条都不做"。** 那一轮的 C0 大概率没配
> `LDSPAD` ——今天的对照里，C0 不加 padding 只有 −22.5，加了 pad4 才是 −73.6，
> **padding 占了这一项收益的三分之二**。而 C2 单独确实接近 0（−13.7），
> 那个判断本身没错，错在**把两者当成独立候选分别估值**（见 7.3.3）。

修正后的一句话：**旧内核 epilogue 的天花板确实由累加器朝向锁死，但解锁它需要两把钥匙。**
朝向（C2）让 Step 1 能宽写，`cshuffle_nlane`（C0）让 Step 2 能宽读宽存，`LDSPAD` 解掉
朝向翻转带来的 bank 冲突。缺任何一把，另外两把的收益都还埋着——单独 −13.7 / −73.6，
合起来 −109.6。

做完之后 **LDS 每 MFMA 0.182 已低于 target 的 0.260、VMEM 写 0.083 与 target 精确相等**。
epilogue 这条线到此为止；剩余指令缺口里 **VALU 占 95%**，下一步只能从别处找
（旧内核 VGPR 198 / LDS 29440 对 target 的 118 / 16384，occupancy 仍是 2 倍差距）。

---

## 附：常用命令

dump 两个内核的最终 ISA 和各级 MLIR：

```bash
cd /data/aiter

# 旧内核（f3 配置）
AITER_FLYDSL_STAGE2_SORTED_PARTIAL=1 FLYDSL_MOE_STAGE2_FASTVALID=1 \
FLYDSL_MOE_STAGE2_SCALAR_ASCALE=1 FLYDSL_MOE_STAGE2_VEC_SCALE=1 \
FLYDSL_MOE_STAGE2_BUFSTORE=1 FLYDSL_MOE_STAGE2_HOIST_PF=1 FLYDSL_MOE_STAGE2_EVEC=4 \
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa_old \
AITER_CONFIG_FMOE=moe_stage2_opt/configs/old.csv HIP_VISIBLE_DEVICES=4 \
python test_qmoe_multi.py --token 32768 --model-dim 4096 --inter-dim 192 --expert 193 --topk 9 \
  --activation silu --dtype bf16 --use-g1u1 1 --doweight-stage1 0 \
  --quant fp8 --quant-type per_tensor --warmup 1 --iters 1 --run perf

# 新内核
AITER_PR1X4_TRITON_REDUCE=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa_new \
AITER_CONFIG_FMOE=moe_stage2_opt/configs/new.csv HIP_VISIBLE_DEVICES=5 \
python test_qmoe_multi.py ...同上
```

产物在 `<dir>/moe_gemm2_0/` 和 `<dir>/moe_2stage_down_prefill_1x4_0/`，
`17_final_isa.s` 是最终 ISA，`00_origin.mlir` 里能直接看到 `fly.make_ordered_layout` 降级出的
layout 和 stride。

数一遍 epilogue 相关的 opcode：

```bash
rg -o "ds_write_b[0-9]+[a-z_0-9]*|ds_read2?_b[0-9]+|buffer_store_[a-z0-9]+|v_perm_b32" \
  /tmp/isa_old/moe_gemm2_0/17_final_isa.s | sort | uniq -c | sort -rn
```
