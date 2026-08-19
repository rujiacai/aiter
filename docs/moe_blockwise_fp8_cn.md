# Blockwise FP8 MoE 代码导读（从 `fused_moe` 到 kernel）

面向第一次接触这块代码的人。目标是把 **DeepSeek 风格 blockwise fp8 MoE** 这条链路从
Python 入口一路讲到 GPU kernel 内部，重点是新加的 FlyDSL 实现
（`8a5d25ec6 [FlyDSL] Add blockwise fp8 MoE (128x128) with swiglu clamp`），
同时说明它之前的 asm / CK 路径长什么样、为什么要新写一套。

配套文档：

- `docs/flydsl_blockwise_fp8_moe_cn.md` — 那个 commit 的**设计笔记**（为什么这么设计、踩了哪些坑、
  验证结果）。本文是**代码导读**，两者互补，尽量不重复。

---

## 0. 先说清楚：什么是 blockwise fp8

MoE 的两个 GEMM 都是 `A(fp8) × W(fp8) → f32`，区别只在**量化 scale 的粒度**。
aiter 用 `QuantType` 枚举描述（`csrc/include/aiter_enum.h:16`，Python 侧同名镜像）：

| QuantType | A scale 粒度 | W scale 粒度 | scale dtype | 典型来源 |
| --- | --- | --- | --- | --- |
| `per_Tensor` | 整个张量 1 个 | 整个张量 1 个 | f32 | 静态量化 |
| `per_Token` | 每行 1 个 | 每个输出通道 1 个 | f32 | 动态 per-token |
| `per_1x32` | 每 32 个元素 1 个 | 每 32 个元素 1 个 | **e8m0**（1 字节，只能表示 2 的幂） | MXFP4 / MXFP8 microscale |
| **`per_1x128` / `per_128x128`** | **每 1×128 一个** | **每 128×128 一个** | **f32** | **DeepSeek V3/V4、Qwen3 等** |

本文说的 "blockwise" 就是最后一行。它的两个关键特征：

1. **scale 是 fp32**，不是 e8m0 字节。表达能力强得多，但也意味着不能用 gfx950 那条
   "scale 作为指令操作数" 的硬件微缩放 MFMA。
2. **W 的 scale 是二维分块**（N 方向 128、K 方向 128 共用一个 scale），
   **A 的 scale 是一维分组**（每个 token 的每 128 个 K 元素一个 scale）。

`per_128x128` 在入口处就被 remap 成 `per_1x128`，所以后面所有代码只看 `per_1x128`：

`aiter/fused_moe.py:1260`

```python
quant_remap = {QuantType.per_128x128: QuantType.per_1x128}
```

张量形状速查（`E` = 本卡上的 expert 数，`K` = `model_dim`，`I` = `inter_dim`）：

| 张量 | 形状 | dtype | 备注 |
| --- | --- | --- | --- |
| `hidden_states` / `a1` | `(tokens, K)` | bf16 或 fp8 | |
| `a1_scale` | `(tokens, K/128)` | f32 | FlyDSL 路径**不转置**；asm 1-stage 要转置 |
| `w1` (gate+up) | `(E, 2I, K)` | fp8 | 需 `shuffle_weight(..., (16,16))` 预混洗 |
| `w1_scale` | `(E, 2I/128, K/128)` | f32 | **不做任何 shuffle**，原样传 |
| `a2` (中间态) | `(tokens, topk, I)` | bf16 → fp8 | |
| `a2_scale` | `(tokens*topk, I/128)` | f32 | |
| `w2` (down) | `(E, K, I)` | fp8 | 同样要预混洗 |
| `w2_scale` | `(E, K/128, I/128)` | f32 | |

> 常见坑：`e8m0_shuffle()` 是给 MXFP4/MXFP8 的 e8m0 scale 用的，它会把张量按 uint8
> 重新解释。**fp32 的 block scale 绝对不能过这个函数**，否则字节会被打乱。
> tuner 里原来就有这个 bug，在上述 commit 里修了
> （`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py:1748`）。

---

## 1. 从 `fused_moe` 入口往下走

### 1.1 三层薄封装

```
fused_moe(...)                aiter/fused_moe.py:454   用户 API，Enum 参数
  └─ fused_moe_(...)          aiter/fused_moe.py:596   torch.compile 的 custom_op 边界，Enum → int
       └─ _fused_moe_impl()   aiter/fused_moe.py:654   真正干活的地方
```

`fused_moe_` 这一层存在只是因为 torch custom_op 的 schema 不接受 Enum 和 `None` 的 int，
所以 `activation` / `quant_type` 在这里被转成 `.value`，`block_size_M=None` 转成 `-1`。
读代码时可以直接跳到 `_fused_moe_impl`。

用户侧最常用的参数：

| 参数 | 含义 |
| --- | --- |
| `w1` | `(E, 2*inter_dim, model_dim)`，gate 和 up 在 N 方向拼在一起（所谓 `g1u1`） |
| `w2` | `(E, model_dim, inter_dim)` |
| `topk_weight` / `topk_ids` | `(tokens, topk)`，路由结果 |
| `quant_type` | 这里传 `QuantType.per_128x128` |
| `w1_scale/w2_scale/a1_scale/a2_scale` | 上表的 scale |
| `expert_mask` | EP（expert parallel）用，标记哪些 expert 在本卡 |
| `doweight_stage1` | topk 权重在 stage1 乘还是 stage2 乘 |
| `swiglu_limit` | SwiGLU 的 clamp 上界，DeepSeek-V4 用 10.0 |
| `block_size_M` | 手动指定 sorting 的 tile M，一般留 `None` 让 metadata 决定 |

### 1.2 `_fused_moe_impl` 的执行顺序

```
1. quant_type = quant_remap[quant_type]                    # per_128x128 -> per_1x128
2. 推导 q_dtype_a / q_dtype_w                               # 见 1.3
3. metadata = get_2stage_cfgs(...)                          # 选后端，见第 2 节
4. moe_sorting(...)                                         # 见 1.4
5. if metadata.run_1stage:  fused_moe_1stage(...)           # 一个 kernel 干完
   else:                    fused_moe_2stages(...)          # stage1 + stage2
```

### 1.3 `q_dtype_a` / `q_dtype_w` 是怎么定的

`q_dtype_w` 直接就是 `w1.dtype`（fp8）。`q_dtype_a` 默认跟随 `w1.dtype`，但有一条 blockwise 专用短路：

`aiter/fused_moe.py:716-721`

```python
    if (
        quant_type == QuantType.per_1x128
        and hidden_states.dtype == dtypes.fp8
        and a1_scale is not None
    ):
        q_dtype_a = dtypes.fp8
```

意思是：如果上游（比如 EP 的 fp8 dispatch）已经把激活量化好了并给了 scale，就别再量化一遍。

### 1.4 `moe_sorting`：整条链路的地基

`moe_sorting`（`aiter/fused_moe.py:337`）把 `(tokens, topk)` 的路由结果重排成
**按 expert 连续、按 `block_size` 对齐**的行序，这样每个 GEMM tile 内的所有行都属于同一个 expert，
可以共用一份权重。它返回 5 个东西：

| 返回值 | 形状 | 含义 |
| --- | --- | --- |
| `sorted_ids` | `(max_padded,)` i32 | 每个排序后行的编码：**低 24 位 = token_id，高 8 位 = slot_id（topk 内的第几个）** |
| `sorted_weights` | `(max_padded,)` f32 | 对应的 topk 权重 |
| `sorted_expert_ids` | `(num_blocks,)` i32 | 每个 M block 属于哪个 expert。**长度就是 grid.y** |
| `num_valid_ids` | `(1,)` i32 | 有效行数（含 padding 之前的截断位置） |
| `moe_buf` | `(tokens, model_dim)` | 最终输出 buffer，atomic 模式下会被清零 |

**padding 约定**：不足一个 block 的位置用 `token_id == tokens`（越界值）填充。
kernel 必须自己判断 `token_id < tokens` 来跳过或置零。这个约定在后面 FlyDSL kernel 的
scale 处理里很关键。

`max_padded = tokens*topk + E*block_size - topk`，也就是最坏情况每个 expert 都要补一个 block。
**这就是 `tile_m` 越大浪费越多的原因**：E=385 的 DSv4 shape 下，`tile_m=64` 比 `tile_m=16`
多出的 padding 行可能比真实行还多。

---

## 2. 后端选择：`get_2stage_cfgs`

`aiter/fused_moe.py:2087`，带 `lru_cache`。它返回一个 `MOEMetadata`
（`aiter/fused_moe.py:1281`），里面装着 stage1 / stage2 的可调用对象、`block_m`、`ksplit` 等。

选择逻辑分两条：

### 2.1 优先：查 tuned CSV

配置文件是 `aiter/configs/tuned_fmoe.csv`（可用 `AITER_CONFIG_FMOE` 换掉），
以及 `aiter/configs/model_configs/a8w8_blockscale_*_fmoe_*.csv` 这类按模型分的表。

索引键（`_INDEX_COLS`，`aiter/fused_moe.py:2114`）：

```
gfx, cu_num, token, model_dim, inter_dim, expert, topk,
act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1
```

注意 `token` 不是真实 token 数，而是 `get_padded_M(M)` 的结果——小于 32768 时向上取 2 的幂，
再往上分 32768 / 131072 两档。所以 tuning 表是按 token 数量级组织的。

命中后拿到的关键列：`block_m`、`ksplit`、`kernelName1`、`kernelName2`、`run_1stage`。
**后端就是靠 `kernelName` 的前缀区分的**：

| 前缀 | 后端 |
| --- | --- |
| `moe_ck2stages_*` | CK |
| `flydsl_moe1_*` / `flydsl_moe2_*` | FlyDSL |
| `opus_moe1_*` / `opus_*` | OPUS |
| `cktile_*` | CK-tile |
| 其它（mangled asm 符号） | asm |

### 2.2 兜底：默认启发式

CSV 没命中时走 `cfg is None` 分支（`aiter/fused_moe.py:2320`）。blockwise 的那条判断：

`aiter/fused_moe.py:2337-2343`

```python
            if q_type == QuantType.per_1x128:
                # for fp8 blockscale, ck has better performance so disable assembly kernel
                run_1stage = (
                    token > 32
                    and (inter_dim % 128 == 0)
                    and os.environ.get("AITER_FLYDSL_BLKFP8", "0") != "1"
                )
```

这是 `AITER_FLYDSL_BLKFP8` 的**第一处门控**：置 1 时强制关掉 asm 1-stage，好让流程往下走到
FlyDSL 分支。（那行注释和代码语义是反的——代码实际是 `token > 32` 时**启用** 1-stage asm，
注释是历史遗留，别被误导。）

### 2.3 FlyDSL blockwise 的插入点

`AITER_FLYDSL_BLKFP8` 的**第二处门控**，就在 `run_1stage` 分支之后：

`aiter/fused_moe.py:2462-2510`

```python
    if (
        q_type == QuantType.per_1x128
        and q_dtype_w == dtypes.fp8
        and q_dtype_a == dtypes.fp8
        and os.environ.get("AITER_FLYDSL_BLKFP8", "0") == "1"
        and is_flydsl_available()
    ):
        ...
        _out_str = "bf16"
        _tile_m = 16 if token < 2048 else 32
        _tile_k2 = 256 if (inter_dim % 512 == 0) else 128
        from aiter.ops.flydsl.moe_kernels import flydsl_kernel_name

        kn1 = flydsl_kernel_name(1, "fp8", "fp8blk", _out_str, _tile_m, 128, 128) + "_w2"
        kn2 = (
            flydsl_kernel_name(
                2, "fp8", "fp8blk", _out_str, _tile_m, 256, _tile_k2, "atomic"
            )
            + "_w2"
        )
        return MOEMetadata(
            functools.partial(_flydsl_stage1_wrapper, kernelName=kn1, ...),
            functools.partial(_flydsl_stage2_wrapper, kernelName=kn2, ...),
            _tile_m,
            1,      # no split-K
            False,  # run_1stage
        )
```

这里的 tile 是**硬编码的实测默认值**，不是 tuner 出来的。三个数字的来由：

- `tile_m = 16/32`：再大就被 per-expert padding 吃掉（见 1.4）。
- stage1 `tile_k=128`、stage2 `tile_k=256`：stage2 的 K 是 `inter_dim`，通常更小，
  需要更大的 tile_k 才能摊薄流水开销；同时 `K/tile_k` 必须是偶数（ping-pong 尾部固定吃两个 tile）。
- `_w2` = `waves_per_eu=2`。**这个后缀很重要**：wrapper 默认的 3 在某些配置下会掉进寄存器压力悬崖
  （同一配置 4220 us vs 17036 us）。

这段就是后续接 tuner 时要替换掉的地方——把这三个数字换成 CSV 查表。

### 2.4 全景图

```
fused_moe(quant_type=per_128x128)
  │
  ├─ remap -> per_1x128
  ├─ get_2stage_cfgs
  │     │
  │     ├─ tuned_fmoe.csv 命中?
  │     │     ├─ run_1stage=1 ─────────────────► [A] 1-stage asm
  │     │     └─ run_1stage=0
  │     │           ├─ kernelName1 = moe_ck2stages_* ──► [C] CK stage1 + CK stage2
  │     │           └─ kernelName1 = asm mangled ─────► [B] asm stage1 + CK stage2
  │     │
  │     └─ 未命中 (默认启发式)
  │           ├─ AITER_FLYDSL_BLKFP8=1 ───────────────► [D] FlyDSL stage1 + stage2  ← 本文重点
  │           ├─ token > 32 且 inter_dim%128==0 ─────► [A] 1-stage asm
  │           └─ 否则 ────────────────────────────────► [C] CK 2-stage
  │
  └─ moe_sorting  →  stage1 [→ a2 quant] → stage2

[A] fmoe_fp8_blockscale_g1u1        hsa/*/fmoe/{silu,gelu}/*.co
[B] moe_stage1_g1u1 + ck_moe_stage2 hsa/*/fmoe_2stages/*.co + CK
[C] moe_ck2stages_gemm{1,2}_*       csrc/ck_gemm_moe_2stages_codegen/
[D] flydsl_moe{1,2}_afp8_wfp8blk_*  aiter/ops/flydsl/kernels/moe_2stage_blockscale.py
```

实测目前生产上跑得最多的是 [C]：`tuned_fmoe.csv` 里 133 行 `per_1x128` 配置中，
127 行是 2-stage，其中 122 行 stage1 用 CK。

---

## 3. 老路径：asm / CK 是怎么做的

理解新实现之前，先知道它在替换什么。blockwise fp8 在 main 上原本有三条路。

### 3.1 1-stage 融合 asm：`fmoe_fp8_blockscale_g1u1`

一个 kernel 从 `hidden_states` 直接算到最终输出，中间的 gate/up GEMM、激活、down GEMM、
topk 归约全在 kernel 内部完成。

调用链：

```
fused_moe_1stage                aiter/fused_moe.py:1016
  └─ aiter.fmoe_fp8_blockscale_g1u1   aiter/ops/moe_op.py:225   (@compile_ops ctypes stub)
       └─ csrc/py_itfs_cu/asm_fmoe.cu:893   host wrapper
            ├─ 选 config_map（按 out dtype / act / A 是否已量化）
            ├─ get_heuristic_kernel(...)     asm_fmoe.cu:243
            └─ AiterAsmKernel(...).launch_kernel<1,2,false>(...)
                 └─ $AITER_ASM_DIR/{gfx}/fmoe/{silu,gelu}/*.co
```

选核不是编译期的模板实例化，而是**运行时查一张 CSV**：
`hsa/{gfx942,gfx950}/fmoe/{silu,gelu}/fmoe_bf16_blockscaleFp8_g1u1_silu.csv`，
列是 `knl_name, co_name, atm, vskip, smf, tg_num_perCU, ps, subGU_m, subGU_n`。
`get_heuristic_kernel` 按 `subGU_m == block_size_M`、`inter_dim % subGU_n == 0` 过滤，
再挑 CU 轮数最少的。

`.co` 文件名的命名法，例如 `fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co`：

| 片段 | 含义 |
| --- | --- |
| `bf16` | 输出 dtype |
| `blockscaleFp8` / `blockscaleBf16` | A 已经是 fp8 / A 是 bf16 由 kernel 内部量化（xbf16 模式） |
| `g1u1` | gate + up 融合布局 |
| `vs` / `novs` | vskip 开关 |
| `1tg` / `ps` / `pf2` | TG 打包 / persistent / prefetch 深度 |
| `32x256` | `subGU_m × subGU_n`，也就是 tile |

scale 怎么进去的：`FMoeKernel::launch_kernel`（`asm_fmoe.cu:101`）把
`input_scale → ptr_XQ`、`fc1_scale → ptr_GUQ`、`fc2_scale → ptr_DQ`，
expert 间的 stride 从 tensor stride 算。**A scale 需要提前转置**：

`aiter/fused_moe.py:1091-1092`

```python
                if quant_type == QuantType.per_1x128:
                    quant_func = functools.partial(quant_func, transpose_scale=True)
```

已经是 fp8 的输入则走 `aiter.partial_transpose`。这一点和 FlyDSL 路径不同，
FlyDSL 直接吃行主序的 `(tokens, K/128)`。

### 3.2 2-stage：asm stage1 + CK stage2

没有 asm 的 blockscale stage2。2-stage 时 stage1 用 asm、stage2 用 CK：

```
asm_stage1                      aiter/fused_moe.py:3373
  └─ aiter.moe_stage1_g1u1      aiter/ops/moe_op.py:248
       └─ csrc/py_itfs_cu/asm_moe_2stage.cu:62 / :149
            └─ cfg_fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1
                 └─ hsa/{gfx}/fmoe_2stages/*.co
ck_moe_stage2_fwd               → CK 模板
```

这条路有一个很容易看懵的细节：**asm stage1 把中间态的 a2 量化融合进去了**，
而且 a2 和 a2_scale 挤在同一个 buffer 里：

`aiter/fused_moe.py:3149-3155`

```python
    elif quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
        ratio = a1_scale.element_size() // a1.element_size()
        a2 = torch.empty(
            (token_num + (token_num * ratio + 127) // 128, topk, inter_dim),
            dtype=q_dtype_a,
            device=device,
        )
```

前 `token_num` 行是 fp8 的 a2，后面多出来的行是 f32 scale 的字节。kernel 跑完后再切开：

`aiter/fused_moe.py:3313-3321`

```python
    elif quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
        a2_v = a2[:token_num, :, :]
        a2_scale = (
            a2[token_num:, ...]
            .view(-1)[: token_num * topk * inter_dim * ratio // 128]
            .view(dtypes.fp32)
            .view(token_num, -1)
        )
        a2 = a2_v
```

**FlyDSL 路径没有这个融合**：stage1 输出 bf16，a2 的量化由 host 侧的 `per_1x128` quant op 完成。
这既是精度更好的原因（少一次量化误差在 GEMM 前引入），也是性能落后的原因之一
（多一趟 HBM 读写）。

### 3.3 CK 2-stage

`csrc/ck_gemm_moe_2stages_codegen/` 在 JIT 时用 `gen_instances.py` 生成 CK 模板实例。
blockscale 对应 `MulABScaleExpertWeightA8W8blkscale` 这个 element-op，
生成的 kernel 名形如：

```
moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleExpertWeightA8W8blkscale_v3_Nswizzle0_Quant4_MulRoutedWeight0_gelu_F8_F8_B16
```

`Quant4` 就是 `per_1x128`。在 `tuned_fmoe.csv` 里，`per_1x128` 的 133 行中有 127 行是
2-stage、且多数是 CK，说明**目前生产上 blockwise 主要跑 CK**。

### 3.4 小结对比

| | 1-stage asm | 2-stage asm+CK | 2-stage CK | **2-stage FlyDSL（新）** |
| --- | --- | --- | --- | --- |
| gfx942 | 有 | 有 | 有 | **有** |
| a2 quant | 内部 | stage1 融合 | stage1 融合 | host（未融合） |
| 可读性 / 可改性 | `.co` 二进制，不可改 | 同左 | C++ 模板 | **Python DSL，可改** |
| `swiglu_limit` | 不支持 | 不支持 | 不支持 | **支持** |
| tile 可调 | 只能在预编译的 `.co` 里选 | 同左 | 模板实例 | **任意重编译** |

新实现的价值主要在最后三行：能改、支持 DSv4 需要的 clamp、tile 空间连续。

前三列的「不支持」现在是**显式报错**而不是静默丢弃（见 5.6）。唯一的例外是
`ActivationType.Swiglu` + `limit=7.0`：asm/CK/cktile 的 OAI swiglu 内部就是这个常量，
语义一致所以放行。

---

## 4. 新路径：FlyDSL blockwise 实现

### 4.1 为什么必须新开一个模块

main 上 fp8 MoE 的新家是 `kernels/mixed_moe_gemm_2stage_common.py`，但 blockwise **接不进去**，
两条硬约束：

1. 那个 pipeline 全文只有一条 MFMA：`mfma_scale_f32_16x16x128_f8f6f4`，**gfx950 专属**。
   gfx942 上 LLVM 直接 `Cannot select`。
2. 那条指令的 scale 操作数是 **e8m0 字节**（用 `opsel_a`/`opsel_b` 从 i32 里选字节），
   只能表示 2 的幂，物理上表达不了 fp32 的 block scale。

所以 blockwise 用 plain `mfma_f32_16x16x32_fp8_fp8` + **显式 f32 FMA**，
单独成模块 `aiter/ops/flydsl/kernels/moe_2stage_blockscale.py`。
它复用 main 仍在的底层 helper（`mfma_preshuffle_pipeline.py` / `mfma_epilogues.py`），
不依赖已被删除的旧 pipeline。

模块内支持两个 `in_dtype`：

| `in_dtype` | 用途 |
| --- | --- |
| `fp8_blk` | 正主，128×128 f32 block scale |
| `fp8` | 同一套 pipeline 的 per-row / per-token f32 scale。**调试对照组**：scale 索引写错时它仍然正确，用来把「索引 bug」和「数据流 bug」分开 |

### 4.2 从 metadata 到 kernel 的编译链

```
_flydsl_stage1_wrapper                aiter/fused_moe.py:1381
  │  解析 kernelName -> {tile_m, tile_n, tile_k, a_dtype, b_dtype, waves_per_eu, ...}
  └─ flydsl_moe_stage1                aiter/ops/flydsl/moe_kernels.py:1988
       │  摊平 scale、算 grid、组装 kernel 参数
       ├─ compile_flydsl_moe_stage1   aiter/ops/flydsl/moe_kernels.py:677
       │    └─ (a_dtype=="fp8" and b_dtype in ("fp8blk","fp8row"))       :771
       │         └─ compile_moe_gemm1  kernels/moe_2stage_blockscale.py:132
       └─ _run_compiled(exe, args)
```

kernel 名的注册表在 `get_flydsl_stage1_kernels_fp8_blk` / `..._stage2_...`
（`moe_kernels.py:582` / `:617`），在 import 时被 `_register_all_configs()` 塞进 `_KERNEL_PARAMS`。
命名规则：

```
flydsl_moe1_afp8_wfp8blk_{out}_t{M}x{N}x{K}[_w{waves}]
flydsl_moe2_afp8_wfp8blk_{out}_t{M}x{N}x{K}_atomic[_w{waves}][_persist]
```

用 `wfp8blk` 而不是 `wfp8`，因为后者已经是 per-1×32 mxfp8 家族的名字。
对照组是 `wfp8row`，只在 compile 分派里认，不进 registry。

注册的候选空间：stage1 `tile_m ∈ {16,32,64,128} × tile_n ∈ {64,128} × tile_k ∈ {128,256} × w ∈ {0..4}`，
stage2 `tile_n ∈ {128,256}`。这就是 tuner 会去扫的集合。

一个特殊之处：**`swiglu_limit` 是编译期参数**，不是 kernel 运行时参数。

`aiter/ops/flydsl/moe_kernels.py:1843-1846`

```python
    if b_dtype == "fp8blk":
        # Blockwise fp8 bakes the clamp into the kernel instead of taking it as a
        # runtime arg, so it has to reach compile time.
        compile_kwargs["swiglu_limit"] = _swiglu_limit_val
```

好处是 `+inf`（不 clamp）时**一条指令都不生成**，坏处是不同 limit 会各自编一份。

### 4.3 host 侧数据流（2-stage）

`fused_moe_2stages`（`aiter/fused_moe.py:2975`）里，blockwise FlyDSL 的路径是：

```
a1, a1_scale = quant_func(hidden_states)         # per_1x128，(tokens, K/128) f32，不转置
a2 = empty((tokens, topk, inter_dim), bf16)      # 中间态就是普通 bf16
metadata.stage1(a1, w1, ..., a1_scale, w1_scale, out=a2)
a2, a2_scale = quant_func(a2, num_rows_factor=topk)   # host 侧 per_1x128 量化
metadata.stage2(a2, ..., a2_scale, w2_scale, out=moe_out)
```

`metadata.prequant` 默认 `True`、`skip_inter_quant` 默认 `False`，所以走的是最朴素的两次
host 量化。注意 `transpose_scale=True` 只对 `asm_stage1` 生效
（`aiter/fused_moe.py:3123`），FlyDSL 拿到的是行主序 scale。

### 4.4 kernel 内部：整体骨架

stage1 的 host launcher 是 `launch_moe_gemm1`：

`aiter/ops/flydsl/kernels/moe_2stage_blockscale.py:2213-2240`

```python
        gx = inter_in // fx.Index(tile_n)
        gy = size_expert_ids_in
        ...
        _k1.launch(
            grid=(gx, gy, k_batch),
            block=(256, 1, 1),
            stream=stream,
        )
```

- **grid.x** = `inter_dim / tile_n`。注意这里是 `inter_dim` 不是 `2*inter_dim`——
  一个 block 同时算 gate 和 up 两个 B tile（所以 kernel 内部有 `gate_list` 和 `up_list` 两套累加器）。
- **grid.y** = `sorted_expert_ids` 的长度，即 M 方向的 block 数。
- **block** = 256 线程 = 4 个 wave。

> 这里埋过一个很痛的坑：kernel 参数 `n_in` 传成 `2*inter_dim` 会让 grid.x 翻倍、
> 输出 buffer resource 也大一倍，越界写到别的显存上，**表现是随机的错误结果**。
> `_n_in = inter_dim * 2 if use_mx_gemm else inter_dim`（`moe_kernels.py:1770`），
> 而 `fp8blk` 的 `use_mx_gemm` 是 False。

单个 workgroup 的主循环是 **ping-pong 双缓冲**：A tile 走 LDS（两份轮换），
B tile 直接 gmem → 寄存器（预混洗过，不进 LDS）。K 方向按 `tile_k` 切，
每个 tile 内再按 **64 字节的 K 微步**（`k_unroll = tile_k_bytes / 64`）展开，
每个微步发 2 条 `mfma_f32_16x16x32_fp8_fp8`。

### 4.5 核心设计：scale 怎么折进累加器

这是整个实现最需要理解的一段。

#### (a) MFMA 累加器的行映射

`16x16x32` MFMA 的 f32x4 累加器，在一个 lane 内对应的输出坐标是
（见 `kernels/mfma_epilogues.py:14` 的注释）：

```
row = bx_m + mi*16 + lane_div_16*4 + ii      # ii = 0..3
col = 由 lane_mod_16 决定
```

也就是说：**一个 lane 手里的 4 个 f32，是 4 个连续的 M 行、同一个 N 列**。

由此推出两个非对称的结论：

- **W scale** 只随 `(N-block, K-block)` 变化，在一个 lane 内是**标量**，可以直接广播。
- **A scale** 随 token（即 M 行）变化，在一个 lane 内是 **4 个不同的值**，必须组成 `f32x4`。

这就是为什么代码里有两个版本的 FMA helper：

`aiter/ops/flydsl/kernels/moe_2stage_blockscale.py:1234-1256`

```python
                    def _acc_scaled_f32(f32_acc_vec, f32_partial_vec, scale_val):
                        """MFMA f32 partial -> scale -> add to f32 accumulator via math.fma on vector."""
                        ...
                        scale_vec = _uw(vector.broadcast(T.f32x4, scale_val))
                        ...

                    def _acc_scaled_f32_vec(f32_acc_vec, f32_partial_vec, scale_vec4):
                        """Like _acc_scaled_f32, but the scale already varies across the
                        4 accumulator lanes (blockwise fp8 has one A scale per M row)."""
```

blockwise 用的是后者。

#### (b) 两级累加：每个 K block 只做一次 FMA

`tile_k=128` 时一个 K tile 正好是一个 scale block。主循环长这样
（`moe_2stage_blockscale.py:1274`）：

```
for 每个 K scale block:
    partial_gate = 0                      # 零初始化的 f32x4
    partial_up   = 0
    for ku in 0..ku_per_kblk:             # ku_per_kblk = scale_blk_k / 64 = 2
        4 × MFMA 链式累加进 partial       # 每个 ku 发 2 条 gate + 2 条 up
    sa = A 的 4 个 f32 scale（每行一个）
    sw = W 的 1 个 f32 scale
    acc += (sa * sw) * partial            # 一次 f32x4 FMA
```

关键代码：

`aiter/ops/flydsl/kernels/moe_2stage_blockscale.py:1317-1341`

```python
                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    pg = zero_f32_acc
                                    pu = zero_f32_acc
                                    for kj in range_constexpr(ku_per_kblk):
                                        a0, a1 = a_packs[kj]
                                        bg0, bg1, _ = b_gate_tile_in[ku_first + kj][ni]
                                        bu0, bu1, _ = b_up_tile_in[ku_first + kj][ni]
                                        pg = mfma_fn(mfma_res_ty, [a0, bg0, pg, 0, 0, 0])
                                        pg = mfma_fn(mfma_res_ty, [a1, bg1, pg, 0, 0, 0])
                                        pu = mfma_fn(mfma_res_ty, [a0, bu0, pu, 0, 0, 0])
                                        pu = mfma_fn(mfma_res_ty, [a1, bu1, pu, 0, 0, 0])
                                    # Fold the scalar W scale into the per-lane A scales
                                    # with 4 scalar muls; a vector mul lowers to the same
                                    # 4 v_mul_f32 anyway.
                                    sg = vector.from_elements(
                                        T.f32x4,
                                        [sa[ii] * sw_gate_blk[ni] for ii in range(4)],
                                    )
                                    su = vector.from_elements(
                                        T.f32x4,
                                        [sa[ii] * sw_up_blk[ni] for ii in range(4)],
                                    )
                                    push_fma(gate_list, acc_idx, pg, sg)
                                    push_fma(up_list, acc_idx, pu, su)
```

相比 microscale 路径每 K32 做一次 scale FMA，这里每 K128 才做一次，**scale FMA 少 4 倍**。
`partial` 只在 `(mi, ni)` 内活 4 条 MFMA，不需要开第二套完整累加器数组。

#### (c) FMA 流水：`_make_scale_fma_pipe`

刚发完 MFMA 就读它的结果会 stall 整个 MFMA 延迟。所以 FMA 被塞进一个 FIFO 延迟执行：

`aiter/ops/flydsl/kernels/moe_2stage_blockscale.py:75-102`

```python
def _make_scale_fma_pipe(apply_fn, depth: int):
    """FIFO for deferred post-MFMA scale FMAs.

    ``push(acc_list, idx, partial, scale)`` records one pending
    ``acc_list[idx] = apply_fn(acc_list[idx], partial, scale)`` and retires the
    oldest entry once more than ``depth`` are outstanding; ``drain()`` flushes
    the rest. Retirement is FIFO, so per-accumulator ordering (and therefore
    the f32 accumulation result) is bit-identical to applying each FMA inline.
    """
```

深度默认 4，`AITER_BLKFP8_FMA_DEPTH` 可调，设 0 退回内联。
FIFO 保序，所以结果和内联**逐位相同**——这点很重要，调这个参数不会影响精度。

#### (d) padded 行：靠 A scale 置零兜住

`moe_sorting` 用 `token_id == tokens` 做 padding。这些行的 A scale 直接给 0：

`aiter/ops/flydsl/kernels/moe_2stage_blockscale.py:925-942`

```python
                            fused_blk = buffer_ops.buffer_load(
                                sorted_rsrc, row_blk, vec_width=1, dtype=T.i32
                            )
                            t_blk = fused_blk & mask24
                            # moe_sorting pads with token_id == tokens; clamp the row so
                            # the scale load stays in bounds and force the scale to 0 so
                            # padded rows accumulate exactly 0 (the epilogue then needs
                            # no separate mask).
                            t_ok = arith.cmpi(
                                arith.CmpIPredicate.ult, t_blk, tokens_i32
                            )
                            t_safe = arith.select(t_ok, t_blk, fx.Int32(0))
                            bases_mi.append(
                                arith.index_cast(T.index, t_safe) * c_num_kb
                            )
                            ok_mi.append(t_ok)
```

于是累加器天然是 0。stage1 的 store 本来就有谓词无所谓，但 **stage2 用 atomic 累加，
脏行会污染输出**，全靠这个 0 兜住。

副作用是 epilogue 里的 `sx` / `sw` 可以直接写常量 1.0（`_epi_sx_one` / `_epi_sw_one`，
`moe_2stage_blockscale.py:466`），因为 scale 已经在 K 循环里折进去了。

另外注意这段 token id 解码放在 **K 循环外面**：`a_blk_scale_base[mi][ii]` 预先算好行基址，
循环内只加一个 block 索引。

#### (e) W scale 的索引：用移位不用除法

`aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:886-896`

```python
    if scale_blk_n & (scale_blk_n - 1) or scale_blk_k & (scale_blk_k - 1):
        raise ValueError(
            f"block scale sizes must be powers of two, got "
            f"scale_blk_n={scale_blk_n}, scale_blk_k={scale_blk_k}"
        )
    n_global = fx.Int32(n_blk) * fx.Int32(16) + fx.Int32(n_intra)
    nb = n_global >> fx.Int32(scale_blk_n.bit_length() - 1)
    kb = fx.Int32(k_pos) >> fx.Int32(scale_blk_k.bit_length() - 1)
    elem_idx = nb * fx.Int32(num_k_blocks) + kb
    return buffer_ops.buffer_load(scale_rsrc, elem_idx, vec_width=1, dtype=T.f32)
```

两个要点：

1. `idx2crd` 返回的是 **i32**，和 `fx.Index` 常量做 `//` **不会** lower 成期望的整数除法。
   实测过：expert 层的偏移对、expert 内的偏移错。所以全部先 `fx.Int32(...)` 归一再移位。
2. 因为 `N_per_expert % scale_blk_n == 0`，**expert 的 stride 自动折进 N-block 索引**，
   不需要单独的 expert 项。调用方传进来的 `n_blk`/`n_intra` 已经带了 `expert_off`。

`k_pos = base_k + ku*64` 携带了 K64 微步，所以 `tile_k=256`（一个 tile 跨 2 个 scale block）
时自动选对块。

#### (f) SwiGLU clamp

`aiter/ops/flydsl/kernels/moe_2stage_blockscale.py:549-557`

```python
            def clamp_gate(x):
                if const_expr(not _clamp_act):
                    return x
                return -((-x).maximumf(_neg_lim))

            def clamp_up(x):
                if const_expr(not _clamp_act):
                    return x
                return (-((-x).maximumf(_neg_lim))).maximumf(_neg_lim)
```

约定：**gate 只 clamp 上界，up 上下都 clamp**，和 triton 的 `fused_clamp_act_mul`
以及 gfx1250 / grouped FlyDSL kernel 一致。
`min(x, lim)` 写成 `-max(-x, -lim)` 是因为只有 `maximumf` 被 wrap 了。
条件必须走 `const_expr`——裸的 Python `if` 在 traced kernel body 里会被改写成
device 端分支，两个分支都会被 trace，no-clamp 那支就会把 `None` 喂给 `maximumf`。

最终 epilogue：`y = silu(clamp_gate(vg)) * clamp_up(vu)`（`:2036` / `:2171`）。

### 4.6 stage2 的差异

stage2（`compile_moe_gemm2`，`moe_2stage_blockscale.py:2247`）结构基本对称，三点不同。

**一、A2 scale 按 `(token, slot)` 索引，不是按 token。**

`aiter/ops/flydsl/kernels/moe_2stage_blockscale.py:2941-2944`

```python
                            ts_row = t_safe * topk_i32 + s_safe
                            bases_mi.append(
                                arith.index_cast(T.index, ts_row) * c_num_kb
                            )
```

`sorted_ids` 的高 8 位 slot_id 在这里被用上了，有效性判断也从 `t_ok` 变成 `t_ok & s_ok`。

**二、输出用 global atomic 累加**（`mode="atomic"`），把 topk 个 slot 的结果加到同一行。
这就是为什么 padded 行必须精确为 0，也是为什么 benchmark 前必须**先快照再计时**——
重复 launch 会一直往里加。

**三、没有 gate/up 两套累加器**，只有一套；`n_in = model_dim`、`k_in = inter_dim`。

---

## 5. 怎么跑、怎么调

### 5.1 直接调 kernel（最快的迭代方式）

```python
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

out = flydsl_moe_stage1(
    a_q, w1_shuffled, sorted_ids, sorted_expert_ids, num_valid_ids,
    out=out, topk=topk,
    tile_m=32, tile_n=128, tile_k=128,
    a_dtype="fp8", b_dtype="fp8blk", out_dtype="bf16", act="silu",
    w1_scale=w1_s,        # (E, 2*inter_dim/128, model_dim/128) f32
    a1_scale=a_s,         # (tokens, model_dim/128) f32
    waves_per_eu=2,
    swiglu_limit=10.0,    # None 表示不 clamp
    smooth_scale=sm,      # 可选，(E, inter_dim) f32；None 表示不用
)
```

权重要先 `shuffle_weight(w.view(int8), layout=(16,16)).view(fp8)`。
scale **不需要**任何 preshuffle。

`smooth_scale` 是可选的 smoothquant 前置缩放，语义和 `torch_moe` 的 `fc2_smooth_scale`
一致——把因子折进 stage1 的激活，让 down projection 看到已缩放的输入：

```
act = silu(clamp_gate(gate)) * clamp_up(up) * smooth_scale[expert, n]
```

它是**编译期开关**（`enable_smooth_scale`），不传时指针和那条乘法完全不生成，
kernel 和改动前逐字节相同。两个限制：只支持 `b_dtype="fp8blk"/"fp8row"`，
且不能和 split-K 同用（split-K 的 stage1 只写 gate/up partial，激活在 host 侧做，
epilogue 里没地方折），两种情况都会 `NotImplementedError`。

`fused_moe` 也接了这个参数：

```python
out = fused_moe(
    hidden_states, w1, w2, topk_weight, topk_ids,
    quant_type=aiter.QuantType.per_128x128,
    w1_scale=w1_s, w2_scale=w2_s,
    smooth_scale=sm,      # (E, inter_dim) f32
)
```

同样有后端校验（`_check_smooth_scale_supported`）：只有 FlyDSL blockwise stage1
实现了它，走到 asm / CK / cktile 会直接 `NotImplementedError` 而不是静默丢弃。
所以用它必须配 `AITER_FLYDSL_BLKFP8=1`，必要时再加 `AITER_BYPASS_TUNE_CONFIG=1`
（见 5.6 的同款陷阱）。

### 5.2 走 `fused_moe`

```bash
export AITER_FLYDSL_BLKFP8=1    # 不设则 per_1x128 仍走原来的 asm / CK
```

### 5.3 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `AITER_FLYDSL_BLKFP8` | `0` | 开启 blockwise fp8 的 FlyDSL 路径 |
| `AITER_BLKFP8_FMA_DEPTH` | `4` | post-MFMA scale FMA 流水深度，0 = 内联 |
| `AITER_CONFIG_FMOE` | — | 指向自定义的 tuned CSV |
| `AITER_BYPASS_TUNE_CONFIG` | `0` | 忽略 CSV，强制走默认启发式 |

### 5.4 测试

harness：`op_tests/flydsl_tests/test_flydsl_moe_blockscale.py`

```bash
cd /data/aiter_main/aiter
export HIP_VISIBLE_DEVICES=0

# 冒烟
python op_tests/flydsl_tests/test_flydsl_moe_blockscale.py \
    -t 128 -dim 1024 -idim 256 -e 8 -k 2 --swiglu-limit 10.0

# DSv4 + 计时
python op_tests/flydsl_tests/test_flydsl_moe_blockscale.py \
    -t 512 -dim 7168 -idim 512 -e 385 -k 7 --swiglu-limit 10.0 --bench

# 端到端
AITER_FLYDSL_BLKFP8=1 AITER_BYPASS_TUNE_CONFIG=1 python op_tests/test_moe_2stage.py \
    -q 5 -t 128 -dim 4096,512 -e 32 -k 4 -sl 10.0 --no-flydsl-csv

# tuner（TUNE_ONLY 必加）
TUNE_ONLY=flydslblk python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
    -i /tmp/untuned_blk.csv -o /tmp/tuned_blk.csv --last
```

几个容易踩的地方：

- **不要加 `--no-legacy`**。它的 help 写的是"跳过内置的固定 shape 扫描"，但
  `-t` / `-dim` / `-e` / `-k` / `-q` 走的是同一个 `_iter_legacy_cases`（"the original
  CLI-driven sweep"），加了它会变成 `scanned 0 cases`，什么都不跑。
- 端到端那条带了 `AITER_BYPASS_TUNE_CONFIG=1`，是为了绕开 tuned CSV。某些 token 档
  CSV 里有 `run_1stage=1` 的行，会把 dispatch 拽回 asm 1-stage，`AITER_FLYDSL_BLKFP8`
  就白设了（详见 5.6）。
- 只想看每个 stage 的耗时就加 `--kernel`；1-stage 路径下 `us_stage2` 会是 `nan`，
  且 `us_stage1` 实际是整个 MoE 的时间。

判据看 `logits_diff`（余弦距离），**不要看 `checkAllclose`**：harness 用 `randn` 造权重不归一化，
输出量级上万，绝对容差 0.01 本来就过不了。

### 5.5 调试开关

harness 的这几个开关是专门为了把故障域切开设计的：

| 开关 | 作用 |
| --- | --- |
| `--scales none/a/w` | 把 A 或 W 的 scale 换成 1。索引写错时仍然正确 → 说明是数据流问题 |
| `--wscale-mode e/n/k` | 让 W scale 只沿 expert / N / K 单轴变化，定位是哪一维索引错 |
| `--in-dtype fp8` | 对照组，走同一 kernel 的 per-row scale 路径 |
| `--repeat N` | 跑 N 次比对是否逐位一致，查非确定性 |
| `--stage 1/2` | 只跑一个 stage |

另外记住：**flydsl 的编译缓存 key 不包含源码**。改了 kernel 但 tile / dtype 没变时可能拿到旧二进制。
结果诡异时先 `rm -rf /root/.flydsl/cache`。

### 5.6 `swiglu_limit` 的后端校验

只有 FlyDSL 和 Opus 的 stage1 真正读 `swiglu_limit`，asm / CK / cktile 没有对应的
kernel 操作数。为避免静默丢弃，`_fused_moe_impl` 在拿到 metadata 之后会校验一次
（`aiter/fused_moe.py` 的 `_check_swiglu_limit_supported`）：

```
swiglu_limit 为 None / 0                      -> 放行（本来就不 clamp）
stage1 是 _flydsl_stage1_wrapper / _opus_...  -> 放行（真参数）
activation=Swiglu 且 limit=7.0                -> 放行（asm/CK/cktile 的 OAI swiglu
                                                 内部就是硬编码 7.0，语义一致）
其它                                          -> raise NotImplementedError
```

所以 `swiglu_limit=10.0` 走 asm/CK 现在会直接报错，而不是返回没 clamp 的结果。
**没有旁路开关**——要么换到支持的后端，要么别传这个参数。

有个容易踩的点：**即使设了 `AITER_FLYDSL_BLKFP8=1` 也可能报错**。如果 tuned CSV 里
那个 token 档命中了 `run_1stage=1` 的行，dispatch 会落到 asm 1-stage——FlyDSL 分支在
`if run_1stage:` 之后（2440 vs 2462），env var 根本走不到。这种情况要配合
`AITER_BYPASS_TUNE_CONFIG=1` 才能确保走 FlyDSL。报错本身就是在提示这件事。

---

## 6. 现状与优化方向

### 6.1 当前数字（gfx942 / MI308X，DSv4 7168/512/385/7）

| tokens | asm 1-stage 基线 | FlyDSL blockwise |
| --- | --- | --- |
| 128 | 1299 us | 1310 us |
| 1024 | 1534 us | 2077 us |
| 8192 | 6197 us | 7782 us |

t=8192 拆开是 stage1 4221 us + stage2 3347 us。
精度上 FlyDSL 更好：`logits_diff` 3.2e-05 vs asm 的 8.5e-05（少一次中间量化）。

### 6.2 优化项，按预期收益排序

1. **把 tuner 结果落进 `configs/`**（最直接）。
   现在 `get_2stage_cfgs` 里的 tile 是硬编码的三个数字。tuner 通路已经接好
   （`TUNE_ONLY=flydslblk`，`gen_flydsl_blockscale_2stages_task`，
   `gemm_moe_tune.py:4236`），候选空间 240 个已验证全部 0.0% 误差。
   缺的只是把产出的 CSV 合进 `aiter/configs/`（注意 shape 去重，见
   `.claude/skills/aiter-config-shape/SKILL.md`）。
   落地后 2.3 节那段硬编码可以整个删掉。

2. **把 a2 量化融进 stage1 epilogue**。
   现在 stage1 写 bf16 → host 读一遍 → 量化 → stage2 再读。
   t=8192 时中间态是 `8192 × 7 × 512 × 2B ≈ 59 MB`，一来一回就是两趟 HBM。
   asm/CK 都融了（见 3.2），这是大 token 落后的主要来源之一。
   参考 mixed pipeline 里 `fuse_quant="fp8"` 的做法，
   MOEMetadata 有现成的 `fuse_quant` / `skip_inter_quant` 字段可以复用。

3. **`tile_m` 与 padding 的权衡**。
   E=385 时 `max_padded = tokens*topk + 385*tile_m - topk`。tile_m=16 时 padding 上限
   6160 行，tile_m=64 时 24640 行。但 tile_m 太小又摊不薄权重加载。
   可以考虑按 `tokens*topk/E` 动态选，或者引入按 expert 实际行数分档的调度。

4. **`waves_per_eu` 的悬崖**。
   同一配置 `w=2` 4220 us、`w=3` 17036 us。说明寄存器压力就在临界点上。
   值得看一眼 `-Rpass-analysis=kernel-resource-usage` 确认 VGPR 用量，
   然后针对性减压（比如减少同时活着的 B tile 寄存器、或调 `_BLK_FMA_DEPTH`
   ——FIFO 深度直接决定有多少组 f32x4 partial + f32x4 scale 同时活着，
   每加一层就多占若干 VGPR，而且它不影响数值结果，是很干净的调节旋钮）。

5. **scale 的加载开销**。
   现在每个 `(ku, ni)` 都调一次 `load_block_scale_f32`，虽然同一个 K block 内
   `ku` 变化时 `kb` 不变（编译器大概率能 CSE 掉），但没有显式提到 K block 外面。
   可以试着把 W scale 的加载显式提出循环。A scale 那边已经做了行基址预解码。

6. **split-K**。
   目前 metadata 里 `ksplit` 写死 1。decode 场景（`tokens*topk < E`）下 grid 很小，
   CU 跑不满，`get_ksplit`（`fused_moe.py:1203`）的启发式对 blockwise 是适用的。
   kernel 侧 `k_batch>1` 的骨架在（`compile_moe_gemm1` 有 `_is_splitk` 分支），但没验证过。

### 6.3 已知限制

- **`inter_dim` / `model_dim` 必须是 128 的倍数**，非对齐 tail block 未实现。
  TP 切出来的 `inter_dim=192` 跑不了（不过 sglang / vLLM 加载 blockwise fp8 权重时
  也会拒绝这种配置，实际部署会 pad 到 256）。
- **`K / tile_k` 必须是偶数**：ping-pong 尾部固定消费两个 tile。编译期有断言。
- **单个权重张量不能超过 4 GiB**：buffer resource 的 `num_records` 是 32 位。
  `E=385 + inter_dim=1536` 单卡放不下。这是既有限制，对所有 `in_dtype` 一样。
- `gemm_moe_tune.py` 在 DSv4 shape 上有一个 HIP illegal memory access，来自
  asm / cktile / opus 那几组候选，与本实现无关（用 `TUNE_ONLY=flydslblk` 隔离验证过）。

---

## 7. 代码索引

### Kernel

| 位置 | 内容 |
| --- | --- |
| `kernels/moe_2stage_blockscale.py:72` | `_BLK_FMA_DEPTH`，post-MFMA scale FMA 流水深度 |
| `kernels/moe_2stage_blockscale.py:75` | `_make_scale_fma_pipe` FIFO |
| `kernels/moe_2stage_blockscale.py:132` | `compile_moe_gemm1`，stage1 入口 |
| `kernels/moe_2stage_blockscale.py:254` | blockwise geometry 校验（`num_k_blocks` / `ku_per_kblk` / `kblk_per_tile`） |
| `kernels/moe_2stage_blockscale.py:466` | `_epi_sx_one` / `_epi_sw_one`，epilogue 用常量 scale |
| `kernels/moe_2stage_blockscale.py:549` | `clamp_gate` / `clamp_up` |
| `kernels/moe_2stage_blockscale.py:909` | A-scale 行基址预解码 |
| `kernels/moe_2stage_blockscale.py:972` | `load_b_tile` 的 blockwise 分支（B pack + 每块一个 f32 W scale） |
| `kernels/moe_2stage_blockscale.py:1244` | `_acc_scaled_f32_vec`，接受 f32x4 scale 的 FMA |
| `kernels/moe_2stage_blockscale.py:1258` | `_a_blk_scales`，取 4 个 A scale、无效行置 0 |
| `kernels/moe_2stage_blockscale.py:1274` | blockwise compute 主体（两级累加） |
| `kernels/moe_2stage_blockscale.py:2189` | `launch_moe_gemm1`，grid / block |
| `kernels/moe_2stage_blockscale.py:2247` | `compile_moe_gemm2`，stage2 入口 |
| `kernels/moe_2stage_blockscale.py:2911` | stage2 的 A2-scale 行基址（`t*topk + s`） |
| `kernels/mfma_preshuffle_pipeline.py:863` | `load_block_scale_f32` |
| `kernels/mfma_epilogues.py:14` | 累加器行映射的权威注释 |

### Host

| 文件 | 内容 |
| --- | --- |
| `aiter/fused_moe.py:454` | `fused_moe` 用户 API |
| `aiter/fused_moe.py:654` | `_fused_moe_impl` |
| `aiter/fused_moe.py:337` | `moe_sorting` |
| `aiter/fused_moe.py:2087` | `get_2stage_cfgs` |
| `aiter/fused_moe.py:2337` | `AITER_FLYDSL_BLKFP8` 门控之一：关掉 1-stage asm |
| `aiter/fused_moe.py:2462` | `AITER_FLYDSL_BLKFP8` 门控之二：FlyDSL dispatch |
| `aiter/fused_moe.py:1381` / `:1460` | `_flydsl_stage1_wrapper` / `_flydsl_stage2_wrapper` |
| `aiter/fused_moe.py:2975` | `fused_moe_2stages` |
| `aiter/ops/flydsl/moe_kernels.py:582` / `:617` | kernel 名注册表 |
| `aiter/ops/flydsl/moe_kernels.py:771` / `:880` | compile 分派的 `fp8blk` / `fp8row` 分支 |
| `aiter/ops/flydsl/moe_kernels.py:1988` / `:2381` | `flydsl_moe_stage1` / `flydsl_moe_stage2` |
| `aiter/aot/flydsl/moe.py` | AOT 预编译用的 dummy scale 形状 |

### asm / CK

| 文件 | 内容 |
| --- | --- |
| `aiter/ops/moe_op.py:224` | `fmoe_fp8_blockscale_g1u1` stub |
| `aiter/ops/moe_op.py:247` | `moe_stage1_g1u1` stub |
| `csrc/py_itfs_cu/asm_fmoe.cu:893` | 1-stage host wrapper |
| `csrc/py_itfs_cu/asm_fmoe.cu:243` | `get_heuristic_kernel` 选核 |
| `csrc/py_itfs_cu/asm_moe_2stage.cu:62` | 2-stage stage1 cfg 选择 |
| `hsa/{gfx942,gfx950}/fmoe/**` | 1-stage `.co` + CSV |
| `hsa/{gfx942,gfx950}/fmoe_2stages/**` | 2-stage stage1 `.co` + CSV |
| `csrc/ck_gemm_moe_2stages_codegen/` | CK codegen |

### 测试 / 调优

| 文件 | 内容 |
| --- | --- |
| `op_tests/flydsl_tests/test_flydsl_moe_blockscale.py` | 单 stage harness + 调试开关 |
| `op_tests/test_moe_2stage.py` | 端到端 |
| `op_tests/test_moe_blockscale.py` | asm 路径的测试 |
| `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py:4236` | `gen_flydsl_blockscale_2stages_task` |
| `aiter/configs/tuned_fmoe.csv` | 调优结果表 |
