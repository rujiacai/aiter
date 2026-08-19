# FlyDSL Blockwise FP8 MoE（DeepSeek 128×128）

fp8 激活 × fp8 权重的 MoE，权重每 128×128 一个 fp32 scale、激活每 1×128 一个 fp32 scale，
即 DeepSeek 系列的 blockwise fp8 量化。带 SwiGLU clamp（`swiglu_limit`），gfx942 / gfx950 通用。

对应 `QuantType.per_128x128`（内部 remap 成 `per_1x128`）。

---

## 1. 为什么是一个新模块

main 上原本承载 fp8 MoE 的 `kernels/moe_gemm_2stage.py` 已在
`4bbc57db1 FlyDSL: port a16wi4 to new pipeline, clean old moe_gemm_2stage (#4646)`
中删除（-3590 行），所有家族迁到了 `kernels/mixed_moe_gemm_2stage_common.py`。

但 blockwise fp8 **接不进那个新 pipeline**，有两条独立的硬约束：

1. 新 pipeline 全文只有一条 MFMA 指令 `mfma_scale_f32_16x16x128_f8f6f4`，是 **gfx950 专属**。
   在 gfx942 上 LLVM 直接 `Cannot select: intrinsic llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`。
2. 那条指令的 scale 操作数是 **e8m0 字节**（通过 `opsel_a` / `opsel_b` 从 i32 里选字节），
   只能表示 2 的幂，物理上表达不了 fp32 的 block scale。

所以 blockwise 必须用 plain `mfma_f32_16x16x32_fp8_fp8` + 显式 f32 FMA，单独成模块：
`kernels/moe_2stage_blockscale.py`。它复用 main 仍在的 `mfma_preshuffle_pipeline.py` /
`mfma_epilogues.py` 底层 helper，不依赖被删的旧 pipeline。

模块内支持两个 `in_dtype`：

| `in_dtype` | 含义 |
| --- | --- |
| `fp8_blk` | 本文主角，128×128 fp32 block scale |
| `fp8` | 同一套 pipeline 的 per-row / per-token f32 scale。**调试对照组**：scale 索引写错时它仍然正确，用来把「索引 bug」和「数据流 bug」分开 |

---

## 2. 设计

### 2.1 Scale 的三个特点

相比已有的 per-1×32 microscale（mxfp4 / mxfp8），blockwise 的差异是：

- **粒度**：K 方向 128 而不是 32
- **dtype/layout**：fp32 的 `(E, N/128, K/128)`，不是 bf16 E8M0 的 `(E, G//2, N, 2)`
- **A scale 随 M 变化** ← 唯一真正的新问题

### 2.2 A scale 为什么必须是向量

MFMA 累加器的行映射（见 `kernels/mfma_epilogues.py` 顶部注释）：

```
row = bx_m + mi*16 + lane_div_16*4 + ii
```

一个 lane 的 `f32x4` 累加器对应 **4 个连续的 M 行**、同一个 N 列。于是：

- **W scale** 只随 (N-block, K-block) 变，在 lane 内是标量 → 可以直接 broadcast
- **A scale** 随 token（即 M）变，在 lane 内是 **4 个不同的值** → 必须是 f32x4

合并后每个 K block 只做一次 FMA：

```
s_vec4[ii] = a_scale[row(mi,ii), kb] * w_scale[e, n_global/128, kb]
acc_final  = fma(s_vec4, acc_partial, acc_final)
```

实现上把 W 的标量直接乘进 A 的 4 个标量再打包成向量（4 条 `v_mul_f32`，
和先打包再做向量乘的代价一样，但只用标量算术，避免 DSL 的向量乘路径）。

### 2.3 K 循环：两级累加

`tile_k=128` 时一个 K tile 正好是一个 scale block：

```
每个 K block:
    partial = 0
    4 × MFMA(16x16x32 fp8) 链式累加进 partial     # k_unroll*2 条
    读 A scale (4 × f32) 和 W scale (1 × f32)
    acc_final = fma(a_vec4 * w_scalar, partial, acc_final)
```

相比 microscale 每 K32 做一次 FMA，这里每 K128 才做一次，**scale FMA 少 4 倍**。
`partial` 只在 `(mi, ni)` 内存活 4 条 MFMA，不需要开第二套完整累加器数组。

FMA 通过 `_make_scale_fma_pipe` 延迟 `_BLK_FMA_DEPTH` 条（默认 4，
`AITER_BLKFP8_FMA_DEPTH` 可调），避免刚发完 MFMA 就读结果而 stall。

`tile_k=256` 时一个 tile 跨 2 个 scale block，靠 `k_pos = base_k + ku*64` 自动选对，
已验证。

### 2.4 padded 行的处理

`moe_sorting` 用 `token_id == tokens` 做 padding。这些行的 A scale 直接置 0，
于是累加器天然是 0：

- stage1 的 store 本来就有谓词，无所谓
- stage2 用 **atomic 累加**，脏行会污染输出，靠这个 0 兜住

代价是 epilogue 里的 `sx` / `sw` 可以直接用常量 1.0（`_epi_sx_one` / `_epi_sw_one`）。

### 2.5 索引用移位而不是除法

`idx2crd` 返回的是 i32 坐标，和 `fx.Index` 常量做 `//` **不会** lower 成期望的整数除法
（这个坑实测过：expert 层偏移对、expert 内偏移错）。所以 `load_block_scale_f32` 里
全部先 `fx.Int32(...)` 归一，再用移位做块索引，并要求 block size 是 2 的幂。

---

## 3. 代码地图

### Kernel

`aiter/ops/flydsl/kernels/moe_2stage_blockscale.py`

| 位置 | 内容 |
| --- | --- |
| `_BLK_FMA_DEPTH`（L72） | post-MFMA scale FMA 的流水深度 |
| `compile_moe_gemm1`（L132） | stage1 入口，`swiglu_limit` 是编译期参数 |
| `is_fp8_blk`（L201 / L2311） | dtype flag |
| geometry 校验（L254 / L2353） | `num_k_blocks` / `ku_per_kblk` / `kblk_per_tile` + 对齐断言 |
| `_epi_sx_one` / `_epi_sw_one`（L466 / L2567） | epilogue 用常量 scale |
| `load_b_tile` 的 blockwise 分支（L683 / 见 stage2 对应处） | B pack + 每块一个 f32 W scale |
| A-scale row bases（L909 / L2911） | 循环外解码 token id 成行基址 |
| `_acc_scaled_f32_vec`（L1244 / L3246） | 接受 f32x4 scale 的 FMA |
| `_a_blk_scales`（L1258 / L3260） | 取 4 个 A scale，无效行置 0 |
| blockwise compute 分支（L1274 / L3276） | 两级累加主体 |
| `compile_moe_gemm2`（L2247） | stage2 入口 |

`aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`

- `load_block_scale_f32`（L863）：按 `(E, N/blk_n, K/blk_k)` fp32 取一个 scale

### Host

| 文件 | 内容 |
| --- | --- |
| `aiter/ops/flydsl/moe_kernels.py` | `get_flydsl_stage{1,2}_kernels_fp8_blk`（L582 / L617）注册 kernel 名；`compile_flydsl_moe_stage{1,2}` 的 `fp8blk` / `fp8row` 分支（L771 / L880） |
| `aiter/fused_moe.py` | `AITER_FLYDSL_BLKFP8` 门控：关掉 1-stage asm（L2342）、走 FlyDSL 的启发式 dispatch（L2466） |
| `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` | `gen_flydsl_blockscale_2stages_task`（L4236）；`run_torch_moe_stage1(skip_inter_quant=)`（L2116） |
| `aiter/aot/flydsl/moe.py` | blockwise 的 dummy scale 形状 |
| `op_tests/test_moe_2stage.py` | `_effective_swiglu_limit` 放行 blockwise |

### Kernel 命名

```
flydsl_moe1_afp8_wfp8blk_{out}_t{M}x{N}x{K}[_w{waves}]
flydsl_moe2_afp8_wfp8blk_{out}_t{M}x{N}x{K}_atomic[_w{waves}][_persist]
```

`wfp8blk` 而不是 `wfp8`，因为后者已经是 per-1×32 mxfp8 家族的名字。
对照组用 `wfp8row`（只在 compile 分派里，不进 registry）。

---

## 4. 用法

### 4.1 直接调 kernel

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
)
```

权重要先 `shuffle_weight(w.view(int8), layout=(16,16)).view(fp8)`。
scale **不需要**任何 preshuffle，fp32 原样传。

### 4.2 走 `fused_moe`

```bash
AITER_FLYDSL_BLKFP8=1   # 不设则 per_1x128 仍走原来的 asm / CK
```

开启后 `get_2stage_cfgs` 会为 `per_1x128 + fp8/fp8` 返回 FlyDSL 的 stage1/stage2 wrapper。
tile 是实测得到的默认值（见下），后续应由 tuner 产出的 CSV 覆盖。

### 4.3 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `AITER_FLYDSL_BLKFP8` | `0` | 开启 blockwise fp8 的 FlyDSL 路径 |
| `AITER_BLKFP8_FMA_DEPTH` | `4` | post-MFMA scale FMA 流水深度 |

---

## 5. 测试

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

# tuner（TUNE_ONLY 必加，见第 7 节）
TUNE_ONLY=flydslblk python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
    -i /tmp/untuned_blk.csv -o /tmp/tuned_blk.csv --last
```

判据看 `logits_diff`，不要看 `checkAllclose`：harness 用 `randn` 造权重不归一化，
输出量级上万，绝对容差 0.01 本来就过不了。

### 调试开关

出问题时用这几个把故障域切开：

| 开关 | 作用 |
| --- | --- |
| `--scales none/a/w` | 把 A 或 W 的 scale 换成 1。索引写错时仍然正确 → 说明是数据流问题 |
| `--wscale-mode e/n/k` | 让 W scale 只沿 expert / N / K 单轴变化，定位是哪一维的索引错 |
| `--in-dtype fp8` | 对照组，走同一 kernel 的 per-row scale 路径 |
| `--repeat N` | 跑 N 次比对是否逐位一致，查非确定性 |
| `--stage 1/2` | 只跑一个 stage |

---

## 6. 验证结果（gfx942 / MI308X）

| 项 | 结果 |
| --- | --- |
| stage1 | `logits_diff = 0.000e+00`（逐位精确） |
| stage2 | `3.99e-06`（差异来自 atomic 归约顺序） |
| `swiglu_limit=10.0` | 通过；端到端 `1.67e-05` |
| tile 扫描 | stage1 / stage2 各 12 组全过，含 `tile_k=256` |
| W scale 索引分解 | unit / expert-only / N-only / K-only / full 全过 |
| DSv4 7168/512/385/7 | stage1 `1.19e-07`、stage2 `8.52e-06`；idim=768 同样通过 |
| 对照组 `--in-dtype fp8` | `0.000e+00` |
| tuner 候选 | 240 个全部 0.0% 误差 |
| 默认路径（不设 env） | `-q 0/2/5` 与改动前一致 |

性能，对比现有 asm 1-stage（DSv4 7168/512/385/7）：

| tokens | asm 基线 | FlyDSL blockwise |
| --- | --- | --- |
| 128 | 1299 us | 1310 us |
| 1024 | 1534 us | 2077 us |
| 8192 | 6197 us | 7782 us |

t=8192 拆开是 stage1 4221 us + stage2 3347 us。
默认 tile 是实测扫出来的：stage1 `tm≤32 / tn=128 / tk=128 / w=2`，
stage2 `tn=256 / tk=256 / w=2`。`waves_per_eu` 很敏感——wrapper 默认的 3 在某些配置下
会掉进寄存器压力悬崖（同一配置 4220 us vs 17036 us），所以 dispatch 里显式带 `_w2`。

精度上 FlyDSL 这条路的 `logits_diff` 稳定在 3.2e-05，比 asm 基线的 8.5e-05 更好，
因为它不做中间 a2 的融合量化。

---

## 7. 已知限制

- **`inter_dim` / `model_dim` 必须是 128 的倍数**，非对齐的 tail block 未实现。
  例如 TP 切出来的 `inter_dim=192` 跑不了（sglang / vLLM 加载 blockwise fp8 权重时
  也会拒绝这种配置，实际部署会 pad 到 256）。
- **`K / tile_k` 必须是偶数**：ping-pong 的 tail 固定消费两个 tile，奇数会漏算。
  编译期有断言。
- **单个权重张量不能超过 4 GiB**：buffer resource 的 num_records 是 32 位。
  `E=385 + inter_dim=1536` 单卡放不下（8.5 GB），把 E 降到 192 可跑。
  这是既有限制，对所有 `in_dtype` 一样。
- **中间 a2 的量化没有融合**：stage1 输出 bf16，由 host 侧现有的 `per_1x128` quant op
  处理。融进 stage1 epilogue 是后续优化项。
- **性能还落后 asm 基线约 25%**（大 token）。tile 通路已经接好，缺的是把 tuner
  产出的 CSV 落进 `configs/`。
- `gemm_moe_tune.py` 在 DSv4 shape 上有一个 HIP illegal memory access，来自
  asm / cktile / opus 那几组候选，**与本实现无关**（用 `TUNE_ONLY=flydslblk`
  隔离验证过）。所以跑 tuner 时建议带 `TUNE_ONLY=flydslblk`。

---

## 8. 排查记录

实现过程中踩到、值得记下来的坑：

1. **i32 坐标与 Index 常量混用做除法会算错**。`idx2crd` 给的是 i32，
   `n_global // fx.Index(128)` 不会 lower 成整数除法。现在统一 `fx.Int32` + 移位。
2. **kernel 参数 `n_in` 传错会静默写坏别的显存**。stage1 的 `n_in` 是 `inter_dim`
   而不是 `2*inter_dim`（gate/up 是两个独立的 B tile）。传大一倍会让 grid.x 翻倍、
   输出 buffer resource 也大一倍，越界写出去，表现是**随机**的错误结果——
   查了很久才定位。
3. **stage2 用 atomic，benchmark 时必须先快照再计时**，否则重复 launch 会累加。
4. **flydsl 的编译缓存 key 不包含源码**。改了 kernel 但 tile / dtype 没变时，
   可能拿到旧二进制。结果诡异时先 `rm -rf /root/.flydsl/cache`。
5. **tuner 的 stage1 参考对 `per_1x128` 会再量化一次**（因为 asm/CK 的 stage1 融合了
   a2 quant），FlyDSL 输出 bf16 所以对不上，会报 99% 误差。加了 `skip_inter_quant`。
6. **`e8m0_shuffle` 会把 fp32 scale 按 uint8 重解释**。tuner 里 `per_1x128` 的
   `w*_scale_aiter` 兜底分支原本会走它，已加 `q_type` 判断绕开。

---

## 9. 与旧分支的关系

同一套设计最早实现在 `/sgl-workspace/aiter`（分支 `port-flydsl-a16w4-a8w4-mxfp4`）的
`kernels/moe_gemm_2stage.py` 上，作为该文件的一个新 `in_dtype`。那个分支落后 main
422 个提交，且 main 已经删掉了那个文件，所以 main 上改成了独立模块。两边的
kernel 主体逻辑一致，差异只在：

- `buffer_ops` / `vector` 在 main 上来自 `aiter.ops.flydsl.kernels`，不是 `flydsl.expr`
- flydsl 0.3.0 删了 `T.f8`，改用 `kernels_common.default_f8_type()`
- main 的 `load_b_pack_k32` 没有 `raw_packed` 参数
