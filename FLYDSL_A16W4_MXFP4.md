# FlyDSL a16w4 (mxfp4) MoE Kernel — 实现、测试与性能对比

## 1. 背景与目标

DeepSeek-V4 的 MoE 在 **MI308X (gfx942 / CDNA3)** 上使用 **a16w4** 精度：
- 激活 (activation)：**bf16**
- 权重 (weight)：**mxfp4** = e2m1 4-bit 码本 + E8M0 per-32 block scale

CDNA3 没有原生 fp4/scaled-MFMA，所以 a16w4 的机制必然是「kernel 内把 mxfp4 权重反量化成 bf16，再走 bf16 MFMA」。

**目标**：用 FlyDSL 实现这条两段式 MoE GEMM（stage1 gate+up、stage2 down-proj），在 kernel 内完成 mxfp4→bf16 反量化，并与 aiter 的 triton a16w4 (`moe_gemm_a16w4`) 对比精度和性能。

**关键设计**：复用 FlyDSL 已有的 `int4_bf16` (W4A16) 两段式 kernel 骨架（pipeline / MFMA / preshuffle / groupwise-scale 全部复用），**只新增 mxfp4 的反量化路径**（e2m1 码本 → bf16，替代对称 int4 → bf16）。

## 2. 运行环境（重要）

| 项 | 值 |
|----|----|
| Host | `hjbog-srdc-47`（不需要 `.mnb.dcgpu` 后缀） |
| Container | `rujia_dsv4_flash_atom_hjbog_47` |
| 挂载 | 容器 `/data` = 宿主 `/data/rujiacai`；容器 `/data/aiter` = 宿主 `/data/rujiacai/aiter`（editable 安装） |
| GPU | MI308X gfx942 (CDNA3) |
| **解释器** | `PYTHONPATH=/data/aiter /opt/venv/bin/python` |

**解释器说明**：只有这个组合同时具备 torch + 你的 editable `/data/aiter` 代码 + flydsl。
- `.aiter` venv 有 flydsl 但**没有 torch**。
- `/opt/venv` 默认把 aiter 指向 `/app/aiter-test`（镜像内旧副本），必须用 `PYTHONPATH=/data/aiter` 覆盖。

**缓存**：改动 kernel `.py` 后必须清 FlyDSL 缓存，否则跑的是旧构建：
```bash
rm -rf /root/.flydsl/cache /data/aiter/aiter/jit/flydsl_cache/*
```

## 3. 改动的文件

全部在 `/data/aiter/aiter/ops/flydsl/`：

### 3.1 `kernels/mfma_preshuffle_pipeline.py`（核心反量化）
新增：
- `_e2m1_byte_to_bf16_bits(code_i32, arith)`：单个 e2m1 4-bit 码 → bf16 位构造（纯位操作 + select 处理 subnormal）。
- `_e2m1x4_in_i32_to_bf16x4_i64(...)`：4 个 e2m1 码 → 4 个 bf16 打包成 i64，含可选 f32 scale。
- `_unpack_mxfp4_nibble_pair(packed32)`：拆 packed fp4x2（只 mask，不像 int4 那样符号扩展）。
- `unpack_b_w4a16_mxfp4(...)` / `unpack_b_w4a16_mxfp4_groupwise(...)`：mxfp4 反量化主入口，对齐 `unpack_b_w4a16` 签名。
- `e8m0_to_f32_scale(...)`：E8M0(uint8) → f32 scale 解码。

### 3.2 `kernels/moe_gemm_2stage.py`（compile 层）
- `compile_moe_gemm1` / `compile_moe_gemm2`：`in_dtype` 新增 `"mxfp4_bf16"`。
- 新增 `is_mxfp4_bf16` 标志，在反量化调用点分流到 `unpack_b_w4a16_mxfp4`；`is_int4_bf16` 仍为 True 以复用所有共享路径。
- mxfp4 强制 `use_gfx950_cvt=False`（gfx950 cvt 是 int4 专用）。

### 3.3 `moe_kernels.py`（host dispatch + 注册）
- `get_flydsl_stage1/2_kernels_mxfp4_bf16`：注册 mxfp4 kernel 变体（kernel 名带 `wmxfp4`）。
- `compile_flydsl_moe_stage1/2`：`b_dtype="mxfp4"` 分支，调 `compile_moe_gemm1/2(in_dtype="mxfp4_bf16", ...)`。

### 3.4 `test_flydsl_moe_a16w4.py`（新单元测试）
仿 `test_flydsl_moe_a4w4.py`，bf16 激活 + mxfp4 权重，含精度对比 + triton 性能对比。

## 4. 关键技术点

### 4.1 e2m1 → bf16 位构造（已验证与码本精确一致）
4-bit e2m1 码格式 `[sign(1) | exp(2) | mant(1)]`，码本 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}：
- normal (exp≥1)：`bf16 = (sign<<15) | ((exp-1+127)<<7) | (mant<<6)`
- subnormal (exp==0)：`bf16 = (sign<<15) | (mant * 0x3F00)`（即 ±0.5 或 ±0）

### 4.2 E8M0 scale
uint8 指数 `u` → f32 值 = `2^(u-127)`，位模式 `u<<23`。与 `fp4_utils.e8m0_to_f32` 精确一致。

### 4.3 权重/scale 的 host 端准备
mxfp4 复用 int4 的字节布局（2 码/字节）：
```python
# 权重：提取 e2m1 码 → i8，再走 int4 的 shuffle+pack
codes_i8 = unpack fp4x2 -> int8 (0..15, low nibble first)
w_shuf = pack_int8_to_packed_int4(shuffle_weight(codes_i8.view(i8), (16,16)))
# scale：E8M0 → bf16，布局 (E, K/32, N)，再 shuffle
scale_bf16 = (2^(u-127)).permute(0,2,1).to(bf16)
scale_shuf = shuffle_scale_for_int4(scale_bf16, group_size=32)
```

## 5. 踩过的坑

1. **算术 vs 逻辑位移导致 NaN**：反量化 f32→bf16 截断时，若把 bitcast 结果包进 `fx.Int32` 再 `>>`，会 emit 算术移位（shrsi），负 bf16 的符号位被扩展进高 16 位 → NaN。**修复**：保留 raw bitcast 值（与 `_i8x4_in_i32_to_bf16x4_i64` 一致），`>>` 才是逻辑移位（shrui）。

2. **stale kernel cache**：改完 kernel 后旧缓存仍被使用，NaN 修复看不到效果。**必须清缓存**。

3. **单 K-tile 边界是伪命题**：一度以为 stage1 需要 scale×0.5，其实那是在 `model_dim==tile_k==256`（单 K-tile）退化情形下"歪打正着"匹配了一个**本身就错的** kernel 输出，反而破坏了正确的 K≥512。**int4 路径在 K=256 也有相同的 ~4× 偏差**——这是共享 pipeline 的单 K-tile 边界 artifact，与 mxfp4 无关。dsv4 实际 model_dim=7168，永远是多 K-tile，不受影响。**结论：不要加任何 scale fudge，测试用 K≥512**。

## 6. 精度验证结果

参考基线：手写的 GEMM+SwiGLU（逐 (token,slot) 显式计算，无歧义），以及 `torch_moe_stage1/2(quant_type=No)` + 手动 e2m1×E8M0 反量化（两者一致）。

**K = model_dim ≥ 512（多 K-tile，dsv4 真实场景）：全部完美**

| 配置 | stage | median_ratio | corr | 结果 |
|------|-------|-------------|------|------|
| (512,512) E8 tk2 | stage1 | 1.000 | 1.0000 | PASS |
| (512,512) E8 tk2 | stage2 | 1.000 | 1.0000 | PASS |
| (512,512) E8 tk2 | e2e | 1.000 | 1.0000 | PASS |
| (2048,768) E32 tk4 | e2e | 1.000 | 1.0000 | PASS |

token=16/128/1024 均通过。

## 7. 性能对比：FlyDSL vs triton moe_gemm_a16w4

triton a16w4 **可以在 gfx942 上跑**（`tl.dot_scaled` 软件仿真到 bf16；`arch_info.is_fp4_avail()=False` 只是 pytest skip 的门槛，不影响 kernel 实际运行）。

对比内容：两者都做 gate+up GEMM + SwiGLU，相同 M/N/K/E/topk、相同 mxfp4 权重。用 `torch.cuda.Event` 计时（50 iter，5 warmup）。

**小尺寸 (model_dim=512, inter_dim=512, E=8, topk=2)：FlyDSL 大幅领先**

| token | FlyDSL (μs) | triton (μs) | 加速比 (triton/flydsl) |
|-------|------------|-------------|----------------------|
| 16 | 99.3 | 264.9 | **2.67×** |
| 128 | 93.2 | 253.9 | **2.73×** |
| 1024 | 120.8 | 258.9 | **2.14×** |

**大尺寸 (model_dim=2048, inter_dim=768, E=32, topk=4)：triton 领先**

| token | FlyDSL (μs) | triton (μs) | 加速比 (triton/flydsl) |
|-------|------------|-------------|----------------------|
| 128 | 351.4 | 265.7 | 0.76× |
| 1024 | 1212.5 | 784.0 | 0.65× |

**分析**：
- 小尺寸下 FlyDSL 快 2-2.7×（triton matmul_ogs 的 routing/gather/scatter 开销占比高）。
- 大尺寸下 triton 快 ~1.3-1.5×，因为 FlyDSL 当前用**固定未 tuning 的 tile**（tile_n=128, tile_k=256），而 triton matmul_ogs 有 autotuning。
- **优化方向**：给 FlyDSL a16w4 加 tile 配置的 autotuning（tile_m/n/k、k_batch split-K、waves_per_eu、xcd_swizzle 等，这些骨架已存在），大尺寸性能应能追平或超过 triton。

## 8. 如何单独测试

```bash
# 1. 进容器
ssh rujiacai@hjbog-srdc-47
docker exec -it rujia_dsv4_flash_atom_hjbog_47 bash
cd /data/aiter
export PYTHONPATH=/data/aiter
PY=/opt/venv/bin/python

# 2. 精度测试（默认 model-dim=512, inter-dim=512）
$PY aiter/ops/flydsl/test_flydsl_moe_a16w4.py --stage all -t 16 -t 128 -t 1024
$PY aiter/ops/flydsl/test_flydsl_moe_a16w4.py --stage stage1     # 单独 stage1
$PY aiter/ops/flydsl/test_flydsl_moe_a16w4.py --stage stage2     # 单独 stage2
$PY aiter/ops/flydsl/test_flydsl_moe_a16w4.py --stage e2e        # 端到端

# 自定义尺寸（dsv4-like，注意 model-dim 要 >= 512）
$PY aiter/ops/flydsl/test_flydsl_moe_a16w4.py --stage e2e \
    --model-dim 2048 --inter-dim 768 -E 32 --topk 4 -t 128

# 3. 与 triton 性能对比
$PY aiter/ops/flydsl/test_flydsl_moe_a16w4.py --compare-triton -t 16 -t 128 -t 1024

# 4. FlyDSL 自身 kernel 计时
$PY aiter/ops/flydsl/test_flydsl_moe_a16w4.py --bench -t 128
```

**命令行参数**：
- `--stage {stage1,stage2,e2e,all}`：测哪个阶段
- `-t/--tokens N`（可多次）：token 数，默认 [16,128,1024]
- `--model-dim / --inter-dim`：维度（**model-dim ≥ 512**）
- `-E/--experts`、`--topk`：专家数、topk
- `--compare-triton`：与 triton a16w4 性能对比
- `--bench`：只测 FlyDSL stage1 时延

## 9. 已知限制与后续

- **单 K-tile (model_dim==256) 不支持**：共享 int4 pipeline 的边界 artifact（int4 同样问题），dsv4 用不到。
- **大尺寸性能待优化**：加 tile autotuning 可提升。
- **triton 对比是「gate+up+SwiGLU 单段」对「FlyDSL stage1」**：triton `moe_gemm_a16w4` 是 matmul_ogs 融合单 GEMM，语义上对应 FlyDSL 的 stage1；FlyDSL 的 stage2 (down-proj) 没有直接的 triton 单算子对照。
- `test_flydsl_moe_a4w4.py` 在 gfx942 上**无法运行**（需要 CDNA4 的 `mfma.scale.f32.16x16x128.f8f6f4`），所以不能作为对照。
