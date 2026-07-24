# FlyDSL a8w4 (fp8 激活 × mxfp4 权重) MoE Kernel — 实现总结（结合代码）

> 目标硬件：**MI300/MI308X (gfx942 / CDNA3)**。解释器：`PYTHONPATH=/data/aiter /opt/venv/bin/python`
> （`/opt/venv` 默认指向镜像内旧副本 `/app/aiter-test`，必须用 `PYTHONPATH=/data/aiter` 覆盖）。
> 本文总结 a8w4 的完整实现过程；早期数值可行性/日志见 `docs/flydsl_a8w4_fp8_*_cn.md`。

---

## 1. a8w4 是什么，为什么要做

DeepSeek-V4 MoE 的一种低比特精度：

- **激活**：per-token **fp8**（e4m3fnuz，gfx942 原生）
- **权重**：**mxfp4** = e2m1 4-bit 码本 `{0,±.5,±1,±1.5,±2,±3,±4,±6}` + E8M0 per-32-block scale

它在三条低比特路线里的定位：

| 方案 | 激活 | 权重存储 | MFMA | 权重 HBM | 说明 |
|---|---|---|---|---|---|
| a16w4 | bf16 | 4-bit e2m1 | **bf16** K16 | 0.5 B/elem | 核内 e2m1→bf16 反量化 |
| a8w8 (mxfp8) | fp8 | **fp8 (8-bit)** | fp8 K32 | 1.0 B/elem | 权重直读，无 unpack |
| **a8w4** | **fp8** | **4-bit e2m1** | **fp8 K32** | **0.5 B/elem** | 核内 e2m1→fp8 unpack + 原生 fp8 MFMA |

**a8w4 的意义**：同时拿到 **fp8 MFMA 的算力**（~2× bf16，胜过 a16w4）和 **4-bit 权重的显存/带宽**（是 a8w8 的一半，胜过 a8w8）。代价是核内要把 e2m1 unpack 成 fp8。

CDNA3 没有原生 scaled-MFMA，所以 per-32 E8M0 scale 必须在 f32 累加时**后乘**（不能靠硬件 scale-MMA）。

---

## 2. 实现演进：Phase-0 → Phase-1

a8w4 分两步落地，两条 host prep 都保留在 `moe_kernels.py`：

### Phase-0（mxfp8 recast，先证明数值 + 复用 mxfp8 内核）
把 mxfp4 权重在 **host** 端 recast 成 fp8（8-bit）存储，per-pair E8M0 base 折进 fp8 指数（幂次移位，无损），kernel 直读 fp8、无 unpack。缺点：**权重仍是 8-bit，没有 a8w4 的显存优势**（等于 a8w8）。

```77:100:/data/aiter/aiter/ops/flydsl/moe_kernels.py
def prep_a8w4_weight_scale(wq_fp4x2, e8m0_scale, E, N, K):
    """mxfp4 weight -> fp8 (per-group-pair base fold) + per-pair-equal E8M0 scale.
    ...
    """
    # ...
    up = u.reshape(E, N, G // 2, 2)
    base = up.amax(dim=-1, keepdim=True)                          # per-pair common exponent
    ratio_exp = (up - base).reshape(E, N, G)                      # <= 0, integer
    wf = e2m1 * torch.exp2(ratio_exp.repeat_interleave(32, dim=2))  # exact power-of-2 shift
    w_fp8_shuf = shuffle_weight(wf.to(FP8).view(torch.int8), layout=(16, 16)).view(FP8)
```

### Phase-1（真 4-bit 存储 + 核内 unpack）—— 真正的 a8w4
权重**保持 4-bit e2m1 打包存储**（= a16w4 的字节布局，HBM 减半），在 **kernel 内** e2m1→fp8 unpack + per-pair ratio-fold。host prep 直接复用 a16w4 的：

```103:111:/data/aiter/aiter/ops/flydsl/moe_kernels.py
def prep_a8w4_w4(wq_fp4x2, e8m0_scale, N, K):
    """a8w4 Phase-1: keep mxfp4 weight PACKED 4-bit (0.5B) + raw per-32 E8M0 bf16 scale.
    ...
    """
    return prep_a16w4_weight(wq_fp4x2, N, K), prep_a16w4_scale(e8m0_scale, N, K)
```

本文后续都以 Phase-1 为主线（`in_dtype="mxfp4_fp8"`）。

---

## 3. MFMA 布局基础：K64 micro-step 与 per-32 scale

内核基座是 `moe_gemm_2stage.py`（与 a16w4 同一套 pipeline）。用的是 `mfma_f32_16x16x32_fp8_fp8`（K=32）。关键结构：

- **一个 K64 micro-step = 2 个 K32 MFMA operand**（记作 `r0`, `r1`）。
- 每个 lane 持有一个 K32 operand 的 8 个 fp8 元素；`lane_div_16 ∈ {0,1,2,3}` 是 K-octet 号，octet j → operand-K `[j*8 : j*8+8]`。
- **权重的 per-32 E8M0 scale**：每 32 个 K 一个 scale。一个 dword 打包 2 个相邻 block 的 bf16 scale（block `2ku` 和 `2ku+1`）。

**关键事实（用 `aiter_logs/derive_mapping.py` 打 marker 实测得到）**：传统 preshuffle 布局（`shuffle_weight(16,16)`）下，**一个 K32 operand 的 32 个元素横跨 2 个 scale block**：
- `lane_div_16 ∈ {0,1}`（octet 0,1，前 16 个 K）→ block A
- `lane_div_16 ∈ {2,3}`（octet 2,3，后 16 个 K）→ block B

这就是 §5 需要 fold、§6 要做 aligned 的根本原因。

---

## 4. 核心机制一：e2m1 → fp8 unpack（三条实现路径）

三条路径可通过环境变量切换，默认 **perm-LUT**：

| 路径 | 环境变量 | 做法 | 相对开销 |
|---|---|---|---|
| f32 construct | `AITER_A8W4_PERMLUT=0` | e2m1→bf16 位构造→f32→(×ratio)→`cvt_pk_fp8` | 基线 |
| **perm-LUT（默认）** | `AITER_A8W4_PERMLUT=1` | 3× `v_perm_b32` 字节 LUT 查 e2m1→fp8 | **unpack int-op ~8× 更少** |
| bitfold | `AITER_A8W4_BITFOLD=1` | 纯整数位构造 e2m1→fp8，ratio 折进指数 | 无 f32 往返 |

**perm-LUT 核心**：把 16 个 e2m1 码 → fp8 字节的映射做成 4 个常量 dword 的 LUT，用 3 条 `v_perm_b32` 完成（低 8 码一组、高 8 码一组，再按 bit3 blend），替代每 nibble ~15 条整数指令：

```586:611:/data/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py
def _e2m1x4_in_i32_to_fp8x4_i32_permlut(val_i32, arith, vector, ratios=None):
    # ...
    sel = v & fx.Int32(0x07070707)
    res_lo = fx.Int32(rocdl.perm_b32(lut_lo_hi, lut_lo_lo, sel))  # codes 0..7
    res_hi = fx.Int32(rocdl.perm_b32(lut_hi_hi, lut_hi_lo, sel))  # codes 8..15
    # blend by bit3 of each code: output byte i from res_lo (sel i) or res_hi (i+4).
    blend_sel = fx.Int32(0x03020100) | ((v >> fx.Int32(1)) & fx.Int32(0x04040404))
    fp8x4 = rocdl.perm_b32(res_hi, res_lo, blend_sel)
    if ratios is None:
        return fp8x4
    # scheme B: fp8 -> f32 -> *ratio -> fp8 (reuse the reliable f32 fold).
```

> **scheme B**：当需要 fold 时（§5），perm-LUT 先出未缩放 fp8，再 `cvt_pk_f32_fp8`→×ratio→`cvt_pk_fp8_f32`。即"可靠的 f32 fold 前端换成 LUT"。这一步是 §6 aligned 想去掉的额外 f32 往返。

perm-LUT 相对 legacy（f32 位构造）实测 **stage1 1.38× / e2e 1.43×，cos 无损**。

---

## 5. 核心机制二：per-pair ratio-fold（为什么需要）

因为一个 K32 operand 横跨 2 个 scale block（§3），而 MFMA 把 `r0`/`r1` 的贡献累加进**同一个 accumulator**，**post-MFMA 只能施加一个 scale**。fold 的做法：

1. 取 pair 内两个 block 的公共 base = `max(scA, scB)`；
2. 把 `2^(exp_g - base)`（ratio，≤1 的幂次）**折进权重**：octet{0,1} 用 ratioA、octet{2,3} 用 ratioB（按 `is_B = lane_div_16>=2` 选）；
3. post-MFMA 只施加一个 `2^base`。

```110:126:/data/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    else:
        base = arith.ArithValue(arith.maximumf(_uw(scA), _uw(scB)))
        _base_raw = _uw(base)
        ratioA = arith.divf(_uw(scA), _base_raw)
        ratioB = arith.divf(_uw(scB), _base_raw)
        ratio = arith.ArithValue(arith.select(is_B, ratioB, ratioA))
        rr = [ratio, ratio, ratio, ratio]
        # ...
        b0 = _unpack(r0, arith, vector, ratios_even=rr, ratios_odd=rr)
        b1 = _unpack(r1, arith, vector, ratios_even=rr, ratios_odd=rr)
    sc_out = _bb | (_bb << fx.Int32(16))   # 2^base packed into both bf16 halves
    return b0, b1, sc_out
```

fold 是**正确但有开销**的：每 operand 多一次 `max/2×div/select` + perm-LUT scheme B 的 f32 往返（`cvt_pk_f32_fp8`×2 + 4×mulf + `cvt_pk_fp8_f32`×2）。

**scale 施加点（mxfp8 compute path，a8w4 复用）**：每个 K32 operand 各自 MFMA 到 zero-acc，再用**各自的 scale** FMA 累加——这天然支持 per-operand scale：

```1448:1461:/data/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                                    scg0 = extract_bf16_scale(arith, scg, 0)
                                    scg1 = extract_bf16_scale(arith, scg, 1)
                                    # ...
                                    pg0 = mfma_fn(
                                        mfma_res_ty, [a0, bg0, zero_f32_acc, 0, 0, 0]
                                    )
                                    gate_list[acc_idx] = _acc_scaled_f32(
                                        gate_list[acc_idx], pg0, scg0
                                    )
```

---

## 6. 核心机制三：A+B aligned（消除 fold，本次核心优化）

**思路**：既然 mxfp8 compute 已经是 per-operand 施加 scale，如果让**一个 K32 operand 恰好 = 一个 32-K scale block**（不再 straddle），就能：
- 权重 unpack 用**纯 perm-LUT，无 fold**（去掉 §5 的 max/div/select + f32 往返）；
- `r0` 用 blockA 的 scale、`r1` 用 blockB 的 scale，直接后乘。

要同时满足 A（激活）和 B（权重）的 operand-K → 原始 K 映射一致，必须 **A+B 协同改**（只改一边 cos=0）。

### B 侧：`shuffle_weight_NK(16,32)` 让 operand 对齐 block

```218:236:/data/aiter/aiter/ops/shuffle.py
def shuffle_weight_NK(
    x: torch.Tensor, inst_N: int, inst_K: int, use_int4=False
) -> torch.Tensor:
    kPerLane = inst_K // (64 // inst_N)
    # ...
    x_ = x_.view(
        -1, x.shape[-2] // inst_N, inst_N, x.shape[-1] // inst_K, 64 // inst_N, kPerLane
    )
    x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
    return x_.view(*x.shape)
```

`inst_K=32` 时 `kPerLane=8`，一个 operand 的 `klane(4)×kPerLane(8)=32` K 全落在一个 block 内。对应的 B layout / 单 operand 加载：

```657:665:/data/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py
def make_aligned_b_layout(arith, *, c_n: ir.Value, c_k: ir.Value):
    """B layout for a8w4 ALIGNED: one K32 fp8-MFMA operand == one per-32 block.
    ...
    """
```

host prep（`shuffle_weight_NK(16,32)` + 复用 a16w4 scale 布局）：

```114:136:/data/aiter/aiter/ops/flydsl/moe_kernels.py
def prep_a8w4_w4_aligned(wq_fp4x2, e8m0_scale, N, K):
    """a8w4 Phase-1 ALIGNED: K32 MFMA operand == one per-32 scale block, NO fold.
    ...
    """
    shuf = pack_int8_to_packed_int4(shuffle_weight_NK(codes.view(dtypes.i8), 16, 32))
    # ...
    scale = prep_a16w4_scale(e8m0_scale, N, K)
    return w4, scale
```

### A 侧：aligned activation loader（一次 16B → 两次 8B）

当前 activation 加载（`lds_load_packs_k64`）里，一个 lane 读 16 连续字节 → `a0`/`a1`，其 operand-K 也 straddle（octet{0,1}=blockA、{2,3}=blockB），正好匹配 straddle 的 B。要 aligned，改成**两次 8B load**，让 `a0`=blockA(K[ku*64+octet*8:+8])、`a1`=blockB(K[ku*64+32+octet*8:+8])：

```1142:1150:/data/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                def lds_load_packs_k64_aligned(curr_row_a_lds, ku, lds_base):
                    # a8w4 ALIGNED activation: each K32 operand == ONE 32-K block so
                    # it pairs with shuffle_weight_NK(16,32) (no in-kernel fold). With
                    # octet=lane_div_16 the operands map to:
                    #   a0 = block(2*ku)   -> K[ku*64 + octet*8 : +8]
                    #   a1 = block(2*ku+1) -> K[ku*64 + 32 + octet*8 : +8]
```

这个 K 序恰好等于 `shuffle_weight_NK(16,32)` 的权重内部序（klane=octet，block-K=octet*8+kp），**A、B 逐元素配对 → cos=1**。

### aligned 分支（无 fold）

```977:993:/data/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
                                    r0 = load_b_operand_aligned(
                                        buffer_ops, arith, vector, b_rsrc=w_rsrc,
                                        layout_b=layout_b, k0=_k0b + fx.Index(2 * ku),
                                        # ...
                                    )
                                    b0 = unpack_b_w4a16_mxfp4_to_fp8_permlut(r0, arith, vector)
                                    b1 = unpack_b_w4a16_mxfp4_to_fp8_permlut(r1, arith, vector)
                                    # raw per-32 scale pair (no fold), applied post-MFMA
```

**门控**：`AITER_A8W4_ALIGNED=1`。为避免 FlyDSL 磁盘缓存串用（cache key 不含该 env），该 flag 被提升为 `compile_moe_gemm1/2` 外层作用域的**闭包标量**（`moe_gemm_2stage.py:428`），从而进入 cache key（见 §10）。

---

## 7. Host 端权重/scale 准备（三个 prep）

| 函数 | 输出权重 | 用途 | env |
|---|---|---|---|
| `prep_a8w4_weight_scale` | fp8 (8-bit) | Phase-0 mxfp8 recast | `AITER_FLYDSL_A8W4` |
| `prep_a8w4_w4` | 4-bit e2m1 (`shuffle_weight(16,16)`) | Phase-1 fold | `AITER_FLYDSL_A8W4_W4` |
| `prep_a8w4_w4_aligned` | 4-bit e2m1 (`shuffle_weight_NK(16,32)`) | Phase-1 **aligned** | `AITER_FLYDSL_A8W4_W4` + `AITER_A8W4_ALIGNED=1` |

三者都以「mxfp4 量化后的权重(fp4x2) + E8M0 scale」为输入。**aligned 的权重必须用 `prep_a8w4_w4_aligned`**——喂 fold 布局给 aligned 内核会算错。

> **注意：这三个都是 host 端的一次性权重预处理，不在 kernel/dispatch 调用链内。**
> GPU kernel 不调它们；`fused_moe` 运行时也**不**调它们（`fused_moe.py:2181` 只在注释里提了一句 `# ...prepared by moe_kernels.prep_a8w4_w4`）。这是 aiter 的标准约定：**权重离线预 shuffle 一次**，把 shuffle 好的权重 + scale 传给 `fused_moe`，运行时不再做 prep。
> 实际调用者都是 host 侧脚本：`aiter_logs/test_a8w4_phase1.py`、`aiter_logs/test_fused_moe_flydsl.py`、`aiter_logs/prof_a8_trigger.py`、benchmark harness `run_moe_bench.py::_prepare_flydsl_weights`。生产中则由离线权重转换流程调用。

---

## 8. Dispatch / fused_moe 集成

`fused_moe` 通过 env flag + `q_dtype_w=fp4x2` 分派到 FlyDSL a8w4：

- `AITER_FLYDSL_A8W4_W4=1` + `per_1x32` + fp4x2 权重 → `flydsl_kernel_name(1,"fp8","mxfp4",...)` → `compile_moe_gemm1(in_dtype="mxfp4_fp8")`。
- aligned 分支由 `AITER_A8W4_ALIGNED` 在 `compile` 内部自动 gate，**dispatch 无需改动**；调用方只需用 `prep_a8w4_w4_aligned` 准备权重。
- 激活由 `fused_moe` 内部做 per-token fp8 量化（调用方传 bf16 hidden + 权重 scale）。

---

## 9. 关键代码地图

| 文件 | 符号 | 作用 |
|---|---|---|
| `moe_kernels.py` | `prep_a8w4_weight_scale` / `prep_a8w4_w4` / `prep_a8w4_w4_aligned` | 三条 host prep（§7）|
| `mfma_preshuffle_pipeline.py` | `_e2m1x4_in_i32_to_fp8x4_i32` (546) | f32 位构造 unpack |
| | `_e2m1x4_in_i32_to_fp8x4_i32_permlut` (586) | **perm-LUT unpack（默认）** |
| | `_e2m1_code_to_fp8_byte_fold` (767) | bitfold 纯整数 unpack |
| | `make_aligned_b_layout` (657) / `load_b_operand_aligned` (688) | aligned B 布局/加载 |
| | `shuffle_weight_NK`（`shuffle.py:218`）| aligned 权重 preshuffle |
| `moe_gemm_2stage.py` | `_mxfp4_fp8_fold_operands` (75) | fold（unpack + ratio-fold + sc_out）|
| | `lds_load_packs_k64_aligned` (1142/3276) | **aligned 激活 loader（2×8B）** |
| | per-operand scale compute (1448) | `mfma → zero-acc → ×scale FMA` |
| | aligned raw-load 分支 (977/3118) | 无 fold 路径 |

---

## 10. 踩过的坑

1. **A+B 必须协同**：只改 B（`shuffle_weight_NK`）不改 A → operand-K 配对错位 → **cos=0**。加 aligned 激活 loader 后 cos=1。用 `aiter_logs/derive_mapping.py`（marker 打标）拿到 operand→block 的 ground truth 才定位清楚。
2. **FlyDSL 缓存 cache-key 串用**（隐藏正确性炸弹）：cache key 的 env 白名单不含 `AITER_A8W4_ALIGNED`，同 shape 下先编 fold 再切 aligned 会**静默加载 fold 二进制**（配 aligned prep → 结果错）。修复：把 `_a8w4_aligned` 提到 `compile_*` 外层作用域成为**闭包标量**，自动进 cache key。基准测试时 fold/aligned 也必须**分进程**（in-process `lru_cache` 按参数缓存、env 在核内读）。
3. **MLIR 类型一致**：stride/坐标计算混用 Python `int` 与 `fx.Index`/`ir.Value` 会 IR type mismatch；统一用 `fx.Index`。
4. **perm-LUT 用 scheme B**（fp8→f32→×ratio→fp8）而非直接在 LUT 里折 scale：复用已验证可靠的 f32 fold，只换 code→f32 前端。

---

## 11. 性能结果

### perm-LUT vs legacy（f32 位构造），a8w4 fold 路径
stage1 **1.38×**、e2e **1.43×**，cos 无损。

### aligned vs fold（`model_dim=4096, inter_dim=512, E=32, topk=6`，清缓存净测）

| token | fold(perm-LUT) | **aligned** | 加速 |
|---|---|---|---|
| stage1 256 | 397 us | **211 us** | **1.89×** |
| stage1 4096 | 3744 us | **3172 us** | **1.18×** |
| e2e 256 | 618 us | **330 us** | **1.87×** |
| e2e 4096 | 6032 us | **5020 us** | **1.20×** |

小 token（unpack 占比高）收益最大，大 token 收敛到 ~1.2×。正确性：uniform/real/diff/stage2 cos=1.0，e2e 0.9997（fold、aligned 一致）。

### dsv4-pro tp8 no-fuse（`7168/384/E384/topk6`）e2e，4 方案全景

| token | fp8 (a8w8) | a8w4 aligned | a8w4 fold | a16w4 |
|---|---|---|---|---|
| 128 | **1024** | 1850 | 3883 | 7323 |
| 4096 | **2436** | 6239 | 12053 | 14336 |
| 16384 | **7277** | 30328 | 36280 | 40084 |

- **aligned 一致优于 fold**（128: 2.10×，4096: 1.93×，16384: 1.20×）——本次优化在真实大模型 shape 再次验证。
- **但本 shape 原生 fp8(a8w8) 最快**，FlyDSL 4-bit 路径慢 1.8×→4.2×。原因：dsv4-pro 是大 `model_dim(7168)`+ 小 `inter_dim(384)`+ 384 experts 的刁钻 shape，FlyDSL 用**未调优的自适应 tile**（MFU 仅 10-20%），而 fp8 是高度调优的 CK/ASM。**优化方向**：为 dsv4-pro shape 扫 tile 配置（tile_m/n/k、k_batch）+ rocprof 定位瓶颈。

---

## 12. 如何测试

单元/正确性 + 性能（`aiter_logs/test_a8w4_phase1.py`）：

```bash
cd /data/aiter/aiter_logs
# 正确性（fold 默认；加 AITER_A8W4_ALIGNED=1 测 aligned）
PYTHONPATH=/data/aiter AITER_A8W4_ALIGNED=1 python test_a8w4_phase1.py            # stage1 uniform
PYTHONPATH=/data/aiter AITER_A8W4_ALIGNED=1 python test_a8w4_phase1.py --real     # 真 per-32 scale
PYTHONPATH=/data/aiter AITER_A8W4_ALIGNED=1 python test_a8w4_phase1.py --diff     # fold 压力(ratio 1/16)
PYTHONPATH=/data/aiter AITER_A8W4_ALIGNED=1 python test_a8w4_phase1.py --stage2 --real
PYTHONPATH=/data/aiter AITER_A8W4_ALIGNED=1 python test_a8w4_phase1.py --e2e
PYTHONPATH=/data/aiter AITER_A8W4_ALIGNED=1 python test_a8w4_phase1.py --fused    # 走 fused_moe

# 性能（stage1 sweep / fused e2e sweep）；fold vs aligned 分进程 + 清缓存
rm -rf /root/.flydsl/cache/*moe_gemm*
PYTHONPATH=/data/aiter AITER_A8W4_ALIGNED=1 python test_a8w4_phase1.py --perf 256 4096 16384
PYTHONPATH=/data/aiter AITER_A8W4_ALIGNED=1 python test_a8w4_phase1.py --fused-perf 256 4096 16384
```

通过 benchmark skill 跑真实模型 shape（`/data/aiter-agent-skills`，gfx942 自动路由到 FlyDSL）：

```bash
S=/data/aiter-agent-skills/aiter-moe-benchmark/scripts
PYTHONPATH=/data/aiter python "$S/run_moe_bench.py" --model dsv4-pro --tp 8 \
    --no-fuse-shared-expert --quant a8w4_aligned --run perf --tokens 128,4096,16384
```

> 关键环境变量：`AITER_A8W4_PERMLUT`（默认 1）、`AITER_A8W4_BITFOLD`（默认 0）、`AITER_A8W4_ALIGNED`（默认 0）、`AITER_A8W4_WIDELOAD`（默认 1，一次 dwordx2 加载两个 K32 operand）。
