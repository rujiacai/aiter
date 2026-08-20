# flydsl kernel 优化建议（结合 gfx942 硬件架构）

> 基线：`flydsl_vs_asm_analysis_cn.md`（shape: token=32768, model_dim=4096, inter_dim=192, E=193, topk=9, fp8 per_tensor, silu, g1u1, gfx942）
>
> 当前状态（该文档已落地的优化后）：
> - gemm2 已从 `t32x128/atomic` 调到 `t64x128/atomic` + `block_m2=64`，`moe_gemm2_0` 7480→5874us（-21.5%），e2e 14545→12931us。
> - 余下两个大头：**flydsl gemm2 GEMM 核本体仍比 asm 慢 ~1900us**（同 tile、同写出方式下 5563 vs 3162）；**per_Tensor 量化 `data_to_scale` 3108us**。

本文把分析文档里"剩余差距"的三个根因（窄访存 / 浅流水 / 重 LDS epilogue）和量化、sorting 的启示，逐一对应到 gfx942 的硬件行为，给出可执行的优化方向，并按"投入产出比"排序。

---

## 一、gemm2 GEMM 核：A2/B2 已是 16B 宽读（建议作废，见更正）

### ⚠️ 更正（2026-06-11 经 codegen + SGPR→参数精确映射核实）

本节最初的前提**不成立**。把 ISA 里每个 buffer 资源映射回内核参数后确认：

| 操作数 | ISA 加载 | 宽度 |
|---|---|---|
| W2 (B) | `buffer_load_dwordx4`（s[0:3]） | **16B 宽读 ✓** |
| A2 (X) | `buffer_load_dwordx4`（s[8:11]←s18=arg1） | **16B 宽读 ✓** |
| sorted_token_ids gather | `buffer_load_dword`（s[4:7]←s26）×18 | 4B 窄读 |
| a2_scale 逐行 | `buffer_load_dword`（s[36:39]←s22）×16 | 4B 窄读 |
| w_scale + 杂项 | ×4 | 4B 窄读 |

codegen 佐证：fp8 路径 `load_b_pack_k32` 用 `vec_elems=16/elem_bytes=1 → vec_width=4 = dwordx4`；X 走 `buffer_copy_gmem16_dwordx4`。**A2/B2 本就已是 16B 宽读**，38 条窄读全是次级元数据（gather 索引 + scale），不是矩阵数据。

此外，flydsl 静态 ISA 只 48 条 MFMA（K-loop 部分卷着 scf.for），而 asm 144 条是完全展开（静态=动态），所以原先"asm 36 宽读 vs flydsl 9 宽读"的静态计数对比并不等价，不能作为"flydsl 访存更碎"的证据。

→ **"强制 A2/B2 宽化"是无效改动（已是宽读），本节作废。** 真正的窄读在次级元数据（见第五节 scale 广播、以及第二节流水深度）。

### 实测探针：消除 16 条 epilogue a2_scale 窄读 → 零收益（2026-06-11）

针对"38 条窄读里 16 条是 per_Tensor 下冗余的逐行 a2_scale 读"这一点，做了 env 门控探针（`FLYDSL_SCALAR_ASCALE=1`）：epilogue 不再逐行 `buffer_load(sx_rsrc, ts2)`，改成入口 hoist 一次 `sx_rsrc[0]` 复用（per_Tensor 下数值等价）。

| | gemm2 us | 正确性 |
|---|---|---|
| 探针 OFF（逐行读） | 5872 | pass |
| 探针 ON（标量广播） | 5873 | **pass，cos=0.999990** |

→ **零收益**。这 16 条窄读在 epilogue（每 M-block 仅一次），被主 GEMM 计算 / 写出 / CShuffle LDS 往返完全掩盖，不在关键路径上。**结论：不值得为它做内核签名改造；探针代码已还原。** 真正的瓶颈仍是 GEMM 核的写出/LDS epilogue/流水深度，而 K=192（仅 3 个 K-tile）限制了加深 K 方向流水的空间。

---

## 二、加深软件流水 / 提高寄存器预算，隐藏 MFMA 与访存延迟

### 硬件依据

- gfx942 的延迟隐藏能力可用 `Wo`（CU 内平均活跃 wave 数）× 每 wave 在途访存（指令数 n × 请求大小 R）来描述；当算术强度低时，必须靠**足够的在途访存 + 软件流水**来填满内存延迟 [2, p.10]。
- MFMA 有显著的 **write-VGPR → 后续读 VGPR 延迟**，需要插入软件 wait：对 `V_MFMA_F32_16x16x32_FP8_FP8` 这类多 pass 指令，"上一条 MFMA 提交结果到 VGPR 前没有内部转发路径"，后续把它当 SrcA/SrcB 读要等十几到几十 cycle 不等 [3, p.18]。
- 因此 **静态展开 K-loop / M-repeat 来制造 ILP**（让相邻 MFMA 之间穿插独立的访存/计算）是隐藏这些延迟的关键。

### 现状对比

| | asm down | flydsl gemm2 |
|---|---|---|
| MFMA 条数 | 144（全展开成直线代码） | 48 |
| 控制流 | ~0 label / 6 分支（直线） | 19 分支（循环+掩码+guard） |
| MFMA 累加器寄存器组 | 16 组 ×4 = 64 VGPR | 11 组 ×4 ≈ 44 VGPR |
| 总 VGPR | 208 | 82 |
| `s_waitcnt` | 21 | 61 |

### 关键澄清：VGPR 利用率不由 K 决定（修正"K 太短没救"的旧判断）

K=192 很小（fp8 16x16x32 仅 6 个 k-step），但 asm 仍能吃满 208 VGPR，因为 VGPR 消耗来自三块、**没有一块跟 K 绑定**：

1. **输出累加器 tile（M×N 常驻寄存器）**：由 `tile_m×tile_n` 决定，与 K 无关。asm 的 M=64×N=128 网格切成 16 个 16×16 子块、各 f32x4 → **64 VGPR 纯累加器**，K 多长都得在。
2. **操作数预取缓冲（大头）**：208−64 ≈ **144 VGPR 用于同时持有多块在途 A/W2**。gemm2 瓶颈是把大 W2 从 HBM 拉入，asm 用大量寄存器让多笔 load 在途、边算边预取 → `s_waitcnt` 仅 21。这只取决于"愿用多少寄存器持有在途数据"，与 K 无关。
3. **全展开成直线代码**：K 小恰恰让"把整个 M×N×K 网格摊平成无分支直线代码"更容易（144 条 MFMA 全展开），便于编译器深度交错 load/MFMA 制造 ILP。

→ 正确因果：**K 小 → 全展开更容易 → 配合大累加器 + 深操作数预取 → 高 VGPR/高 ILP/低 waitcnt**。K 小是有利条件而非障碍。flydsl 的 82 VGPR 是**保守寄存器策略**（累加器更少 + 预取浅 + 保留较多控制流）的结果，不是 K 限制的结果。

### 实测：强行提 VGPR（n_per_wave=32→64）→ VGPR 上去了但更慢 22%（2026-06-11）

直接试了把 per-wave 累加器 tile 翻倍来逼近 asm：放开 codegen 校验允许 `n_per_wave=64`（→ `num_acc_n=4`、`num_waves=tile_n/64=2`），用 kernel 名后缀 `_n64` 走通。hot_loop_scheduler 本就有 `num_acc_n>=4` 分支，编译/正确性均无问题。

| | 基线 n_per_wave=32 | n_per_wave=64 |
|---|---|---|
| VGPR | 82 | **150**（↑ 向 208 靠拢）|
| MFMA（静态） | 48 | **96**（↑ 向 144 靠拢）|
| 累加器寄存器组 | 11 | 23 |
| `s_waitcnt` | 61 | **95（↑ 更差）**|
| 正确性 | pass | **pass，cos=0.999990** |
| **gemm2 us** | **5872** | **7199（慢 22%）** |

**根因 + 关键教训**：`n_per_wave=64` 让每 WG 只剩 **2 waves（128 线程）**，而 asm 实际是 **4 waves（256 线程，与基线 n32 相同）**。asm 的 208 VGPR 不是靠"减 wave 换大累加器"，而是 **同样 4 waves 下持有更多在途操作数预取缓冲**。减半 wave 数牺牲了 wave 级延迟隐藏，且 num_acc_n=4 的 MFMA write→read 依赖链更长，waitcnt 反升。

→ **VGPR 是必要非充分条件**：单纯堆高 VGPR（经 n_per_wave 这个唯一可用旋钮）会同时砍掉 occupancy/wave 并行，净负优化。要真正复刻 asm，必须 **保持 4 waves 不变、只加深操作数预取（在 K 循环里多持有几块在途 B/A）**，并配套手工 load/MFMA 交错调度——这属于 flydsl codegen inner-loop 的深度改写，且 K=192（仅 3 个 K-tile）限制了预取深度收益。该实验改动已还原。

### 建议（按可行性）

1. **保持 n_per_wave=32（4 waves，与 asm 一致），在 K 循环里加深操作数预取深度**（从 1-ahead 到 2-ahead 持有更多在途 B），这才是 asm 高 VGPR 的真实来源。但需配套 inner-loop 调度改写，且 K=192 仅 3 tile，预取深度收益受限。
2. **修复 / 启用 async copy（global→LDS 直达）**：`use_async_copy=True` 在 t64x128/fp8/gfx942 上 LLVM lowering 崩溃。async copy 能把 load 从 `s_waitcnt` 关键路径摘下，是隐藏访存延迟的标准手段，值得作为 codegen bug 单独推进。
3. 结论性判断：**flydsl gemm2 在该 shape 已接近其 codegen 自动调度的实际上限（5874us）**；剩余与 asm 的差距是"手工汇编调度" vs "JIT 自动调度"的固有鸿沟，靠现有旋钮/局部改动无法跨越。ROI 更高的是换战场（按 M 分流走 asm / 量化🅱）。

---

## 三、精简 CShuffle 的 LDS epilogue

### 硬件依据

gfx942 每 CU 的 LDS 为 **64KB / 32 banks**（实验中可双倍/64 banks；扩容无收益，但 **64 banks 可带来 1.2~1.6× 性能**，说明 LDS 瓶颈主要在 **bank 冲突/吞吐** 而非容量）[4][5]。epilogue 走 LDS 往返会占用这部分 bank 带宽并制造额外依赖等待。

### 现状对比

flydsl epilogue 的 LDS 流量重：`ds_write/ds_read` 共 76 条，asm 仅 37 条。asm 的 epilogue 是**寄存器内联**完成的：`v_pk_mul_f32`（acc×scale 反量化）→ `v_perm_b32`（两个 f32 打包成 bf16）→ `global_store_dwordx4` 宽写，**几乎不经 LDS**。

### 建议

1. **对该 shape 评估"绕过 LDS 的寄存器直写 epilogue"**：tile_n=128、输出 bf16，若能像 asm 那样在寄存器里做 dequant + pack 后直接宽写/宽原子，可省掉 CShuffle 的 76 条 LDS 往返与相应 wait。
2. 若 CShuffle 必须保留（用于跨 wave 重排以获得合并写），则**优化 LDS 访问的 bank 映射**避免冲突（结合 64-banks 收益的结论），并尽量用 `ds_read_b128`/`ds_write_b128` 宽 LDS 指令减少指令数。

---

## 四、gemm2 写出（atomic）：维持现状，但注意 cache-line 串行化

### 硬件依据（修正旧判断）

分析文档步骤 4/5 已实测证明：大 M 下 **bf16 global atomic 不是瓶颈**（atomic 5892 仅比非原子 store 5563 慢 ~329us，且省掉 679us 的独立 reduce）。从硬件看也合理：

- 输出 `out[token, model_dim]`≈268MB 小缓冲，9 份 topk 累加进同地址 → **L2 友好、HBM 写流量小**；reduce 模式要写 2.4GB partial + 读回 2.4GB，带宽翻数倍。
- 但 gfx942 上 **device-scope 原子在 TCP 是 forced-miss lookup + evict update**，且有 `bypass_pending` 机制：对同一 cache line，一个 RMW 之后的读/写/RMW 会被 stall 到完成响应回来 [6, p.13][7, p.14]。topk=9 抢同一行 → 存在 cache-line 级串行。

### 建议

1. **保持 atomic 模式**（已是 flydsl 两种模式里更优解，勿回退到 reduce）。
2. 微优化方向（收益有限，~329us 上限内）：尽量让不同 topk 的部分和落到**不同 cache line**（错开 N 维 tile 与 token 的映射），减少 `bypass_pending` 串行；或在 split-K=1 时确认没有引入多余的同地址原子。
3. gfx950+ 才有 `buffer_atomic_pk_add_bf16`（快路径），gfx942 只有 `global_atomic_pk_add_bf16`（慢路径，raw pointer）——这是架构固有限制，gfx942 上无法绕过，只能从"减少同 line 争用"着手。

---

## 五、量化：让 2-stage 路径走 per_Token 融合（投入产出比最高的"非 gemm"项）

这一项不属于 flydsl gemm 内核本身，但在 e2e 里占比极大（per_Tensor 量化 3608us vs per_Token 759us，差 ~2.85ms），且改动小、风险可控。

### 硬件依据

`data_to_scale_kernel` 对中间结果 `[294912, 192]` 的线程映射极差：cols=192 → `num_vecs=12`，**256 线程里只有 12 个在干活（~5% 利用率）**，却起 ~29.5 万个 block，每个 block 末尾对**唯一一个全局 float** 做 `atomicMaxFloat`。结合上面的 gfx942 原子语义（device-scope 原子 forced-miss + 单点地址争用），~29.5 万 block 抢一个地址 → 极端串行，这正是 3108us 远超访存理论值的根因。

### 建议（按优先级）

1. **首选：2-stage 路径激活量化也改 per_Token 融合 kernel**（像 asm `fused_moe_asmjit_aot.py:120` 那样硬走 `QuantType.per_Token`），一遍读数据、每行独立 scale、写各自地址无原子争用。单这一步可省 ~2.8ms。**前提是确认 flydsl gemm2 能吃 per-token 的激活 scale**（gemm 里权重 scale 用法不变）。
2. 若必须保留 per_Tensor：改用代码库已有的**融合 per_tensor 量化**（`fused_moe.py` 的 `dynamic_per_tensor_quant_fp8_i8_fused_small` / `_direct_per_tensor_quant_cached`），避免"两遍 + 低效 block 映射"。
3. 若短期都不动接口：**重写 `data_to_scale_kernel` 对窄 cols 的映射**——一个 block 吃多行、提高线程利用率、用 **LDS 原子做 workgroup 内规约**再少量 global 原子（文档明确建议"workgroup 内的原子用 LDS atomics 性能更好"）[8, p.14]，把 ~29.5 万次单点 global 原子降到每 block 一次甚至更少。

---

## 六、moe_sorting：统一 block_m 省掉第二次排序（低优先级）

sorting kernel 两条路径完全相同，差别只是 flydsl 2-stage 因 `block_m=128`（stage1 CK）≠ `block_m2=64`（stage2 flydsl，已从 32 改为 64）而排两次（~328us vs ~161us）。

- 注意：步骤 4 把 `block_m2` 从 32 调成 **64** 后，与 stage1 的 128 仍不相等，仍会触发两次排序。
- 建议：若能让 stage1 与 stage2 统一 block_m（例如让 CK gemm1 也用 64，或评估 stage2 用 128 的 gemm 损失），可回到单次排序省 ~167us。需权衡：被改的那个 stage 偏离最优 tile 的损失是否小于省下的 167us，要重新 tune 验证。优先级低于一/五。

---

## 优先级汇总

| 优化项 | 预期收益 | 改动量/风险 | 优先级 |
|---|---|---|---|
| 五. 量化改 per_Token 融合 | ~2.8ms | 小（配置/接口），需验证 gemm2 吃 per-token scale | ★★★★★ |
| 一. gemm2 加载 4B→16B 宽合并读 | 大（gemm2 核 ~1900us 差距的主因） | 中（codegen 向量宽度 + 对齐） | ★★★★★ |
| 二. 加深软件流水 / 提 VGPR / 修 async copy | 中~大 | 中~重（codegen 调度，async copy 是 bug 修复） | ★★★★ |
| 三. 精简 CShuffle LDS epilogue | 中 | 重（codegen epilogue 重写） | ★★★ |
| 六. 统一 block_m 省第二次排序 | ~167us | 小，但需重 tune 权衡 | ★★ |
| 四. atomic 写出维持 + 减同 line 争用 | 小（≤329us） | 小 | ★★ |

> 核心判断：**"非 gemm"的量化（五）改动最小、收益最大，应最先做**；gemm2 内核本体的差距（一/二/三）本质是 JIT codegen（窄访存 + 浅流水 + 重 LDS epilogue）相对手写汇编的固有差异，其中**访存向量化（一）是单点收益最高、最该先攻的 codegen 项**，与 gfx942 的 128B cache line / dwordx4 合并机制直接对应。

────────────────────────────────────────────────────────────────
[1] mi300_gpu_bandwidth.pdf
[2] mi300_gpu_bandwidth.pdf
[3] MI300_SP_MAS.docx
[4] MI300_Custom_Memory_Request_nonSP.pptx
[5] PVonCM_MLB_MI300_v1.pptx
[6] mi300_gpu_bandwidth.pdf
[7] MI300 GFX L2 Coherence.docx
[8] MI300 GFX L2 Coherence.docx
────────────────────────────────────────────────────────────────
