# FlyDSL atomicAdd MoE Sorting — 实现 / 性能 / 回退根因 / 复现

> 目标 shape：`token=32768, model_dim=4096, inter_dim=192, expert=193, topk=9, fp8 per_tensor`，硬件 MI308X（gfx942, 80 CU, max sclk 1420MHz）。

## 1. 这是什么

用 **FlyDSL** 实现的一版 MoE token sorting，思路参考 hpc-ops 的 `count_and_gather`：
- **atomicAdd** 做计数 / 定位（分层：block 内 LDS 原子 → 跨 block 通过 partial 直方图归约）。
- **惰性索引（lazy index）**：只产出 `sorted_token_ids` 等索引，不搬运 token 数据（gather 融合进下游 GEMM）。

输出契约与 CK 的 `moe_sorting_fwd` 对齐，可 drop-in。

## 2. 涉及的文件

| 文件 | 作用 |
|---|---|
| `aiter/ops/flydsl/kernels/moe_sorting_atomic.py` | FlyDSL kernel 构造（fill / zero / count(融合zero) / cumsum / write_eids / scatter） |
| `aiter/ops/flydsl/moe_sorting_api.py` | host 封装：`moe_sorting_atomic_fwd`（输出预分配）/ `moe_sorting_atomic`（自分配） |
| `aiter/fused_moe.py` | 集成点：`_moe_sorting_impl` 内按环境变量分发到 FlyDSL，EP 场景自动回退 CK |

## 3. 算法（5 个 kernel）

```
K0  fill        : sorted_ids 预填 sentinel=(topk<<24)|M, sorted_weights=0
K1  count(+zero): 每 block 用 LDS atomicAdd 建直方图 -> partial[block, E]（plain store，无跨 block 原子）
                  额外的 block 同时向量化(dwordx4) 清零 moe_buf（与计数并发，藏带宽）
K2  cumsum      : 归约 partial[*,e] -> total -> padded 独占前缀和 -> 每 block 的 base slot（原地写回 partial）
K2b write_eids  : grid=E，每个 expert 一个 block，写 sorted_expert_ids
K3  scatter     : 读本 block 的 base slot 进 LDS；单遍：lpos=LDS atomicAdd(wcur[eid]); slot=base[eid]+lpos
                  写 packed_id=(topk_slot<<24)|token 和对应权重
```

打包格式：`(topk_slot<<24)|token`，padding sentinel `(topk<<24)|M`，与 CK/native 一致。

## 4. 开关（环境变量）

| 变量 | 默认 | 含义 |
|---|---|---|
| `AITER_USE_FLYDSL_MOE_SORTING` | 0 | 1=fused_moe 的 `moe_sorting` 走 FlyDSL（EP/local-token 场景自动回退 CK）|
| `AITER_SORT_BLOCKS` | 32 | count/scatter 的 block 数（争用最优点≈32）|
| `AITER_SORT_SKIP_ZERO` | 0 | 1=跳过 moe_buf 清零（**仅纯排序基准**，e2e 会算错）|
| `AITER_SORT_REORDER` | 0 | 1=**诊断用**：host 端把每个 expert 段内 token 重排成升序（用来隔离"顺序"的影响）|
| `AITER_ZERO_OCC` | 8 | zero kernel 的 occupancy 系数 |

## 5. 性能结论

### 5.1 纯排序 microbench（CK vs FlyDSL，E=193, topk=9, block=16）

| token | CK | FlyDSL | 比值 |
|---:|---:|---:|---:|
| 1 | 10.0 us | 28.7 us | 0.35x |
| 256 | 11.0 | 31.4 | 0.35x |
| 4096 | 36.4 | 50.5 | 0.72x |
| 16384 | 93.6 | 104.0 | 0.90x |
| 32768 | 169.8 | 175.5 | 0.97x |

- 小 batch：FlyDSL 因为 5 次 kernel launch（每次 ~5us dispatch）远慢于 CK 的单 oneshot kernel。
- **去掉 moe_buf 清零后，FlyDSL 纯排序在 32k 只要 ~76.5us**（CK 把 ~100us 的清零焊死在排序里，拆不出来）。所以"排序算法本身" FlyDSL 更快。

### 5.2 e2e（32k，through `moe_sorting` 路径，PASS cos=0.99999）

| 配置 | stage2 (moe_gemm2_0) | e2e |
|---|---:|---:|
| CK 排序（基线，env off） | 4184 us | 8702 us |
| FlyDSL 排序（env on） | 6522 us | 11241 us |
| FlyDSL + `AITER_SORT_REORDER=1`（诊断） | 4182 us | 9560 us |

**开了 FlyDSL 排序后 e2e 反而慢了 ~2540us。**

## 6. 为什么会回退（根因，已用 trace 实证）

退化**不在排序耗时**（排序在 32k 只占 e2e <2%），而在 **stage2 GEMM +2340us**。

### 6.1 因果证明（控制变量）
`AITER_SORT_REORDER=1` 只把每个 expert 段内的 token 改成升序，其余（padding/num_valid/分段/计算量）全不变 → stage2 从 6522→4182us。**顺序是唯一变量**。

根因：**atomicAdd 的 scatter 打乱了 expert 段内的 token 顺序**（块内由 wave 调度决定、块间 grid-stride 跨步），而 CK 的计数排序保证 **token 升序**。stage2(down-proj) 按 `sorted_token_ids` 做 `inter_states` gather + `moe_buf` scatter-add，对访存顺序敏感。

### 6.2 微架构铁证（rocprofv3 计数器，stage2=moe_gemm2_0）

| 计数器 | 乱序(atomicAdd) | 有序 | 差异 |
|---|---:|---:|---:|
| TCP_TOTAL_CACHE_ACCESSES（L1 访问数） | 1,793,010,960 | 1,793,010,960 | **完全相同** |
| TCP_TCC_READ_REQ（L1→L2 请求） | 344,438,196 | 344,014,349 | +0.1% |
| L2 hit rate（TCC_HIT/(HIT+MISS)） | 60.1% | 60.1% | 相同 |
| TCC_REQ（L2 总请求） | 544M | 550M | +1% |
| **TCP_PENDING_STALL_CYCLES（L1 访存停顿周期）** | **2,447,855,887** | **1,102,948,374** | **+122% (2.2x)** |

**解读**：访存的"量"完全没变（L1 访问数一字不差、L2 命中率/请求数相同、HBM 字节也相同）；唯一暴涨的是 **L1 访存停顿周期（+122%）**。即：触碰的内存一样，但散射访问让**每个请求的有效延迟变高 / 内存级并行(MLP)下降** → L1 miss 队列打满、访存流水线停顿翻倍。stage2 是 K=192 的瘦 GEMM（访存受限，MFU~18%），被访存延迟主导，所以对乱序极度敏感。

时间对得上：stage2 6522/4182 ≈ 1.56x；停顿周期 2.45B/1.10B ≈ 2.2x（停顿是运行时间主导成分）。

### 6.3 一句话
> atomicAdd 给的是"正确但乱序"的索引；下游 GEMM 按它做 gather/scatter，**有序=低延迟访存，乱序=访存停顿翻倍**。排序少花的几十 us，在 stage2 上以 +2340us 的访存停顿还了回去。

## 7. 结论与方向
- **大 batch e2e：用 CK**（`AITER_USE_FLYDSL_MOE_SORTING=0`）。排序占比 <2%，且乱序伤 stage2。
- 要让 FlyDSL 在 e2e 不亏：scatter 必须**保序**（连续分区 + 块内按 expert 稳定前缀和定位，替代 LDS 原子），代价是排序变慢（接近 CK），但能消除 stage2 的 +2340us。即便如此 e2e 也最多和 CK 持平。
- 另一条正交优化：把 moe_buf 清零从排序里挪到 stage1 init 并与 GEMM 重叠。

## 8. 复现

```bash
cd /opt/aiter

# (1) 正确性 + 纯排序 CK vs FlyDSL token 扫描
python flydsl_moe_sorting/bench_sort.py

# (2) e2e 根因（控制变量：乱序 vs 段内升序），看 stage2 / e2e
bash flydsl_moe_sorting/repro_e2e_reorder.sh

# (3) 微架构铁证（L2 + L1 停顿计数器），看 TCP_PENDING_STALL_CYCLES
bash flydsl_moe_sorting/repro_counters.sh
```

脚本说明见各文件头部注释。`repro_counters.sh` 用单趟原始计数器（快）；派生指标(FETCH_SIZE/MemUnitStalled)需多趟、每趟重新 JIT，很慢，故未用。
