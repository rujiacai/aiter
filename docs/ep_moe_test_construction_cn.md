# EP 模式下 MoE 单测构造指南

本文说明如何在 **Expert Parallel (EP)** 部署下，用 `op_tests/test_moe_2stage.py` 或相关测试**同构**构造用例。以 GLM-5.3 vLLM dump（`bs128`、MTP3、EP16、DP16）为例，给出理论公式、实测 shape 对照，以及推荐参数。

相关文件：

| 文件 | 用途 |
|------|------|
| `op_tests/test_moe_2stage.py` | 合成 micro-bench；`--ep > 1` 时自动构造 EP 路由 |
| `op_tests/test_moe_ep.py` | 带 `expert_mask` 的 EP 路径（含 shared expert 等复杂语义） |
| `op_tests/test_fmoe_vllm_dump.py` | 真实 dump 回放（最高保真） |
| `fmoe_dump_20260908/FMOE_DUMP_20260908.md` | dump 采集说明与 bs↔token 换算 |

---

## 1. 背景：为什么需要 `--ep`

`test_moe_2stage.py` 在 **`--ep 1`（默认）** 时，在 **单卡、E 个本地专家**上生成路由：

```python
score = torch.randn((token, E), ...)   # E=16，全是本地专家
topk_ids = fused_topk(input, score, topk)
```

因此 `-k 8` 时 **8 个 slot 全部命中本地专家**，`pairs = token × topk`。

真实 EP 部署（dump）则是：

- **256 全局专家**，`topk_ids` 为全局 id（0~255）
- 本 rank 仅拥有 **16 个本地专家**（`expert_map` / `expert_mask`）
- 每个 token 全局选 8 个专家，平均只有 **~0.5 个**落在本 rank；经 EP 筛选后，有效本地 topk **~1.2~1.3**

| 场景 | token | topk | 本地 pairs | 说明 |
|------|------:|-----:|-----------:|------|
| dump bs128 layer3 | 221 (recv) | 8 全局 | **288** | 真实 EP 稀疏路由 |
| `test_moe_2stage -t 221 -k 8`（无 `--ep`） | 221 | 8 本地 | **1768** | 高估 **~6×** 算力 |
| `test_moe_2stage -t 512 -k 8 --ep 16` | 512 (buffer) | 8 全局 | **~264** | **`--ep` 合成 EP，推荐** |
| `test_moe_2stage -t 221 -k 1`（无 `--ep`） | 221 | 1 本地 | **221** | 旧近似，仅粗对齐 recv |

---

## 2. 术语与 bs128 实测 shape

### 2.1 三个「token」不要混用

| 名称 | bs128 稳态值 | 含义 |
|------|-------------:|------|
| **bs** | 128 | 并发 decode **请求数**（非 token 数） |
| **rows** | 512 | dispatch 后 FMOE **输入 buffer 行数** = `bs × (1+MTP)` = `128×4` |
| **recv** | 177~227 | 本 EP rank **实际有效行数**（`expert_num_tokens`） |
| **pairs** | 213~289 | 本 rank **本地 (token, expert) 对数**（真实 GEMM 工作量） |

MTP3：每请求 1 个 verified + 3 个 draft → 每请求 4 token。

```
全局 rows = bs × (1 + K_mtp) = 128 × 4 = 512
每 DP rank dispatch 前 token = (bs/16) × 4 = 8 × 4 = 32
rows = 32 × 16(EP layout) = 512   # mori compact_recv_layout
```

### 2.2 bs128 dump 各层实测（`mstep51`，DP rank 0）

数据来源：`fmoe_dump_20260908/dump/bs128/bs128_mstep51_dp0_layer*.pt`

| layer | hidden_states | topk_ids | expert_map | w1 | w2 | recv | pairs | eff_k |
|------:|---|---|---|---|---|---:|---:|------:|
| 3 | `[512, 6144]` fp8 | `[512, 8]` i32 | `[257]` i32 | `[16, 4096, 6144]` fp8 | `[16, 6144, 2048]` fp8 | 221 | 288 | 1.30 |
| 20 | 同上 | 同上 | 同上 | 同上 | 同上 | 219 | 259 | 1.18 |
| 40 | 同上 | 同上 | 同上 | 同上 | 同上 | 227 | 289 | 1.27 |
| 60 | 同上 | 同上 | 同上 | 同上 | 同上 | 177 | 213 | 1.20 |
| 77 | 同上 | 同上 | 同上 | 同上 | 同上 | 204 | 256 | 1.26 |

公共 meta：

| 字段 | 值 |
|------|----|
| `pre_dispatch_tokens` | 32（每 DP rank） |
| `num_decode_tokens` | 32（全局 decode token） |
| `num_decodes` | 8 |
| `experts_per_token` (topk) | 8 |
| `global_num_experts` | 256 |
| `hidden_dim` / `inter_dim` | 6144 / 2048 |
| quant | `per_128x128` fp8×fp8（blockwise a8w8） |

`a1q_scale` shape：`[512, 48]`（`6144/128=48` 个 per-128 块 scale）。

`expert_map.shape = [257]`：256 全局专家 + 1 个 sentinel/fake expert id。

### 2.3 pairs 统计方式（与 `test_fmoe_vllm_dump.py` / `--ep` 一致）

```python
valid_ids = topk_ids[:recv]
pairs = expert_mask[valid_ids.long()].gt(0).sum().item()
eff_k = pairs / recv
```

`expert_mask[global_id] == 1` 表示该全局专家属于本 rank；对 `ep_id=0` 时本地专家为 `[0..15]`，对 `ep_id=1` 则为 `[16..31]`，**不能用 `global_id < 16` 判断**。

---

## 3. 均匀路由理论模型

### 3.1 假设

- 全局专家数 \(E = 256\)，EP 规模 \(R = 16\)，每 rank 本地专家 \(E_\ell = E/R = 16\)
- 每 token 从 \(E\) 个专家中**均匀无放回**选 \(k=8\) 个（topk）
- 本 rank 只处理「至少命中 1 个本地专家」的 token（EP dispatch 筛选）

### 3.2 单 token 本地命中数

**随机变量 \(X\) 的定义**：对每个 token，设其全局 topk 路由选出了 \(k\) 个专家 id，则

\[
X = \text{「这 } k \text{ 个全局专家里，落在本 rank 本地专家上的个数」}
\]

例如 EP16、本 rank 拥有本地专家 `[0..15]`，`topk=8`：

| topk_ids（全局） | 本地命中 | \(X\) |
|------------------|----------|------:|
| `[100, 50, 200, 88, 150, 33, 120, 77]` | 无 | **0** |
| `[3, 100, 50, 200, 88, 150, 33, 120]` | 专家 3 | **1** |
| `[1, 5, 12, 200, 88, 150, 33, 120]` | 专家 1, 5, 12 | **3** |

**\(X=0\) 的含义**：该 token 的 \(k\) 个全局专家**全部都不是**本 rank 的本地专家。EP dispatch 后，这类 token 在本 rank **没有任何 GEMM 工作**，不会进入 `recv` 统计。

因此 \(P(X=0)\) 就是「一个 token 在本 rank 完全无本地命中」的概率；\(P(X \ge 1) = 1 - P(X=0)\) 才是该 token 会被本 rank 接收、计入 `recv` 的概率。

在均匀无放回抽样假设下：

\[
X \sim \mathrm{Hypergeometric}(N=E,\; n=E_\ell,\; k)
\]

即从 \(E\) 个全局专家中无放回抽 \(k\) 个，其中恰好有 \(X\) 个落在本 rank 的 \(E_\ell\) 个本地专家上。

| 量 | 精确公式 | EP16, k=8 数值 |
|----|----------|----------------|
| \(\mathbb{E}[X]\) | \(k \cdot E_\ell / E = k/R\) | **0.500** |
| \(P(X=0)\) | \(\binom{E-E_\ell}{k} / \binom{E}{k}\) | **0.592** |
| \(P(X \ge 1)\) | \(1 - P(X=0)\) | **0.408** |
| \(\mathbb{E}[X \mid X \ge 1]\) | \(\mathbb{E}[X] / P(X \ge 1)\) | **1.226** |

近似（\(E\) 大时）：\(P(X=0) \approx (1 - 1/R)^k = (15/16)^8 \approx 0.592\)

### 3.3 N 个 token 时的 recv / pairs

设 \(N\) 为进入 EP dispatch 的 **buffer 行数**（bs128 稳态 \(N=512\)）：

\[
\boxed{\mathbb{E}[\mathrm{recv}] = N \cdot P(X \ge 1)}
\]

\[
\boxed{\mathbb{E}[\mathrm{pairs}] = N \cdot \mathbb{E}[X] = N \cdot \frac{k}{R}}
\]

\[
\boxed{\mathrm{eff\_k} = \mathbb{E}[X \mid X \ge 1] = \frac{k/R}{P(X \ge 1)}}
\]

\[
\mathrm{pairs} \approx \mathrm{recv} \times \mathrm{eff\_k}
\]

**recv 的标准差**（Bernoulli 近似，每 token 独立）：

\[
\sigma_{\mathrm{recv}} = \sqrt{N \cdot p(1-p)}, \quad p = P(X \ge 1)
\]

### 3.4 N=512、EP16、topk=8 理论值

| 量 | 理论 | dump layer3 | 偏差 |
|----|-----:|------------:|-----:|
| \(P(X \ge 1)\) | 40.8% | — | — |
| **recv** | **208.7 ± 11.1** | **221** | +5.9%（在 1σ 内） |
| **pairs** | **256.0** | **288** | +12.5% |
| **eff_k** | **1.226** | **1.303** | +6.3% |

dump 略高于均匀假设，符合真实 gate **非均匀**（热门专家负载倾斜）。layer77 pairs=256 与理论 **完全一致**。

### 3.5 理论 vs 五层 dump 汇总

| layer | recv (实测) | recv (理论 512) | pairs (实测) | pairs (理论 512) | eff_k (实测) | eff_k (理论) |
|------:|------------:|----------------:|-------------:|-----------------:|-------------:|-------------:|
| 3 | 221 | 209 | 288 | 256 | 1.30 | 1.23 |
| 20 | 219 | 209 | 259 | 256 | 1.18 | 1.23 |
| 40 | 227 | 209 | 289 | 256 | 1.27 | 1.23 |
| 60 | 177 | 209 | 213 | 256 | 1.20 | 1.23 |
| 77 | 204 | 209 | 256 | 256 | 1.26 | 1.23 |
| **均值** | **210** | **209** | **261** | **256** | **1.24** | **1.23** |

五层 recv 均值 210 vs 理论 209；pairs 均值 261 vs 理论 256。**均匀超几何模型可解释量级与 eff_k，非均匀路由带来 ±10~15% 层间波动。**

### 3.6 性能对标（无环境变量，default tuned config）

以下数据在 **gfx942** 上测量，`iters=50, warmup=5`，**不设置任何 `AITER_*` 环境变量**。

#### A. buffer=512 同表对比：dump replay vs 合成 EP

**Dump replay**（真实 vLLM 路由 + activation，随机权重）：

```bash
python op_tests/test_fmoe_vllm_dump.py -b 128 -l 3 20 40 60 77
```

**合成 EP**（均匀随机全局路由 + `expert_mask`，随机权重；buffer 与 dump 对齐）：

```bash
python op_tests/test_moe_2stage.py -q 5 -t 512 -dim 6144,2048 -e 16 -k 8 \
  --no-flydsl-csv --ep 16 --ep-id 0
```

| 来源 | layer / 命令 | rows | recv | pairs | eff_k | us (µs) | TFLOPS | 路径 |
|------|-------------|-----:|-----:|------:|------:|--------:|-------:|------|
| dump | 3 | 512 | 221 | 288 | 1.30 | **262.9** | 82.7 | asm **1stage** |
| dump | 20 | 512 | 219 | 259 | 1.18 | **261.2** | 74.9 | asm **1stage** |
| dump | 40 | 512 | 227 | 289 | 1.27 | **269.3** | 81.0 | asm **1stage** |
| dump | 60 | 512 | 177 | 213 | 1.20 | **232.8** | 69.1 | asm **1stage** |
| dump | 77 | 512 | 204 | 256 | 1.26 | **259.0** | 74.6 | asm **1stage** |
| dump | **均值** | 512 | **210** | **261** | **1.24** | **257.0** | **76.5** | asm **1stage** |
| synth EP | `--ep 16 -t 512 -ep-id 0` | 512 | **210** | **264** | **1.26** | **263.4** | **75.7** | FlyDSL **2stage** |

合成 EP 行的 recv/pairs/eff_k 与 dump 五层均值（210 / 261 / 1.24）几乎一致，说明 `--ep` 构造的路由稀疏度可信；us 也在同一量级（263 vs 257 µs），但 **tuned config 在 `M=512` 为 2stage FlyDSL，dump 为 asm 1stage，路径不同故 us/TFLOPS 不宜当作同实现对比**。

#### B. 更大 buffer — `test_moe_2stage.py --ep 16 -t 1024`

```bash
python op_tests/test_moe_2stage.py -q 5 -t 1024 -dim 6144,2048 -e 16 -k 8 \
  --no-flydsl-csv --ep 16 --ep-id 1
```

| buffer (`-t`) | ep_id | recv | pairs | eff_k | us (µs) | 路径 |
|-------------:|------:|-----:|------:|------:|--------:|------|
| **1024** | 1 | 401 | 481 | 1.20 | **~308–361** | asm **1stage** |

> **对比注意**：`-t 1024` 时 recv/pairs 约为 dump 均值的 1.8×（401 / 481），us 约为 1.2–1.4×，符合算力量级线性放大；与 dump 同为 asm 1stage 路径，可比性优于 §3.6 A 中的 2stage 行。

### 3.7 EP vs TP 单 rank 性能对比（全局 shape 固定）

**全局模型 shape**（GLM-5.3 量级）：

| 字段 | 全局值 |
|------|-------:|
| `model_dim` | 6144 |
| `inter_dim` | 2048 |
| `E`（全局专家） | 256 |
| `topk` | 8 |
| `token`（buffer） | 512 |

在 `test_moe_2stage.py` 中模拟 **单 rank** 负载：

- **EP**：`-dim 6144,2048`，`-e = 256/ep`，加 `--ep`，全局 topk 路由 + `expert_mask` 稀疏筛选
- **TP**：不加 `--ep`，`-e 256`（本 rank 拥有全部专家），`-dim 6144,{2048/tp}`（按 TP 切分 `inter_dim`），**无稀疏**，`pairs = token × topk = 4096`

测量条件：gfx942，`iters=50, warmup=5`，无环境变量，四条命令均走 **asm 1stage** blockscale。

#### 命令与实测（2026-09-10 复跑）

```bash
# EP16 — 推荐，对齐 GLM-5.3 部署
python op_tests/test_moe_2stage.py -q 5 -t 512 -dim 6144,2048 \
  -e 16 -k 8 --no-flydsl-csv --ep 16

# EP8
python op_tests/test_moe_2stage.py -q 5 -t 512 -dim 6144,2048 \
  -e 32 -k 8 --no-flydsl-csv --ep 8

# TP8 — 无 --ep，inter_dim 切为 2048/8
python op_tests/test_moe_2stage.py -q 5 -t 512 -dim 6144,256 \
  -e 256 -k 8 --no-flydsl-csv

# TP16 — inter_dim 切为 2048/16
python op_tests/test_moe_2stage.py -q 5 -t 512 -dim 6144,128 \
  -e 256 -k 8 --no-flydsl-csv
```

| 模式 | `-e` | `--ep` | `inter_dim` | recv | pairs | eff_k | us (µs) | 相对 EP16 |
|------|-----:|-------:|------------:|-----:|------:|------:|--------:|----------:|
| **EP16** | 16 | 16 | 2048 | 189 | 235 | 1.24 | **245** | 1.00× |
| **EP8** | 32 | 8 | 2048 | 337 | 521 | 1.55 | **470** | 1.92× |
| **TP8** | 256 | — | 256 | 512 | 4096 | 8.00 | **475** | 1.93× |
| **TP16** | 256 | — | 128 | 512 | 4096 | 8.00 | **415** | 1.69× |

理论 recv/pairs（均匀路由，\(N=512, k=8\)）：

| 模式 | 理论 recv | 理论 pairs | 理论 eff_k |
|------|----------:|-----------:|-----------:|
| EP16 | 209 | 256 | 1.23 |
| EP8 | 339 | 512 | 1.51 |
| TP8/16 | 512（无筛选） | 4096 | 8.00 |

#### 解读

1. **EP16 最快（~245 µs）**：本 rank 仅 16 个本地专家，且约 59% token 无本地命中被滤掉（recv≈190 vs buffer 512），真实 GEMM 工作量 pairs≈235，远低于 TP 的 4096。
2. **EP8 vs TP8 us 接近（~470 vs ~475 µs）**：EP8 的 pairs（521）远小于 TP8（4096），但 EP8 保持 **完整 inter_dim=2048**；TP8 的 inter 缩为 1/8（256），单次 GEMM 更小。两者总算力在同一量级，故延迟接近。
3. **TP16（~415 µs）快于 TP8**：同样 pairs=4096，但 inter_dim 再减半（128），down-proj GEMM 更小。
4. **EP 的核心收益是稀疏路由**：不是单纯切 weight，而是通过 `expert_mask` 把大部分 token 和远程专家排除，使单 rank 实际 pairs 从 4096 降到 ~235（EP16，约 **17×** 算力缩减）。

---

## 4. 如何同构 `test_moe_2stage.py` 构造用例

分三档保真度；**推荐档位 C + `--ep 16`**（见 §4.3）。

### 4.1 档位 A：最高保真 — `test_fmoe_vllm_dump.py`

```bash
python op_tests/test_fmoe_vllm_dump.py -b 128 -l 3
```

真实 shape、路由、EP mask 全部来自 dump，**性能与正确性对标首选**（默认 asm 1stage，见 §3.6）。

### 4.2 档位 B：EP 语义 — `test_moe_ep.py`

使用 **256 全局专家 + expert_mask**，`fused_topk` 在全局 score 上选 topk，再由 mask 过滤本地专家。适合验证 EP 路径逻辑（含 shared expert、MORI dispatch 仿真等复杂语义）。

### 4.3 档位 C（推荐）：`test_moe_2stage.py --ep`

#### 命令示例（对齐 GLM-5.3 bs128）

```bash
python op_tests/test_moe_2stage.py \
  -q 5 -d bf16 -a silu \
  -dim 6144,2048 \
  -e 16 -k 8 -t 512 \
  --ep 16 --ep-id 0 \
  --no-flydsl-csv --no-situv2
```

#### CLI 参数说明

| 参数 | 默认值 | `--ep > 1` 时的含义 |
|------|--------|---------------------|
| `--ep` | `1` | EP 规模 \(R\)；`16` 表示 EP16 |
| `--ep-id` | `0` | 模拟的 EP rank，范围 `[0, ep)` |
| `-e` | `257` | **本地专家数** \(E_\ell\)（GLM-5.3 取 `16`） |
| `-k` | `9` | **全局 topk**（GLM-5.3 取 `8`） |
| `-t` | 多档 sweep | **dispatch buffer 行数** \(N\)（bs128 取 `512`） |

全局专家数：\(E_\mathrm{global} = e \times \mathrm{ep} = 16 \times 16 = 256\)。

#### 内部构造流程（`test_fmoe()` 中 `ep > 1` 分支）

与 `test_moe_ep.py` / vLLM dump 语义对齐，核心步骤如下：

```
1. expert_mask
   shape = [E_global + 1]          # 256 + 1 sentinel，与 dump expert_map 一致
   expert_mask[ep_id*E_local : (ep_id+1)*E_local] = 1
   expert_mask[-1] = 0             # fake/sentinel expert 恒为 0

2. 全局路由
   score = randn(N, E_global)      # N = -t，在 256 个全局专家上打分
   fused_topk → topk_ids [N, k]    # 全局 expert id（0~255）

3. MORI dispatch 仿真（compact）
   recv_mask[t] = any(expert_mask[topk_ids[t]] > 0)   # 至少 1 个本地命中
   recv = recv_mask.sum()
   将 recv 个有效行 compact 到 buffer 前部 [:recv]
   buffer 尾部 [recv:N) 为 dead padding（随机值，kernel 不处理）

4. fused_moe 入参
   w1/w2 shape = [E_local, ...]  # 仅本地权重
   topk_ids 保留全局 id
   expert_mask=expert_mask
   num_local_tokens=tensor([recv])

5. torch reference
   ref_topk_ids = global_to_local(topk_ids)   # 远程专家映射为 -1，不参与计算
   仅在 out[:recv] 上与 kernel 比对

6. 输出统计（summary / CSV）
   recv, pairs, eff_k, E_global
   pairs = expert_mask[topk_ids[:recv]].sum()
```

对应代码位置：`op_tests/test_moe_2stage.py` 中 `_build_ep_expert_mask`、`_build_ep_dispatch_buffer`、`_count_ep_pairs`。

#### 参数对照表（GLM-5.3 bs128）

| 参数 | dump 含义 | `--ep 16` 推荐值 |
|------|-----------|------------------|
| `-dim` | model_dim, inter_dim | `6144,2048` |
| `-e` | 本地专家数 \(E_\ell\) | `16` |
| `-k` | 全局 topk | **`8`** |
| `-t` | buffer rows \(N\) | **`512`** |
| `--ep` | EP 规模 \(R\) | **`16`** |
| `--ep-id` | 本 rank id | `0`（或 `0..15`） |
| `-q 5` | per_128x128 fp8 | blockwise a8w8 |

理论预期（\(N=512, R=16, k=8\)）：recv≈209±11，pairs≈256，eff_k≈1.23（见 §3.4–3.5）。性能对标见 §3.6。

#### C2. 旧近似：无 `--ep`，用 `-k 1` 凑算力

无 `--ep` 时无法模拟全局路由稀疏性，只能用 **等效本地 topk≈1** 粗对齐：

```bash
python op_tests/test_moe_2stage.py \
  -q 5 -d bf16 -a silu \
  -dim 6144,2048 \
  -e 16 -k 1 -t 261 \
  --no-flydsl-csv --no-situv2
```

| 对齐目标 | `-t` | `-k` | 预期 pairs | 说明 |
|----------|-----:|-----:|-----------:|------|
| pairs（五层均值） | **261** | **1** | ~261 | 仅算力量级 |
| recv（layer3） | **221** | **1** | ~221 | 仅行数量级 |
| ❌ 错误 | 221 | **8** | **1768** | 无 `--ep` 时勿用全局 topk=8 |

**规则**：有 `--ep` 时用 **`-k 8`（全局 topk）**；无 `--ep` 时用 **`-k 1`（等效本地 topk）**。

---

## 5. 反推公式（构造单测用）

### 5.1 正推：已知 buffer 行数 N

```python
from math import comb

def ep_uniform_expect(N, E=256, R=16, k=8):
    n = E // R
    p0 = comb(E - n, k) / comb(E, k)
    p_hit = 1 - p0
    return {
        "p_hit": p_hit,
        "recv": N * p_hit,
        "pairs": N * k / R,
        "eff_k": (k / R) / p_hit,
        "std_recv": (N * p_hit * (1 - p_hit)) ** 0.5,
    }

# bs128 稳态: N=512
# → recv≈209±11, pairs≈256, eff_k≈1.23
```

### 5.2 反推：目标 recv 或 pairs

\[
N = \frac{\mathrm{recv}_\mathrm{target}}{P(X \ge 1)} \approx \frac{\mathrm{recv}_\mathrm{target}}{0.408}
\]

\[
N = \frac{\mathrm{pairs}_\mathrm{target}}{k/R} = \frac{\mathrm{pairs}_\mathrm{target}}{0.5}
\]

| 目标 | 反推 N | 说明 |
|------|-------:|------|
| recv=221 | **542** | 略高于 rows=512，因真实路由略偏本地 |
| pairs=288 | **576** | layer3 算力对应等效 token |

### 5.3 不同 EP 规模速查（均匀，k=8）

| R | \(P(X\ge1)\) | E[recv]/N | E[pairs]/N | eff_k |
|---|-------------:|----------:|-----------:|------:|
| 4 | 0.900 | 0.900 | 2.000 | 2.22 |
| 8 | 0.656 | 0.656 | 1.000 | 1.52 |
| **16** | **0.408** | **0.408** | **0.500** | **1.23** |
| 32 | 0.224 | 0.224 | 0.250 | 1.12 |

---

## 6. 推荐单测构造流程

```
1. 确定要对标的量：算力 → pairs；延迟形状 → recv；全 fidelity → dump 回放
2. 查表或 ep_uniform_expect(N) 算理论 recv/pairs
3. 选档位：
   - 全 fidelity → test_fmoe_vllm_dump.py
   - EP 复杂语义 → test_moe_ep.py
   - 快速 bench → test_moe_2stage.py --ep 16（推荐）
4. 跑完后核对：pairs/recv ∈ [1.1, 1.4]；recv ≈ N×0.408
```

### 6.1 示例：为 bs128 / layer3 构造 `test_moe_2stage` 用例

```bash
# 推荐：--ep 合成 EP，buffer/recv/pairs 与 dump 同量级
python op_tests/test_moe_2stage.py \
  -q 5 -d bf16 -a silu -dim 6144,2048 \
  -e 16 -k 8 -t 512 \
  --ep 16 --ep-id 0 \
  --no-flydsl-csv --no-situv2

# 分 stage 计时
python op_tests/test_moe_2stage.py \
  -q 5 -d bf16 -a silu -dim 6144,2048 \
  -e 16 -k 8 -t 512 \
  --ep 16 --ep-id 0 \
  --no-flydsl-csv --no-situv2 --kernel
```

预期输出（每次随机路由略有波动）：

```
recv ≈ 210, pairs ≈ 260, eff_k ≈ 1.2~1.3
us ≈ 260 µs（视 tuned config 走 1stage asm 或 2stage FlyDSL）
```

与 dump 对比：`python op_tests/test_fmoe_vllm_dump.py -b 128 -l 3`（见 §3.6）。

---

## 7. 常见误区

| 误区 | 正确理解 |
|------|----------|
| `bs=128` → `-t 128` | bs 是请求数；buffer rows=512，recv≈221 |
| dump topk=8 → `-k 8` 且无 `--ep` | 8 个 slot 全本地，算力高估 ~6×；应加 **`--ep 16`** |
| `-e 256` 表示全局专家 | `--ep > 1` 时 **`-e` 是本地专家数**；全局 = `e × ep` |
| `-t 221` 对齐 dump recv | `-t` 是 **buffer 容量**（512），recv 由路由自动算出（~210） |
| rows=512 行全部有效 | 仅前 `recv` 行有效；`num_local_tokens=recv` |
| recv > bs 不合理 | recv 是 token 行数，可远大于并发请求数（EP 汇聚 + MTP） |

---

## 8. 参考：bs 与 rows 换算（MTP3 + DP16 + EP16）

```
每 rank dispatch 前 token = (bs / 16) × (1 + K_mtp)     # K_mtp=3 → ×4
全局 FMOE buffer rows    = 每 rank token × 16           # mori layout
                         = bs × (1 + K_mtp)

bs128 → rows = 128 × 4 = 512
bs64  → rows = 64 × 4  = 256
```

---

## 9. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-09-10 | 初版：bs128 dump 实测 + 超几何均匀模型 + test_moe_2stage 同构指南 |
| 2026-09-10 | `test_moe_2stage.py` 新增 `--ep` / `--ep-id`；§4.3 更新为推荐构造方式；§3.6 性能对标 |
| 2026-09-10 | §3.7：EP16/EP8/TP8/TP16 单 rank 性能对比（全局 shape 6144×2048, E=256, N=512） |
