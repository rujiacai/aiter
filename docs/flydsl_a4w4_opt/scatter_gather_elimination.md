# 消除 stage1 scatter / stage2 gather —— sorted_a2 与 compact_a2

> 面向 FlyDSL a4w4（`QuantType.per_1x32`, fp4x2/fp4x2）MoE 2-stage。
> 两条路径均为**编译期开关，默认关闭**，不影响现有行为。
> 结论先行：功能已实现并验证正确，但在实测 shape 上**性能中性**（gather/read 不是 stage2 瓶颈）。作为"已验证、默认关"的可选特性落地。

---

## 1. 背景：为什么会有 scatter/gather

MoE 2-stage 的中间激活 `a2`（stage1 输出 = stage2 输入）默认按 **token\*topk 序**存放，即第 `t*topk+s` 行放 (token `t`, slot `s`) 的激活。

但 stage2 是 **grouped GEMM**：每个 expert 有独立权重 `w2[e]`，必须把"路由到同一 expert 的 token"凑成一个稠密 tile 连续计算。`moe_sorting` 把 `(t,s)` 按 expert 分组排好，产出 `sorted_token_ids`（简称排序序）。

于是"存储顺序（token\*topk）"与"遍历顺序（expert 排序序）"不一致，产生两处错位访问：

- **stage1 scatter 写**：kernel 按排序序算出一个 tile 的结果，却要把每行"散射"到 `t*topk+s` 的分散地址。
- **stage2 gather 读**：kernel 按排序序处理一个 block，却要从 `t*topk+s` 的分散地址"聚集"每行输入。

`sorted_token_ids[p]` 是 fused i32：低 24 位 = `t`，高 8 位 = `s`。默认路径就是靠它做 `p → (t,s) → t*topk+s` 的地址翻译——这就是 scatter/gather 的根源。

完整链路里其实有 4 处错位（首尾两处躲不掉）：

| 位置 | 操作 | 索引 | 能否消除 |
|---|---|---|---|
| stage1 输入 | gather 读 hidden | 按 `t` | 不能（hidden 是用户按 token 序给的）|
| stage1 输出 | scatter 写 a2 | 按 `t*topk+s` | **可以**（本特性）|
| stage2 输入 | gather 读 a2 | 按 `t*topk+s` | **可以**（本特性）|
| stage2 输出 | scatter 写 out | 按 `t`（atomic）| 不能（out 要按 token 序还给用户）|

本特性消除的是**中间那两处**（stage1 写 + stage2 读）。

---

## 2. 两种方案

核心思想一致：**让 stage1 写地址和 stage2 读地址都改成"块内连续"，并共用同一个映射**（写进去的行 stage2 一定读得回来）。

| | 默认 | **sorted_a2** | **compact_a2**（接入 fused_moe 的）|
|---|---|---|---|
| a2 物理行号 | `t*topk+s` | `sorted_pos`（排序序全局行号）| `compact_map[sorted_pos]`（紧凑序）|
| a2 值 buffer 大小 | `token*topk` | **padded 排序序长度**（膨胀）| `token*topk`（**无膨胀**）|
| 定位元数据 | 无 | 无（直接用 `bx_m+row_local`）| `compact_map`（前缀和数组，小）|
| 适用 | 现状 | 教学/参考实现 | 推荐（无膨胀）|

- **sorted_a2**：a2 直接按排序序存。写/读都用 `bx_m+row_local`，最简单，但 buffer 要按 padded 排序序长度分配 → decode/多 expert 场景膨胀严重（可达 10~30×）。
- **compact_a2**：a2 仍是 `token*topk` 大小，但物理行按"紧凑序"排（去掉每个 expert block 的 padding 空洞），通过 `compact_map` 定位。**无膨胀**，是最终接入 fused_moe 的方案。

---

## 3. compact_map 的构造（host 侧）

`aiter/fused_moe.py :: _build_compact_row_map`

```python
def _build_compact_row_map(sorted_ids, token_num, topk):
    fused = sorted_ids.to(torch.int64)
    t = fused & 0xFFFFFF
    s = (fused >> 24) & 0xFF
    valid = ((t < token_num) & (s < topk)).to(torch.int64)
    cm = torch.cumsum(valid, 0) - valid   # exclusive prefix sum
    return cm.clamp(max=token_num * topk - 1).to(torch.int32)
```

- 与 `sorted_ids` 等长（padded 排序序长度）。
- `compact_map[p]` = **排序序位置 p 之前的有效行个数**（exclusive 前缀和）= 紧凑 a2 里的物理行号。
- 原理：每个 expert 的有效行在排序序里本就连续（padding 只在 block 尾部），所以前缀和天然满足"**同一 block 内 `compact_map[p]` 连续递增**" → stage1 能连续写、stage2 能连续读。
- padding 行 `clamp` 到 `token*topk-1` 防越界（这些行的 GEMM 结果会被 epilogue mask，值无所谓）。

### 例子（block_size=4, topk=2, token=3）

| p | 解码 | valid | cm(exclusive) | 紧凑行 |
|---|---|---|---|---|
| 0 | (t0,s0) | 1 | 0 | expert A 行0 |
| 1 | (t2,s1) | 1 | 1 | expert A 行1 |
| 2 | pad | 0 | 2 | (借位，被mask) |
| 3 | pad | 0 | 2 | (借位) |
| 4 | (t1,s0) | 1 | 2 | expert B 行2 |
| 5 | (t0,s1) | 1 | 3 | expert B 行3 |
| 6 | pad | 0 | 4 | (借位) |
| 7 | pad | 0 | 4 | (借位) |

有效行被压成 0,1,2,3 连续密排；buffer 从 8 行压到 4 行（=token\*topk），padding 空洞被挤掉。

---

## 4. gating 与传参（fused_moe.py）

`fused_moe_2stages` 里的启用条件（`AITER_FLYDSL_COMPACT_A2=1` 且 a4w4 flydsl 双 stage 且非 fuse_quant）：

```python
_compact_a2 = (
    os.environ.get("AITER_FLYDSL_COMPACT_A2", "0") == "1"
    and stage1_func is _flydsl_stage1_wrapper
    and stage2_func is _flydsl_stage2_wrapper
    and quant_type == QuantType.per_1x32
    and w1.dtype == dtypes.fp4x2
    and q_dtype_a == dtypes.fp4x2
    and not metadata.fuse_quant
)
if _compact_a2:
    _compact_map = _build_compact_row_map(sorted_ids, token_num, topk)
    extra_stage1_args["compact_a2"] = True; extra_stage1_args["compact_map"] = _compact_map
    extra_stage2_args["compact_a2"] = True; extra_stage2_args["compact_map"] = _compact_map
```

**同一个 `_compact_map` 同时喂 stage1 和 stage2** —— 写/读地址一致的保证。

中间量化（gated 分支）：stage1 输出是"紧凑序 bf16"，用 torch 逐行量化成紧凑 fp4 值 + 紧凑-linear scale，再用 `_compact_scale_to_tt` 把 scale 换回 token\*topk 序喂给标准 `moe_mxfp4_sort`（值保持紧凑、供 stage2 直接连续读）。
> 注：这一步用 torch 量化是**测试脚手架**做法（trace 里会看到额外 aten 算子拖慢 e2e）；生产级需替换成 compact-aware 的 HIP 量化 kernel。

参数透传链：`_flydsl_stage1/2_wrapper → flydsl_moe_stage1/2 → compile_flydsl_moe_stage1/2 → compile_mixed_moe_gemm1/2`，并进入 lru_cache 的 `_cache_tag`（避免不同开关命中同一份编译产物）。

---

## 5. kernel 侧实现（mixed_moe_gemm_2stage.py）

### 5.1 新增指针参数 + buffer resource

两个 kernel 都加 `arg_compact_map: fx.Pointer`，并在开关打开时建 resource（`const_expr` = 编译期分支，关闭时零开销）：

```python
compact_map_rsrc = 1
if const_expr(compact_a2):
    _cm_rows = size_expert_ids_in * arith.constant(_sort_block_m, index=True)  # padded 长度
    _cm_nbytes = arith.index_cast(T.i32, _cm_rows * arith.constant(4, index=True))
    compact_map_rsrc = _ptr_buffer_resource(arg_compact_map, _cm_nbytes)
```

> `_cm_rows` 用 padded 长度是**正确且必须的**——它 sizing 的是"索引数组 compact_map 的描述符"（i32 小数组），不是值 buffer。stage2 会读 `compact_map[bx_m+row_local]`，最大到 `size_expert_ids_in * sort_block_m`，描述符必须覆盖到。
> 值 buffer（`x_rsrc`）在 compact 下走 `_x_rows = tokens_in * c_topk`（无膨胀），这才是大头。

### 5.2 stage1 消除 scatter 写（`precompute_row`）

```python
if const_expr(compact_a2):
    _cr = buffer_ops.buffer_load(compact_map_rsrc, row, vec_width=1, dtype=T.i32)
    ts_idx = arith.index_cast(ir.IndexType.get(), _cr)      # 写到 compact_map[row]
elif const_expr(sorted_a2):
    ts_idx = row                                             # 写到排序序行
else:
    t_idx = ...; s_idx = ...
    ts_idx = t_idx * topk + s_idx                            # 默认：散射到 t*topk+s
row_byte_base = out_base_idx + ts_idx * _out_row_stride
```

`(t,s)` 解码**仍保留**——`row_valid`（padding/越界判定）要用，只是不再用它算地址。

### 5.3 stage2 消除 gather 读（X-load 地址生成）

```python
sorted_row_i = bx_m + row_local
if const_expr(compact_a2):
    cr_i32 = buffer_ops.buffer_load(compact_map_rsrc, sorted_row_i, vec_width=1, dtype=T.i32)
    row_ts_idx = arith.index_cast(ir.IndexType.get(), cr_i32)   # 读 compact_map[sorted_row]
elif const_expr(sorted_a2):
    row_ts_idx = sorted_row_i                                   # 直接读排序序行
else:
    fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, ...)
    t_i32 = fused_i & mask24; s_i32 = fused_i >> 24
    row_ts_idx = t_safe * topk + s_safe                         # 默认：gather t*topk+s
x_row_base_div4.append(row_ts_idx * c_k_div4)
```

block 内 `compact_map[sorted_row_i]` 连续 → 连续块读（gather 消除）。

### 5.4 padding 行的正确性

compact 下 a2 保持 `token*topk` 大小，每个 expert 尾块的 padding 行会 borrow 到下一 expert 紧凑区开头（读到别人的数据），但这些行 `row_valid=false`，GEMM 结果在 epilogue 被 mask 不写出 → 无害。这也是 `(t,s)` 解码在 compact 下仍要保留的原因。

---

## 6. 涉及文件

| 文件 | 改动 |
|---|---|
| `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` | 两 kernel 加 `sorted_a2`/`compact_a2` 开关 + `arg_compact_map`；stage1 compact 写 / stage2 compact 读；值 buffer sizing 分支 |
| `aiter/ops/flydsl/moe_kernels.py` | compile 包装 + `flydsl_moe_stage1/2` + `_s1/s2_args_fp4` 透传 `compact_a2`/`compact_map` |
| `aiter/fused_moe.py` | gated compact 路径 + `_build_compact_row_map` / `_compact_scale_to_tt` + 两 wrapper 透传 |
| `aiter/ops/flydsl/test_flydsl_moe_a4w4.py` | `--sorted-a2` / `--compact-a2` 独立验证用例 |

---

## 7. 验证与用法

### 正确性（已验证）
- 独立测试：stage1 compact 写逐值 bit-exact（max|Δ|=0）；stage2 compact 读与 baseline 一致。
- fused_moe 端到端（7168/384/E384/k6, token=16384）：`cos_sim=0.999991`。
  （`checkAllclose failed` 是 fp4 大数值超 atol 的老误报，非精度问题。）

独立验证命令：
```bash
python aiter/ops/flydsl/test_flydsl_moe_a4w4.py -t 16 64 256 --block-m 32 --compact-a2 --stage stage1
```

fused_moe 端到端：
```bash
AITER_FLYDSL_COMPACT_A2=1 AITER_LOG_MORE=1 \
  python op_tests/test_moe_2stage.py -q 4 -dim 7168,384 -e 384 -k 6 --no-flydsl-csv -t 16384
```

### 性能（实测中性）
- gemm2 compact 读 vs 默认 gather ≈ **0.98–1.00×**（跨 shape 一致）。
- 原因：a2 输入读只占 gemm2 访存的 ~4%（输出写是它的 ~75 倍：`model_dim*2B` vs `inter_dim*0.5B`），且原 gather 是"按行大 burst"本就合并良好、被计算与输出写盖住。
- **互斥原理**：gather 惩罚大 ⟺ 行小（inter_dim 小）⟺ a2 读总量小（非瓶颈）；a2 读占大头 ⟺ 行大 ⟺ gather 天然合并。两头堵死，任何 shape 都逼近 1.0×。

### 结论
特性功能完整、正确、默认关。**在当前 shape 上不改善性能**（stage2 真正瓶颈是 topk 放大的输出写 + reduction readback，不在 a2 读）。作为已验证的可选实现保留；若未来出现 read-bound 的 MoE 形状可直接启用。
