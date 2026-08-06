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

## 3. 权重数据流与布局：fp4 → shuffle → load → fp8 MFMA（为何一条 MFMA 横跨两个 scale block）

内核基座是 `moe_gemm_2stage.py`（与 a16w4 同一套 pipeline），用 `mfma_f32_16x16x32_fp8_fp8`。本节从头到尾把「权重从 HBM 的 fp4 到喂进 MFMA 的 fp8」这条链讲清，并推导出 straddle（一条 MFMA 横跨两个 scale block）的精确成因。

### 3.1 这条 MFMA 指令（fp8）
一条指令算一个小块：**C[16×16] += A[16×32] × B[32×16]**（M=16, N=16, **K=32**，fp8 输入 / f32 累加），由一个 wavefront 的 **64 个 lane 协同**完成。A、B 各 512 个 fp8，512/64 = **每个 lane 装 8 个 fp8**。

> **命名**：MFMA `C = A × B` 里 **A = 激活(M×K)、B = 权重(K×N)**。preshuffle（`shuffle_weight`）**只作用于 weight(B)**；激活 A 从不 preshuffle。

### 3.2 operand-K 与 octet
- **operand-K**：这一条 MFMA 指令内部的 K 索引 **0..31**（"逻辑 K"，**不是** HBM 里的原始 K）。MFMA 算 `Σ_{k=0..31} A[m,k]·B[k,n]`，A/B **共享**同一个 operand-K —— 同一个 k 必须是同一个原始 K 才乘得对（§6 "A+B 协同"的根源）。
- **octet**：64 lane 按 `octet = lane//16` 分 4 组（各 16 lane），每 lane 的 8 个 fp8 = operand-K 的一段 `[octet*8 : +8]`；`lane%16` = A 的行 M / B 的列 N。

| octet=lane//16 | 持有的 operand-K | | lane 例 | lane%16 | octet | B 侧含义 |
|---|---|---|---|---|---|---|
| 0 | `[0:8]`   | | 0  | 0 | 0 | B[K0–7, n=0]  |
| 1 | `[8:16]`  | | 16 | 0 | 1 | B[K8–15, n=0] |
| 2 | `[16:24]` | | 33 | 1 | 2 | B[K16–23, n=1]|
| 3 | `[24:32]` | | 48 | 0 | 3 | B[K24–31, n=0]|

代码里 `lane_div_16` = octet，`lane_mod_16` = M/N。

### 3.3 权重的四个阶段（表示 / 打包 / 排布）

| 阶段 | 表示 | 打包 | 一个 lane·一个 operand |
|---|---|---|---|
| ① HBM 原始 | e2m1 **fp4** | 2 码/字节 (fp4x2) | 8 码 = 4 字节 |
| ② host shuffle 后 | e2m1 **fp4** | 2 码/字节 | 同上（只重排字节） |
| ③ load 进寄存器 | e2m1 **fp4** | dwordx2=8 字节 | 8 字节 = 16 码 = r0(8)+r1(8) |
| ④ dequant 后 | **fp8** | 1 值/字节 | 8 fp8 = 8 字节(i64) |

- **① 原始 mxfp4**：权重 `(E, N, K)` 的 e2m1 码，K 连续；**per-32 E8M0 scale**：block `b` = 原始 K `[b*32 : (b+1)*32]`。
- **② host shuffle**（`prep_a8w4_w4` = `shuffle_weight(16,16)` + `pack`）：
  - `shuffle_weight(16,16)`：`view(E, N/16, 16, K/32, 2, 16)` → `permute` → 内存序 **`(n0, k0=K/32, kk=2, n=16, k_in=16)`**，原始 K = **`k0*32 + kk*16 + k_in`**。
  - `pack_int8_to_packed_int4`：每连续 8 码 → 4 字节（偶低/奇高，见 §7.1）。
- **③ load**（`make_preshuffle_b_layout`，`kpack_bytes=8`）：内核按 **64-K 单位** 布局 **`(n0, k0'=K/64, klane=4, n=16, kpack=8字节)`**。一次 wide load 取一个 lane 的 8 字节 = 16 码 = `r0`(前 4 字节) + `r1`(后 4 字节)。
- **④ dequant**（`unpack_b_w4a16_mxfp4_to_fp8*`，§4）：e2m1 码 → fp8 值，**逐元素、无损、不重排 K**（第 i 码 → 第 i 个 fp8）。每 lane 8 fp8 = 一个 K32 operand 的该 lane 数据 → 喂 MFMA。

**要点**：K 的归属（哪个原始 K 落到哪个 `(operand, octet, 位置)`）在 **② host shuffle** 就定死了；③ load 只搬运、④ dequant 只换数值精度，**都不重排**。所以 straddle 与 fp4→fp8 无关，是 ② 的布局决定的。

#### 3.3.1 ③ load 展开：layout 5 维 + stride 怎么算

`make_preshuffle_b_layout(kpack_bytes=8)` 构造的是一个 **5 维连续 layout**，告诉 buffer_load 每个 lane 该读哪几个字节：

```227:229:/data/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py
    layout_b = fx.make_layout(
        (n0_i32, c_k0_i32, klane_dim, 16, kpack_elems_static), stride_b
    )
```

| 维（外→内） | 值 | 含义 |
|---|---|---|
| `n0` | `N/16` | N 方向每 16 行一组 |
| `k0'` | `K/64` | K 方向**每 64 个一步**（一个 K64 micro-step）|
| `klane` | `4` | 组内 4 个 klane = 4 个 **octet**（lane//16），每 octet 管 16 K |
| `n` | `16` | 组内 16 行 N（lane%16）|
| `kpack` | `8 字节` | 一个 lane 持有 **8 字节 = 16 nibble = 16 个 e2m1 码** |

**stride = 「该维索引 +1 时，线性字节地址跳多少字节」**。因为字节连续无空洞，铁律是「**外层维 stride = 它内部所有维 size 连乘**」，于是从最内维往外累乘：

```210:216:/data/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py
        c64 = fx.Index(64)
        c4 = fx.Index(4)
        c_k0 = c_k_bytes // c64
        klane_dim = 4
        stride_klane = c16 * stride_nlane
        stride_k0 = c4 * stride_klane
        stride_n0 = c_k0 * stride_k0
```

```
维:      kpack(8)   n(16)      klane(4)     k0'(K/64)      n0(N/16)
stride:    1     →    8    →     128     →     512      →  (K/64)·512
累乘:      起点  ×kpack=8  ×n=16→128   ×klane=4→512   ×k0'
含义:  连续字节  跳1个lane  跳16行×8B   跳4个octet     跳整个K维
              的kpack    (一个octet)  (一个64-K组)
```

- `kpack` stride=1：一个 lane 的 8 字节相邻。
- `n`   stride=8：走下一行 N，跳过上一行那个 lane 的整 8 字节 kpack → 16 行占 `16×8=128` B。
- `klane` stride=128：走下一个 octet，跳过里面整个 `n×kpack` 块（`16×8=128`）→ 4 octet 占 `4×128=512` B。
- `k0'` stride=512：走下一个 64-K 组，跳过里面整个 `klane×n×kpack` 块（`4×128=512`）。
- `n0`  stride=`(K/64)·512`：走下一个 N 组，跳过它覆盖的全部 K。

**校验一个 `(n0,k0')` tile**：`4(klane)×16(n)×8(kpack) = 512 B = 1024 nibble`，覆盖 `16 行 N × 64 K = 1024` 码 ✓（无空洞、无重叠）。

**地址例子**：`n0=0, k0'=1, klane=2, n=3, kpack=0` → `addr = 1·512 + 2·128 + 3·8 = 792` B；从此起一条 `dwordx2` 读 8 字节 = 该 lane 的 16 个码。

**wide load**（`load_b_pack_k32_pair_raw`）：一条 `dwordx2` 读 8 字节 → bitcast 成 `vec(2,i32)`：

```761:764:/data/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py
    b_i32x2 = vector.bitcast(T.vec(2, T.i32), b8)
    r0 = vector.extract(b_i32x2, static_position=[0], dynamic_position=[])
    r1 = vector.extract(b_i32x2, static_position=[1], dynamic_position=[])
    return r0, r1
```

- `r0` = 前 4 字节 = 8 码 = 第 1 个 K32 operand 该 lane 的数据
- `r1` = 后 4 字节 = 8 码 = 第 2 个 K32 operand 该 lane 的数据

**为什么 load 单位是 64-K（而非 32-K）**：纯访存指令数优化——一条 `dwordx2` 一次拉两个 K32 operand，B-load 指令数减半（旧版是两条 `dword`）。unpack 的 ALU 基本被 MFMA 掩盖，访存指令数才是 a8w4 Phase-1 的真实瓶颈。这个 64-K vs 32-K（scale/shuffle）的错配，正是 §3.4 straddle 的根源。

#### 3.3.2 ④ dequant 展开：e2m1 码 → fp8（默认 perm-LUT 路径）

**默认路径**（`AITER_A8W4_PERMLUT=1`，见 §4/§8）走 `unpack_b_w4a16_mxfp4_to_fp8_permlut`。它把一个 `packed32`（8 码）变成一个 `i64`（8 fp8）= 一个 K32 operand 该 lane 的 8 元素，**每个 operand 只花 9 条 VALU**：

```657:660:/data/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py
    fe = _e2m1x4_from_packed_to_fp8x4_permlut(packed32, 0, arith, vector,
                                              ratios=ratios_even)
    fo = _e2m1x4_from_packed_to_fp8x4_permlut(packed32, 1, arith, vector,
                                              ratios=ratios_odd)
```

注意**没有独立的「拆 nibble」步骤**：两个 nibble 组（`nib=0` 低半→前 4 fp8、`nib=1` 高半→后 4 fp8）都直接从 `packed32` 里取，省掉了 `& 0x0F0F0F0F` 提取。

**每 4 码 → 4 fp8** `_e2m1x4_from_packed_to_fp8x4_permlut`：**1 次查表 + 1 次贴符号 +（可选）scheme B fold**。

```634:646:/data/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py
    lut_lo = fx.Int32(_A8W4_FP8_LUT[0])  # codes 0..3
    lut_hi = fx.Int32(_A8W4_FP8_LUT[1])  # codes 4..7
    p = fx.Int32(packed32)
    sh = 4 * int(nib)
    base = p if sh == 0 else (p >> fx.Int32(sh))
    sel = base & fx.Int32(0x07070707)
    mag = fx.Int32(rocdl.perm_b32(lut_hi, lut_lo, sel))
    # sign = code bit3 -> fp8 bit7; for the high nibble it is already in place.
    if sh == 0:
        sign = (p & fx.Int32(0x08080808)) << fx.Int32(4)
    else:
        sign = p & fx.Int32(0x80808080)
    fp8x4 = mag | sign
```

##### 展开：为什么 8 项 LUT + 符号位就够（省掉一半 perm）

**先记住 `perm_b32(A, B, sel)` 的语义**（AMD 字节重排指令）：把低位操作数 `B` 的 4 字节 + 高位操作数 `A` 的 4 字节拼成一个 **8 字节池** `pool[0..7]`（`pool[0..3]=B`、`pool[4..7]=A`）；输出 4 字节，**第 i 个输出字节 = `pool[sel 的第 i 个字节]`**（sel 每字节是 0–7 的下标）。即「按下标从 8 字节里挑 4 个」。

一条 perm 的池只有 8 字节，而 e2m1 码有 16 个 —— 看上去必须查两次再按 bit3 合并（那就是 3 条 perm）。但把码表摊开就会发现**上半区就是下半区置上符号位**：

```
码 0–7  (低半区，幅值):  00, 38, 40, 44, 48, 4c, 50, 54
码 8–15 (高半区，负值):  00, b8, c0, c4, c8, cc, d0, d4
                         ↑    ↑
                    例外!  0x38|0x80 = 0xb8 ✓  0x40|0x80 = 0xc0 ✓ …
```

除了**码 8**，其余每一项都严格满足 `LUT[c+8] == LUT[c] | 0x80`。所以只要能处理掉码 8，就可以「查 8 项拿幅值 + 把 bit3 搬到 bit7 当符号」，**一条 perm 就够**。

**码 8 的例外**：它是 `-0.0`，而 e4m3fnuz 的 `0x80` 是 NaN，所以码表里写的是 `0x00` 而非 `0x80`。解法在 host 端：`_mxfp4_codes_i8` 把码 8 归一成码 0（两者数值都是 `0.0`，严格等价，且只在 prep 时做一次、零运行时成本），例外就消失了。

**具体走一遍**（设低 nibble 的 4 个码 = `3, 10, 1, 15`，期望 fp8 = `44, c0, 38, d4`）：

```
① sel = p & 0x07             码低 3 位 → 池内下标（不需要先 & 0x0F）
   3→3, 10→2, 1→1, 15→7      sel = [3,2,1,7]

② mag = perm(lut_hi, lut_lo, sel)     一次查 8 项幅值表
   池 = [00,38,40,44, 48,4c,50,54]    (低4=lut_lo=码0-3, 高4=lut_hi=码4-7)
   pool[3,2,1,7] → mag = [44, 40, 38, 54]

③ sign = (p & 0x08) << 4              码 bit3 → fp8 bit7
   码 3,10,1,15 的 bit3 = 0,1,0,1  →  sign = [00, 80, 00, 80]

④ fp8x4 = mag | sign
   [44, 40|80, 38, 54|80] → [44, c0, 38, d4]   ✓ 与期望一致
```

高 nibble（`nib=1`）只差一点：`sel` 要先 `p >> 4` 再取低 3 位，而符号位**已经天然落在 bit7**（码 bit3 = 字节 bit7），所以直接 `p & 0x80808080` 即可，连移位都省了。

**代价与收益**：整条路径从「3 perm + blend 构造 + 先拆 nibble」的 17 条 VALU 降到 **9 条**。因为 a8w4 是 VALU-bound（tile_m=32 时 VALU 占 72.8% SIMD 周期，MFMA 只占 23.8%），这是直接的端到端收益。

然后按 `ratios` 分两种：

```645:646:/data/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py
    if ratios is None:
        return fp8x4
    return _fold_fp8x4_by_ratios(fp8x4, ratios, arith, vector)
```

- `ratios is None`（aligned / no-fold）：**直接返回裸 `fp8x4`**，scale 走 post-MFMA。
- `ratios` 非空（默认 fold 路径，straddle）：**scheme B** —— `fp8 → f32`（`cvt_pk_f32_fp8`）→ `×ratio` → 归一 -0 → `cvt_pk_fp8_f32` 重打包。只把「码→数值」前端换成 LUT，后端仍复用可靠的 f32 ratio-fold。

**拼 i64** `_pack_i32_pair_to_i64`：`fe`(前4) + `fo`(后4) → i64（8 fp8）直接喂 MFMA。

**为什么"逐元素、无损、不重排 K"**：

- **不重排**：第 `i` 码 → 第 `i` fp8，顺序原样；K 归属仍由 ② 决定。
- **无损**：`_A8W4_FP8_LUT` 是对全部 16 个 e2m1 码验证过 **byte-exact** 的 e4m3fnuz 码表（e2m1 尾数只有 `{0,1}` bit，在 e4m3fnuz normal range 内可精确表示）；「查低 8 项 + 贴符号」与查完整 16 项等价（唯一例外码 8 已在 host 端折成码 0，见上）。
- **逐元素**：8 码各自独立查表转换。

**`ratios_even/odd`（fold 在此注入）**：默认 fold 路径传入 §5 的 per-element `2^(exp-base)` 因子（straddle 时把 A/B 两块归一到公共 base，**顺带在 dequant 乘掉、不额外加 pass**）；aligned 路径传 `None`（出裸 fp8、scale 走 post-MFMA）。

**备选路径**（非默认，见 §8 env）：`AITER_A8W4_PERMLUT=0` → `unpack_b_w4a16_mxfp4_to_fp8`（`_e2m1x4_in_i32_to_fp8x4_i32` 位构造 e2m1→bf16→f32→×ratio→`cvt_pk_fp8`）；`AITER_A8W4_BITFOLD=1`（override）→ `unpack_b_w4a16_mxfp4_to_fp8_bitfold`（纯整数，ratio 折进 fp8 指数、无 cvt）。

### 3.4 为什么一条 MFMA 横跨两个 scale block（精确推导）

矛盾点：scale block、MFMA operand 都是 32-K，为什么还跨两块？—— 因为 **② shuffle 的分块单位是 32-K（`k0=K/32`，块内再 `kk=2` 分成 2×16），而 ③ load 的单位是 64-K（`k0'=K/64`，`klane=4`）**，两者错配。把二者的线性 K 索引对齐（`load 的 k0'×4+klane` = `shuffle 的 k0×2+kk`）展开：

| load 的 klane(=octet) | = shuffle `(k0, kk)` | 原始 K 段（相对 `64k0'`）| 属 scale block |
|---|---|---|---|
| 0 | `(2k0', 0)`   | `[0:16]`  | **block 2k0'  (A)** |
| 1 | `(2k0', 1)`   | `[16:32]` | **block 2k0'  (A)** |
| 2 | `(2k0'+1, 0)` | `[32:48]` | **block 2k0'+1 (B)** |
| 3 | `(2k0'+1, 1)` | `[48:64]` | **block 2k0'+1 (B)** |

（推导：原始 K = `k0*32+kk*16+k_in`。klane0=`(2k0',0)`→`64k0'+[0:16]`；klane2=`(2k0'+1,0)`→`64k0'+32+[0:16]` …）

结论：**内核 64-K 单位里，`klane{0,1}` 落在 scale block A、`klane{2,3}` 落在 block B**（`derive_mapping.py` 打 marker 实测一致）。而 klane = octet，**一条 K32 MFMA operand 要用满全部 4 个 octet**（operand-K 0–31）→ 它同时碰到 block A(octet 0,1) 和 block B(octet 2,3)：

```
r0 这一条 MFMA 的 32 个 operand-K：
  octet 0,1 (operand-K 0–15)  → block A 的 K（scale sA）
  octet 2,3 (operand-K 16–31) → block B 的 K（scale sB）
```

### 3.5 后果 → fold / aligned
MFMA 把这 32 项累加成**一个数**，post-MFMA 只能乘**一个** scale；但里面前 16 属 A(sA)、后 16 属 B(sB) → 摆不平：
- **§5 fold**（`shuffle_weight(16,16)`，block 分界在 **octet 轴**）：dequant 时先把 A/B 的值各按 `ratio=2^(exp-base)` 归一到公共 base，再统一乘 `2^base`。
- **§6 aligned**（`shuffle_weight_NK(16,32)`，把 block 分界挪到 **r0/r1 轴**：r0=整块 A、r1=整块 B）：一条 MFMA = 一个 32-block → 各乘各的 scale，**免 fold**。

---

## 4. 核心机制一：e2m1 → fp8 unpack（三条实现路径）

三条路径可通过环境变量切换，默认 **perm-LUT**：

| 路径 | 环境变量 | 做法 | 相对开销 |
|---|---|---|---|
| f32 construct | `AITER_A8W4_PERMLUT=0` | e2m1→bf16 位构造→f32→(×ratio)→`cvt_pk_fp8` | 基线 |
| **perm-LUT（默认）** | `AITER_A8W4_PERMLUT=1` | 1× `v_perm_b32` 查 8 项幅值 LUT + 贴符号位 | **每 operand 17→9 条 VALU** |
| bitfold | `AITER_A8W4_BITFOLD=1` | 纯整数位构造 e2m1→fp8，ratio 折进指数 | 无 f32 往返 |

**perm-LUT 核心**：e2m1 码表的上半区恰好等于下半区置上符号位，所以只需 **1 条 `v_perm_b32` 查 8 项幅值表**，再把码 bit3 搬到 fp8 bit7 当符号；nibble 也直接从 packed dword 取，不必先 `& 0x0F`。替代每 nibble ~15 条整数位构造（完整推导、码 8 的例外处理、逐码走一遍见 §3.3.2「展开：为什么 8 项 LUT + 符号位就够」）：

```634:646:/data/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py
    lut_lo = fx.Int32(_A8W4_FP8_LUT[0])  # codes 0..3
    lut_hi = fx.Int32(_A8W4_FP8_LUT[1])  # codes 4..7
    p = fx.Int32(packed32)
    sh = 4 * int(nib)
    base = p if sh == 0 else (p >> fx.Int32(sh))
    sel = base & fx.Int32(0x07070707)
    mag = fx.Int32(rocdl.perm_b32(lut_hi, lut_lo, sel))
    # sign = code bit3 -> fp8 bit7; for the high nibble it is already in place.
    if sh == 0:
        sign = (p & fx.Int32(0x08080808)) << fx.Int32(4)
    else:
        sign = p & fx.Int32(0x80808080)
    fp8x4 = mag | sign
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

### 7.1 weight 怎么 shuffle（`prep_a8w4_w4` = `prep_a16w4_weight`）

```52:63:/data/aiter/aiter/ops/flydsl/moe_kernels.py
def prep_a16w4_weight(wq_fp4x2, N, K):
    # ...
    codes = _mxfp4_codes_i8(wq_fp4x2, N, K)
    shuf = pack_int8_to_packed_int4(shuffle_weight(codes.view(dtypes.i8), (16, 16)))
    return shuf.view(E, N, K // 2).view(dtypes.fp4x2)
```

**总览**（全程搬"码 index"，不碰数值；形状标注在每一步右侧）：

```
  wq_fp4x2 (E, N, K/2)  fp4x2        每字节 = [ 高4位 hi | 低4位 lo ] = [ K_odd | K_even ]
      │
      │ ① _mxfp4_codes_i8            拆码：低半→偶K位、高半→奇K位（low nibble first）
      ▼
  codes   (E, N, K)  int8            1 码/字节，值 0–15（是"码 index"，不是数值/不是 fp8）
      │
      │ ② shuffle_weight(16,16)      view+permute，按【单个码】重排到 MFMA 布局
      ▼                              内存序 (n0=N/16, k0=K/32, kk=2, n=16, k_in=16)
  shuf    (E, N, K)  int8            每个 (n0,k0) tile = 16 行 N × 32 连续 K = 一个 K32 operand
      │
      │ ③ pack_int8_to_packed_int4   每 8 码打回 4 字节（间隔-4 配对，见下图）
      ▼
  out     (E, N, K/2)  fp4x2         2 码/字节，正是核内 load + unpack（§4）期望的字节序
```

输入 `wq_fp4x2` `(E, N, K/2)`（每字节 2 个 e2m1 码），**三步**：

1. **拆码** `_mxfp4_codes_i8`：fp4x2 → e2m1 codes `int8 (E,N,K)`，低半字节在前（`codes[...,0::2]`=低 4 位、`[...,1::2]`=高 4 位）。此时 1 码占 1 字节，值 0–15。
2. **MFMA preshuffle** `shuffle_weight(codes, (16,16))`（use_int4 路径，BK=32）：把 `(N,K)` 重排为
   `(E, N/16, K/32, klane=?, n=16, kPerLane)` —— 每个 `(n0=N/16, k0=K/32)` tile = **16 行 N × 32 连续 K**，正好是一个 K32 MFMA operand 的数据摆放（lane 读连续的 K 段）。这一步把权重摆成 kernel 期望的 register/LDS 布局。
3. **重打包成 4-bit** `pack_int8_to_packed_int4`：每连续 8 码 `[v0..v7]` → 4 字节 `b_i = v_i | (v_{i+4}<<4)`。即一个 lane 在一个 K32 operand 上持有的 8 个码，**偶数码进低半字节（对应前 4 个 fp8）、奇数码进高半字节（对应后 4 个 fp8）** —— 正是核内 unpack（perm-LUT / fold，§4）读取的字节布局。

**第 3 步「间隔-4 打包」示意**（一个 lane 在一个 K32 operand 上持有的 8 个码 `[v0..v7]`）：

```
  拆码后 8 个 int8 码:   v0  v1  v2  v3   v4  v5  v6  v7
                        └───前 4 个───┘   └───后 4 个───┘
                              │  │  │  │     │   │   │   │
                              │  │  │  └──┐  │   │   │   │   (v_i 配 v_{i+4})
                              ▼  ▼  ▼     ▼  ▼   ▼   ▼   ▼
  pack 成 4 字节:   b0 = v0 | (v4<<4)     低半:v0  高半:v4
                   b1 = v1 | (v5<<4)     低半:v1  高半:v5
                   b2 = v2 | (v6<<4)     低半:v2  高半:v6
                   b3 = v3 | (v7<<4)     低半:v3  高半:v7

  核内 unpack（逆）: even = b & 0x0F0F0F0F        → [v0 v1 v2 v3]（前 4 个 fp8）
                    odd  = (b >> 4) & 0x0F0F0F0F  → [v4 v5 v6 v7]（后 4 个 fp8）
```

> 第 1 步「拆码」的字节例子：某字节 `0x5A` → 低 4 位 `0xA` = 码 10（偶数 K 位）、高 4 位 `0x5` = 码 5（奇数 K 位）。"low nibble first" = 低半字节对应更小的 K 索引。

**为什么走「拆码 → shuffle → 再打包」这条弯路（而不是直接对 `fp4x2` shuffle）？**

- **`shuffle_weight` 只能按"元素（字节）"搬**（本质是 `view + permute`）。打包的 `fp4x2` 是 **2 码/字节**，一个字节里两个 e2m1 码被"绑"在一起，`permute` 拆不开、没法把单个 4-bit 码搬到不同的 `(klane, 位置)`。所以先 **拆成"1 码 1 字节"的 int8**（`_mxfp4_codes_i8`）让每个码可独立寻址、按码重排；重排完再 `pack` **压回 4-bit**（恢复 0.5 B/elem 紧凑存储 + 摆成内核 unpack 期望的字节序）。
- **全程搬的是"码 index (0–15)"，不是数值**：中间的 int8 只是"码的可搬运形态"，**不是 fp8**；e2m1 码 → fp8 值的 dequant 在**内核 unpack（§4）**才做，host prep 完全不碰数值。
- 之所以能直接复用这套 `shuffle_weight + pack_int8_to_packed_int4`：mxfp4 与 int4（W4A16）的**字节布局完全一样**，只是码值含义不同（e2m1 vs 有符号 int4）—— 见 `prep_a16w4_weight` 的 docstring「Byte layout is identical to the int4_bf16 path; only the code values are e2m1」。

> aligned 版（§6）把第 2 步换成 `shuffle_weight_NK(16,32)`，让一个 K32 operand 恰好对齐**单个 32-block**（而非 (16,16) 的横跨 2 block）。

### 7.2 scale 怎么 shuffle（`prep_a16w4_scale`）

```66:74:/data/aiter/aiter/ops/flydsl/moe_kernels.py
def prep_a16w4_scale(e8m0_scale, N, K, scale_mul=1.0):
    # ...
    scale_f32 = torch.pow(2.0, ws_u8.float() - 127.0) * scale_mul
    scale_bf16 = scale_f32.permute(0, 2, 1).contiguous().to(torch.bfloat16)  # (E,K/32,N)
    return shuffle_scale_for_int4(scale_bf16, group_size=32).view(-1).contiguous()
```

输入 `e8m0_scale` `(E, N, K/32)` uint8（每个 `(N, 32-K-block)` 一个 E8M0 指数），**四步**：

1. **解码** E8M0 → f32：`2^(u-127)`（E8M0 是纯 2 的幂指数，解码无损）。
2. **转置** `(E,N,G) → (E,G,N)`（`G=K/32`），转 bf16。
3. **打包** `shuffle_scale_for_int4`（bf16 分支）：`(E,G,N) → view(E,G/2,2,N) → permute → (E, G/2, N, 2)` —— **同一 N 位置、相邻两个 K-block `(g, g+1)` 打进一个 dword**（末维 `2` = 2 个 bf16 = 1 dword）。
4. **flatten** 成 1D contiguous。

**总览**（`G = K/32` = 每行 N 的 per-32 K-block 数）：

```
  e8m0_scale (E, N, G)  uint8         每个 (N, 32-K-block) 一个 E8M0 指数 u
      │
      │ ① 解码  2^(u-127)              E8M0 是纯 2 的幂，解码无损
      ▼
  scale_f32  (E, N, G)  f32
      │
      │ ② 转置 (E,N,G)→(E,G,N) + →bf16  让 K-block 维在前，对齐 kernel 访问序
      ▼
  scale_bf16 (E, G, N)  bf16
      │
      │ ③ shuffle_scale_for_int4       view(E,G/2,2,N)→permute→(E, G/2, N, 2)
      ▼                                末维 2 = 相邻两 K-block (g,g+1) = 1 dword
  packed     (E, G/2, N, 2)  bf16
      │
      │ ④ flatten → 1D contiguous
      ▼
  out        (E * G/2 * N * 2,)  bf16
```

**第 ③ 步「相邻两块打进一个 dword」示意**（固定某个 N 位置，沿 K-block 方向）：

```
  转置后 (沿 g):   g0   g1   g2   g3   g4   g5  ...   每个 = 该 N 行、一个 32-K-block 的 bf16 scale
                  └──┬──┘   └──┬──┘   └──┬──┘
                     ▼         ▼         ▼
  打成 dword:     [g0|g1]   [g2|g3]   [g4|g5]        末维 2：低 16 位 = 偶块 g、高 16 位 = 奇块 g+1

  核内一次 dword load 拿到 [g|g+1]：
      extract_bf16_scale(sc, 0) → scA = block g
      extract_bf16_scale(sc, 1) → scB = block g+1
```

这个「`(g,g+1)` 一个 dword」正是内核里 `extract_bf16_scale(sc, 0)`/`(sc, 1)` 读到的 `scA`(block g) / `scB`(block g+1) —— 一次 dword load 拿到相邻两块的 per-32 scale，喂给 fold 的 ratio-fold（fold 路径 §5）或直接做 per-operand 后乘（aligned / mxfp8 路径 §6）。

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
| | `_e2m1x4_from_packed_to_fp8x4_permlut` (615) | **perm-LUT unpack（默认，9 VALU/operand）** |
| | `_e2m1_code_to_fp8_byte_fold` (791) | bitfold 纯整数 unpack |
| | `make_aligned_b_layout` (681) / `load_b_operand_aligned` (712) | aligned B 布局/加载 |
| | `shuffle_weight_NK`（`shuffle.py:218`）| aligned 权重 preshuffle |
| `moe_gemm_2stage.py` | `_mxfp4_fp8_fold_operands` (75) | fold（unpack + ratio-fold + sc_out）|
| | `lds_load_packs_k64_aligned` (1147/3288) | **aligned 激活 loader（2×8B）** |
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
