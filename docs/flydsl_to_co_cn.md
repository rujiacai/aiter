# 把 FlyDSL kernel 导出成 `.co` 交付

> 目标：产出不依赖 FlyDSL、也不含内核源码的独立 AMDGPU code object，
> 让部署方只拿到二进制就能跑。
> 现有实现以 blockwise-fp8 MoE（`moe_gemm1_0` / `moe_gemm2_0`）为例。

---

## 0. 为什么能这么做

FlyDSL 的编译产物最终就是一个**普通的 AMDGPU code object**（ELF）。它被
`hipModuleLoadData` 加载后，和手写汇编内核没有任何区别——aiter 里已有 2900 多个
`.co` 就是这么用的。

所以"导出"本质上只有两件事：

1. **把 ELF 从 FlyDSL 的编译缓存里取出来**，连同启动所需的元数据（kernel 符号名、
   kernarg 大小、LDS、workgroup 大小）；
2. **写一个不认识 FlyDSL 的 C++ launcher**，按元数据把参数打包好丢给 `hipModuleLaunchKernel`。

导出物里不出现 FlyDSL 任何字样，运行时也不需要装 FlyDSL。

---

## 1. 全流程

```
                 ┌──────────────────────────────────────────┐
   tune          │ moe_blk_tuned.csv  (每 shape×token 一行)   │
   （可选）       │   block_m + kernelName1/2 → 隐含 tile      │
                 └───────────────┬──────────────────────────┘
                                 │ tiles_for()
                 ┌───────────────▼──────────────────────────┐
   export        │ hsa/flydsl_export.py                     │
                 │   编译 → 取 ELF → 写 .co + manifest.csv    │
                 └───────────────┬──────────────────────────┘
                                 │
                 ┌───────────────▼──────────────────────────┐
   runtime       │ hsa/{arch}/moe_blk/*.co                  │
                 │   moe_blk.py（查表+命名）                  │
                 │   moe_blk.cu（打包 kernarg + launch）      │
                 └───────────────┬──────────────────────────┘
                                 │
   verify        │ op_tests/.../test_moe_blk_co.py           │
```

四个环节各自独立，但**命名必须全程一致**，见 §5。

---

## 2. 导出器做了什么

`hsa/flydsl_export.py`，三步。

### 2.1 在隔离的缓存里编译

```python
_TMP_CACHE = tempfile.mkdtemp(prefix="flydsl_export_")
os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = _TMP_CACHE   # 必须在 import aiter 之前
```

> 重定向缓存是必需的，否则会读到开发者本地缓存里的旧产物，或者反过来污染它。
> 因为要在 `import aiter` 之前生效，这几行只能放在文件最顶部。

编译走的是 AOT 辅助函数而不是 `compile_flydsl_moe_stage1`：

```python
from aiter.aot.flydsl.moe import _precompile_to_cache
```

> **这一点很容易踩坑**：`compile_flydsl_moe_*` 只构造 jit 对象，
> **真正的 MLIR 编译要等到内核被真实参数调用时才发生**——而缓存文件也是那时才写。
> 直接调它拿不到任何产物。

### 2.2 从产物里抠出 ELF

编译产物是个 pickle，里面存着 MLIR 文本。ELF 以字符串字面量的形式嵌在 `gpu.binary` 里：

```python
blob = re.search(r'"((?:[^"\\]|\\.){512,})"', ir_text, re.S)
elf  = mlir_unescape(blob.group(1))
assert elf[:4] == b"\x7fELF"
```

> **`mlir_unescape` 不能用 Python 自带的解码。** MLIR 的 `\XX` 是**十六进制**，
> 而 Python 字符串字面量里 `\XX` 是**八进制**。直接用 `codecs.decode(s,'unicode_escape')`
> 会得到一堆错位的字节，且往往还能通过长度检查，只在最后加载时才报错。

同一段 MLIR 里还能读到启动元数据，**从二进制里读回来，不要手写假设**：

| 字段 | 来源 |
|---|---|
| `kernel_name` | `#gpu.kernel_metadata<"...">` |
| `arch` | `#rocdl.target<chip = "...">` |
| `lds_bytes` | `group_segment_fixed_size` |
| `workgroup_size` | `max_flat_workgroup_size` |
| `vgpr_count` / `sgpr_count` | 同名字段 |
| `kernarg_size` | 由函数签名累加：`ptr<1>` 8 字节、`i32` 4 字节 |

### 2.3 落盘

```
hsa/{arch}/moe_blk/
├── moe_blk_stage1_bf16_d6144x256_e256k8_t16x64x128_w2.co
├── moe_blk_stage1_..._w2_smooth.co
├── moe_blk_stage2_...co
└── manifest.csv          ← 每个 .co 的启动元数据
```

`manifest.csv` 不参与运行时，但它是**排查"加载成功却算错"的唯一依据**——
比如 launcher 的 kernarg 结构体和 `kernarg_size` 对不上时。

---

## 3. 用法

```bash
cd /data/aiter_main/aiter

# 先看矩阵，不编译
python hsa/flydsl_export.py --dry-run

# 全量导出（默认 4 个 shape，约 50 个 .co / 546 KiB / 5 分钟）
python hsa/flydsl_export.py

# 指定 shape 和 token 桶
python hsa/flydsl_export.py --shape 6144,256,256,8 --token-bucket 1 64 256
```

| 参数 | 说明 |
|---|---|
| `--shape MD,ID,E,K` | 可重复；这四个值**烘焙进二进制**，新 shape 必须重新导出 |
| `--token-bucket` | 代表性 token；tile 由 `tiles_for()` 从中推导，重复的 tile 自动去重 |
| `--smooth 0 1` | stage1 的 smooth_scale 变体（编译期特化）。stage2 无激活，自动跳过 |
| `--waves` | 默认取 `tiles_for` 的值 |
| `--dry-run` | 只列文件名 |

单个配置失败不会中断整批，最后统一列出，退出码非 0。

---

## 4. 运行时如何找到并启动

### 4.1 Python 侧：查表 + 拼名字

`aiter/ops/moe_blk.py`

```python
tiles_for(token, model_dim, inter_dim, expert, topk)  # 查 tuned CSV，未覆盖则启发式
co_name(stage, ..., tile_m, tile_n, tile_k, waves, out_dtype, smooth_scale)
have_co_for(token, ...)   # 二进制是否真的发布了；没有就退回 asm/CK
```

> `have_co_for` 是必需的。没有它，未导出的 shape 会去加载不存在的文件而**硬崩**。
> 它检查的目录必须来自 `AITER_ASM_DIR`（launcher 用的同一个），
> **不能从 `__file__` 自己推**——打包安装时路径是 `<root>/aiter_meta/hsa/`。

### 4.2 C++ 侧：打包 kernarg

`csrc/py_itfs_cu/moe_blk.cu`

```cpp
struct __attribute__((packed)) MoeBlkStage1Args { ... };
static_assert(sizeof(MoeBlkStage1Args) == 100, "stage1 kernarg layout drifted");
constexpr int MOE_BLK_BLOCK = 256;   // 与 manifest 的 workgroup_size 一致
```

> `static_assert` 对着 manifest 里的 `kernarg_size` 写死。内核签名一旦变动，
> **编译期就会失败**，而不是运行时给出一个错误的结果。这是最便宜的一道防线。

kernel object 按名字缓存在进程内（`hipModuleLoadData` 太慢，不能每次调用都做）。

---

## 5. 三个必须对齐的地方

导出、命名、启动这三者只要有一个不一致，症状都是**加载了错误的内核但不报错**。

**① 文件名**：导出器不自己拼名字，直接调运行时的 `co_name`：

```python
def co_filename(stage, cfg):
    from aiter.ops.moe_blk import co_name   # 单一来源
    return co_name(...)
```

**② tile 集合**：导出器枚举用的也是运行时的 `tiles_for`，所以能被请求的名字必然被导出过。

**③ kernarg 布局**：`static_assert` 锁死，见 §4.2。

> 交付分支上 `moe_blk.py` / `moe_blk.cu` 的注释里提到 FlyDSL 和导出器的地方要改写——
> 既是避免暴露来源，也因为那些文件在交付分支上根本不存在。

---

## 6. 验证

```bash
python op_tests/flydsl_tests/test_moe_blk_co.py     # 逐位对比 .co 与 FlyDSL 源路径
python op_tests/flydsl_tests/verify_moe_blk_co.py   # 全 shape×token 对 asm 的正确性+性能
```

两者职责不同，都要跑。

> ⚠️ **只跟 asm 比是不够的。** 它只能发现"`.co` 和它替代的内核不一致"，
> 发现不了"两者以同样的方式一起错"。举个真实的例子：给任一路径喂**未 shuffle**
> 的权重，两边都返回垃圾，但彼此吻合、cos 接近 1。
>
> 布局本身有疑问时，要拿 `torch_moe` + 量化前的 bf16 权重做独立参考，
> 预期 **cos ≈ 0.998**（fp8 量化误差）。

---

## 7. 已知约束

| 约束 | 说明 |
|---|---|
| shape 烘焙进二进制 | `model_dim/inter_dim/expert/topk` 都是编译期常量，新 shape 必须重新导出 |
| `smooth_scale` 是编译期的 | 两个变体各一个 `.co`；`swiglu_limit` 则是**运行时**参数，不用分裂二进制 |
| arch 绑定 | `.co` 只对导出时那颗 GPU 的 arch 有效，换 arch 要在目标机上重新导出 |
| 未导出的 shape | 由 `have_co_for` 拦下并退回 asm/CK，不会崩，但也没有加速 |

### `swiglu_limit` 为什么能做成运行时参数

`compile_moe_gemm1` 带 `@functools.lru_cache`，**所有入参都是特化维度**。
把 clamp 值留在签名里，`None/3.0/10.0` 就会各编一个二进制。
改成运行时参数后，内核恒定做 clamp，"不 clamp" 用 `+inf` 表示（恒等），
一个二进制覆盖所有取值。

> 新增可调项时先问一句：**它真的需要特化吗？** 能进 kernarg 就别进签名。

---

## 8. 换一个新内核族要做什么

1. 确认 `_precompile_to_cache` 支持该内核（否则照它加一个入口）
2. 定义命名函数（照 `co_name`），导出器和运行时共用
3. 写 launcher：从 manifest 抄 kernarg 布局，加 `static_assert`
4. 在 `optCompilerConfig.json` 注册 launcher 模块
5. 接进 dispatch，并加 `have_co_for` 式的存在性检查
6. 用独立参考（不是被替代的那条路径）验证正确性

`parse_artifact` / `mlir_unescape` 与内核无关，可直接复用。
