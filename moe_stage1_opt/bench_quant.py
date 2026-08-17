"""Micro-benchmark for the fused_moe per-tensor input quant path at 32k.

Measures the current two-kernel path against two cheap alternatives, plus the
cost of the zero-input guard, so the e2e budget numbers can be replaced with
measured ones.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.quant import (
    _per_tensor_amax_kernel,
    _quant_from_per_tensor_amax_kernel,
)
from aiter.ops.triton.quant import dynamic_per_tensor_quant_fp8_i8_nozero

import sys

# a2 (stage1 output) is token*topk x inter_dim*? -> pass the element count in.
if len(sys.argv) > 1 and sys.argv[1] == "a2":
    M, N = 32768 * 9, 192  # 56,623,104 elements
else:
    M, N = 32768, 4096  # 134,217,728 elements
FP8 = torch.float8_e4m3fnuz
DEV = "cuda"

x = torch.randn(M, N, dtype=torch.bfloat16, device=DEV)
x_flat = x.reshape(-1)
n_elements = x_flat.numel()
qx = torch.empty(M, N, dtype=FP8, device=DEV)
qx_flat = qx.reshape(-1)
scale = torch.empty(1, dtype=torch.float32, device=DEV)
DTYPE_MAX = torch.finfo(FP8).max

read_bf16 = n_elements * 2 / 1e9
write_fp8 = n_elements / 1e9


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000.0 / iters


def report(name, us, gb):
    print(f"{name:<52} {us:8.1f} us   {gb / (us * 1e-6) / 1e3:6.2f} TB/s")


print(f"shape {M}x{N} bf16 -> fp8   read {read_bf16:.3f} GB  write {write_fp8:.3f} GB\n")

# --- current production path -------------------------------------------------
amax_cur = torch.empty((n_elements + 1023) // 1024, dtype=torch.float32, device=DEV)
report(
    "A. 现状 dynamic_per_tensor_quant_..._nozero(bs=2048)",
    bench(lambda: dynamic_per_tensor_quant_fp8_i8_nozero(qx, x, scale, amax_cur)),
    read_bf16 * 2 + write_fp8,
)

# the two halves of A, separately
BS = 2048
n_blocks = triton.cdiv(n_elements, BS)
amax_bs = triton.next_power_of_2(n_blocks)
report(
    "   A1. _per_tensor_amax_kernel (只读)",
    bench(lambda: _per_tensor_amax_kernel[(n_blocks,)](x_flat, amax_cur, n_elements, BLOCK_SIZE=BS)),
    read_bf16,
)
report(
    f"   A2. _quant_from_..._kernel (AMAX_BLOCK_SIZE={amax_bs})",
    bench(
        lambda: _quant_from_per_tensor_amax_kernel[(n_blocks,)](
            qx_flat, x_flat, amax_cur, scale, n_elements, n_blocks,
            DTYPE_MAX=DTYPE_MAX, BLOCK_SIZE=BS, AMAX_BLOCK_SIZE=amax_bs,
        )
    ),
    read_bf16 + write_fp8,
)

# --- fix 1: bigger amax block  ->  fewer amax entries to re-reduce -----------
for bs in (8192, 32768, 65536):
    nb = triton.cdiv(n_elements, bs)
    abs_ = triton.next_power_of_2(nb)
    amax_b = torch.empty(nb, dtype=torch.float32, device=DEV)
    report(
        f"B. bs={bs:<6d} (n_blocks={nb}, AMAX_BLOCK_SIZE={abs_})",
        bench(
            lambda bs=bs, nb=nb, abs_=abs_, amax_b=amax_b: (
                _per_tensor_amax_kernel[(nb,)](x_flat, amax_b, n_elements, BLOCK_SIZE=bs),
                _quant_from_per_tensor_amax_kernel[(nb,)](
                    qx_flat, x_flat, amax_b, scale, n_elements, nb,
                    DTYPE_MAX=DTYPE_MAX, BLOCK_SIZE=bs, AMAX_BLOCK_SIZE=abs_,
                ),
            )
        ),
        read_bf16 * 2 + write_fp8,
    )

# --- fix 2: reduce the amax array once, then quant with a 1-element amax -----
amax_one = torch.empty(1, dtype=torch.float32, device=DEV)


def three_stage():
    _per_tensor_amax_kernel[(n_blocks,)](x_flat, amax_cur, n_elements, BLOCK_SIZE=BS)
    torch.amax(amax_cur[:n_blocks], dim=0, out=amax_one[0].reshape(()))
    _quant_from_per_tensor_amax_kernel[(n_blocks,)](
        qx_flat, x_flat, amax_one, scale, n_elements, 1,
        DTYPE_MAX=DTYPE_MAX, BLOCK_SIZE=BS, AMAX_BLOCK_SIZE=1,
    )


report("C. amax 先归约成 1 个数, 再 quant (AMAX_BLOCK_SIZE=1)", bench(three_stage), read_bf16 * 2 + write_fp8)

# --- the zero-input guard ----------------------------------------------------
q_u8 = qx.view(torch.uint8)


def guard():
    scale.clamp_(min=1e-12)
    q_u8.masked_fill_(q_u8.eq(0x80), 0)


report("D. _guard_zero_input (clamp_ + eq + masked_fill_)", bench(guard), write_fp8 * 4)

# --- reference ceiling: a pure bf16->fp8 cast with a known scale -------------
report("E. 参照上限 torch (x * r).to(fp8)", bench(lambda: torch.mul(x, 2.0, out=None).to(FP8)), read_bf16 * 2 + write_fp8 * 2)
