"""Bit-parity checks for the per-tensor quant changes.

1. Widening the amax tile must not change a single byte: max is associative.
2. With the in-kernel scale clamp, an all-zero input must give clean fp8 zeros
   without the host-side guard (that is what makes the guard removable).
"""

import torch

from aiter.ops.triton.quant import dynamic_per_tensor_quant_fp8_i8_nozero

FP8 = torch.float8_e4m3fnuz
DEV = "cuda"
M, N = 32768 * 9, 192  # the a2 shape at token=32768, topk=9, inter_dim=192


def quant(x, block_size):
    qx = torch.empty(x.shape, dtype=FP8, device=DEV)
    s = torch.empty(1, dtype=torch.float32, device=DEV)
    amax = torch.empty((x.numel() + 1023) // 1024, dtype=torch.float32, device=DEV)
    dynamic_per_tensor_quant_fp8_i8_nozero(qx, x, s, amax, block_size=block_size)
    return qx, s


def guard(qx, s):
    s.clamp_(min=1e-12)
    q_u8 = qx.view(torch.uint8)
    q_u8.masked_fill_(q_u8.eq(0x80), 0)


fail = 0

# --- 1. tiling parity, over a spread of magnitudes ---------------------------
for tag, x in (
    ("randn", torch.randn(M, N, dtype=torch.bfloat16, device=DEV)),
    ("randn*1e4", torch.randn(M, N, dtype=torch.bfloat16, device=DEV) * 1e4),
    ("randn*1e-4", torch.randn(M, N, dtype=torch.bfloat16, device=DEV) * 1e-4),
):
    q_ref, s_ref = quant(x, 2048)
    guard(q_ref, s_ref)  # current production behaviour
    q_new, s_new = quant(x, 32768)  # widened tile, no guard
    same_bytes = torch.equal(q_ref.view(torch.uint8), q_new.view(torch.uint8))
    same_scale = torch.equal(s_ref, s_new)
    print(
        f"[tile] {tag:<11} scale={s_ref.item():.6e} vs {s_new.item():.6e}  "
        f"bytes_identical={same_bytes}  scale_identical={same_scale}"
    )
    fail += not (same_bytes and same_scale)

# --- 2. all-zero input: no NaN without the guard -----------------------------
for bs in (2048, 32768):
    z = torch.zeros(4096, N, dtype=torch.bfloat16, device=DEV)
    qz, sz = quant(z, bs)  # deliberately no guard() call
    nan_bytes = int(qz.view(torch.uint8).eq(0x80).sum())
    nonzero = int(qz.view(torch.uint8).ne(0).sum())
    print(
        f"[zero] block_size={bs:<6} scale={sz.item():.3e}  "
        f"fp8_NaN(0x80)={nan_bytes}  nonzero_bytes={nonzero}"
    )
    fail += not (nan_bytes == 0 and nonzero == 0 and sz.item() > 0)

print("\nRESULT:", "PASS" if fail == 0 else f"FAIL ({fail} checks)")
