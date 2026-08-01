# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness for mla_prefill_v4_bf16.

The shape list is built around what actually varies in V4 prefill: whether a
region is empty, whether the tile is partial, whether the rows carry `-1`
sentinels, and how the prefix/extend split falls relative to the kv tile
boundary. That last one matters here specifically because the kernel walks the
two regions as one virtual row rather than as two loops, so a tile that
straddles the boundary has to pull its rows from both buffers.

Every head count that maps to a distinct head-tile count is covered too. That is
not padding: a gap in the launcher's dispatch runs a kernel for fewer head tiles
than the grid was sized for, which leaves most heads unwritten without failing
the launch.
"""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops.attention import mla_prefill_v4_bf16, mla_prefill_v4_bf16_supported

D = 512

pytestmark = pytest.mark.skipif(
    not mla_prefill_v4_bf16_supported(dtypes.bf16, dtypes.bf16, 16, D),
    reason="mla_prefill_v4_bf16 is gfx942-only",
)


def build(N, H, p_len, e_len, sentinel=0.0, ragged=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = max(p_len * 4, 64)
    tokens = max(e_len * 4, 64)
    q = torch.randn(N, H, D, generator=g, device="cuda", dtype=dtypes.bf16) * 0.25
    unified_kv = torch.randn(pages, D, generator=g, device="cuda", dtype=dtypes.bf16) * 0.25
    kv = torch.randn(tokens, D, generator=g, device="cuda", dtype=dtypes.bf16) * 0.25
    attn_sink = torch.randn(H, generator=g, device="cuda", dtype=dtypes.fp32)

    def csr(per_len, limit):
        lens = []
        for _ in range(N):
            n = per_len
            if ragged and per_len > 0:
                n = int(torch.randint(0, per_len + 1, (1,), generator=g, device="cuda"))
            lens.append(n)
        indptr = torch.tensor([0] + lens, device="cuda", dtype=torch.int32).cumsum(0).int()
        nnz = int(indptr[-1])
        if nnz == 0:
            return torch.zeros(0, device="cuda", dtype=torch.int32), indptr
        idx = torch.randint(0, limit, (nnz,), generator=g, device="cuda").int()
        if sentinel > 0:
            drop = torch.rand(nnz, generator=g, device="cuda") < sentinel
            idx = torch.where(drop, torch.full_like(idx, -1), idx)
        return idx, indptr

    ip, pp = csr(p_len, pages)
    ie, pe = csr(e_len, tokens)
    return q, unified_kv, ip, pp, kv, ie, pe, attn_sink


def ref(q, unified_kv, ip, pp, kv, ie, pe, sink, scale):
    """fp32 reference: each token reads its own two CSR rows."""
    N, H, _ = q.shape
    out = torch.empty(N, H, D, dtype=q.dtype, device=q.device)
    qf, sinkf = q.float(), sink.float()
    for t in range(N):
        rows = []
        for indices, indptr, src in ((ip, pp, unified_kv), (ie, pe, kv)):
            base, end = int(indptr[t]), int(indptr[t + 1])
            if end <= base:
                continue
            idx = indices[base:end].long()
            idx = idx[idx >= 0]
            if idx.numel():
                rows.append(src.index_select(0, idx).float())
        if not rows:
            out[t] = 0
            continue
        kvt = torch.cat(rows, dim=0)
        scores = (qf[t] @ kvt.t()) * scale
        m = torch.maximum(scores.max(dim=-1).values, sinkf)
        p = torch.exp(scores - m[:, None])
        denom = p.sum(dim=-1) + torch.exp(sinkf - m)
        out[t] = ((p @ kvt) / denom[:, None]).to(q.dtype)
    return out


CASES = [
    # (name, N, H, p_len, e_len, sentinel, ragged)
    ("both regions", 8, 16, 64, 32, 0.0, False),
    # One per reachable head-tile count.
    ("H=32", 4, 32, 64, 32, 0.0, False),
    ("H=48", 4, 48, 64, 32, 0.0, False),
    ("H=64", 4, 64, 64, 32, 0.0, False),
    ("H=80", 4, 80, 64, 32, 0.0, False),
    ("H=96", 4, 96, 64, 32, 0.0, False),
    ("H=112", 4, 112, 64, 32, 0.0, False),
    ("H=128", 4, 128, 64, 32, 0.0, False),
    # The splice point between the two sources.
    ("splice mid-tile", 8, 16, 50, 30, 0.0, False),
    ("splice at tile edge", 8, 16, 32, 32, 0.0, False),
    ("splice one before", 8, 16, 31, 33, 0.0, False),
    ("splice one after", 8, 16, 33, 31, 0.0, False),
    ("prefix only", 8, 16, 96, 0, 0.0, False),
    ("extend only", 8, 16, 0, 96, 0.0, False),
    ("both empty", 4, 16, 0, 0, 0.0, False),
    ("single row each", 8, 16, 1, 1, 0.0, False),
    ("partial tail tile", 8, 16, 37, 5, 0.0, False),
    ("sentinels", 8, 16, 64, 32, 0.25, False),
    ("heavy sentinels", 8, 16, 96, 48, 0.6, False),
    ("ragged", 16, 16, 80, 40, 0.0, True),
    ("ragged + sentinels", 16, 16, 80, 40, 0.3, True),
    ("long prefix", 4, 16, 1152, 128, 0.0, False),
    ("long prefix H=128", 2, 128, 1152, 128, 0.0, False),
    ("prefill chunk", 105, 16, 518, 105, 0.0, True),
]


@pytest.mark.parametrize("name,N,H,p_len,e_len,sentinel,ragged", CASES)
def test_mla_prefill_v4_bf16(name, N, H, p_len, e_len, sentinel, ragged):
    scale = 1.0 / (D**0.5)
    args = build(N, H, p_len, e_len, sentinel, ragged, seed=abs(hash(name)) % 9973)
    got = mla_prefill_v4_bf16(*args, scale, sentinel > 0)
    want = ref(*args, scale)

    if want.float().norm() == 0:
        assert got.float().norm() == 0, f"{name}: expected an all-zero result"
        return
    # bf16 carries ~3 decimal digits and the reference accumulates in fp32 while
    # the kernel rounds P to bf16 before PV, so a few 1e-3 is expected.
    err = ((got.float() - want.float()).norm() / want.float().norm()).item()
    assert err < 6e-3, f"{name}: relative error {err:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
