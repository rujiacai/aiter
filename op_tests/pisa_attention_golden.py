"""Customer PISA golden (op_tests only; not part of aiter.ops).

Vendored from VSA_CSR/requirement/pisa_attention.py.

Minimal BF16 PISA attention using a VSA-compatible input ABI.

The VSA baseline consumes Q/K/V, ``q2k_index``, ``q2k_num``, and
``variable_block_sizes``.  PISA keeps those inputs unchanged and splits the
single VSA sparse descriptor into two descriptors:

* ``exact_q2k_index`` / ``exact_q2k_num``;
* ``approx_q2k_index`` / ``approx_q2k_num``.

Both descriptors use VSA's fixed-width padded layout.  There is no CSR, H
correction, cache manager, route scorer, quantization, serialization, or CLI.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # The PyTorch reference remains usable without Triton.
    triton = None
    tl = None


BLOCK = 128
HEAD_DIM = 128
APPROX_GROUP = 16


def block_masks_to_vsa_indices(
    exact_block_mask: torch.Tensor,
    approx_block_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert exact/approx block masks to the same descriptors as VSA.

    Args:
        exact_block_mask: bool ``[B,H,Q_blocks,KV_blocks]``.
        approx_block_mask: same shape, disjoint from ``exact_block_mask``.

    Returns:
        ``exact_q2k_index`` and ``approx_q2k_index`` have the same full shape
        as their input masks.  Selected key-block ids occupy the beginning of
        every row in ascending order; unused slots are ``-1``.  Corresponding
        ``*_q2k_num`` tensors have shape ``[B,H,Q_blocks]``.

    The conversion is self-contained and produces a fixed-width
    VSA-compatible descriptor.
    """

    if exact_block_mask.shape != approx_block_mask.shape:
        raise ValueError("exact and approximate masks must have identical shapes")
    if exact_block_mask.ndim != 4:
        raise ValueError("block masks must have shape [B,H,Q_blocks,KV_blocks]")
    exact_block_mask = exact_block_mask.bool()
    approx_block_mask = approx_block_mask.bool()
    if bool((exact_block_mask & approx_block_mask).any().item()):
        raise ValueError("exact and approximate block masks must be disjoint")

    def convert(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv_blocks = mask.shape[-1]
        number = mask.sum(dim=-1, dtype=torch.int32)
        index = torch.full(
            mask.shape, -1, device=mask.device, dtype=torch.int32
        )
        selected = mask.nonzero(as_tuple=False)
        if selected.numel():
            _, heads, query_blocks, _ = mask.shape
            row = (
                (selected[:, 0] * heads + selected[:, 1]) * query_blocks
                + selected[:, 2]
            )
            flat_number = number.reshape(-1).long()
            row_offsets = flat_number.cumsum(0) - flat_number
            position = torch.arange(
                selected.shape[0], device=mask.device, dtype=torch.long
            ) - row_offsets[row]
            # torch.nonzero enumerates the final dimension in ascending order,
            # so selected key ids remain sorted within every descriptor row.
            index.view(-1, kv_blocks)[row, position] = selected[:, 3].to(
                torch.int32
            )
        return index.contiguous(), number.contiguous()

    exact_q2k_index, exact_q2k_num = convert(exact_block_mask)
    approx_q2k_index, approx_q2k_num = convert(approx_block_mask)
    return exact_q2k_index, exact_q2k_num, approx_q2k_index, approx_q2k_num


def split_block_mask_by_score(
    block_mask: torch.Tensor,
    block_score: torch.Tensor,
    resparsity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split VSA-selected routes into exact and approximate PISA routes.

    For each ``(batch, head)``, all entries where ``block_mask`` is true form
    one candidate pool across query-block and key-block dimensions.  Heads do
    not share a budget.  Finite scores are sorted within that head and lower
    scores are approximated first.  For ``N`` routes selected by one head, the
    target is ``floor(N * resparsity)`` (capped by the number of finite
    scores).  No selected route is dropped:

    ``exact_mask | approx_mask == block_mask``.

    ``resparsity=0`` is a deliberate fast path: the original VSA block mask is
    returned as exact and the approximate mask is empty, without sorting.
    """

    if block_mask.shape != block_score.shape or block_mask.ndim != 4:
        raise ValueError(
            "block_mask and block_score must have identical [B,H,QB,KVB] shapes"
        )
    resparsity = float(resparsity)
    if not 0.0 <= resparsity <= 1.0:
        raise ValueError("resparsity must be in [0,1]")
    selected = block_mask.bool()
    if resparsity == 0.0:
        return selected.contiguous(), torch.zeros_like(selected)

    finite_selected = selected & torch.isfinite(block_score)
    batch, heads = selected.shape[:2]
    selected_flat = selected.reshape(batch, heads, -1)
    finite_flat = finite_selected.reshape(batch, heads, -1)
    score_flat = block_score.float().reshape(batch, heads, -1)
    target = torch.floor(selected_flat.sum(-1).float() * resparsity).to(torch.int64)
    target = torch.minimum(target, finite_flat.sum(-1, dtype=torch.int64))
    approx_flat = torch.zeros_like(selected_flat)
    # Sort only selected, finite routes.  This has the same stable ranking as
    # sorting the full dense Q_block x KV_block plane, while keeping the
    # reference usable for long sparse sequences.
    for batch_index in range(batch):
        for head_index in range(heads):
            count = int(target[batch_index, head_index].item())
            if count == 0:
                continue
            candidates = finite_flat[batch_index, head_index].nonzero(
                as_tuple=True
            )[0]
            order = torch.argsort(
                score_flat[batch_index, head_index, candidates], stable=True
            )
            approx_flat[
                batch_index,
                head_index,
                candidates[order[:count]],
            ] = True
    approx_mask = approx_flat.view_as(selected)
    exact_mask = selected & ~approx_mask
    return exact_mask.contiguous(), approx_mask.contiguous()


def build_pisa_descriptors(
    block_mask: torch.Tensor,
    block_score: torch.Tensor,
    resparsity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank selected VSA routes and materialize exact/approx VSA descriptors."""

    exact_mask, approx_mask = split_block_mask_by_score(
        block_mask,
        block_score,
        resparsity,
    )
    return block_masks_to_vsa_indices(exact_mask, approx_mask)


def validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_q2k_index: torch.Tensor,
    exact_q2k_num: torch.Tensor,
    approx_q2k_index: torch.Tensor,
    approx_q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Validate the VSA-compatible PISA public input boundary."""

    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical [B,H,T,D] shapes")
    batch, heads, tokens, dim = q.shape
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        raise ValueError("minimal PISA attention requires BF16 q, k, and v")
    if dim != HEAD_DIM or tokens % BLOCK:
        raise ValueError("minimal PISA requires D=128 and T divisible by 128")
    blocks = tokens // BLOCK
    if variable_block_sizes.numel() != blocks:
        raise ValueError("variable_block_sizes must contain T/128 values")
    if variable_block_sizes.dtype != torch.int32:
        raise ValueError("variable_block_sizes must use INT32 like VSA")
    if variable_block_sizes.device != q.device:
        raise ValueError("variable_block_sizes and q/k/v must share a device")
    if bool(
        ((variable_block_sizes < 0) | (variable_block_sizes > BLOCK)).any().item()
    ):
        raise ValueError("variable block sizes must be in [0,128]")

    index_shape = (batch, heads, blocks, blocks)
    number_shape = (batch, heads, blocks)
    for name, index, number in (
        ("exact", exact_q2k_index, exact_q2k_num),
        ("approx", approx_q2k_index, approx_q2k_num),
    ):
        if index.shape != index_shape:
            raise ValueError(f"{name}_q2k_index must have shape {index_shape}")
        if number.shape != number_shape:
            raise ValueError(f"{name}_q2k_num must have shape {number_shape}")
        if index.dtype != torch.int32 or number.dtype != torch.int32:
            raise ValueError(f"{name} descriptors must use INT32")
        if index.device != q.device or number.device != q.device:
            raise ValueError(f"{name} descriptors and q/k/v must share a device")
        if bool(((number < 0) | (number > blocks)).any().item()):
            raise ValueError(f"{name}_q2k_num must be in [0,KV_blocks]")
        lanes = torch.arange(blocks, device=q.device).view(1, 1, 1, blocks)
        active = lanes < number.unsqueeze(-1)
        invalid = active & ((index < 0) | (index >= blocks))
        if bool(invalid.any().item()):
            raise ValueError(f"{name}_q2k_index contains an invalid active id")
    return batch, heads, tokens, dim


def compute_block_statistics(
    k: torch.Tensor,
    v: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the BF16 PISA K center and V sum from current-step K/V."""

    batch, heads, tokens, dim = k.shape
    blocks = tokens // BLOCK
    k_blocks = k.view(batch, heads, blocks, BLOCK, dim).float()
    v_blocks = v.view(batch, heads, blocks, BLOCK, dim).float()
    token_lane = torch.arange(BLOCK, device=k.device)
    valid = token_lane.view(1, 1, 1, BLOCK, 1) < variable_block_sizes.view(
        1, 1, blocks, 1, 1
    )
    denominator = variable_block_sizes.clamp_min(1).view(1, 1, blocks, 1)
    k_center = (k_blocks * valid).sum(3) / denominator
    value_sum = (v_blocks * valid).sum(3)
    return k_center.bfloat16().contiguous(), value_sum.bfloat16().contiguous()


def _route_row(
    index: torch.Tensor,
    number: torch.Tensor,
    batch: int,
    head: int,
    query_block: int,
) -> torch.Tensor:
    count = int(number[batch, head, query_block].item())
    return index[batch, head, query_block, :count].long()


@torch.no_grad()
def pisa_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_q2k_index: torch.Tensor,
    exact_q2k_num: torch.Tensor,
    approx_q2k_index: torch.Tensor,
    approx_q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """FP32 reference for the VSA-compatible exact/approx PISA equation."""

    batch_size, heads, tokens, dim = validate_inputs(
        q,
        k,
        v,
        exact_q2k_index,
        exact_q2k_num,
        approx_q2k_index,
        approx_q2k_num,
        variable_block_sizes,
    )
    blocks = tokens // BLOCK
    scale = dim**-0.5
    output = torch.empty_like(q)
    for batch in range(batch_size):
        for head in range(heads):
            for query_block in range(blocks):
                query_start = query_block * BLOCK
                query = q[batch, head, query_start : query_start + BLOCK].float()

                exact_keys: list[torch.Tensor] = []
                exact_values: list[torch.Tensor] = []
                exact_blocks = _route_row(
                    exact_q2k_index,
                    exact_q2k_num,
                    batch,
                    head,
                    query_block,
                )
                for key_block in exact_blocks.tolist():
                    length = int(variable_block_sizes[key_block].item())
                    key_start = key_block * BLOCK
                    exact_keys.append(
                        k[batch, head, key_start : key_start + length].float()
                    )
                    exact_values.append(
                        v[batch, head, key_start : key_start + length].float()
                    )
                if exact_keys:
                    exact_key = torch.cat(exact_keys)
                    exact_value = torch.cat(exact_values)
                    exact_logits = query @ exact_key.T * scale
                else:
                    exact_logits = query.new_empty((BLOCK, 0))
                    exact_value = query.new_empty((0, dim))

                centers: list[torch.Tensor] = []
                value_sums: list[torch.Tensor] = []
                lengths: list[int] = []
                approx_blocks = _route_row(
                    approx_q2k_index,
                    approx_q2k_num,
                    batch,
                    head,
                    query_block,
                )
                for key_block in approx_blocks.tolist():
                    length = int(variable_block_sizes[key_block].item())
                    key_start = key_block * BLOCK
                    if length:
                        centers.append(
                            k[batch, head, key_start : key_start + length]
                            .float()
                            .mean(0)
                            .bfloat16()
                            .float()
                        )
                        value_sums.append(
                            v[batch, head, key_start : key_start + length]
                            .float()
                            .sum(0)
                            .bfloat16()
                            .float()
                        )
                    else:
                        centers.append(query.new_zeros(dim))
                        value_sums.append(query.new_zeros(dim))
                    lengths.append(length)
                if centers:
                    center = torch.stack(centers)
                    value_sum = torch.stack(value_sums)
                    length_tensor = torch.tensor(
                        lengths, device=q.device, dtype=torch.float32
                    )
                    approx_logits = query @ center.T * scale
                else:
                    value_sum = query.new_empty((0, dim))
                    length_tensor = query.new_empty(0)
                    approx_logits = query.new_empty((BLOCK, 0))

                if exact_logits.shape[1] + approx_logits.shape[1] == 0:
                    output[
                        batch, head, query_start : query_start + BLOCK
                    ].zero_()
                    continue
                row_max = torch.cat((exact_logits, approx_logits), 1).amax(
                    1, keepdim=True
                )
                exact_weight = torch.exp(exact_logits - row_max)
                approx_weight = torch.exp(approx_logits - row_max)
                denominator = exact_weight.sum(1)
                numerator = exact_weight @ exact_value
                if centers:
                    denominator += (approx_weight * length_tensor).sum(1)
                    numerator += approx_weight @ value_sum
                output[batch, head, query_start : query_start + BLOCK] = (
                    numerator / denominator.clamp_min(1.0e-20)[:, None]
                ).bfloat16()
    return output


if triton is not None:

    @triton.jit
    def _pisa_attention_kernel(
        Q,
        K,
        V,
        EXACT_Q2K_INDEX,
        EXACT_Q2K_NUM,
        APPROX_Q2K_INDEX,
        APPROX_Q2K_NUM,
        VARIABLE_BLOCK_SIZES,
        K_CENTER,
        VALUE_SUM,
        OUT,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        BATCH,
        HEADS,
        N_CTX,
        Q_TILES: tl.constexpr,
        MAX_KV_BLOCKS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        APPROX_N: tl.constexpr,
    ):
        """VSA's exact loop plus a second approximate descriptor loop."""

        query_block = tl.program_id(0)
        off_hz = tl.program_id(1)
        batch = off_hz // HEADS
        head = off_hz % HEADS
        meta_base = off_hz * Q_TILES + query_block
        exact_count = tl.load(EXACT_Q2K_NUM + meta_base).to(tl.int32)
        approx_count = tl.load(APPROX_Q2K_NUM + meta_base).to(tl.int32)
        exact_count = tl.minimum(tl.maximum(exact_count, 0), MAX_KV_BLOCKS)
        approx_count = tl.minimum(tl.maximum(approx_count, 0), MAX_KV_BLOCKS)
        exact_ptr = EXACT_Q2K_INDEX + meta_base * MAX_KV_BLOCKS
        approx_ptr = APPROX_Q2K_INDEX + meta_base * MAX_KV_BLOCKS

        qkv_offset = (
            batch.to(tl.int64) * stride_qz + head.to(tl.int64) * stride_qh
        )
        kv_offset = (
            batch.to(tl.int64) * stride_kz + head.to(tl.int64) * stride_kh
        )
        value_offset = (
            batch.to(tl.int64) * stride_vz + head.to(tl.int64) * stride_vh
        )
        output_offset = (
            batch.to(tl.int64) * stride_oz + head.to(tl.int64) * stride_oh
        )
        query_start = query_block * BLOCK_M
        q_ptr = tl.make_block_ptr(
            Q + qkv_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(query_start, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        k_ptr = tl.make_block_ptr(
            K + kv_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        v_ptr = tl.make_block_ptr(
            V + value_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        out_ptr = tl.make_block_ptr(
            OUT + output_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(query_start, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        offsets_n = tl.arange(0, BLOCK_N)
        offsets_d = tl.arange(0, HEAD_DIM)
        query = tl.load(q_ptr)
        qk_scale = (1.0 / tl.sqrt(float(HEAD_DIM))) * 1.4426950408889634
        running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        running_sum = tl.full((BLOCK_M,), 1.0, tl.float32)
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

        # This loop is structurally the VSA BF16 sparse-attention loop.  Its
        # only descriptor change is the explicit "exact" name.
        for route in range(0, exact_count):
            key_block = tl.load(exact_ptr + route).to(tl.int32)
            key_block = tl.minimum(tl.maximum(key_block, 0), MAX_KV_BLOCKS - 1)
            length = tl.load(VARIABLE_BLOCK_SIZES + key_block).to(tl.int32)
            length = tl.minimum(tl.maximum(length, 0), BLOCK_N)
            key_start = key_block * BLOCK_N
            key = tl.load(tl.advance(k_ptr, (0, key_start)))
            logits = tl.dot(query, key) * qk_scale
            logits = tl.where((offsets_n < length)[None, :], logits, -float("inf"))
            new_max = tl.maximum(running_max, tl.max(logits, axis=1))
            weight = tl.math.exp2(logits - new_max[:, None])
            old_scale = tl.math.exp2(running_max - new_max)
            running_sum = running_sum * old_scale + tl.sum(weight, axis=1)
            accumulator *= old_scale[:, None]
            value = tl.load(tl.advance(v_ptr, (key_start, 0)))
            accumulator = tl.dot(weight.to(value.dtype), value, acc=accumulator)
            running_max = new_max

        # PISA adds this loop.  It reads the same VSA descriptor layout but
        # substitutes one K center and one V sum for all tokens in a block.
        for start in range(0, approx_count, APPROX_N):
            route_offsets = start + tl.arange(0, APPROX_N)
            live = route_offsets < approx_count
            key_blocks = tl.load(approx_ptr + route_offsets, mask=live, other=0).to(
                tl.int32
            )
            key_blocks = tl.minimum(
                tl.maximum(key_blocks, 0), MAX_KV_BLOCKS - 1
            )
            statistic_rows = (
                (off_hz.to(tl.int64) * MAX_KV_BLOCKS + key_blocks.to(tl.int64))[
                    :, None
                ]
                * HEAD_DIM
                + offsets_d[None, :]
            )
            centers = tl.load(K_CENTER + statistic_rows, mask=live[:, None], other=0.0)
            value_sums = tl.load(
                VALUE_SUM + statistic_rows, mask=live[:, None], other=0.0
            )
            lengths = tl.load(
                VARIABLE_BLOCK_SIZES + key_blocks, mask=live, other=0
            ).to(tl.float32)
            logits = tl.dot(query, tl.trans(centers)) * qk_scale
            logits = tl.where(live[None, :], logits, -float("inf"))
            new_max = tl.maximum(running_max, tl.max(logits, axis=1))
            old_scale = tl.math.exp2(running_max - new_max)
            weight = tl.math.exp2(logits - new_max[:, None])
            running_sum = running_sum * old_scale + tl.sum(
                weight * lengths[None, :], axis=1
            )
            accumulator *= old_scale[:, None]
            accumulator = tl.dot(
                weight.to(value_sums.dtype), value_sums, acc=accumulator
            )
            running_max = new_max

        result = accumulator / tl.maximum(running_sum, 1.0e-20)[:, None]
        result = tl.where((running_sum > 0)[:, None], result, 0.0)
        tl.store(out_ptr, result.to(OUT.dtype.element_ty))


@torch.no_grad()
def pisa_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_q2k_index: torch.Tensor,
    exact_q2k_num: torch.Tensor,
    approx_q2k_index: torch.Tensor,
    approx_q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Run PISA with VSA inputs plus exact/approx descriptor separation."""

    if triton is None:
        raise RuntimeError("Triton is required for pisa_attention_triton")
    batch, heads, tokens, _ = validate_inputs(
        q,
        k,
        v,
        exact_q2k_index,
        exact_q2k_num,
        approx_q2k_index,
        approx_q2k_num,
        variable_block_sizes,
    )
    if q.device.type != "cuda":
        raise ValueError("Triton PISA attention requires a ROCm/CUDA device")
    blocks = tokens // BLOCK
    sizes = variable_block_sizes.to(device=q.device, dtype=torch.int32).contiguous()
    exact_q2k_index = exact_q2k_index.contiguous()
    exact_q2k_num = exact_q2k_num.contiguous()
    approx_q2k_index = approx_q2k_index.contiguous()
    approx_q2k_num = approx_q2k_num.contiguous()
    k_center, value_sum = compute_block_statistics(k, v, sizes)
    output = torch.empty_like(q)
    _pisa_attention_kernel[(blocks, batch * heads)](
        q,
        k,
        v,
        exact_q2k_index,
        exact_q2k_num,
        approx_q2k_index,
        approx_q2k_num,
        sizes,
        k_center,
        value_sum,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        batch,
        heads,
        tokens,
        Q_TILES=blocks,
        MAX_KV_BLOCKS=blocks,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        APPROX_N=APPROX_GROUP,
        num_warps=4,
        num_stages=1,
        waves_per_eu=2,
        matrix_instr_nonkdim=32,
    )
    return output


@torch.no_grad()
def pisa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_score: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    resparsity: float,
) -> torch.Tensor:
    """High-level PISA entry: VSA block mask plus score and resparsity.

    It ranks the selected routes, creates exact/approx VSA descriptors, and
    invokes the fused attention kernel.
    """

    descriptors = build_pisa_descriptors(block_mask, block_score, resparsity)
    return pisa_attention_triton(q, k, v, *descriptors, variable_block_sizes)


@torch.no_grad()
def vsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Standalone VSA baseline using the identical exact kernel path.

    It is intentionally expressed as all selected routes exact and no routes
    approximate.  Therefore ``pisa_attention(..., resparsity=0)`` and this
    function launch the same kernel with identical active descriptors.
    """

    descriptors = block_masks_to_vsa_indices(
        block_mask,
        torch.zeros_like(block_mask, dtype=torch.bool),
    )
    return pisa_attention_triton(q, k, v, *descriptors, variable_block_sizes)
