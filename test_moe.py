import torch
from aiter.fused_moe_bf16_asm import asm_moe
from aiter.fused_moe import fused_moe
from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx
from aiter import QuantType
from aiter.ops.shuffle import shuffle_weight
device = "cuda"
seq_len = 2#19147
x0 = torch.randn(seq_len, 4096, dtype=torch.bfloat16, device=device)
w1 = torch.randint(low=0, high=128, size=(400, 1536, 4096), dtype=torch.int8, device=device)
w1 = shuffle_weight(w1, (16, 16))
w2 = torch.randint(low=0, high=128, size=(400, 4096, 1536), dtype=torch.int8, device=device)
w2 = shuffle_weight(w2, (16, 16))
x1 = torch.randn(seq_len, 20, dtype=torch.float32, device=device)
x2 = torch.load("topk_ids.pt", weights_only=False) # torch.Size([seq_len, 20]), torch.int32
x2 = x2[:seq_len,...]
fc1_scale = torch.randn(400, 1536, 1, dtype=torch.float32, device=device)
fc2_scale = torch.randn(400, 4096, 1, dtype=torch.float32, device=device)
fc1_smooth_scale = torch.ones([400, 4096], dtype=torch.float32, device=device)
fc2_smooth_scale = torch.ones([400, 1536], dtype=torch.float32, device=device)
a1_scale = torch.randn(400, 1, 4096, dtype=torch.float32, device=device)
a2_scale = torch.randn(400, 1, 1536, dtype=torch.float32, device=device)
fused_moe(
    hidden_states=x0,
    w1=w1,
    w2=w2,
    topk_weight=x1,
    topk_ids=x2,
    expert_mask=None,
    activation=ActivationType.Gelu,
    quant_type=QuantType.per_Token,
    doweight_stage1=False,
    w1_scale=fc1_scale,
    w2_scale=fc2_scale,
    #a1_scale=a1_scale,
    #a2_scale=a2_scale,
    dtype=torch.bfloat16,
)
