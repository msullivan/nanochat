"""Reproduce the FlexAttention prefill shared-memory OOM on sm120 (e.g. RTX 6000
Blackwell). At head_dim=128, flex's full-attention Triton template wants more shared
memory than the GPU's ~99KB per-block opt-in cap, so Inductor finds no valid config
("no valid triton configs ... OutOfMemoryError ... Required: N Hardware limit: M").

  CUDA_VISIBLE_DEVICES=<sm120 gpu index> python dev/repro_flex_oom.py

fp32 reproduces reliably (it doubles per-tile shared mem vs bf16). The real code
avoids this by pinning kernel_options={"BLOCK_M":64,...} for prefill -- see
nanochat/flash_attention.py (_FLEX_PREFILL_KERNEL_OPTIONS). Set DT=bf16 below or
FORCE=1 to see the bf16 / force-the-big-tile variants.
"""
import os, torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
assert torch.cuda.is_available(), "needs a CUDA GPU (an sm120 / consumer Blackwell)"
dev = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0), "compute_cap", torch.cuda.get_device_capability(0))
flex = torch.compile(flex_attention)
B, H, D, T = 4, 12, 128, 512                 # head_dim=128 (D) is the knob that breaks it
dt = torch.bfloat16 if os.environ.get("DT") == "bf16" else torch.float32
q, k, v = (torch.randn(B, H, T, D, device=dev, dtype=dt) for _ in range(3))
bm = create_block_mask(lambda b, h, qi, ki: qi >= ki, B=None, H=None, Q_LEN=T, KV_LEN=T, device=dev)
# no kernel_options -> the autotuner picks the big tile (BLOCK_M=128, num_stages=3).
# FORCE=1 pins it explicitly to guarantee the OOM regardless of dtype/autotune luck.
ko = {"BLOCK_M": 128, "BLOCK_N": 64, "num_stages": 3} if os.environ.get("FORCE") else None
print(f"dtype={dt}, kernel_options={ko}")
try:
    flex(q, k, v, block_mask=bm, kernel_options=ko); torch.cuda.synchronize()
    print("compiled+ran OK -- did NOT reproduce (try FORCE=1, or DT default fp32)")
except Exception as e:
    print("REPRO:", str(e).strip().splitlines()[-1])
    raise
