"""Minimal runtime for the standalone nanochat HF artifact.

This is INFRASTRUCTURE the copied gpt.py needs, trimmed to what inference
requires (no Muon optimizer, no distributed training, no filelock). It is
intentionally small and stable -- the actual model architecture lives entirely
in the copied gpt.py (single source of truth), this file only provides:

  - COMPUTE_DTYPE (same auto-detection as nanochat.common)
  - print0 / get_dist_info (trivial single-process stubs)
  - KVCache (copied verbatim from nanochat.engine -- stable inference infra)
"""

import os
import torch
import torch.nn as nn

# --- compute dtype (mirrors nanochat.common._detect_compute_dtype) ----------
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env]
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability() >= (8, 0):
            return torch.bfloat16
        return torch.float32
    return torch.float32


COMPUTE_DTYPE = _detect_compute_dtype()


def print0(*args, **kwargs):
    print(*args, **kwargs)


def get_dist_info():
    # single-process inference: (ddp, rank, local_rank, world_size)
    return False, 0, 0, 1


# --- KV cache (verbatim from nanochat.engine.KVCache) ------------------------
class KVCache(nn.Module):
    """KV Cache as an nn.Module with all state held in registered per-layer
    buffers (k_cache_{i}/v_cache_{i}). Position bookkeeping is external (callers
    pass input_pos to model.forward)."""

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, dtype, n_embd, device=None):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.n_embd = n_embd
        self.dtype = dtype
        for i in range(num_layers):
            self.register_buffer(f"k_cache_{i}",
                torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device), persistent=False)
            self.register_buffer(f"v_cache_{i}",
                torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device), persistent=False)
        self.register_buffer("prev_embedding",
            torch.zeros(batch_size, 1, n_embd, dtype=dtype, device=device), persistent=False)
        self.register_buffer("prev_token_id",
            torch.zeros(batch_size, dtype=torch.long, device=device), persistent=False)
        for b in self.buffers():
            torch._dynamo.mark_static_address(b)

    def reset(self):
        self.prev_embedding.zero_()
        self.prev_token_id.zero_()

    def get_layer_cache(self, layer_idx):
        return getattr(self, f"k_cache_{layer_idx}"), getattr(self, f"v_cache_{layer_idx}")
