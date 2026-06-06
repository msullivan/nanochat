"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Bigram value embeddings: index VE by (prev_byte, curr_byte) instead of curr_byte alone.
    # Uses 8 bits of prev + low 7 bits of curr (= 15 bits = 32768 entries) — UTF-8-aware
    # since the dropped high bit of curr is recoverable when curr is a UTF-8 cont byte
    # (forced to 1 by validity). Only meaningful for byte-level tokenizers (vocab_size <= 256).
    bigram_value_embeds: bool = False
    # Skip value-embedding contribution in attention entirely: the
    # value_embeds tables and per-block ve_gate weights are not allocated.
    # Cross-loading from non-disabled checkpoints uses strict=False at load.
    disable_value_embeds: bool = False


# 15-bit bigram VE table size. 8 bits of prev (full byte) × 7 bits of curr (low 7).
# Token IDs above 255 collapse modulo 256 -- byte tokenizer only uses 0-255 so this is fine.
BIGRAM_VE_VOCAB_SIZE = 32768
BIGRAM_VE_CURR_BITS = 7
BIGRAM_VE_CURR_MASK = (1 << BIGRAM_VE_CURR_BITS) - 1  # 0x7F


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def compute_bigram_idx(idx, prev_token_id=None):
    """Compute bigram value-embedding indices.

    For each position t, output[t] = (prev_byte * 128) | (curr_byte & 0x7F)
    where prev_byte at t=0 is BOS (0x00) for training/prefill and prev_token_id for decode.

    8 bits of prev byte + low 7 bits of curr byte = 15 bits = 32768 entries. Lossless on
    UTF-8 multi-byte intra-char bigrams under validity (curr's redundant high bit when it
    is a continuation byte is recoverable from curr's full bits, so we drop it). Token IDs
    are masked to byte range modulo 256, which is exact for the byte tokenizer (vocab=256).

    Args:
        idx: (B, T) input token ids.
        prev_token_id: optional (B,) tensor giving the byte BEFORE idx[:, 0]. Used during
                       KV-cache decode where we have the previous step's emitted token.
                       If None, BOS=0x00 is used (training/prefill default).
    Returns:
        (B, T) int64 tensor of bigram indices in [0, 32768).
    """
    B, T = idx.size()
    curr_byte = (idx & 0xFF).long()
    if prev_token_id is None:
        # Training / prefill: shift idx right by 1, prepend BOS (0x00) at position 0.
        bos_col = torch.zeros((B, 1), dtype=torch.long, device=idx.device)
        prev_byte = torch.cat([bos_col, curr_byte[:, :-1]], dim=1)
    else:
        # Decode: prev_token_id provides the byte just before idx[:, 0].
        # idx is typically (B, 1) here; for T>1 prefill-with-cache, shift internally too.
        prev_first = (prev_token_id & 0xFF).long().view(B, 1)
        prev_byte = torch.cat([prev_first, curr_byte[:, :-1]], dim=1) if T > 1 else prev_first
    return (prev_byte << BIGRAM_VE_CURR_BITS) | (curr_byte & BIGRAM_VE_CURR_MASK)

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        ve_layer = has_ve(layer_idx, config.n_layer) and not config.disable_value_embeds
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if ve_layer else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, input_pos, key_padding_mask=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window.
            # key_padding_mask (B, T) marks valid (non-pad) key positions; only
            # passed for left-padded batched inference, None for training.
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size,
                                           key_padding_mask=key_padding_mask)
        else:
            # Inference: use flash_attn_with_kvcache. input_pos tells the cache
            # where to insert these k,v and is consumed by the mask construction
            # in the SDPA fallback path / cache_seqlens in the FA3 path.
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                input_pos=input_pos,
                causal=True,
                window_size=window_size,
            )

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, input_pos, key_padding_mask=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, input_pos, key_padding_mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # KV cache (None during training; set by setup_caches() for inference).
        # Attaching as a model attribute -- rather than passing through forward
        # as a parameter -- means torch.compile mode="reduce-overhead" treats
        # cache mutations as graph-owned state instead of as mutations to
        # external inputs (which would block cudagraph capture).
        self.kv_cache = None
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included.
        # When bigram_value_embeds=True, the embedding table is indexed by a 15-bit
        # (prev_byte, curr_byte_low7) pair instead of curr_byte alone.
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        ve_vocab_size = BIGRAM_VE_VOCAB_SIZE if config.bigram_value_embeds else padded_vocab_size
        if config.disable_value_embeds:
            self.value_embeds = nn.ModuleDict({})
        else:
            self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(ve_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std).
        # Empty dict when disable_value_embeds is set, so this loop is a no-op.
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral.
        # ve_gate is None on every block when disable_value_embeds is set.
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context, -1 = no window), S=short (quarter of training seq len)

        L uses -1 (the FA3 "unlimited" sentinel) rather than config.sequence_len so
        that eval/inference at sequences longer than the training seq still routes
        to the dense-causal fast path everywhere (FA3, FlexAttention's hybrid
        dispatch, SDPA's is_causal=True). With long_window = sequence_len, longer
        eval prompts would partially fall into the sliding-window codepath and
        cause flex_attention recompiles per-shape during CORE eval.
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        short_window = -(-config.sequence_len // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (-1, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (-1, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit.
        # Position MATTERS: PyTorch optimizers match groups by index when
        # load_state_dict is called, so the value_embeds group must stay at
        # its historical position (3rd AdamW group, between embedding and
        # resid) for back-compat with existing checkpoints. When VE is
        # disabled the slot is simply omitted; that's fine because a
        # disabled-VE run can only resume from a disabled-VE checkpoint
        # (which also omits the slot), so positions still line up.
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        ]
        if value_embeds_params:
            param_groups.append(
                dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            )
        param_groups.extend([
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ])
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def setup_caches(self, batch_size, max_seq_length, dtype):
        """Attach a KV cache to the model for inference.

        Allocates the cache as a registered submodule so its buffers are
        treated as model state (rather than as forward arguments), which is
        what torch.compile mode="reduce-overhead" needs to capture decode
        steps into a CUDA graph.

        Idempotent: if a cache exists at the requested (batch_size,
        max_seq_length) or larger, reuses it and just resets transient
        state. Otherwise reallocates.
        """
        from nanochat.engine import KVCache
        c = self.config
        existing = self.kv_cache
        if (existing is not None
                and existing.batch_size == batch_size
                and existing.max_seq_len >= max_seq_length
                and existing.k_cache.dtype == dtype):
            existing.reset()
            return
        # device is taken from an existing parameter (model already on the right device by now)
        device = self.lm_head.weight.device
        self.kv_cache = KVCache(
            batch_size=batch_size,
            num_heads=c.n_kv_head,
            seq_len=max_seq_length,
            head_dim=c.n_embd // c.n_head,
            num_layers=c.n_layer,
            dtype=dtype,
            n_embd=c.n_embd,
        ).to(device)

    def forward(self, idx, targets=None, input_pos=None, loss_reduction='mean', attention_mask=None):
        """
        Args:
            idx: (B, T) token ids
            targets: (B, T) target token ids for training (None for inference)
            input_pos: (T,) absolute positions of the tokens in `idx`. Required
                       for inference (kv-cache present); ignored for training.
            loss_reduction: 'mean' or 'sum'
            attention_mask: optional (B, T) mask, 1 for real tokens and 0 for
                       padding. Only used on the no-cache path for LEFT-PADDED
                       batched inference: it masks pad keys in attention AND
                       suppresses the smear contribution at pad->real boundaries
                       (so the first real token isn't smeared with pad). None for
                       training / single-stream -> fully bit-identical fast path.

        For inference, the KV cache is read from self.kv_cache (attached via
        setup_caches). input_pos tells the cache where to write the new k,v
        and where in the rotary table to look up positions. Passing input_pos
        explicitly (rather than reading from cache_seqlens internally) means
        forward has no host syncs -- prerequisite for cudagraph capture.
        """
        # Use kv cache only when an explicit input_pos is provided. The naive
        # GPT.generate (no input_pos) and training (targets given) both want
        # the no-cache full-sequence path.
        kv_cache = self.kv_cache if (targets is None and input_pos is not None) else None
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # Slice the rotary table at the absolute positions where these tokens live.
        # For training/no-cache: input_pos is None and we use the dense slice [0:T].
        # For inference: input_pos is the (T,) tensor of absolute positions, fed in
        # externally so the model doesn't need to read kv_cache state internally
        # (which would force a host sync and break cudagraph capture).
        if kv_cache is None:
            cos_sin = self.cos[:, :T], self.sin[:, :T]
        else:
            assert input_pos is not None, "input_pos required when kv_cache is present"
            cos_sin = self.cos.index_select(1, input_pos), self.sin.index_select(1, input_pos)

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info).
        # KV-cache path uses an in-place copy_ into kv_cache.prev_embedding (rather than
        # a Python attribute assignment) so the buffer's memory address stays stable
        # across forwards, which is what CUDA graph capture requires for correctness on
        # replay. Order matters: compute the smear contribution FIRST using the OLD
        # prev_embedding, THEN write the current pre-smear last position back into the
        # buffer, THEN apply the smear to x.
        assert attention_mask is None or kv_cache is None, "attention_mask is only supported on the no-cache path"
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            smear_add = gate * x[:, :-1]
            if attention_mask is not None:
                # Left-pad boundary: zero the smear into a token whose predecessor
                # is padding (so the first real token isn't smeared with pad). No-op
                # path when attention_mask is None -> bit-identical to before.
                smear_add = smear_add * attention_mask[:, :-1, None].to(x.dtype)
            x = torch.cat([x[:, :1], x[:, 1:] + smear_add], dim=1)
        elif T > 1:
            # Prefill: smear within x like training; save last pre-smear position
            # for the next decode step's smear.
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            new_x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            kv_cache.prev_embedding.copy_(x[:, -1:, :])  # pre-smear last pos
            x = new_x
        else:
            # Decode: single token. Compute smear from OLD prev_embedding first
            # (using read), THEN overwrite prev_embedding with current pre-smear x.
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
            smear_contribution = gate * kv_cache.prev_embedding
            kv_cache.prev_embedding.copy_(x[:, -1:, :])
            x = x + smear_contribution

        # Compute the index used for value-embedding lookups. Default = curr token.
        # When bigram_value_embeds is on, fold (prev_byte, curr_byte) into a 15-bit index.
        # KV-cache path uses in-place copy_ into kv_cache.prev_token_id (stable address
        # for CUDA graph capture). The buffer is initialized to zeros, which matches
        # the original `None → BOS=0` semantics in compute_bigram_idx for the first call.
        if self.config.bigram_value_embeds:
            if kv_cache is None:
                ve_idx = compute_bigram_idx(idx, prev_token_id=None)
            else:
                ve_idx = compute_bigram_idx(idx, prev_token_id=kv_cache.prev_token_id)
                kv_cache.prev_token_id.copy_(idx[:, -1])
        else:
            ve_idx = idx

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](ve_idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, input_pos, attention_mask)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
