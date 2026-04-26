"""
Unified Flash Attention interface with automatic backend dispatch.

Exports `flash_attn` module that matches the FA3 API exactly. Backends, in
preference order:
    fa3   - Flash Attention 3, Hopper (sm_90) only, fastest path
    flex  - PyTorch FlexAttention (Triton-generated), works on Ada/Blackwell etc.
    sdpa  - PyTorch SDPA, universal fallback (CPU/MPS/older GPUs)

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need a different backend until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


def _flex_attention_available():
    """FlexAttention requires PyTorch 2.5+ and CUDA with Triton."""
    if not torch.cuda.is_available():
        return False
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask  # noqa: F401
        return True
    except ImportError:
        return False


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
HAS_FLEX = _flex_attention_available()

# Override for testing: 'fa3', 'flex', 'sdpa', or None (auto)
_override_impl = None


def _resolve_backend():
    """Decide once which backend to use, based on availability, override, dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return 'fa3'
    if _override_impl == 'flex':
        assert HAS_FLEX, "Cannot override to FlexAttention: not available"
        return 'flex'
    if _override_impl == 'sdpa':
        return 'sdpa'
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use a different backend
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return 'fa3'
    if HAS_FLEX:
        return 'flex'
    return 'sdpa'


BACKEND = _resolve_backend()
USE_FA3 = BACKEND == 'fa3'  # kept for backward compat with anything that imports this


# =============================================================================
# FlexAttention helpers (block mask cache + lazy compile)
# =============================================================================
_block_mask_cache = {}
_compiled_flex_attention = None


def _get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention
        # max-autotune-no-cudagraphs: pick best Triton kernel per shape; skip
        # cudagraphs since they don't compose well with DDP / grad accumulation.
        # First call pays significant compile/autotune cost (tens of seconds);
        # subsequent calls reuse the cached kernel.
        _compiled_flex_attention = torch.compile(
            flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
    return _compiled_flex_attention


def _get_block_mask(window, q_len, k_len, device):
    """
    Causal block mask with optional left sliding window. window<0 means full
    causal context. Cached per (window, q_len, k_len, device) -- training has
    a tiny number of distinct shapes so this stays small.
    """
    cache_key = (int(window), int(q_len), int(k_len), str(device))
    if cache_key in _block_mask_cache:
        return _block_mask_cache[cache_key]

    from torch.nn.attention.flex_attention import create_block_mask

    # Chunked inference (q_len < k_len) needs offset between query and key positions.
    offset = k_len - q_len
    use_window = 0 <= window < k_len

    if use_window:
        def mask_fn(b, h, q_idx, kv_idx):
            abs_q = q_idx + offset
            return (abs_q >= kv_idx) & ((abs_q - kv_idx) <= window)
    else:
        def mask_fn(b, h, q_idx, kv_idx):
            return (q_idx + offset) >= kv_idx

    block_mask = create_block_mask(
        mask_fn, B=None, H=None, Q_LEN=q_len, KV_LEN=k_len, device=device,
    )
    _block_mask_cache[cache_key] = block_mask
    return block_mask


def _flex_attention(q, k, v, window_size, enable_gqa):
    """
    q, k, v in (B, H, T, D) layout. Returns same layout.

    Hybrid dispatch on the flex backend: full-causal layers (L in window_pattern)
    go through SDPA's is_causal=True fast path -- on Blackwell that's cuDNN's
    hand-tuned dense flash kernel, faster than flex's dense-causal kernel.
    Only true sliding-window layers (S) get the flex/Triton sparse kernel.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full causal context with aligned shapes -> SDPA fast path
    if (window < 0 or window >= Tk) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    block_mask = _get_block_mask(window, Tq, Tk, q.device)
    flex = _get_compiled_flex_attention()
    return flex(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if BACKEND == 'fa3':
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # Both flex and SDPA paths use (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)

    if BACKEND == 'flex':
        y = _flex_attention(q, k, v, window_size, enable_gqa)
    else:
        y = _sdpa_attention(q, k, v, window_size, enable_gqa)

    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our fallback does the same.

    The FlexAttention backend routes inference through SDPA: block_mask depends
    on cache_seqlens which changes every step, so the cache that makes flex fast
    for training doesn't help here. Could be added later with per-step block_mask
    rebuilds if inference latency becomes a bottleneck.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if BACKEND == 'fa3':
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache (used by both BACKEND='flex' and 'sdpa')
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
