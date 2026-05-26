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
        # FA3 kernels are compiled for Hopper (sm90) only.
        # Other GPUs need a different backend (FlexAttention or SDPA):
        #   Ada sm89, Blackwell sm100 (B200/GB200), Blackwell sm120 (RTX 5090,
        #   RTX PRO 6000 workstation). FA4 currently targets sm100 only; sm120
        #   support is requested but uncommitted (Dao-AILab/flash-attention#2307).
        #   FA2 covers Ampere/Ada but has known bf16 training convergence issues
        #   on sm120 (#2151) so we don't fall back to it.
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
        import os
        from torch.nn.attention.flex_attention import flex_attention
        # Default mode. max-autotune-no-cudagraphs OOMs at d24 on Blackwell
        # (autotune keeps multiple kernel candidates resident) and didn't help
        # throughput when it did fit, so not worth it.
        # NANOCHAT_FLEX_DYNAMIC=1 forces dynamic-shape compilation: one shared
        # kernel across (B, T) instead of one specialised kernel per shape.
        # Useful for variable-shape eval (e.g. CORE) where static recompilation
        # blows past dynamo's recompile_limit and the function silently falls
        # back to eager flex_attention.
        dynamic = os.environ.get("NANOCHAT_FLEX_DYNAMIC", "0") == "1"
        _compiled_flex_attention = torch.compile(flex_attention, dynamic=dynamic)
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


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, input_pos=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert at positions input_pos, shape (B, T_new, H_kv, D)
        input_pos: (T_new,) absolute positions where the new k,v are written into
                   the cache, and which the resulting queries attend to causally.
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)

    For FA3 (Hopper): forwards to flash_attn_with_kvcache, deriving
    cache_seqlens from input_pos[:1]. FA3 advances the cache position via
    cache_seqlens and writes new k,v at that offset internally.

    For the SDPA fallback (everything else): writes new k,v at input_pos via
    index_copy_, then computes SDPA over the FULL preallocated cache with a
    bool mask built from input_pos. Shapes are constant across decode steps
    (no variable `[:, :end_pos]` slice), which is what torch.compile mode=
    "reduce-overhead" needs to capture the decode forward into a cuda graph.
    Costs a few extra microseconds per layer per step over zero-padded cache
    positions (which get masked-to-zero contribution anyway), but the
    cudagraph wins back far more.
    """
    if BACKEND == 'fa3':
        # FA3 wants cache_seqlens (per-batch position). Derive from input_pos[0]
        # broadcast across the batch dim -- our input_pos is shared across batch.
        B = q.shape[0]
        cache_seqlens = input_pos[:1].to(torch.int32).expand(B)
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: full-cache + mask, no `.item()`, no variable-shape slicing.
    B, T_new, H, D = q.shape
    _, T_max, H_kv, _ = k_cache.shape
    device = q.device

    # In-place insert into cache at positions input_pos.
    if k is not None and v is not None:
        k_cache.index_copy_(1, input_pos, k)
        v_cache.index_copy_(1, input_pos, v)

    # Build attention mask: query at absolute position input_pos[i] attends to
    # keys at positions [max(0, input_pos[i] - window), input_pos[i]].
    # Shape is (T_new, T_max).
    key_positions = torch.arange(T_max, device=device)
    q_positions = input_pos  # (T_new,)
    attn_mask = key_positions.unsqueeze(0) <= q_positions.unsqueeze(1)  # causal
    window = window_size[0]
    if 0 <= window < T_max:
        attn_mask = attn_mask & (key_positions.unsqueeze(0) >= (q_positions.unsqueeze(1) - window))

    # SDPA expects (B, H, T, D). attn_mask broadcasts (T_new, T_max) over (B, H).
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_cache.transpose(1, 2)
    v_sdpa = v_cache.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)

    # Force cuDNN backend: SDPA's default selection on Blackwell sm120 picks
    # the MATH backend (slow) for our (full-cache + bool mask + GQA) combo
    # because Flash refuses non-null masks and Efficient/cuDNN are runtime-
    # disabled in the default scoring. cuDNN explicitly supports masks and
    # is ~2.4x faster than MATH on this hardware for our shapes (microbench
    # in dev/sdpa_bench.py).
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        y_sdpa = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask, enable_gqa=enable_gqa,
        )

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
