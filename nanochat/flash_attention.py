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

# Flex tile config for masked PREFILL (Tq>1, headdim=128). The autotuned default
# (BLOCK_M=128, num_stages=3) needs ~145KB shared memory > sm120's ~100KB and fails
# to compile ("no valid triton configs"). BLOCK_M=64 divides the 128 sparse block
# (so SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0 holds) and fits. Prefill is one-shot so the
# smaller tile costs ~nothing. Decode (q_len=1) and no-cache/training use the default.
_FLEX_PREFILL_KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 32, "num_stages": 2}


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
    causal context. Cached per (window, q_len, k_len, device) -- a tiny number of
    distinct shapes so this stays small.
    """
    cache_key = (int(window), int(q_len), int(k_len), str(device))
    if cache_key in _block_mask_cache:
        return _block_mask_cache[cache_key]

    from torch.nn.attention.flex_attention import create_block_mask

    offset = k_len - q_len  # chunked use (q_len < k_len) offsets query vs key positions
    use_window = 0 <= window < k_len

    if use_window:
        def mask_fn(b, h, q_idx, kv_idx):
            abs_q = q_idx + offset
            return (abs_q >= kv_idx) & ((abs_q - kv_idx) <= window)
    else:
        def mask_fn(b, h, q_idx, kv_idx):
            return (q_idx + offset) >= kv_idx

    block_mask = create_block_mask(mask_fn, B=None, H=None, Q_LEN=q_len, KV_LEN=k_len, device=device)
    _block_mask_cache[cache_key] = block_mask
    return block_mask


# =============================================================================
# Attention cores. _attend_full = full attention (training, no-cache generation, AND
# cache prefill -- prefill is just attention over the written cache slice).
# _attend_decode = the incremental q_len=1 step. Dispatch is by MASK, not layer type:
# no mask (plain causal) -> SDPA is_causal (its mask-free flash kernel benchmarks ~2x
# faster than flex here); anything needing a mask (sliding window, left-pad, or
# decode's causal-up-to-cur over a fixed cache) -> flex. SDPA is NEVER handed an
# attn_mask -- that drops it off the flash path, which was the whole point of using it.
# =============================================================================
def _attend_full(q, k, v, window, left_pad, enable_gqa, kernel_options=None):
    """Full causal self-attention over aligned q,k,v (Tq == Tk), with optional sliding
    window and per-row left-pad. (B, H, T, D) layout.

    No mask (plain causal) -> SDPA is_causal (fastest; also the CPU path). Any mask
    (sliding window and/or left-pad) -> flex; masked attention requires the flex
    backend. kernel_options forces the flex tile config -- the prefill passes
    BLOCK_M=64 so the headdim=128 kernel fits sm120 shared memory (the autotuned
    default BLOCK_M=128/num_stages=3 needs ~145KB vs the ~100KB limit; BLOCK_M=64
    divides the 128 sparse block so the kernel's divisibility assert holds). None lets
    the autotuner choose (training / no-cache).
    """
    T = q.size(2)
    window = int(window)
    if not ((0 <= window < T) or (left_pad is not None)):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
    assert BACKEND == 'flex', "masked attention (sliding window / left-pad) requires the FlexAttention backend"
    block_mask = _get_block_mask(window, T, T, q.device)
    score_mod = None
    if left_pad is not None:
        neg_inf = float("-inf")
        def score_mod(s, b, h, q_idx, kv_idx):
            return torch.where(kv_idx >= left_pad[b], s, neg_inf)
    return _get_compiled_flex_attention()(
        q, k, v, block_mask=block_mask, score_mod=score_mod,
        enable_gqa=enable_gqa, kernel_options=kernel_options)


def _attend_decode(q, k_cache, v_cache, input_pos, window, left_pad, enable_gqa):
    """One incremental decode step: q is (B, H, 1, D) attending over the cache.

    Flex backend: score_mod over the FULL cache (B, H, T_max, D) -- constant shape so
    it composes with the reduce-overhead cudagraph. Mask per (b, kv): kv <= input_pos
    (causal up to the write position) [& window] [& kv >= left_pad[b]]; causal is
    needed even unpadded since the cache holds zeros past input_pos. Raw flex_attention
    so the outer compile owns it.

    CPU / no-flex: SDPA is_causal over the FILLED slice [0:cur+1] -- a single query at
    the last position attends all prior keys, so is_causal is exact and no mask is
    built. Unpadded full-causal only (windowed / left-pad decode requires flex).
    """
    window = int(window)
    if BACKEND == 'flex':
        from torch.nn.attention.flex_attention import flex_attention
        use_window = window >= 0
        neg_inf = float("-inf")
        def score_mod(score, b, h, q_idx, kv_idx):
            abs_q = input_pos[q_idx]
            keep = kv_idx <= abs_q
            if left_pad is not None:
                keep = keep & (kv_idx >= left_pad[b])
            if use_window:
                keep = keep & ((abs_q - kv_idx) <= window)
            return torch.where(keep, score, neg_inf)
        return flex_attention(q, k_cache, v_cache, score_mod=score_mod, enable_gqa=enable_gqa)

    assert left_pad is None and window < 0, "windowed / left-pad decode requires FlexAttention"
    end = int(input_pos[-1]) + 1  # filled cache length (host sync ok: CPU, no cudagraph)
    return F.scaled_dot_product_attention(
        q, k_cache[:, :, :end], v_cache[:, :, :end], is_causal=True, enable_gqa=enable_gqa)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), left_pad=None):
    """No-cache attention (training / generation prefill of a fresh sequence).
    q, k, v: (B, T, H, D). left_pad: optional (B,) int count of left-pad columns
    per row (left-padded batches). Returns (B, T, H, D).
    """
    if left_pad is None and BACKEND == 'fa3':
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # -> (B, H, T, D)
    y = _attend_full(q, k, v, window_size[0], left_pad, enable_gqa=q.size(1) != k.size(1))
    return y.transpose(1, 2)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, input_pos=None,
                            causal=False, window_size=(-1, -1), left_pad=None):
    """Attention with a KV cache. q: (B, T_new, H, D); cache: (B, T_max, H_kv, D);
    k, v: new keys/values written at input_pos. left_pad: optional (B,) left-pad
    column counts. Returns (B, T_new, H, D).

    Writes the new k,v into the cache, then:
      - PREFILL (T_new>1): full attention over the just-written [0:T_new) slice --
        i.e. exactly _attend_full, the same core training/no-cache use.
      - DECODE (T_new==1): incremental _attend_decode over the full cache.
    FA3 (Hopper, unpadded) keeps its native fused kvcache path.
    """
    if BACKEND == 'fa3' and left_pad is None:
        # FA3 advances the cache via cache_seqlens (shared across batch) and writes internally.
        B = q.shape[0]
        cache_seqlens = input_pos[:1].to(torch.int32).expand(B)
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size,
        )
    if BACKEND == 'fa3' and left_pad is not None:
        raise NotImplementedError(
            "FA3 left-pad batched decode (leftpad_k) is not wired up yet; "
            "run the batched eval with the flex backend for now"
        )

    if k is not None and v is not None:
        k_cache.index_copy_(1, input_pos, k)
        v_cache.index_copy_(1, input_pos, v)

    T_new = q.size(1)
    enable_gqa = q.size(2) != k_cache.size(2)
    if T_new > 1:
        # Prefill == full attention over the written slice [0:T_new); force the small
        # tile config so the headdim=128 flex kernel fits sm120 shared memory.
        y = _attend_full(q.transpose(1, 2), k_cache[:, :T_new].transpose(1, 2),
                         v_cache[:, :T_new].transpose(1, 2), window_size[0], left_pad, enable_gqa,
                         kernel_options=_FLEX_PREFILL_KERNEL_OPTIONS)
    else:
        y = _attend_decode(q.transpose(1, 2), k_cache.transpose(1, 2), v_cache.transpose(1, 2),
                           input_pos, window_size[0], left_pad, enable_gqa)
    return y.transpose(1, 2)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
