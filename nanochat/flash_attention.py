"""
Unified Flash Attention interface with automatic backend dispatch.

Exports `flash_attn` module that matches the FA3 API exactly. Primary backend:
    fa3   - Flash Attention 3, Hopper (sm_90) only, fastest path (bf16/fp8)
    flex  - PyTorch FlexAttention everywhere else (the default; runs eager on CPU)
Within these, plain causal attention uses SDPA is_causal (its mask-free flash kernel
benchmarks fastest); any masked attention (sliding window, left-pad, decode) uses flex.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import os
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


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
# FlexAttention is assumed always available (PyTorch >= 2.5; runs eager on CPU, fused
# on GPU). It's the default backend whenever FA3 isn't used.

# Override for testing: 'fa3', 'flex', or None (auto)
_override_impl = None


def _resolve_backend():
    """Pick the primary attention backend: 'fa3' (Hopper, bf16) or 'flex' (otherwise)."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return 'fa3'
    if _override_impl == 'flex':
        return 'flex'
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16/fp8; other dtypes fall through to flex.
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return 'fa3'
    return 'flex'


BACKEND = _resolve_backend()  # 'fa3' or 'flex'
USE_FA3 = BACKEND == 'fa3'  # kept for backward compat with anything that imports this


# =============================================================================
# FlexAttention helpers (block mask cache + lazy compile)
# =============================================================================
_block_mask_cache = {}
_compiled_flex_attention = None

# OPT-IN small tile for masked PREFILL (Tq>1, headdim=128), enabled by
# NANOCHAT_FLEX_PREFILL_PIN. OFF by default because it's ~1.6-2x slower than the
# autotuned default tile, and the default fits the production path (Blackwell/sm120
# bf16). Turn it on where the default overflows the ~99KB per-block shared-memory cap
# on consumer GPUs and fails with "no valid triton configs": fp32 anywhere (incl.
# Blackwell, ~115KB) and bf16 on the 3090/sm86 (~106KB). BLOCK_M=64 divides the 128
# sparse block (SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0) so it fits. Training is unaffected
# either way (no-cache always uses the autotuner). Datacenter GPUs (164-228KB/SM) fit
# the default, which is why gpt-fast never pins.
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

    No mask (plain causal) -> SDPA is_causal (benchmarks ~2x faster than flex here, and
    it's the CPU path too). Any mask (sliding window and/or left-pad) -> flex.
    kernel_options pins the flex tile config; the prefill passes
    _FLEX_PREFILL_KERNEL_OPTIONS so the headdim=128 kernel fits the GPU's per-block
    shared-memory cap (see that constant). None lets the autotuner choose (training /
    no-cache).
    """
    T = q.size(2)
    window = int(window)
    if not ((0 <= window < T) or (left_pad is not None)):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
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
    """One incremental decode step: q is (B, H, 1, D) attending over the FULL cache
    (B, H, T_max, D) via a flex score_mod. Constant cache shape across steps, so it
    composes with the reduce-overhead cudagraph; raw flex_attention so the outer
    compile owns it (eager on CPU). Mask per (b, kv): kv <= input_pos (causal up to
    the write position) [& window] [& kv >= left_pad[b]] -- causal is needed even
    unpadded since the cache holds zeros past input_pos.
    """
    from torch.nn.attention.flex_attention import flex_attention
    window = int(window)
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
        # Prefill == full attention over the written slice [0:T_new). Autotuned tile by
        # default (fastest, fits on Blackwell bf16); pin the small tile only when
        # NANOCHAT_FLEX_PREFILL_PIN is set (fp32 / smaller cards). See the constant.
        ko = _FLEX_PREFILL_KERNEL_OPTIONS if os.environ.get("NANOCHAT_FLEX_PREFILL_PIN") else None
        y = _attend_full(q.transpose(1, 2), k_cache[:, :T_new].transpose(1, 2),
                         v_cache[:, :T_new].transpose(1, 2), window_size[0], left_pad, enable_gqa,
                         kernel_options=ko)
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
