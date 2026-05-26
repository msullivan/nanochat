"""
Microbenchmark + backend introspection for SDPA in our decode shape.

Decode pattern: q (B, H, 1, D), k/v (B, H_kv, T_max, D), with our bool
attn_mask blocking positions > current_pos. Compare against three
alternatives:
  1. Our path: full-cache + bool mask  (currently used)
  2. Sliced cache + is_causal=True     (gets Flash backend on supported HW)
  3. Force each SDPA backend explicitly + report which ones succeed

If (2) is dramatically faster, the per-token cost we're paying is the
slow attention backend, and switching to FlexAttention (which compiles
a Triton kernel tuned to our mask pattern) would help.
"""
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def time_call(fn, n_warmup=10, n_iter=100):
    """Time fn() over n_iter iterations after n_warmup warmups. Returns ms/call."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n_iter


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    device = "cuda"
    dtype = torch.bfloat16

    # Matches nanochat d24 inference decode shapes
    B = 1
    H = 8
    H_KV = 4
    D = 64
    T_MAX = 2048
    CURRENT_POS = 50  # arbitrary mid-prefill position

    print(f"Device: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")
    print(f"Shapes: q (B={B}, H={H}, T=1, D={D}) | k/v (B={B}, H_kv={H_KV}, T_max={T_MAX}, D={D})")
    print(f"Current valid prefix length: {CURRENT_POS+1}")
    print()

    q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    k_full = torch.randn(B, H_KV, T_MAX, D, device=device, dtype=dtype)
    v_full = torch.randn(B, H_KV, T_MAX, D, device=device, dtype=dtype)

    # Our path: full-cache + bool mask
    key_positions = torch.arange(T_MAX, device=device)
    q_pos = torch.tensor([CURRENT_POS], device=device)
    attn_mask_full = (key_positions.unsqueeze(0) <= q_pos.unsqueeze(1))  # (1, T_MAX)

    # Sliced-cache + is_causal: only the valid prefix of the cache
    k_sliced = k_full[:, :, :CURRENT_POS + 1, :]  # (B, H_kv, CURRENT_POS+1, D)
    v_sliced = v_full[:, :, :CURRENT_POS + 1, :]

    # 1. Default SDPA backend selection with our mask
    def f_mask_default():
        return F.scaled_dot_product_attention(q, k_full, v_full,
                                              attn_mask=attn_mask_full, enable_gqa=True)
    t_mask_default = time_call(f_mask_default)
    print(f"[1] SDPA default backend, full cache + bool mask : {t_mask_default:7.3f} ms/call")

    # 2. is_causal=True on the sliced cache (NOT shape-stable; this is the "old" pattern)
    def f_sliced_causal():
        return F.scaled_dot_product_attention(q, k_sliced, v_sliced,
                                              is_causal=False, enable_gqa=True)
    # For decode (Tq=1, Tk=CURRENT_POS+1), causal-from-end means "attend to all valid keys",
    # which == is_causal=False with no mask (since Tq=1 attends to everything in Tk anyway).
    t_sliced_causal = time_call(f_sliced_causal)
    print(f"[2] SDPA default backend, sliced cache, no mask : {t_sliced_causal:7.3f} ms/call")

    print()
    print("Backend probe: force each SDPA backend with our shapes + mask")
    backends = [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]
    for name, backend in backends:
        # Test with mask
        try:
            def f():
                with sdpa_kernel([backend]):
                    return F.scaled_dot_product_attention(q, k_full, v_full,
                                                          attn_mask=attn_mask_full, enable_gqa=True)
            t = time_call(f)
            print(f"  [{name:24}] with mask        : {t:7.3f} ms/call")
        except Exception as e:
            print(f"  [{name:24}] with mask        : REFUSED  ({type(e).__name__}: {str(e)[:80]})")
        # Test with is_causal on sliced cache
        try:
            def f():
                with sdpa_kernel([backend]):
                    return F.scaled_dot_product_attention(q, k_sliced, v_sliced,
                                                          is_causal=False, enable_gqa=True)
            t = time_call(f)
            print(f"  [{name:24}] sliced, no mask  : {t:7.3f} ms/call")
        except Exception as e:
            print(f"  [{name:24}] sliced, no mask  : REFUSED  ({type(e).__name__}: {str(e)[:80]})")


if __name__ == "__main__":
    main()
