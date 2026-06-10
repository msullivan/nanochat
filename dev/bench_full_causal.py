"""Does the full-causal SDPA is_causal path actually beat flex? Compare SDPA
is_causal vs compiled flex (causal block_mask) for full attention at training/
prefill shapes. head_dim=128, bf16. flex full-causal needs BLOCK_M=64 to fit sm120."""
import time, torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
dev = torch.device("cuda"); dt = torch.bfloat16
flex_c = torch.compile(flex_attention)
H, D = 12, 128
def bench(fn, it=100, wu=25):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e6
for B, T in [(8, 512), (8, 2048), (4, 2048), (1, 2048)]:
    q = torch.randn(B, H, T, D, device=dev, dtype=dt)
    k = torch.randn(B, H, T, D, device=dev, dtype=dt)
    v = torch.randn(B, H, T, D, device=dev, dtype=dt)
    fs = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    bm = create_block_mask(lambda b, h, qi, ki: qi >= ki, B=None, H=None, Q_LEN=T, KV_LEN=T, device=dev)
    ko = {"BLOCK_M": 64, "BLOCK_N": 32, "num_stages": 2}
    ff = lambda: flex_c(q, k, v, block_mask=bm, kernel_options=ko)
    ff_def = None
    # also try flex default config if it fits
    def _try_default():
        try:
            flex_c(q, k, v, block_mask=bm); torch.cuda.synchronize(); return bench(lambda: flex_c(q, k, v, block_mask=bm))
        except Exception:
            return None
    ts, tf = bench(fs), bench(ff)
    td = _try_default()
    ds = f"  flex_default={td:.1f}us" if td else "  flex_default=OOM/fail"
    print(f"B={B} T={T:5d}: SDPA_is_causal={ts:7.1f}us  flex_causal(BM64)={tf:7.1f}us  ratio={tf/ts:.2f}x{ds}")
