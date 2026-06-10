"""Isolate the per-step decode attention kernel: score_mod (dense) vs block_mask
(sparse) vs SDPA-cuDNN-mask, at d24-byte-l decode shapes. Decides whether a block
mask is worth wiring into the decode path."""
import time, argparse
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.attention import SDPBackend, sdpa_kernel

def bench(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6  # us/call

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()
    dev = torch.device("cuda"); dt = torch.bfloat16
    B, H, D = args.B, args.H, args.D
    flex_c = torch.compile(flex_attention)

    for T_max in (512, 2048, 8192):
        cur = T_max // 2  # half-filled cache (typical mid-generation)
        q = torch.randn(B, H, 1, D, device=dev, dtype=dt)
        k = torch.randn(B, H, T_max, D, device=dev, dtype=dt)
        v = torch.randn(B, H, T_max, D, device=dev, dtype=dt)
        left_pad = torch.randint(0, 8, (B,), device=dev)
        input_pos = torch.tensor([cur], device=dev)

        # 1) dense score_mod (current decode path)
        def score_mod(s, b, h, qi, ki):
            keep = (ki <= input_pos[qi]) & (ki >= left_pad[b])
            return torch.where(keep, s, float("-inf"))
        f_score = lambda: flex_c(q, k, v, score_mod=score_mod)

        # 2) block_mask (sparse: skips future + all-pad blocks)
        def mask_mod(b, h, qi, ki):
            return (ki <= input_pos[qi]) & (ki >= left_pad[b])
        bm = create_block_mask(mask_mod, B=B, H=None, Q_LEN=1, KV_LEN=T_max, device=dev)
        f_block = lambda: flex_c(q, k, v, block_mask=bm)

        # 3) SDPA + cuDNN with bool mask (old decode path)
        kp = torch.arange(T_max, device=dev)
        m = (kp.unsqueeze(0) <= input_pos.unsqueeze(1))  # (1, T_max) causal
        m = m.unsqueeze(0).unsqueeze(0) & (kp.view(1,1,1,T_max) >= left_pad.view(B,1,1,1))
        def f_sdpa():
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

        # 4) block_mask REBUILT each step (worst case: pays create_block_mask every step)
        def f_block_rebuild():
            bm2 = create_block_mask(mask_mod, B=B, H=None, Q_LEN=1, KV_LEN=T_max, device=dev)
            return flex_c(q, k, v, block_mask=bm2)

        # 5) AMORTIZED: rebuild once per 128-block, reuse (cur tensor updated in between)
        BLK = 128
        def f_block_amortized():
            for i in range(BLK):
                if i == 0:
                    bm2 = create_block_mask(mask_mod, B=B, H=None, Q_LEN=1, KV_LEN=T_max, device=dev)
                input_pos.fill_(cur + i if cur + i < T_max else T_max - 1)  # cur advances within block
                flex_c(q, k, v, block_mask=bm2)

        ts, tb, td = bench(f_score), bench(f_block), bench(f_sdpa)
        tbr = bench(f_block_rebuild, iters=50)
        tam = bench(f_block_amortized, iters=10, warmup=3) / BLK  # per-step
        print(f"T_max={T_max:5d} cur={cur:5d}: score_mod={ts:7.1f}us  block={tb:7.1f}us  "
              f"block_amortized={tam:7.1f}us  block_rebuild={tbr:7.1f}us  sdpa={td:7.1f}us")

if __name__ == "__main__":
    main()
