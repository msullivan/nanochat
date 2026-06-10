"""Validate the BlockMask-SLICE decode pattern (precompute once, slice per step,
captured offset) for correctness vs dense score_mod, and measure per-step cost.

This is the pattern from the PyTorch FlexAttention-for-Inference blog: no
create_block_mask per step, just block_mask[:, :, cur//BLOCK] + a mutated offset.
"""
import time, argparse
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def bench(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6

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
        q = torch.randn(B, H, 1, D, device=dev, dtype=dt)
        k = torch.randn(B, H, T_max, D, device=dev, dtype=dt)
        v = torch.randn(B, H, T_max, D, device=dev, dtype=dt)
        left_pad = torch.randint(0, 8, (B,), device=dev)

        # precompute ONCE: shared causal block mask over the full cache (B=None).
        # q-block size 1 (kv-block 128) so a single decode token is its own q-block
        # -> slicing yields q_len=1 directly (no _adjust below block granularity).
        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        base = create_block_mask(causal, B=None, H=None, Q_LEN=T_max, KV_LEN=T_max,
                                 device=dev, BLOCK_SIZE=(1, 128))

        # captured offset (absolute query position); mutated in place each step
        offset = torch.zeros((), dtype=torch.long, device=dev)
        def sliced_mask_mod(b, h, q_idx, kv_idx):
            aq = q_idx + offset
            return (aq >= kv_idx) & (kv_idx >= left_pad[b])

        def step(cur):
            offset.fill_(cur)
            sl = base[:, :, torch.tensor([cur], device=dev, dtype=torch.int32)]  # q-block=cur, q_len=1
            sl.mask_mod = sliced_mask_mod
            return flex_c(q, k, v, block_mask=sl)

        # correctness vs dense score_mod at a few positions
        def dense(cur):
            ip = torch.tensor([cur], device=dev)
            def sm(s, b, h, qi, ki):
                keep = (ki <= ip[qi]) & (ki >= left_pad[b])
                return torch.where(keep, s, float("-inf"))
            return flex_c(q, k, v, score_mod=sm)
        maxerr = 0.0
        for cur in (left_pad.max().item()+1, T_max//3, T_max//2, T_max-1):
            a = step(cur); d = dense(cur)
            maxerr = max(maxerr, (a.float() - d.float()).abs().max().item())

        # per-step cost (slicing, no rebuild). cur fixed mid-cache (steady state)
        cur = T_max // 2
        t = bench(lambda: step(cur))
        print(f"T_max={T_max:5d} BLK={BLK}: slice_step={t:7.1f}us  max|diff vs dense|={maxerr:.2e}")

if __name__ == "__main__":
    main()
