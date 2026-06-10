"""Decode throughput bench: single-stream + batched, steady-state (post-compile)."""
import os, time, argparse
import torch
from nanochat.checkpoint_manager import build_model
from nanochat.engine import Engine
import nanochat.flash_attention as fa

def timed(fn, *a, **k):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = fn(*a, **k)
    torch.cuda.synchronize(); return r, time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    step = args.step or max(int(f[6:-3]) for f in os.listdir(args.src) if f.startswith("model_") and f.endswith(".pt"))
    dev = torch.device("cuda")
    model, tok, meta = build_model(checkpoint_dir=args.src, step=step, device=dev, phase="eval")
    seqlen = meta["model_config"]["sequence_len"]
    print(f"backend={fa.BACKEND} seqlen={seqlen} dtype(default)={'bf16' if dev.type=='cuda' else 'f32'}")
    eng = Engine(model, tok)
    vocab = meta["model_config"]["vocab_size"]
    torch.manual_seed(0)
    prompt = torch.randint(1, vocab, (16,)).tolist()
    K = args.max_tokens

    # ---- single stream ---- (warm up with the SAME max_tokens so the cache bucket
    # matches the timed run -- otherwise a recompile lands inside the timed window)
    eng.generate_batch(prompt, num_samples=1, max_tokens=K)        # warmup (compile+capture)
    (_r, _m), dt = timed(eng.generate_batch, prompt, num_samples=1, max_tokens=K)
    print(f"single-stream: {K/dt:8.1f} tok/s  ({dt*1000:.0f} ms for {K} tok)")

    # ---- batched ----
    prompts = [torch.randint(1, vocab, (16,)).tolist() for _ in range(args.batch)]
    eng.generate_batched(prompts, max_tokens=K)                    # warmup (same bucket)
    _r, dt = timed(eng.generate_batched, prompts, max_tokens=K)
    print(f"batched B={args.batch}: {args.batch*K/dt:8.1f} tok/s aggregate  ({K/dt:.1f} tok/s/row, {dt*1000:.0f} ms)")

if __name__ == "__main__":
    main()
