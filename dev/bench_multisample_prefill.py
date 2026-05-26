"""
Benchmark: prefill cost as a function of (prompt_len, num_samples).

The cudagraph-branch refactor changed multi-sample prefill from "B=1 prefill
then broadcast cache to B=N for decode" to "B=N prefill (prompt duplicated
across batch dim)". For short prompts this is irrelevant; for long prompts +
N>1 it could be an Nx slowdown of prefill.

This script measures wallclock for several (prompt_len, num_samples) cells
so we can decide if the regression matters in practice. Run on both master
and the cudagraph branch and compare.

Usage:
    .venv/bin/python dev/bench_multisample_prefill.py [--model-tag TAG] [--source base|sft|...]
"""
import argparse
import time

import torch

from nanochat.common import autodetect_device_type, compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="base", choices=["base", "sft", "rl", "cute"])
    p.add_argument("--model-tag", default=None)
    p.add_argument("--max-tokens", type=int, default=32,
                   help="decode tokens per generation; keep low to isolate prefill cost")
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                        model_tag=args.model_tag)
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()

    cells = [
        # (prompt_len, num_samples)
        (1,    1),    # baseline: trivial prefill
        (1,    8),    # multi-sample, trivial prefill (this is base_eval's path)
        (128,  1),    # medium prompt, single sample
        (128,  8),    # medium prompt + multi-sample (the regressed case)
        (512,  1),    # long prompt, single sample
        (512,  8),    # long prompt + multi-sample (worst case for regression)
        (1024, 1),
        (1024, 8),
    ]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_tag or args.source}")
    print(f"max_tokens={args.max_tokens}, repeats={args.repeats}")
    print()
    print(f"{'prompt_len':>10} {'num_samples':>11} {'mean_s':>8} {'min_s':>8}")

    for prompt_len, num_samples in cells:
        # Build a synthetic prompt of the requested length using random valid tokens.
        # (Real text would be more realistic but isn't necessary for timing.)
        torch.manual_seed(42)
        prompt_tokens = [bos] + [int(t) for t in torch.randint(
            low=1, high=min(256, model.config.vocab_size), size=(prompt_len - 1,))]

        # Warmup: triggers cache setup, cudagraph capture (one-time costs we
        # don't want to count). One generation warmup per (prompt_len, num_samples)
        # combo, because cudagraph capture happens per num_samples.
        _, _ = engine.generate_batch(prompt_tokens, num_samples=num_samples,
                                     max_tokens=args.max_tokens, temperature=1.0)
        torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, _ = engine.generate_batch(prompt_tokens, num_samples=num_samples,
                                         max_tokens=args.max_tokens, temperature=1.0)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        mean = sum(times) / len(times)
        print(f"{prompt_len:>10} {num_samples:>11} {mean:>8.3f} {min(times):>8.3f}")


if __name__ == "__main__":
    main()
