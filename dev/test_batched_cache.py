"""Correctness test for Engine.generate_batched (cached, left-padded batching).

Compares, with greedy decoding:
  (a) generate_batched(prompts)      -- batched, left-padded, masked cache
  (b) generate_batched([p]) per p    -- B=1, no padding (isolates the pad/mask)
  (c) generate_batch(p) per p        -- the established single-prompt cached path

(a)==(b) proves the left-pad masking is correct; (b)==(c) proves the new method
matches the existing cached generation.

  PYTHONPATH=. python dev/test_batched_cache.py --src <ckpt>
"""
import os
os.environ.setdefault("NANOCHAT_DTYPE", "float32")
import argparse
import torch
from nanochat.checkpoint_manager import build_model
from nanochat.engine import Engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    step = args.step
    if step is None:
        step = max(int(f[6:-3]) for f in os.listdir(args.src) if f.startswith("model_") and f.endswith(".pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, meta = build_model(checkpoint_dir=args.src, step=step, device=device, phase="eval")
    vocab = meta["model_config"]["vocab_size"]
    eng = Engine(model, tok)

    torch.manual_seed(0)
    prompts = [torch.randint(1, vocab, (torch.randint(4, 14, (1,)).item(),)).tolist() for _ in range(args.batch)]
    MNT = args.max_new_tokens

    batched = eng.generate_batched(prompts, max_tokens=MNT, temperature=0.0)
    b1 = [eng.generate_batched([p], max_tokens=MNT, temperature=0.0)[0] for p in prompts]
    ref = []
    for p in prompts:
        res, _ = eng.generate_batch(p, num_samples=1, max_tokens=MNT, temperature=0.0)
        ref.append(res[0][len(p):])

    m_ab = sum(a == b for a, b in zip(batched, b1))
    m_bc = sum(b == c for b, c in zip(b1, ref))
    print(f"(a) batched == (b) B=1 padded-isolated : {m_ab}/{len(prompts)}")
    print(f"(b) B=1     == (c) Engine.generate_batch: {m_bc}/{len(prompts)}")
    ok = m_ab == len(prompts) and m_bc == len(prompts)
    if not ok:
        for i, (a, b, c) in enumerate(zip(batched, b1, ref)):
            if a != b or b != c:
                print(f"  row {i} len(prompt)={len(prompts[i])}: ab={a==b} bc={b==c}")
    print("CACHE-BATCH OK" if ok else "CACHE-BATCH: divergence (bf16 argmax ties if on GPU)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
