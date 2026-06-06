"""Batched, left-padded greedy/sampled generation using the no-cache masked forward.

Distinct prompts are left-padded into one batch and decoded together (one forward
per step over the whole batch), with per-row EOS stopping. No KV cache -- O(n^2) in
length but batched across prompts, which is the win for short-answer evals (CUTE).

Run modes:
  --test   : batched outputs == per-prompt (B=1) outputs  [correctness]
  --bench  : time batched vs per-prompt loop              [speedup]

  PYTHONPATH=. python dev/batched_generate.py --src <ckpt> --test
  PYTHONPATH=. python dev/batched_generate.py --src <ckpt> --bench --batch 64
"""
import os
import time
import argparse
import torch
from nanochat.checkpoint_manager import build_model


@torch.inference_mode()
def batched_generate(model, prompts, max_new_tokens, eos_ids=(), pad_id=0,
                     temperature=0.0, top_k=None, seed=42):
    """prompts: list[list[int]]. Returns list[list[int]] of generated tokens
    (prompt and terminal eos excluded)."""
    device = model.get_device()
    B = len(prompts)
    L = max(len(p) for p in prompts)
    ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros(B, L, dtype=torch.float32, device=device)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p, device=device)
        mask[i, L - len(p):] = 1.0

    eos = set(eos_ids)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    out = [[] for _ in range(B)]
    rng = None
    if temperature > 0:
        rng = torch.Generator(device=device); rng.manual_seed(seed)

    for _ in range(max_new_tokens):
        logits = model(ids, attention_mask=mask)[:, -1, :]  # (B, vocab)
        if temperature == 0.0:
            nxt = logits.argmax(-1)
        else:
            lg = logits / temperature
            if top_k:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg[lg < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=rng)[:, 0]
        nxt_list = nxt.tolist()
        for i in range(B):
            if not done[i]:
                if nxt_list[i] in eos:
                    done[i] = True
                else:
                    out[i].append(nxt_list[i])
        if bool(done.all()):
            break
        nxt_col = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        ids = torch.cat([ids, nxt_col.unsqueeze(1)], dim=1)
        mask = torch.cat([mask, (~done).float().unsqueeze(1)], dim=1)
    return out


def _load(src, step):
    if step is None:
        step = max(int(f[6:-3]) for f in os.listdir(src) if f.startswith("model_") and f.endswith(".pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, meta = build_model(checkpoint_dir=src, step=step, device=device, phase="eval")
    return model, tok, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    model, tok, meta = _load(args.src, args.step)
    vocab = meta["model_config"]["vocab_size"]
    torch.manual_seed(0)
    prompts = [torch.randint(1, vocab, (torch.randint(4, 14, (1,)).item(),)).tolist()
               for _ in range(args.batch)]

    if args.test:
        ref = [batched_generate(model, [p], args.max_new_tokens)[0] for p in prompts]
        bat = batched_generate(model, prompts, args.max_new_tokens)
        n_match = sum(r == b for r, b in zip(ref, bat))
        print(f"batched vs per-prompt greedy: {n_match}/{len(prompts)} exact-match")
        if n_match < len(prompts):
            for i, (r, b) in enumerate(zip(ref, bat)):
                if r != b:
                    # first divergence position
                    j = next((k for k in range(min(len(r), len(b))) if r[k] != b[k]), min(len(r), len(b)))
                    print(f"  row {i}: diverge at {j} (len ref={len(r)} bat={len(b)})")
        print("TEST OK" if n_match == len(prompts) else "TEST: some divergence (greedy argmax ties under float noise?)")

    if args.bench:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        _ = batched_generate(model, prompts, args.max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_bat = time.time() - t0

        t0 = time.time()
        for p in prompts:
            _ = batched_generate(model, [p], args.max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_loop = time.time() - t0
        print(f"batch={args.batch} max_new={args.max_new_tokens}  "
              f"batched={t_bat:.2f}s  per-prompt-loop={t_loop:.2f}s  speedup={t_loop/t_bat:.1f}x")


if __name__ == "__main__":
    main()
