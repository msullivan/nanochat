"""Correctness test for left-padded batched forward (left_pad support).

Checks that GPT.forward(ids, left_pad=counts) on a LEFT-PADDED batch produces,
at each prompt's real positions, the same logits as forwarding that prompt alone
unpadded. The first real token is the strict case: it exercises the smear-boundary
masking (without it, the first real token would be smeared with the pad token).

NOTE: the left-padded path now routes attention through FlexAttention, which is
CUDA-only -- this test requires a Flex-capable GPU (run it on genai). float32 keeps
the comparison tight.

Usage: python dev/test_batched_padding.py --src ~/.cache/nanochat/base_checkpoints/d8-byte
"""
import os
os.environ.setdefault("NANOCHAT_DTYPE", "float32")
import argparse
import torch
from nanochat.checkpoint_manager import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--atol", type=float, default=2e-4)
    args = ap.parse_args()

    step = args.step
    if step is None:
        step = max(int(f[6:-3]) for f in os.listdir(args.src) if f.startswith("model_") and f.endswith(".pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _tok, meta = build_model(checkpoint_dir=args.src, step=step, device=device, phase="eval")
    model = model.to(torch.float32)
    vocab = meta["model_config"]["vocab_size"]

    torch.manual_seed(0)
    lengths = [12, 5, 9, 7, 3]
    prompts = [torch.randint(0, vocab, (n,)).tolist() for n in lengths]

    # per-prompt unpadded reference
    with torch.no_grad():
        per = [model(torch.tensor([p], device=device))[0] for p in prompts]  # each (len, vocab)

    # left-padded batch
    L = max(lengths)
    B = len(prompts)
    ids = torch.zeros(B, L, dtype=torch.long, device=device)
    left_pad = torch.tensor([L - len(p) for p in prompts], dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p, device=device)
    with torch.no_grad():
        batched = model(ids, left_pad=left_pad)  # (B, L, vocab)

    ok = True
    worst = 0.0
    first_tok_worst = 0.0
    for i, p in enumerate(prompts):
        real = batched[i, L - len(p):, :]              # (len, vocab)
        d = (real - per[i]).abs().max().item()
        first = (real[0] - per[i][0]).abs().max().item()  # first real token (smear boundary)
        last_argmax = (real[-1].argmax() == per[i][-1].argmax()).item()  # what generation reads
        worst = max(worst, d)
        first_tok_worst = max(first_tok_worst, first)
        ok &= d < args.atol and last_argmax
        print(f"  len={len(p):2d}  max|diff|={d:.2e}  first-token|diff|={first:.2e}  last-argmax-match={last_argmax}")

    print(f"worst max|diff|={worst:.2e}  worst first-token|diff|={first_tok_worst:.2e}")
    print("BATCHED-PAD OK" if ok else "BATCHED-PAD FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
