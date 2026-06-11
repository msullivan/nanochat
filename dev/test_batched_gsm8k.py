"""Validate batched GSM8K (calculator tool via generate_batched) against the
per-problem path (generate_batch). fp32 + greedy -> completions should match
exactly, and the tool path must actually fire (some completions contain a
<|python_start|> block). Run on a tool-trained chat checkpoint.

  PYTHONPATH=. python dev/test_batched_gsm8k.py --src ~/.cache/nanochat/chatsft_checkpoints/d24-byte-l-ext-chat
"""
import os
os.environ.setdefault("NANOCHAT_DTYPE", "float32")
os.environ.setdefault("NANOCHAT_FLEX_PREFILL_PIN", "1")  # fp32 cache-prefill needs the small flex tile
import argparse
import torch
from nanochat.checkpoint_manager import build_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    step = args.step
    if step is None:
        step = max(int(f[6:-3]) for f in os.listdir(args.src) if f.startswith("model_") and f.endswith(".pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, meta = build_model(checkpoint_dir=args.src, step=step, device=device, phase="eval")
    eng = Engine(model, tok)
    py_start = tok.encode_special("<|python_start|>")

    task = GSM8K(subset="main", split="test")
    N = min(args.n, len(task))
    convs = [task[i] for i in range(N)]
    prompts = [tok.render_for_completion(c) for c in convs]

    # Reference: per-problem (generate_batch -> generate -> _generate_rows), tools on.
    ref_comp, ref_pass, tool_used = [], [], 0
    for c, p in zip(convs, prompts):
        results, _ = eng.generate_batch(p, num_samples=1, max_tokens=args.max_new_tokens, temperature=0.0)
        gen = results[0][len(p):]
        ref_comp.append(gen)
        ref_pass.append(bool(task.evaluate(c, tok.decode(gen))))
        if py_start in gen:
            tool_used += 1

    # Batched: generate_batched, tools on, in fixed-size chunks.
    bat_comp, bat_pass = [None] * N, [None] * N
    for s in range(0, N, args.batch):
        chunk = prompts[s:s + args.batch]; cc = convs[s:s + args.batch]
        gens = eng.generate_batched(chunk, max_tokens=args.max_new_tokens, temperature=0.0)
        for j in range(len(chunk)):
            bat_comp[s + j] = gens[j]
            bat_pass[s + j] = bool(task.evaluate(cc[j], tok.decode(gens[j])))

    exact = sum(a == b for a, b in zip(ref_comp, bat_comp))
    verdict_agree = sum(a == b for a, b in zip(ref_pass, bat_pass))
    print(f"tool-path fired on {tool_used}/{N} problems (per-problem ref)")
    print(f"exact completion match: {exact}/{N}")
    print(f"pass/fail agreement:    {verdict_agree}/{N}  (ref acc {sum(ref_pass)}/{N}, batched acc {sum(bat_pass)}/{N})")
    if exact < N:
        for i in range(N):
            if ref_comp[i] != bat_comp[i]:
                print(f"  mismatch i={i}: ref_len={len(ref_comp[i])} bat_len={len(bat_comp[i])} "
                      f"ref_pass={ref_pass[i]} bat_pass={bat_pass[i]}")
    ok = (verdict_agree == N) and (tool_used > 0)
    print("BATCHED-GSM8K OK" if ok else "BATCHED-GSM8K: check mismatches above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
