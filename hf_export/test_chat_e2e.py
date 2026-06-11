"""End-to-end chat generation through the exported HF artifact.

Exercises the full serving path that --verify does NOT cover: a real
apply_chat_template -> cached generate -> decode round-trip on a 265-vocab CHAT
checkpoint (the base ckpts --verify uses are vocab-256, so the chat specials
>=256 and the chat template are never exercised there).

Path notes: chat checkpoints are window_pattern "L" (full causal), so prefill
takes the SDPA is_causal core and decode takes flex (q_len=1) -- neither hits the
flex headdim-128 prefill SMEM cap, so this runs on consumer GPUs and CPU alike.
Run with NANOCHAT_DTYPE=float32 to match the bit-exact verified config; bf16 is
fine for a smoke test but diverges numerically.

Usage (point --dst at a generated artifact, e.g. from generate_hf_model.py):
    CUDA_DEVICE_ORDER=PCI_BUS_ID NANOCHAT_DTYPE=float32 \
        python hf_export/test_chat_e2e.py --dst /tmp/hf_chat
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "What is 17 + 25?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True, help="generated HF artifact dir (265-vocab chat model)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    print(f"device={dev} ({name})")

    tok = AutoTokenizer.from_pretrained(args.dst, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.dst, trust_remote_code=True, dtype=torch.float32).to(dev).eval()
    print(f"loaded: eos={tok.eos_token_id} (<|assistant_end|>) pad={tok.pad_token_id}")

    for prompt in PROMPTS:
        # add_generation_prompt=True appends <|assistant_start|>; return_dict=True
        # because transformers v5 apply_chat_template returns a BatchEncoding.
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(dev)
        mask = enc.get("attention_mask")
        mask = mask.to(dev) if mask is not None else torch.ones_like(ids)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=mask, max_new_tokens=args.max_new_tokens,
                do_sample=False, use_cache=True,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        new = out[0, ids.shape[1]:]  # only the freshly generated continuation
        resp = tok.decode(new, skip_special_tokens=True)
        print(f"\nUSER: {prompt}\nASSISTANT: {resp}\n[{new.shape[0]} toks in {time.time() - t0:.1f}s]\n" + "-" * 60)


if __name__ == "__main__":
    main()
