"""Tokenizer parity: HF NanochatByteTokenizer vs nanochat ByteTokenizer.

Checks:
  - special-token ids land at exactly 256..264 (baked into embedding rows)
  - raw byte round-trip (incl. multibyte UTF-8) is lossless
  - apply_chat_template matches render_conversation / render_for_completion
    byte-for-byte (system-merge, generation prompt, multibyte content)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from tokenization_nanochat import NanochatByteTokenizer, SPECIAL_TOKENS
from nanochat.byte_tokenizer import ByteTokenizer


def main():
    tok = NanochatByteTokenizer()
    tok.chat_template = open(os.path.join(HERE, "chat_template.jinja")).read()
    ref = ByteTokenizer()  # canonical 265-wide scheme

    ok = True

    # 1) special-token ids
    ids = [tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS]
    special_ok = ids == list(range(256, 265)) and len(tok) == 265
    print(f"special ids 256..264 + len==265: {special_ok}  ({ids}, len={len(tok)})")
    ok &= special_ok

    # 2) raw byte round-trip
    for s in ["Hello, world!", "Héllo 世界 — 🚀\ttabs\nnewlines", "".join(chr(i) for i in range(32, 127))]:
        enc = tok.encode(s, add_special_tokens=False)
        rt_ok = all(0 <= i < 256 for i in enc) and tok.decode(enc, skip_special_tokens=True) == s
        ok &= rt_ok
        if not rt_ok:
            print(f"  ROUND-TRIP FAIL: {s!r}")
    print("raw byte round-trip lossless: True" if ok else "raw byte round-trip: FAIL")

    # 3) chat template vs reference
    cases = [
        [{"role": "user", "content": 'Spell "cat".'}, {"role": "assistant", "content": "c a t"}],
        [{"role": "system", "content": "You are nanochat."},
         {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}],
        [{"role": "user", "content": "Héllo 世界"}, {"role": "assistant", "content": "hey"}],
    ]
    for i, msgs in enumerate(cases):
        hf_full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)["input_ids"]
        rf_full, _ = ref.render_conversation({"messages": msgs})
        hf_gen = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)["input_ids"]
        rf_gen = ref.render_for_completion({"messages": msgs})
        case_ok = hf_full == rf_full and hf_gen == rf_gen
        ok &= case_ok
        print(f"chat case {i}: full={hf_full == rf_full} gen={hf_gen == rf_gen}")

    print("TOKENIZER PARITY OK" if ok else "TOKENIZER PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
