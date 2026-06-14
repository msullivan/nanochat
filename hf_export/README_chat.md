---
language:
- en
license: mit
library_name: transformers
base_model: BASE_REPO_ID
pipeline_tag: text-generation
tags:
- nanochat
- byte-level
- custom_code
datasets:
- karpathy/climbmix-400b-shuffle
- HuggingFaceTB/smoltalk
- cais/mmlu
- openai/gsm8k
---

# nanochat-byte (d24-byte-l-ext-chat-anneal) <!-- TODO: final ckpt tag/step -->

<!-- TODO: one-paragraph intro -->

Byte-level (tokenizer-free) chat model from a fork of [nanochat](https://github.com/karpathy/nanochat).
One token = one UTF-8 byte.

## Quick facts

| | |
|---|---|
| Parameters | 686M total / 680M scaling (matmul + lm_head) |
| Layers / heads / width | 24 / 12 (head_dim 128) / 1536 |
| Attention | full causal, all layers (`window_pattern="L"`) |
| Context length | 8192 bytes |
| Vocab | 265 (256 raw bytes + 9 special tokens), padded to 320 |
| Position encoding | RoPE (base 1e5) |
| Extras | QK-norm (×1.2), ReLU² MLP, value embeddings (alternate layers), token smear, logit softcap 15, untied embeddings |
| Pretraining | ~25.1B bytes of ClimbMix (23,900 steps × 1,048,576 bytes) |
| Post-training | SFT (1 epoch, ~854K rows) + short CUTE-heavy consolidation anneal |
| Weights dtype | bfloat16 |

## Usage

Requires `trust_remote_code=True` (custom architecture; modeling code is bundled and standalone — it needs only `torch` + `transformers`).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("REPO_ID", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("REPO_ID", trust_remote_code=True)

messages = [{"role": "user", "content": "Which has more letters, 'strawberry' or 'blueberry'?"}]
enc = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)
out = model.generate(
    enc["input_ids"],
    attention_mask=enc["attention_mask"],
    max_new_tokens=256,
    do_sample=True, temperature=0.6, top_k=50,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Chat format

```
<|bos|><|user_start|>...<|user_end|><|assistant_start|>...<|assistant_end|>
```

Special tokens (ids 256–264): `<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`. EOS = `<|assistant_end|>`. A system message is folded into the first user turn.

## Training

<!-- TODO: narrative -->

1. **Pretrain** — see the base model: [BASE_REPO_ID](https://huggingface.co/BASE_REPO_ID).
2. **Vocab extension** — 256 → 265 (chat special tokens appended; embedding rows for ids 0–255 unchanged).
3. **SFT** — one epoch over a mixture of SmolTalk (460K), MMLU aux-train ×3, GSM8K ×4, synthetic char-level tasks (CUTE-style, 4K/subtask × 8 subtasks), arithmetic/spelling tasks, and small custom sets (identity, PEP 827 Q&A, D&D setting). Final val: 0.316 bpb.
4. **Anneal** — short consolidation anneal from the SFT checkpoint on a CUTE-heavy mixture (50% char-level tasks), re-warmed LR annealed to zero. <!-- TODO: steps/recipe of the released ckpt -->

Trained on a single RTX 6000 PRO Blackwell.

## Evals

<!-- numbers below are d24-byte-l-ext-chat-anneal-lr0.3 @ step 300; TODO: re-run if a different ckpt is released -->

| Benchmark | Score |
|---|---|
| ChatCORE | 0.4552 |
| ARC-Easy / ARC-Challenge | 0.6284 / 0.4898 |
| MMLU | 0.3507 |
| GSM8K | 0.1433 |
| HumanEval | 0.1220 |
| SpellingBee | 0.9609 |
| CUTE (char-level mean, zero-shot chat) | 0.9263 |

## Limitations

<!-- TODO -->

## Provenance

Exported with [`hf_export/generate_hf_model.py`](https://github.com/msullivan/nanochat/tree/master/hf_export) from nanochat checkpoint `d24-byte-l-ext-chat-anneal-lr0.3` step TODO. Bit-exact logits parity with the reference implementation verified in fp32.
