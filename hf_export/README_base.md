---
language:
- en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- nanochat
- byte-level
- custom_code
datasets:
- karpathy/climbmix-400b-shuffle
---

# nanochat-byte-base (d24-byte-l-ext)

<!-- TODO: one-paragraph intro -->

Byte-level (no tokenization) base language model from a fork of [nanochat](https://github.com/karpathy/nanochat).

I wanted to experiment with

nanochat uses a technique called "value embeddings", where token-indexed vectors get


## Quick facts

| | |
|---|---|
| Parameters | 686M total / 680M scaling (matmul + lm_head) |
| Layers / heads / width | 24 / 12 (head_dim 128) / 1536 |
| Attention | full causal, all layers |
| Context length | 8192 bytes |
| Vocab | 265 (256 raw bytes + `<|bos|>` at 256; ids 257–264 reserved for chat fine-tuning, unused here), padded to 320 |
| Position encoding | RoPE (base 1e5) |
| Extras | QK-norm (×1.2), ReLU² MLP, degenerate value embeddings (alternate layers), token smear, logit softcap 15, untied embeddings |
| Training data | ~25B bytes of ClimbMix  |
| Final val loss | 0.711 bits/byte |
| Weights dtype | bfloat16 |

## Usage

Requires `trust_remote_code=True` (custom architecture; modeling code is bundled and standalone — it needs only `torch` + `transformers`).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("BASE_REPO_ID", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("BASE_REPO_ID", trust_remote_code=True)

ids = tokenizer("The capital of France is", return_tensors="pt")
out = model.generate(
    ids["input_ids"],
    attention_mask=ids["attention_mask"],
    max_new_tokens=64,
    do_sample=True, temperature=0.8, top_k=50,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

This is a base model: it continues text, it does not follow instructions.

## Training

<!-- TODO: narrative -->

d24 nanochat arch, byte tokenizer, 8192-byte context, on [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle). 23,900 steps at 1,048,576 bytes/step ≈ 25.1B bytes, WSD schedule (extended from a mid-stable-phase checkpoint of a shorter run). Trained on a single RTX 6000 PRO Blackwell.

Note on the vocab: the checkpoint was trained with a 256-wide vocab (BOS at row 0) and converted to the 265-wide layout for release (byte rows unchanged, BOS moved to 256). Ids 257–264 exist so chat fine-tunes share the embedding layout but received no training here.

## Evals

<!-- TODO: fill in / run on the exported checkpoint -->

| Benchmark | Score |
|---|---|
| Val bits/byte (ClimbMix) | 0.711 |
| CORE | 0.2366 |
| CUTE (char-level mean, few-shot completion) | TODO |

## Limitations

<!-- TODO -->

## Provenance

Exported with [`hf_export/generate_hf_model.py`](https://github.com/msullivan/nanochat/tree/master/hf_export) from nanochat checkpoint `d24-byte-l-ext` step 23900 (vocab-converted). Bit-exact logits parity with the reference implementation verified in fp32.
