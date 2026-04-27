# Byte-level LM Literature Review

Written 2026-04-26 to ground design choices for the nanochat-byte fork (d24, ~600M params, vocab=256, BPB-comparison-against-BPE).

## Canonical works

### ByT5 — Xue et al., 2022 ([arXiv:2105.13626](https://arxiv.org/abs/2105.13626))
T5 encoder-decoder with vocab=256, no patching. Compensates by going *deeper, narrower `d_ff`, encoder-heavy* (3:1 enc:dec layer ratio); span-corruption noise span length raised 3 → 20. Competitive with mT5 at matched params, much more robust on character-level tasks. Pays 3-10× compute.

### Charformer / GBST — Tay et al., ICLR 2022 ([arXiv:2106.12672](https://arxiv.org/abs/2106.12672))
Differentiable soft-tokenization: scores candidate n-gram blocks of multiple sizes, mean-pools to a downsampled sequence. Encoder-only; soft-pooling doesn't generalize to causal LM (future leakage). Mostly superseded by entropy-patching.

### CANINE — Clark et al., 2021 ([arXiv:2103.06874](https://arxiv.org/abs/2103.06874))
Local windowed attention on chars → 4× strided conv downsample → deep transformer on coarse stream → upsample. Encoder-only. Conceptual ancestor of every "local module + global on coarser stream" byte LM.

### MegaByte — Yu et al., 2023 ([arXiv:2305.07185](https://arxiv.org/abs/2305.07185))
Two-stage: bytes grouped into *fixed-size* patches (P=4-8); small local transformer within each patch, large global across patches. Quadratic cost paid only on the N/P-length sequence. SOTA on raw audio / ImageNet density. Fixed P doesn't honor word boundaries.

### SpaceByte — Slagle, NeurIPS 2024 ([arXiv:2404.14408](https://arxiv.org/abs/2404.14408))
Flat byte transformer with extra "global" blocks inserted only after specific bytes (default: spaces, punctuation). Beats MegaByte; matches subword transformers within noise on PG19/Stories/arXiv/GitHub at 300M-1B. Limit: English-script bias.

### MambaByte — Wang et al., COLM 2024 ([arXiv:2401.13660](https://arxiv.org/abs/2401.13660))
Mamba SSM on bytes, no patching. **MambaByte-972M beats MegaByte at 1/3 compute** (33.0 BPB vs 36.4 on PG19). Standard SSM tradeoff (worse in-context recall) applies independently of byte-vs-token.

### BLT (Byte Latent Transformer) — Pagnoni et al., Meta 2024 ([arXiv:2412.09871](https://arxiv.org/abs/2412.09871))
Local encoder (1-3 layers, hash n-gram embeddings n=3..8) → latent transformer at patch rate → local decoder (7-9 layers) cross-attending back to bytes. Patch boundaries from a small (100M) auxiliary entropy LM, threshold 0.6, **avg ~4.5 bytes/patch**. First FLOP-controlled scaling study to 8B / 4T bytes. **BLT-Entropy 8B beats Llama-3 8B by +1.1pts averaged on 7 downstream tasks at matched FLOPs.** Headline scaling finding: **byte/BPE crossover sits at ~2.5-3× compute-optimal at 8B class**. AdamW (β=0.9/0.95), LR=4e-4, WD=0.1, clip 1.0, **constant batch in bytes**.

### EvaByte — HKU+SambaNova, ICLR 2025 ([blog](https://hkunlp.github.io/blog/2025/evabyte/))
*Flat* byte LM, vocab=320. Two key tricks: **multibyte prediction** (predict next k bytes per position) + **EVA attention** (chunked linear-attention variant). 6.5B / 1.5T bytes (~5× less than typical BPE budgets). Beats BLT with 3-4× fewer training bytes; better on code. Limit: SambaNova-specific infra.

### H-Net — Hwang/Wang/Gu 2025 ([arXiv:2507.07955](https://arxiv.org/abs/2507.07955))
End-to-end hierarchical with **dynamic chunking** (learned, no auxiliary entropy model). Multi-stage. Beats BPE at matched compute; multi-stage matches transformers of *2× the size*. Biggest gains on Chinese, code, DNA (≈4× data efficiency on DNA). **H-Net++** ([arXiv:2508.05628](https://arxiv.org/abs/2508.05628)) reports 0.159 BPB reduction (~12% better compression) over BPE GPT-2-fa on Persian. Dynamic chunking causes load-balance issues — slower wall-clock than isotropic transformers despite better FLOPs.

### Bolmo 7B / 1B — Ai2, Dec 2025 ([blog](https://allenai.org/blog/bolmo) · [HF](https://huggingface.co/allenai/Bolmo-7B))
**Byteifies** an existing Olmo-3 BPE checkpoint via 2-stage distill+finetune at <1% of pretrain cost (~9.8B + 39.3B byte budget). Beats BLT-7B, TFree-Hat-7B, EvaByte-6.5B on CUTE/EXECUTE; matches base Olmo-3 on standard benchmarks. **Suggests byteification may dominate from-scratch byte at every fixed budget.**

### Smaller / adjacent
- **MrT5** ([arXiv:2410.20771](https://arxiv.org/abs/2410.20771)) — learned delete gate drops up to 75% of bytes after a fixed encoder layer. Encoder-only; gate idea is reusable.
- **MBLM** ([arXiv:2502.14553](https://arxiv.org/abs/2502.14553)) — model-agnostic hierarchical stacks (transformer + Mamba blocks); 5M-byte context on a single GPU.
- **Bit-level BPE** ([arXiv:2506.07541](https://arxiv.org/abs/2506.07541)) — goes below the byte boundary for multilingual fairness; orthogonal but useful framing.

## Implications for nanochat-byte (clear, actionable)

1. **The d24 byte run will almost certainly lose on raw BPB, and that is fine.** BLT puts the byte/BPE crossover at ~2.5-3× compute-optimal at 8B, and the speedrun budget is well below that. The honest pitch is character-level competence (CUTE-style spelling/manipulation) and noise robustness, *not* lower BPB. Reframe expectations and pick a primary eval metric that reflects this (CUTE, character-substitution robustness, byte-level text completion).

2. **A flat byte transformer is the riskiest design in the literature.** Every successful pure-transformer byte LM in 2024-2025 (SpaceByte, BLT, H-Net) uses some form of hierarchy. The only flat-transformer success (EvaByte) needed both EVA attention *and* multibyte prediction. MambaByte works flat but only because it's an SSM. So: a flat baseline is fine as v1, but expect to need hierarchy for v2.

3. **Adopt ByT5's depth-over-width prescription.** Spend the ~100M embedding-table savings on more layers or longer training, not wider `d_model`. The encoder-heavy 3:1 doesn't port to a causal decoder, but the deeper-narrower spirit does.

4. **Switch to constant-bytes batching.** Single biggest fairness fix vs. the BPE speedrun. Without this, the BPE/byte BPB comparison is muddled by per-step examples-vs-bytes mismatch.

5. **Sequence length: 8K-16K bytes.** Below 4K, you're paying tokenizer-free cost without long-context payoff. Above 32K, you'll need linear attention or hierarchy.

6. **Hyperparameters: AdamW (0.9/0.95), LR ~3-4e-4, WD ~0.1, clip 1.0.** Nothing in the byte literature requires departing from the speedrun defaults at this scale. Watch grad norms early; consider slightly longer warmup. Consistent with the d12 WD ablation finding.

7. **Cheap upgrades to queue after the flat baseline:**
   - **Multibyte prediction head** (EvaByte) — additive, k=4 or k=8, predicts k future bytes per position. Densifies loss to compensate for the low per-byte signal.
   - **Hash n-gram input embeddings** (BLT) — additive on the embedding side; n=3..8, ~500K hash buckets. Reportedly worth tens of BPB millibits.
   - **SpaceByte-style global blocks on whitespace** — smallest structural change to get hierarchy without an auxiliary entropy model.

8. **The Bolmo result deserves serious thought.** If "byteify a trained BPE Olmo-3" beats "train byte from scratch" at every budget — and Ai2's numbers suggest it does — then the path-to-deployment story for from-scratch byte is weak. Two possible reframings:
   - **Research-curiosity track:** nanochat-byte is a from-scratch proof-of-concept, deployment-quality byte models would be byteified.
   - **Replicate-and-extend track:** try the Bolmo recipe on the existing d24 BPE base. Distill+finetune is <1% of pretrain cost, so it's well within the speedrun budget. Could be the strongest single-result story available at this scale.

## What I personally learned (with clearest implications for us)

In rough order of "I should change my recommendations because of this":

- **Bolmo (Ai2, Dec 2025)** — biggest update. I had been treating from-scratch byte as the obvious approach because that's what BLT/EvaByte/H-Net do. Bolmo says: don't bother, byteify a BPE base for <1% of the cost and beat all of them. This is fresh enough that the field hasn't fully digested it yet.
- **BLT's crossover number (~2.5-3× compute-optimal at 8B).** Concrete and FLOP-controlled. Means at speedrun scale, raw BPB *will* lose to BPE — this isn't pessimism, it's measured. Frame the experiment around capabilities, not loss.
- **EvaByte's "flat-but-with-EVA+multibyte" recipe.** Only flat-transformer success at scale; the multibyte-prediction head in particular is cheap and additive — easy to bolt onto a flat baseline.
- **Hash n-gram embeddings (BLT).** Additive embedding-side trick that costs ~zero compute and reportedly buys real BPB. Would slot cleanly into the existing model.
- **Constant-bytes batching is non-negotiable for a fair BPB comparison.** I'd been thinking about it as a tweak; it's actually the fairness foundation.
- **ByT5's depth>width prescription is robust** — predates the patching era and still holds. Worth applying directly to the d24 config.
- **No optimizer surprises across the literature.** Standard AdamW settings work; nothing in byte papers contradicts the d12 WD ablation. Means we don't need to retune optimizer, only architecture and batch.

## Sources
[ByT5](https://arxiv.org/abs/2105.13626) · [Charformer](https://arxiv.org/abs/2106.12672) · [CANINE](https://arxiv.org/abs/2103.06874) · [MegaByte](https://arxiv.org/abs/2305.07185) · [SpaceByte](https://arxiv.org/abs/2404.14408) · [MambaByte](https://arxiv.org/abs/2401.13660) · [BLT](https://arxiv.org/abs/2412.09871) · [EvaByte](https://hkunlp.github.io/blog/2025/evabyte/) · [H-Net](https://arxiv.org/abs/2507.07955) · [H-Net++](https://arxiv.org/abs/2508.05628) · [MrT5](https://arxiv.org/abs/2410.20771) · [MBLM](https://arxiv.org/abs/2502.14553) · [Bolmo](https://allenai.org/blog/bolmo) · [Awesome-Byte-LLM](https://github.com/zjysteven/Awesome-Byte-LLM)
