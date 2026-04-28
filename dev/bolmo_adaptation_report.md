# Adapting Bolmo to nanochat — feasibility and plan

Written 2026-04-27. Source: Minixhofer et al., *Bolmo: Byteifying the Next Generation of Language Models*, [arXiv:2512.15586](https://arxiv.org/abs/2512.15586). Local cache: `~/.cache/nanochat/knowledge/2512.15586/`.

## TL;DR

Bolmo's recipe is mostly portable to nanochat, but two of its choices are sticky for us: the **mLSTM local encoder/decoder** (new dep, no reusable code in this repo) and the **non-causal boundary predictor** (small but invasive plumbing for pooling/depooling/output-boundary fusion). The **Stage 1 / Stage 2 training split** and the **reuse of an existing BPE checkpoint as the global model** are the two cleanest ideas we should adopt. Stage 1 alone — frozen global model, train only the byte-side scaffolding to mimic BPE behavior — fits in <1% of pretrain compute and would be the highest-information experiment we can run before committing to a full from-scratch byte run.

The strongest pitch: **byteify the existing d24 BPE checkpoint** and produce a like-for-like byte/BPE pair that share a global model, at <$50 of compute. That's a more legible result at speedrun scale than "from-scratch byte loses on BPB," which is the predicted outcome of the current `no-tokenizing` track.

## What Bolmo actually does

### Architecture (Latent Tokenizer LM, "LTLM")

Pipeline: byte → embed → local encoder → boundary predictor → pool → **global model** → depool → local decoder → LM head.

| Component | Bolmo-7B | Bolmo-1B | Role |
|---|---|---|---|
| Byte embed table | 256 entries → 512 (boundary fusion) | same | Cheap, plus an additive *subword-suffix* lookup that retains the BPE source's embedding table indexed by longest matching subword suffix |
| Local encoder | 1 mLSTM+FFN layer, d=4096 | 1 layer, d=2048 | Contextualize bytes before pooling |
| Boundary predictor | Cosine-distance between byte rep at *t* and *t+1* — uses **one byte of future context** during prefill | same | Decides where to place patch boundaries |
| Pool | Take last byte of each patch as the patch rep (no projection — local and global use same d) | same | Compress to ~BPE-token-rate sequence |
| Global model | **Olmo-3 7B verbatim** (frozen in Stage 1) | OLMo-2 1B verbatim | The expensive part — reused from the source BPE LLM |
| Depool | Add latest patch rep to a linear projection of the byte rep | same | Distribute patch info back to per-byte positions |
| Local decoder | 4 mLSTM+FFN layers, d=4096 | 4 layers, d=2048 | Contextualize depooled byte reps |
| LM head | 512-way (byte × {boundary, no-boundary}) | same | Predicts next byte AND whether it ends the patch in one step (`<b>` fusion) |

Param overhead vs source BPE model: **+4.5% (7B), −0.7% (1B)**. Most of the gain comes from the local decoder; the local encoder is deliberately tiny because the subword-suffix embedding lookup carries most of the input-side capacity.

### Two-stage training

**Stage 1** (~9.8B tokens / ~43B bytes; ~20% of total budget): freeze global model. Train only local encoder, local decoder, boundary predictor, LM head. Loss has four terms:

- $\mathcal{L}_\mathcal{B}$: BCE against true subword boundaries — boundary predictor learns to mimic BPE's tokenization decisions. Hits >99% accuracy quickly.
- $\mathcal{L}_\mathcal{E}$: pool the local encoder output, run it through the **first 4 layers** of the global model, and L2-match against running the BPE model's input embeddings through those same 4 layers. Backprop into the local encoder only. (n=4 layers vs n=0 layers makes a real difference — model-stitching insight: matching the first-layer input doesn't guarantee matching downstream representations.)
- $\mathcal{L}_{\mathcal{D},\text{Distill}}$: per-patch likelihood matching — the byte model's product of per-byte probabilities for a patch should equal the BPE model's probability for that subword token. Done in log-space with τ=5 temperature smoothing.
- $\mathcal{L}_{\mathcal{D},\text{CE}}$: optional auxiliary cross-entropy on bytes.

Stage 1 cost: ~2× FLOPs through the global model (1 forward; backward only through layers 0..n=4). Substantially cheaper than full E2E.

**Stage 2** (~39.3B tokens; ~80% of total budget): unfreeze global model, drop the distillation losses, keep only $\mathcal{L}_\mathcal{B}$ and $\mathcal{L}_\text{CE}$. Standard byte-level LM training, but starting from a checkpoint that already approximately reproduces the BPE model's behavior. Lower LR for global model (1.8e-5) than for local models (3.7e-5) to protect the inherited weights.

Total budget: 49.1B tokens for Bolmo-7B, which they call "<1% of typical pretrain." For OLMo 2 1B (1.48B params, trained on multi-T tokens), 49.1B is roughly 1-2% of that pretrain.

### Headline results (relative to nanochat scale)

- **Bolmo 1B vs OLMo 2 1B (subword)**: matches on average (58.2 vs 58.3 on the 1B suite), wins by **+32.5 pts on CUTE** (60.0 vs 27.5). This is the result that's most directly informative for us.
- **Bolmo 1B vs BLT 1B (from-scratch byte)**: Bolmo trails BLT 1B slightly on aggregate (58.2 vs 58.5), wins on CUTE (60.0 vs 37.6), but BLT has 4.53B parameters (counts hash-embedding params Bolmo doesn't include in its number) so it's not a clean param-match.
- **Continued training is most of the byteification cost**: Olmo-3-with-continued-training (same data, no byteification) actually drops below Olmo 3 on most tasks. Bolmo beats Olmo-3-CT on character/code, ties on math/MC, loses small on GenQA. This means **the gap between Bolmo and the source BPE model is mostly a continued-training gap, not a byte-vs-subword gap.** Important framing.
- **Compression sweep**: byteified models can be retrained with merged boundaries (BPE-style merge of the original subword tokens) up to ~7 bytes/patch, smoothly trading performance for inference speed; the BPE source can't do this past ~6 bytes/patch (softmax cost dominates).
- **Task arithmetic transfer of post-training**: take the diff between Olmo-3-base and Olmo-3-RL-instruct (in BPE space), add it to Bolmo's global-model weights — Bolmo's IFEval jumps from 31% → 67%, matching the BPE post-trained checkpoint. **Zero extra training.** This is wild.

## What's adaptable to nanochat

We have a d24 BPE base from `runs/speedrun.sh`. The natural Bolmo-style experiment is to byteify *that* checkpoint.

### Cheap and clean to port
- **The two-stage training split.** Conceptually free; the only thing it requires is a way to freeze the global model and run only its forward pass during Stage 1. nanochat already has a clean `GPT` class; freezing is one parameter-iteration loop.
- **Subword-suffix additive embedding.** Trivial — it's a lookup table indexed by "longest subword suffix matching the byte sequence ending at position i." We already have the BPE tokenizer (`nanochat/tokenizer.py`) and we can precompute a byte-position → subword-id map via a single tokenize-and-expand pass per training example.
- **Boundary supervision (Stage 1 BCE loss).** Same precompute as above gives us the gold boundary mask.
- **Per-patch distillation loss.** Needs simultaneous BPE + byte forward passes during Stage 1, which doubles memory but is straightforward.
- **Lower LR on global model in Stage 2.** Trivial — we already have per-parameter-group LRs in the optimizer.
- **Boundary-symbol vocabulary fusion (256→512).** Twelve-line LM-head change.

### Sticky to port
- **mLSTM local models.** Bolmo specifically chose mLSTM (xLSTM family, Tiled Flash Linear Attention) for wallclock decode throughput — they show Mamba2 and Gated DeltaNet underperform at the same FLOPs/byte. We don't have any of this in the repo; it would be a new dep (`xlstm` or `mlstm-kernels`) and a new attention path. **A pragmatic substitute: use 1-layer / 4-layer transformer blocks** (with FlashAttention which we already have). Bolmo's reasoning for mLSTM is *inference speed* not *modeling capacity*; for an experiment that prioritizes "does byteification work" over "does it match prefill latency," a transformer local stack is a reasonable shortcut.
- **Non-causal boundary predictor (uses 1 byte of future context during prefill, then a separate output-boundary head during decoding).** The architectural insight is real — they show in §6.1 that causal predictors force a choice between *correct boundary* and *correct patch content*, while non-causal can do both. **Cheaper variant: just use the BPE tokenizer for prefill boundaries** (Bolmo explicitly considers this and rejects it because it brings back tokenization bias and defeats the compression-sweep story; we don't care about either at the experiment stage). Then we only need an output boundary predictor for decoding, which is what the 512-vocab fusion already does.
- **Pool/depool plumbing with variable patch boundaries.** This is the largest engineering item. Variable-length pooling and depooling against a frozen global model means writing a "gather last-byte-of-patch" / "scatter latest-patch-rep" pair that plays nicely with `torch.compile` and the existing FlashAttention path. ~150 LOC of careful indexing.

### Doesn't apply at our scale
- **The compression sweep (training at higher bytes/patch with merged BPE boundaries).** Not interesting until we have a working byteified d24 baseline.
- **Task-arithmetic transfer of post-training.** Premature for us — we don't have a strong BPE post-trained d24 checkpoint to transfer from. (Could be revisited later as: take a d24-SFT checkpoint and add its `(SFT − base)` weight delta to the byteified d24-base.)

## Concrete proposal for nanochat

### Phase 0 — feasibility check (~$5 of compute)

Build the byteification scaffolding *without* the new architecture pieces, just to exercise the pipeline:

1. New file `scripts/byteify_d24.py`. Loads the d24 BPE checkpoint as the frozen global model.
2. Add a tiny `nanochat/byteify.py` module:
    - byte embed table (256 entries, expand to 512 later)
    - subword-suffix lookup (precompute byte→token-id map per batch using the existing BPE tokenizer)
    - **simplification: use BPE-tokenizer-supplied boundaries** for prefill — skip the boundary predictor entirely in Phase 0
    - pool/depool functions over variable patch boundaries
    - 1-layer transformer local encoder, 4-layer transformer local decoder (reuse `Block` from `nanochat/gpt.py`)
3. Stage 1 only, ~1B tokens of FineWeb-Edu (the existing speedrun data). Loss = $\mathcal{L}_\mathcal{E}$ + $\mathcal{L}_{\mathcal{D},\text{Distill}}$.
4. Eval on the byte side after Stage 1: BPB on the held-out shard, and CUTE numbers via existing `scripts/cute_eval.py`. The interesting question for Phase 0 is **does the BPE model's behavior survive the byte wrapper** (i.e., do we recover the BPE model's loss within ε after Stage 1, with global frozen).

Cost estimate: 1B tokens × 8×H100 forward + tiny backward = ~30 min ≈ $4. If this works, Phase 0 alone gives us "byteified d24 has BPB within 5% of d24 BPE on a held-out shard," which is a publishable line.

### Phase 1 — full byteification (~$30 of compute)

If Phase 0 lands:
1. Add the non-causal boundary predictor (cosine-distance variant) and the 512-way LM head.
2. Stage 1: 5B tokens, full loss with all four terms.
3. Stage 2: 10B tokens, unfreeze global, lower its LR by ~2×.
4. Evals: BPB, CUTE, the existing nanochat eval suite. Compare against (i) d24 BPE base and (ii) the in-progress from-scratch d24 byte run on the `no-tokenizing` branch.

Cost estimate: 15B tokens × 8×H100 ≈ 8 hours ≈ $30. (For reference, the speedrun is ~11B tokens and takes ~4 hours on the same hardware.)

### Phase 2 — character-task SFT mix (free, just a data change)

Bolmo adds 75M tokens of CUTE-style synthetic data to the *pretraining* mix. We can replicate a smaller version of this: ~10M synthetic CUTE-style examples interleaved into Stage 2, generated once via the existing `tasks/` infrastructure. This is the cheapest 10x-CUTE-score lever in the paper.

## What would change about the current `no-tokenizing` track

The current branch is doing from-scratch byte training (per the literature review's "from-scratch is the riskiest design"). Bolmo doesn't supersede that effort — they're complementary. From-scratch tells us how byte models behave when nothing is given to them; byteification tells us how much of a BPE model's behavior carries through when we wrap it. The byteification track has lower variance and (per Bolmo's own results) a higher likely BPB ceiling at fixed compute.

If forced to pick one, **byteification is the higher-EV bet at speedrun scale.** The reason: at our compute budget, from-scratch byte will likely lose on BPB (per BLT's crossover at ~2.5-3× compute-optimal), and we're betting on a CUTE win to make up for it. Byteification *also* gets the CUTE win (Bolmo 1B: 60% vs OLMo 2 1B: 27.5%) but gives us close-to-BPE-parity on the rest of the eval suite for free, because the global model is the BPE model.

## Open questions before committing

1. **Does a transformer-based local encoder/decoder work as well as mLSTM?** Bolmo's argument for mLSTM is wallclock decode speed, not modeling capacity, so this is a soft constraint at experiment stage. Worth pre-checking by running a 1-layer transformer through Stage 1 only and checking distillation loss curves vs Bolmo's reported numbers.
2. **Is BPE-tokenizer-supplied prefill boundary acceptable as a Phase 0 shortcut?** It defeats the long-term story (compression sweep, full tokenizer-free pipeline) but isolates the Stage 1 plumbing for debugging.
3. **Do we have enough memory for Stage 1's dual-forward-pass?** Stage 1 needs both BPE and byte models live simultaneously for the distillation loss. d24 BPE is ~700M params; the byte wrapper adds ~50-100M; total well within 80GB H100 even at the speedrun's batch size.
4. **Is the d24 BPE base good enough as a starting checkpoint?** Bolmo started from heavily mid-trained Olmo-3. d24 base from `runs/speedrun.sh` is ~11B-token-trained, much weaker. Byteification quality probably degrades when the source model is weaker, but the relative comparison ("byteified d24 vs d24-CT vs d24-byte-from-scratch") is still meaningful.

## What I'd actually recommend doing first

Phase 0 — **the feasibility check**. It's 30 minutes of GPU time and answers the load-bearing question: can a byte wrapper around a frozen d24 BPE checkpoint reproduce its loss after Stage 1 distillation? If yes, the rest of the recipe falls out and we have a clear path to a strong byte model at speedrun scale. If no, we've learned that something specific about Bolmo's setup (model scale, mLSTM, longer training, the suffix embedding) is load-bearing — and we know which to investigate next.

---

# Follow-up findings (2026-04-27)

After writing the report above, several issues surfaced in discussion that change the picture materially. Captured here so the original isn't lost.

## Source checkpoint: use the fully-baked one

Bolmo started from the **fully-baked** Olmo 3 7B checkpoint, not a partially-cooked one. Direct quote from §4: *"We use the pretrained Olmo 3 7B checkpoint after mid-training and long-context extension as our starting point for byteifying into Bolmo."* That's base pretrain → mid-training (Olmo's anneal-equivalent) → long-context extension. Most-cooked available BPE checkpoint.

Why this works without re-pretraining:
- Stage 1 freezes the global model entirely — staleness of optimizer state is irrelevant.
- Stage 2 protects inherited weights with a much lower LR: 1.8e-5 for global vs 3.7e-5 for local. That's fine-tune-scale, won't blow out the learned features.
- The whole pitch of byteification is "free-ride on a finished BPE LLM"; using a half-trained source defeats the value.

For nanochat: **use the final pretrained d24 base checkpoint**, no re-pretraining. Mirror Bolmo's LR split (global at 1-2e-5 peak in Stage 2, local at 2-4×). The speedrun's cosine decays to ~10% of peak by end; a Stage 2 warm-up to 1.8e-5 is below that decayed value, so no shock.

**Cheap upgrade we could grab "for free":** Olmo 3 had mid-training (cleaner data, comparable to nanochat's existing `dev/anneal_workflow.md`). The d24 base from `runs/speedrun.sh` doesn't routinely run anneal. If we want to maximize source quality without retraining, running the existing anneal workflow on the d24 base before byteifying is the single highest-EV cheap upgrade — it's what Bolmo got for free from Olmo's pipeline.

## The architecture mismatch is real but narrower than first stated

The original report claimed three places where token IDs leak past the input embedding (value embeddings, `x0_lambdas`, `smear`). On closer reading of `nanochat/gpt.py`, only **one** is actually hostile to the Bolmo recipe.

**Value embeddings (gpt.py:190, 464) — actually hostile.** A separate `Embedding(vocab_size, kv_dim)` table at every other layer (plus the last). At each block: `ve = self.value_embeds[str(i)](idx); v = v + gate * ve`. That's `~n_layer/2` token-id-indexed lookups happening *inside* the global model, each one a vocabulary table that needs a byte-side replacement. Bolmo's "global model only sees patch reps, never token IDs" assumption breaks here.

**`x0_lambdas` (gpt.py:181, 463) — false alarm.** Residually re-mixes the layer-0 *activation* `x0` back in at every layer with a per-layer learned scale. It's not a token-ID lookup — it's a residual connection. In the byteified pipeline, `x0` becomes "whatever is fed into layer 0 of the global model" = the pooled patch rep. Same activation flows in residually at every depth; Bolmo's "global model only sees patch reps" property still holds. Ports cleanly.

**`smear` (gpt.py:182-184) — false alarm.** Mixes the previous token's embedding into the current via a gated linear projection. Pre-trunk, operating at the layer-0 input granularity. For byteification, "previous token" becomes "previous patch" if applied at the global-model-input position, or "previous byte" if applied inside the local encoder. Either is fine — it's a one-position shift-and-mix. The gate weights port cleanly.

So the actually-hostile feature is just **value embeddings**.

## Levers for the value embedding problem

Karpathy's LOG.md (2026-01-17 entry, lines 600-617) is unambiguous about VEs:

> "the models *love* Value Embeddings. It is a way to add a huge amount of capacity (parameters) to the model at almost zero cost of FLOPs... **Any attempt to reduce the capacity of value embeddings (param sharing, low rank, projections) fail.** The model wants many of them, and with all the capacity, and doing so wins across all x axes."

Magnitudes from his d12 breakdown: VEs are **151M of 412M total params (~37%)**. Bigger than the token embedding table. After adding VEs, the optimal tokens-to-params ratio **halved from 8 to 4** — the model is genuinely embedding-bloated, and the "free FLOPs from cheap params" is what makes it win.

So we cannot just delete VEs. Levers, ranked by likely viability:

1. **Reuse VEs in-place via the longest-suffix subword ID.** The BPE base's VE tables are `Embedding(vocab_size, kv_dim)` keyed by token ID. In a byteified world, the global model only runs at patch positions. Bolmo's subword-suffix lookup already gives us a way to map any patch to "the subword whose bytes are the suffix of this patch." That's exactly the index the existing VE tables want. The boundary predictor's Stage 1 target (match the BPE tokenizer) ensures the VE lookup matches what the BPE model would have computed. Existing VE weights preserved verbatim, no retraining of the tables. **Leading lever — ~30 LOC of plumbing.**

2. **Hybrid: (1) plus a small learned residual that depends on the patch rep.** Initialize the residual at zero so Stage 1 starts from "VE = exactly the BPE lookup," then let Stage 2 learn the byte-aware delta. Best of both: preserves the BPE-trained weights as a strong prior, lets byte-side info modulate the contribution. Probably the right place to land if (1) shows VE values that are *almost-but-not-quite* what the byte signal wants.

3. **Replace each VE with a learned projection from the patch rep.** `Linear(d_model, kv_dim)` per VE-layer, additive at the value residual. Lets the model build a byte-aware analog of VEs from the patch rep. Cons: throws away 151M params of trained weight, has to learn replacement signal in Stage 2 — and Karpathy explicitly tested low-rank-projection alternatives and found they fail. **Skip — Karpathy's negative result is strong evidence this won't fly.**

4. **Train a VE-free d24 base from scratch and byteify that.** Karpathy's data says the model would be substantially worse — VEs win across all axes. "Byteify a deliberately handicapped base." Probably ~10-20% worse on BPB at the same compute. Only justifiable if (1)/(2) prove to break in some way.

5. **Re-spend the VE capacity in the byte-side stack.** Drop VEs from the global, pour the 151M params into a deeper local encoder/decoder or a much larger byte input embedding table (Bolmo's "increase the size and sparsity of the embedding table" idea). Real architectural redesign — skip unless byteification is the permanent track.

**The general insight:** Bolmo's suffix-embedding trick isn't just an input-side hack, it's a general bridge for any token-ID-indexed component in the BPE checkpoint. Anything that does `Embedding(vocab)(idx)` — value embeddings, bigram embeddings, lm_head — can in principle be reused via the same suffix-id lookup, applied at the relevant point in the byte pipeline. nanochat's embedding-heavy architecture is less hostile than it first looks; the "wte is the only token-aware interface" assumption Bolmo uses turns out to be a relaxable one, as long as the BPE base's embedding tables stay subword-indexed.

## How Bolmo actually gets character understanding (mechanism, not training)

Three mechanisms compose:

1. **Output-side: byte-by-byte generation is the structural win.** The local decoder generates bytes sequentially. For "spell alphabet" the model emits 'a', 'l', 'p', 'h', 'a', 'b', 'e', 't' as separate steps. A BPE model has to either memorize the spell-out as a token sequence or decompose the embedding post-hoc. Bolmo just emits bytes — character-level addressability is *native, not learned*. This alone explains a lot of the CUTE win without invoking any global-model learning.

2. **Input-side: the patch rep is richer than a BPE embedding.** For BPE, "alphabet" is a fixed lookup vector — the embedding table can't encode "what's the 3rd character" because it's not a function of the bytes, it's just an integer ID. For Bolmo, the patch rep for "alphabet" is the last byte's mLSTM output after consuming a, l, p, h, a, b, e, t — a sequence model's contextualized output over the constituent bytes. Carries byte-level structural information that BPE physically cannot. Stage 1 forces this rep to *match* the BPE embedding (so early on it looks like BPE); Stage 2 lets the local encoder drift toward producing reps with more byte info, and the global model learns to read it.

3. **Training: the 75M tokens of CUTE-style synthetic data does the targeted teaching** that makes the global model actively *use* the byte info now available in patch reps. Bolmo is explicit: *"trained with synthetic data encouraging character understanding... which speeds up the acquisition of this skill."*

The global model is still pattern-matching at the patch level (≈ token level), but the patches themselves are no longer opaque IDs. They're sequence-encoder outputs over bytes, so "what byte is at position 3 of this patch" is at least in principle encoded in the patch rep — meaning the global model can learn to address character-level features that for BPE are physically inaccessible.

For nanochat, the encouraging implication: **even a Phase-0-only result** (frozen global model, just Stage 1 distillation) should already get *some* CUTE win from the output-side mechanism alone, before any of the trickier Stage 2 + synthetic-data work. Cheap signal you can validate the architecture with.

## The data-vs-architecture decomposition (the load-bearing finding)

The headline Bolmo result is "+32.5 pts on CUTE vs source BPE model." The byteify-vs-CT table in the appendix decomposes this for the 7B comparison and the picture is very different from the headline:

**CUTE:**
- Olmo 3 base → Olmo 3 CT (same data, no byteification): 56.9 → 72.9 = **+16.0**
- Olmo 3 CT → Bolmo: 72.9 → 78.6 = **+5.7**
- Total: **+21.7**

**EXECUTE:**
- Olmo 3 → Olmo 3 CT: 55.1 → 69.2 = **+14.1**
- Olmo 3 CT → Bolmo: 69.2 → 71.6 = **+2.4**

So at 7B, **roughly 75% of the character-understanding gain is just "train on CUTE-style synthetic data"**, and only 25% is the byte architecture. The headline Bolmo number is mostly a data story dressed as an architecture story.

This significantly changes our prioritization. Cheapest experiment by a wide margin is now:

**Phase −1 (~$10 of compute):** Generate ~10M CUTE-style synthetic examples (using existing `tasks/` infrastructure — we already have most of the templates from the CUTE eval work), interleave them into a d24 anneal or SFT run on the existing BPE base. No model changes, no byteification, no new code. Probably gets us most of the available CUTE win. One day of work + brief continued-pretraining run.

If that's true, the byteification track has to be re-justified on the +5.7 marginal pts (or whatever its analog is at our scale), not on the +21.7 headline. That's a much weaker pitch.

### Caveats pulling the other direction

- **The 25% byte-specific gain is probably concentrated in catastrophic-BPE-ceiling tasks** (`swap_char` at 10%, `ins_char` at 9% in the published numbers). Bolmo doesn't break out per-subtask numbers in the byteify-vs-CT table, just the CUTE average. The +5.7 average could be `+0` on `spell` (already saturated by data) and `+15` on `swap_char` (where BPE structurally can't compete). The "byte models can do char manipulation that BPE structurally can't" claim still holds even if data closes the gap on easier tasks.
- **At our scale (1B param-class, 11B tokens), the data-only effect might be smaller.** Olmo-CT's result is at 7B trained for 39B additional tokens; smaller models with less compute may absorb the synthetic data less well. We genuinely don't know.
- **The Bolmo-1B vs OLMo-2-1B comparison (+32.5 CUTE) lacks a CT control**, so we can't decompose it at 1B. The 1B story is more compelling on the headline number, but it's quite plausibly also mostly-data-driven and we just can't see it.

### Updated experiment ladder

1. **Phase −1 (~$10):** CUTE-style synthetic data into d24 BPE anneal/SFT. Cheapest way to find the data-only ceiling at our scale. Necessary baseline for evaluating any byteification result — without it we can't separate architecture wins from data wins.
2. **Decide whether to byteify based on the Phase −1 gap.** If d24-BPE+CUTE-data hits ~60% on CUTE, byteification's marginal value is just the catastrophic-BPE-ceiling tasks — interesting but narrow. If it stalls at ~30%, byteification has a much larger pitch.
3. **If byteification proceeds:** Phase 0 → Phase 1 from the original report, but plan around the value-embedding lever (suffix-id reuse + Stage 2 learned residual) rather than treating the global model as VE-free.

### Honest reframing

Bolmo demonstrates that **byteification doesn't *cost* much performance**, not that it *gains* much character understanding above what the same data buys an unbyteified model. The character understanding gain is mostly the data; the byte architecture is what makes the data ride a model that's structurally able to express character-level outputs (the byte-by-byte generation mechanism above). For our purposes, that means:

- The byteification track and the synthetic-data track are **separable experiments**. Run the data one first; it's cheap and directly comparable to BPE.
- The byteification track's value at speedrun scale is the **structural advantage on output-addressable character tasks** (swap, ins, del, sub, reverse) — not a generic CUTE win.
- The "from-scratch byte" track on the `no-tokenizing` branch is *only* about the structural advantage; there's no continued-training-from-BPE component. So its pitch hasn't moved — it's still "does a from-scratch byte model at speedrun scale beat BPE on CUTE manipulation tasks." That's still worth running and is unaffected by this finding.
