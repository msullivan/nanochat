# Experiment Log

A running summary documenting some experiments and findings. Started ~Jan 7 2026.

---

## 2026-06-06: torch.compile DOES beat manual CUDA graphs — the 05-25 result was a stacked-cache bug; + batched eval

**Supersedes the 2026-05-25 conclusion** ("manual cudagraph beats reduce-overhead
by ~2x"). That was real on the code at the time, but the cause was NOT
reduce-overhead's wrapper overhead — it was our KV-cache *layout*. Fixed, compiled
`reduce-overhead` is the faster and simpler path. torch was 2.9.1+cu128 on BOTH
dates (per uv.lock) — not a version effect.

### The bug: a stacked KV-cache makes torch.compile copy the whole cache every step
`KVCache` was one stacked tensor `(n_layer, B, T_max, H, D)`; each layer's in-place
`index_copy_` into a slice gets **functionalized** by Inductor into a *full-cache
copy* (it can't prove a slice-write to a shared tensor is safe). So every decode
step copied the entire multi-layer cache → cost O(cache_size) per step:

| single-stream decode (tok/s) | cache=512 | cache=2048 | cache=8192 |
|---|---:|---:|---:|
| compiled reduce-overhead, **stacked** cache | 246 | 150 | 30 |
| manual cudagraph (raw replay) | 214 | — | 183 |

Profiler at 8192: six cache-sized triton copy/pointwise kernels = **84%** of
runtime. Manual cudagraph captures the raw in-place op (no copy), so it's flat with
cache size — which is why it "won" at the real 8192 cache. My earlier
reduce-overhead benchmarks all used a 512 cache, hiding the blowup entirely.

### The fix: per-layer cache buffers (gpt-fast's structure)
Split the cache into per-layer standalone buffers (`k_cache_{i}`/`v_cache_{i}`) +
`mark_static_address`. Inductor's reinplace pass then keeps each write in place (no
copy), so compiled decode is flat with cache size and **beats** manual:
eager 56 → manual cudagraph 183 → **compiled+per-layer 232 tok/s** (decode @8192).
This is exactly why gpt-fast works at big context — its cache is per-layer; our
05-25 stacked cache was the anomaly. `Engine` is now compile-only (manual cudagraph
machinery deleted).

### Batched generation for fast eval (~40x)
Generative evals ran one prompt at a time. Added `Engine.generate_batched`: distinct
prompts left-padded into one batch, one forward/step over all rows, per-row EOS.
~40x over per-prompt on short-answer CUTE (21x even vs cudagraphed per-prompt at
B=64). Wired into `cute_eval` and tool-free `chat_eval` tasks.

Gotchas (each cost real time):
- **Smear breaks left-pad.** The smear mixes adjacent tokens, so left-padding makes
  the first real token smear in the pad. Must mask the smear at pad→real boundaries,
  not just attention — nanochat-specific; generic left-pad batching misses it.
- **Compile/cudagraph need fixed shapes.** Bucket the cache length (round to 256) and
  fix the batch size (pad partial batches), else recompile (~20s) every batch. Within
  a bucket, varying prompt lengths are free (one compile).
- **Mask must match the ACTUAL cache width** (`kv_cache.max_seq_len`), not the
  per-call bucket — `setup_caches` reuses a larger existing cache (idempotent `>=`),
  so a mixed-length task crashed until the mask was sized to the real cache.
- bf16 batching is non-deterministic at argmax ties (~0–2/64 differ vs per-prompt);
  fp32 is exact. Cross-run comparisons are noise-limited to ~1 point.

### Confirmed with data
- **Only GSM8K** (of the chat-eval tasks) uses the calculator tool (emits `python`
  blocks) → can't batch (needs the per-problem tool-use state machine).
  Addition/Multiplication/SpellingBee/HumanEval are raw (74/71/96/12% are genuine;
  SpellingBee's `.count()` tool path is commented out). Batched tool-free results
  match per-problem exactly (SpellingBee 96.09=96.09, Addition 74.22=74.22).
- **HF route:** compiling `transformers.generate` would hit the same two recompile
  knobs (batch size; prompt length if prefill is compiled) but NOT the cache-copy
  bug — HF `StaticCache` is per-layer by design. Default HF generate is eager.

### Anneal / minimal-finetuning (separate thread)
CUTE mini-eval curve of the 50%-CUTE anneal (wandb `byte-anneal-lr0.3`) is
**saturating-exponential to a ~96% ceiling** (gap-to-ceiling halves ~every 80–90
steps), not linear/log: 30→86.75, 60→90.75, 120→93.5, 300→95.75. **Dedicated short
anneals** (decay-to-0 in 30/60 steps) ≈ truncating the 300-run at the same step
(uncapped: 30→82.2 vs 81.9; 60→87.7 vs 86.8) → the low-LR "consolidation tail" buys
~nothing; CUTE tracks the *volume* of CUTE-mix steps, not the schedule shape. Strict
CUTE (must end the turn + nothing after the answer) == lenient on every subtask →
the model's outputs are already clean.

---

## 2026-06-03: Matched byte-vs-BPE CUTE learning curves + WSD/LR ablation + dip

Follow-on to the 2026-05-28 mix work. All runs are 50% mix of ClimbMix +
synthetic CUTE (no-demos), 300k words, resumed from the respective base
checkpoint, single RTX 6000 on genai. Figures + reproducible pipeline live in
`dev/` (`make -f dev/graphs.mk`); numbers pulled from wandb, not hand-typed.

### Headline: matched byte-vs-BPE learning curves (mix50, 300k, LR0.05, WD0.28)
Identical recipe, only the base model differs. x-axis = finetune tokens (log),
in-training full-char CUTE @100 + CORE@100/task, log-spaced eval points.
Spell-task 50%-crossing (tokens to learn to spell):

| model            | spell→50% | CUTE-mean endpoint | CORE (flat throughout) |
|------------------|-----------|--------------------|------------------------|
| d24-byte-l-ext   | ~ft14     | ~0.96              | ~0.25 (base 0.253)     |
| d24-byte-l-early | ~ft21     | ~0.96              | ~0.16 (base 0.169)     |
| d24-bpe (d24)    | ~ft127    | ~0.67 @ ft458      | ~0.29 (base 0.260)     |

**Findings:**
- Byte learns the char tasks ~**7-10x faster (in tokens)** than BPE, despite
  byte-tokens being ~3x denser (so byte sees ~3x FEWER underlying chars at
  matched tokens). Byte advantage ≫ amount-of-byte-pretraining: early vs ext
  differ only ~1.5x, both ~order-of-magnitude ahead of BPE.
- **CORE stays flat for all three** across the whole finetune (each at its own
  base floor) — mixing preserves capability regardless of how much CUTE is
  learned. No forgetting at LR0.05.
- contains_char is the one task BPE keeps pace early on (binary yes/no, no char
  decomposition needed) — confirms the byte edge is specifically char-access.

### WSD / high-LR ablation (LR0.8, SFT-style, vs the LR0.05 default)
Tested whether a hot stable LR (0.8, matching chat_sft) + short warmdown beats
the gentle 0.05 "barely nudge". It does NOT, for capability:
- bpe @ LR0.8: CUTE saturates fast but CORE settles **0.216** (vs LR0.05 mix50-e4
  0.276) — warmdown did not recover it.
- byte @ LR0.8: CORE dips to ~0.10 early, partially recovers to **~0.16** (base
  0.253) — byte is ~2x more LR-fragile than bpe. CUTE saturated throughout.
- Conclusion: **flat low LR (0.05) is on a strictly better CORE/CUTE frontier.**
  High-LR WSD buys speed at a real, bounded CORE cost; not worth it here. The
  anneal-from-checkpoint idea is undercut because hot-phase checkpoints are all
  CORE-eroded — run the stable phase cool, don't rely on annealing to repair it.

### spell_inverse transient dip (byte-l-early)
Dense dip-probe (checkpoints every ~2 steps via `SAVE_AT_EVALS=1`) caught it:
spell_inverse rises to ~0.27 (ft16), **craters to ~0.01 at ft24-28**, recovers
to 0.91 by ft64. `--debug-n` on the trough checkpoint: the model passes its
spaced input straight through unjoined (`" t h e "` → `t h e`) instead of
joining it (→ `the`) — i.e. forward-spell's "emit spaced letters" output reflex
transiently contaminates inverse-spell during the forward-spell rise. Real
task-interference transient, recovers on its own; minor (spell_inverse is easy
= "delete the spaces"), not a knowledge-loss finding.

### Compute accounting (all nanochat* wandb projects, `dev/wandb_compute.py`)
~268 wall-clock h / ~274 GPU-h across 100 runs. ~248 GPU-h on genai's RTX 6000
(the friend's box — single GPU; metadata mislabels it "RTX 3090" because the
box has both a 3090 and a 6000 and device 0 enumerates first). Base pretraining
dominates (the two big d24 byte bases = ~143h). Real multi-GPU work was on
Runpod: d24-stock BPE base pretrain (8xH100, ~978K tok/s, 59% MFU, 2.79e19
FLOPs / ~28 EFLOP) + two SFT jobs, all in the nanochat-sft project. NOTE: the
logged `total_training_flops` is a resume-cumulative counter — do NOT sum it
across finetunes (double-counts base pretraining); only `_runtime` sums.

### Tooling added this session
- `cute_mix.sh`: `LOG_EVALS` (log-spaced 1,2,4,…), `EXTRA_OFFSETS` (1.4x
  infill), `SAVE_AT_EVALS` (checkpoint at every eval point), `CORE_MAX_PER_TASK`.
- `base_train.py`: `--cute-every`/`--cute-subtasks char` (in-training mini-CUTE
  + cute/mean), `--cute-at-steps`/`--core-at-steps`, `--log-step-offset` (ft_step
  axis for log plots), `--save-at-evals`.
- `dev/graphs.mk` + `dev/plotsave.py`: one command regenerates all 7 figures
  (png+svg) from data. plot_core now reads model identity from base_eval/<tag>/
  subdirs (the base_eval migration) instead of a step-number lookup.
- dataset.py: dropped the silent FineWeb-EDU fallback (was masking a
  missing-CUTE-dir bug as "training fine").

---

## 2026-05-28: CUTE experiment inventory + mixed-data capability preservation

**Why this entry exists**: the CUTE work started as a clean grid and drifted
into several ad-hoc clusters. This is the map. Each cluster is coherent on its
own; they don't stack into one grid because **each holds a different thing
constant**. Numbers below are CUTE char-subtask mean (8 subtasks, 100 problems
each, zero-shot completion) and CORE (max-per-task 500, single GPU).

### The datasets on disk (`~/.cache/nanochat/cute_sweep/`)

- `results_sft-mask.csv` — **the main grid.** sft-mask recipe (FT_LRM=0.8,
  WARMDOWN_FRAC=0.5, WD=0, MASK_BEFORE='Answer: "'). 4 base models ×
  up to 7 sizes (1k,3k,10k,30k,50k,100k,300k). Holds **recipe + ~2-epoch
  CUTE exposure** constant; varies base model and data size.
- `results_mix.csv` — the capability-preservation experiments (below).
- `results_scratch.csv` — from-scratch byte LM, 2 cells only (30k, 100k).
  Abandoned (see below).
- `results.csv`, `results_sft-mask-OLD.csv` — dead/superseded (old "nodemos"
  recipe and a prior grid version). Ignore.

### Base-model CORE (no CUTE training), for reference
d24 (BPE) = 0.260, byte-l-ext = 0.253, byte-l = 0.195, byte-l-early = 0.169.
These are the "capability ceilings" the CUTE-trained variants are measured against.

### Cluster A — cute_pt destroys capability (the motivating finding)
100% CUTE midtraining (no general data). CUTE accuracy is high but CORE collapses:

| cell | CUTE mean | CORE |
|------|-----------|------|
| d24-BPE @ 300k cute_pt        | 0.75 | **−0.008** |
| byte-l-ext @ 30k cute_pt      | 0.925 | 0.003 |
| byte-l-ext @ 10k cute_pt      | —     | 0.087 |

Catastrophic forgetting: the model learns CUTE format and loses everything else.
`contains_char` also bias-collapses (≈0.48, constant-ish answer) at high saturation.

### Cluster B — byte mix @ 30k (holds CUTE-tokens-seen constant)
byte-l-ext base, mix fraction ∈ {10,20,50}%, MASK_BEFORE applied only to the
CUTE stream. **Each cell sees the same ~2 epochs of the 30k CUTE data**, so
total step count varies with 1/mix_frac (458 / 229 / 92). This holds CUTE
exposure fixed and varies dilution.

| cell    | ft_steps | CUTE mean | CORE  |
|---------|----------|-----------|-------|
| mix10   | 458      | 0.967     | 0.250 |
| mix20   | 229      | 0.968     | 0.243 |
| mix50   | 92       | 0.908     | 0.242 |

All preserve CORE (≈ base 0.253). mix10/20 saturate CUTE; mix50 dips on
`contains_char` (under-trained at 92 steps). Byte models pay almost no CUTE
cost for mixing — they're already character-native.

### Cluster C — BPE mix @ 300k (two sub-experiments, different held-constant)
d24-BPE base. This is where it got ad-hoc. Two crossing slices:

*C1 — hold CUTE-tokens-seen constant (~2 epochs), vary mix fraction:*
| cell      | ft_steps | CUTE mean | CORE  |
|-----------|----------|-----------|-------|
| mix50-e1  | 229      | 0.51      | 0.268 |
| mix70-e1  | 164      | 0.48      | (not run) |

*C2 — hold mix fraction, vary CUTE exposure (epoch bumps, the -e3/-e4 tags):*
| cell        | ft_steps | CUTE mean | CORE  |
|-------------|----------|-----------|-------|
| mix70-e3    | 246      | 0.57      | 0.268 |
| mix50-e4    | 458      | 0.67      | **0.276** |

**Conclusion for BPE**: every mix cell preserves CORE (0.268–0.276, all ≥ base
0.260, even slightly above). CUTE accuracy is **compute/exposure-limited, not
mix-ratio-limited** — mix50-e4 (more total training) beats mix70-e3 (higher CUTE
share, less training) on both CUTE and CORE. mix50-e4 reaches 0.67 CUTE (89% of
cute_pt's 0.75) with zero capability loss. The last ~0.08 of CUTE that cute_pt
buys costs the entire model.

`contains_char` recovers from cute_pt's collapsed 0.48 to 0.88 under mix50-e4 —
mixing fixes the bias-collapse failure mode.

### Cluster D — from-scratch byte (abandoned)
Train a byte LM from random init on CUTE-only data; does pretraining matter for
CUTE? 100k-words cell: spell 26% / spell_inverse 23% (vs ~100% for
pretrained+cute_pt at the same size). Pretraining is essential; from-scratch is
not competitive. Stopped after 100k. (Note: used base_train's default warmdown
schedule, which at ~150 steps is mostly warmup+warmdown with little plateau —
but the schedule wasn't judged to be the cause; the gap is too large.)

### Headline
The capability-preservation thesis holds for **both tokenizers**: mixing ~50%
general (ClimbMix) data with CUTE preserves CORE fully while still teaching the
character tasks. Pure cute_pt's CORE collapse is avoidable. The cost is some
CUTE accuracy (recoverable with more compute), not capability.

### If re-cleaning the grid
The clean version of Cluster C would be one (model, size) with mix_fraction ×
total-steps as explicit axes, rather than the current two crossing slices that
each hold a different thing fixed. Tooling supports it: `CELL_TAG_SUFFIX` lets
variant runs coexist without clobbering, and `--cute-every` logs mid-training
CUTE curves to wandb.

### Code produced (this session, branch `cudagraph`)
- `nanochat/dataloader.py`: per-row source selection — loader takes
  `[(stream, weight), ...]`, Bernoulli-picks a source per row, best-fits within
  it. mask_before applies per-stream. (Replaced two earlier attempts: within-row
  best-fit pushed small CUTE docs to row-tail crops and chopped the mask marker;
  sequential+shuffle fixed that but lost best-fit's low waste.)
- `nanochat/dataset.py`: removed the silent FineWeb-EDU fallback (it masked a
  missing-CUTE-dir bug as "training fine on web text").
- `runs/cute_mix.sh`, `dev/sweep_cute_mix.sh`: mixed-data launcher + sweep.
- `scripts/base_train.py`: `--mix-data-dir`/`--mix-fraction`, `--cute-every`
  in-training mini-CUTE eval.

---

## 2026-05-25: Inference speedup on Blackwell — manual CUDA graphs beat torch.compile

**Epistemic-status caveat**: this entire entry was written by Claude during
the debugging session. The **wallclock numbers and microbench results are
empirical** (msully ran them and reported them). Everything else —
explanations for *why* one approach was faster than another, claims about
what PyTorch's reduce-overhead wrapper does internally, claims about
per-call kernel-launch overheads, claims about why gpt-fast's pattern works
for them but not us, etc. — is the LLM's best-guess interpretation, not
verified against PyTorch source code or profiler data. Treat the rationale
as plausible-but-unconfirmed. The measurements stand; the stories about
them might be wrong in places.

Multi-hour debugging session trying to make cute_eval decode faster on the
RTX 6000 PRO (Blackwell sm120, no FA3 available). Final result: **1:22 for
200 prompts of contains_char (~13.7 ms/token at d24), ~3× faster than the
eager baseline of 4:00**.

### Headline tally of approaches tried

| approach | 200-prompt wallclock |
|---|---:|
| Eager `--no-compile` (original) | 4:00 |
| Eager + `sdpa_kernel([CUDNN_ATTENTION])` forcing | 4:21 (slower — per-call ctx mgr overhead in eager) |
| Manual cudagraph, per-prompt capture (f3449ef era) | ~5:00 (extrapolated from 2:30/100) |
| `torch.compile(mode="reduce-overhead")` + cudagraph_trees | 9:25 |
| `torch.compile(mode="reduce-overhead")` + cuDNN forcing | 7:39 |
| **Manual cudagraph + cached across calls + cuDNN forcing** | **1:22** ← winner |

### The non-obvious lesson: gpt-fast's pattern is wrong for small models

Meta's gpt-fast (the reference for fast PyTorch-native inference) uses
`torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)`.
Their benchmarks show this winning vs eager on Llama 7B/13B/70B. We naively
followed this pattern and got **2× slower than eager**, with no warnings.

Why: `reduce-overhead` has per-call wrapper overhead (Dynamo guard checks,
Inductor dispatch, cudagraph_trees machinery) that's negligible relative to
GPU work for 7B+ models but dominates at d24 (770M) where per-op GPU work is
already small. Manual `torch.cuda.CUDAGraph().replay()` is a single C++ call
with no Python tax.

If you re-attempt this in the future, **start with manual capture, not
reduce-overhead**.

### Why our cudagraph is so much faster than eager (when it works)

At d24 single-token decode, ~150 CUDA kernels launch per token across 24
layers. Each launch has ~5-10 µs of Python+dispatch overhead in eager →
~1-2 ms per token of pure overhead, plus actual GPU work. Cudagraph
captures all 150 launches as one replay-able sequence; replay is one
`graph.replay()` call. The Python overhead disappears.

Training doesn't hit this because B*T tokens per forward amortize the
overhead. (We get 50K tps in training vs 73 tps in eager inference at d24.)

### What's in the code now

- `nanochat/engine.py`: `KVCache` is an `nn.Module` with `register_buffer`s,
  attached to the model via `GPT.setup_caches(...)`. `Engine` lazily
  captures a decode-step graph on first call, caches per `num_samples`,
  reuses across all subsequent `generate()` calls.
- `nanochat/gpt.py`: `forward(idx, targets=None, input_pos=None, ...)` —
  `input_pos` is an explicit tensor argument (no `.item()` host syncs
  inside forward). Smear and bigram_value_embeds refactored to use
  in-place `copy_` into preallocated buffers (Python attribute mutations
  on the cache would break cudagraph correctness).
- `nanochat/flash_attention.py`: SDPA fallback uses full-cache + bool mask
  (fixed shape across decode steps — needed for cudagraph). Forces cuDNN
  backend via `sdpa_kernel([CUDNN_ATTENTION])` context manager — without
  this, SDPA's default selection picks MATH (2.4× slower) because Flash
  refuses non-null masks and Mem-Efficient/cuDNN are "runtime disabled"
  in default scoring on sm120. The context manager's per-call overhead
  matters in eager mode (the 4:00→4:21 regression above) but is traced
  out by graph capture, so it's a net win for the cudagraph path.

### Things rejected with data (don't redo)

- **`torch.compile(mode="reduce-overhead", fullgraph=True)`**: slower than
  manual cudagraph by 2-4× at our scale, see table above. Right answer for
  7B+ models; wrong for us.
- **FlexAttention with BlockMask**: bench showed compiled flex_attention at
  0.163 ms/call vs cuDNN at 0.085 ms/call for our exact shape. The
  Triton-generated kernel doesn't beat cuDNN's hand-written kernel at our
  shape, so integration would have made things slower. (gpt-fast uses
  FlexAttention because it gives them a clean way to express sliding-window
  + causal + paged-cache; we have a simpler attention pattern.)
- **SDPA default backend selection**: picks MATH for our (bool mask + GQA)
  on sm120 because the fast backends are "runtime disabled" in the default
  scoring. Force cuDNN explicitly.
- **Compiling the whole model with `torch.compile`**: per-prompt
  recompiles on variable prompt length killed wallclock. Compile only the
  decode step (fixed shape) — and even then, manual capture wins.

### Still on the table for future work (if you want to push further)

- `torch._inductor.config.coordinate_descent_tuning = True` (gpt-fast uses
  this; tunes Triton kernel block sizes per shape). Could help marginally.
- Fused custom attention kernel for our exact shape (Triton or CUDA C++).
  Would need to actually saturate Blackwell's memory bandwidth (~1.7 TB/s);
  we're probably at 20-30% utilization with the cuDNN call.
- FA3 sm120 port lands (tracking: Dao-AILab/flash-attention#2307). Would
  give us a dedicated KV-cache kernel; no port committed as of 2026-05.

### Diagnostic tooling

- `dev/sdpa_bench.py` — microbenchmark + backend probe for SDPA on our
  decode shapes. Forces each `SDPBackend` explicitly and reports which
  accept our inputs and how fast each runs. Also benchmarks FlexAttention
  compiled vs uncompiled. Useful for spotting backend selection issues.

---

## 2026-05-23: Byte vs BPE on CUTE char-level tasks (sweep + recipe ablation)

Setting up an experiment to measure whether byte-tokenizer models have a structural advantage over BPE on character-level tasks (the CUTE benchmark — Edman et al. EMNLP 2024). Approach: train a cheap "midtraining" (cute_pt) on synthetic CUTE-format examples, then eval zero-shot on the real CUTE benchmark. Sweep across (base model × dataset size × training recipe).

### Setup

- 4 base models, all d24 (1.38B total / 730M scaling params), single-GPU pretrained on friend's RTX 6000 PRO Blackwell:
  - `d24-byte-l-early` — byte, shortest pretrain (~5.5B byte tokens, ratio ≈ 8)
  - `d24-byte-l` — byte, full speedrun (~10.9B tokens)
  - `d24-byte-l-ext` — byte, extended (~22-25B tokens)
  - `d24` — BPE stock (5.84B BPE tokens, ratio 8)
- CUTE training data: `dev/gen_cute_pt_data.py --no-demos` generates bare `Question: ... \n\nAnswer: " w o r d "` documents (no 4-shot demo prefix). 8 char-level subtasks per word.
- Eval: `scripts/cute_eval --prompt-style zero` strips the published 4-shot demo prefix so eval surface form matches training.
- Sweep driver: `dev/sweep_cute_pt.sh` with SKIP_DONE=1 default, per-cell ETA from rolling-dt window (fix for inherited cumulative `total_training_time` from 8×H100 base pretrain).

### Headline results (no-demos recipe, BUDGET_MODE=epochs)

Mean accuracy across 8 char-level subtasks, eval at 100 problems/subtask:

| model            | 10k   | 30k   | 100k  |
|------------------|------:|------:|------:|
| d24-byte-l-ext   |  —    | 0.529 |  —    |
| d24-byte-l       |  —    | 0.365 |  —    |
| d24-byte-l-early | 0.120 | 0.244 | 0.840 |
| d24 (BPE stock)  |  —    | 0.065 | 0.171 |

The d24-byte-l-early jump 30k → 100k (mean 0.24 → 0.84) is the dramatic result. Manipulation subtasks (ins/del/sub/swap_char) go from ~0% to 70-93% — they don't just need pretraining strength (ext at 30k gets only 14-39% on those), they need CUTE training data volume. At 100k, the LEAST-pretrained byte model crushes BPE.

### Per-subtask gap at 100k (early-byte vs BPE-stock)

| subtask        | early (byte) | d24 (BPE) | gap   |
|----------------|-------------:|----------:|------:|
| spell          | 1.00         | 0.05      | 20×   |
| spell_inverse  | 1.00         | 0.00      | ∞     |
| contains_char  | 0.57         | 0.65      | BPE wins! |
| orth           | 0.71         | 0.54      | 1.31× |
| ins_char       | 0.93         | 0.10      | 9.3×  |
| del_char       | 0.91         | 0.00      | ∞     |
| sub_char       | 0.91         | 0.02      | 45×   |
| swap_char      | 0.69         | 0.00      | ∞     |
| **mean**       | **0.84**     | **0.17**  | ~5×   |

BPE saw *more* underlying text in pretraining (~24 GB vs early-byte's ~5.5 GB) and the same cute_pt budget, yet on edit tasks it's at ≤10% while byte's at 70-93%.

### But — BPE *can* do this under proper SFT

Caveat that reshapes the framing: user's earlier full SFT pass (200K rows of SimpleSpelling + SpellingBee mixed with smoltalk/mmlu/gsm8k) hit ~100% on SimpleSpelling for d24-stock. With 1K SimpleSpelling rows it dropped to 60-70% — still way above what no-demos cute_pt achieves on CUTE spell. So BPE isn't structurally incapable; the no-demos cute_pt recipe is specifically BPE-unfriendly.

### Why cute_pt is BPE-unfriendly (analysis of SFT vs cute_pt code)

1. **Loss masking** — SFT computes loss only on assistant tokens (`mask=1` for assistant content in `tokenizer.render_conversation`, `target=-1` elsewhere). cute_pt is bare text continuation, loss on every token. For BPE specifically, half the gradient signal is wasted on predicting the `Question:` prompt text the model never needs to *produce*.
2. **Format diversity** — SFT trains on a TaskMixture (SmolTalk 460K + MMLU 100K×3 + GSM8K 8K×4 + SimpleSpelling 1K rendered to 200K + others). The model learns generic instruction-following with spelling as one small instance. cute_pt has only CUTE.
3. **Chat format tokens** — SFT uses `<|user_start|>`/`<|assistant_start|>` clean structural separators. cute_pt has no boundaries.
4. **Spelling output format** — SFT SimpleSpelling: `nadorite:n,a,d,o,r,i,t,e` (bare ASCII-range tokens 97-116, comma separators). CUTE spell: `" n a d o r i t e "` (space-prefix tokens like ` n`, ` a` — common individually but the *sequence* of space-letter-space-letter is essentially never seen in pretraining text).
5. **LR schedule** — SFT uses `init_lr_frac=0.8` with `warmdown_ratio=0.5`. cute_pt anchors at `FT_LRM=0.05` with 10% warmdown.

### Recipe ablation plan

Added two flag-gated interventions (defaults off, sweep-aware):

- **`--mask-before STRING`** (base_train.py + dataloader): per-document loss masking. Sets targets to -1 for positions predicting the prompt region of each sub-doc, defined as everything up to (and including) the tokenized form of STRING. Pair with `MASK_BEFORE="Answer: "`.
- **`SFT_STYLE=1`** (cute_pt.sh env var): swap the LR schedule from "barely nudge" (FT_LRM=0.05, 10% warmdown) to SFT-style "actively finetune" (FT_LRM=0.8, 50% warmdown).

`dev/sweep_cute_pt.sh` extended to auto-derive a `RECIPE` tag from the flag combination (`nodemos` / `mask` / `sft` / `sft-mask`), include it in DST_TAG (`${model}-cute-${recipe}-${size}w`), and write to a per-recipe CSV (`results.csv` for nodemos to preserve back-compat; `results_${recipe}.csv` otherwise).

4-cell ablation matrix to run:

| Recipe   | SFT_STYLE | MASK_BEFORE | Hypothesis |
|----------|-----------|-------------|------------|
| nodemos  | 0         | (empty)     | Baseline (already have data) |
| mask     | 0         | "Answer: "  | Loss masking alone (probably the biggest single-knob win) |
| sft      | 1         | (empty)     | Aggressive LR alone |
| sft-mask | 1         | "Answer: "  | Both — closest analogue to SFT, expected best |

Invocation reference: `dev/cute_sweep_recipes.md`.

### Likely blog post framing

Not "byte structurally beats BPE on char tasks" (false — SFT proves BPE can do char tasks). Instead:

> "Byte tokenizers are more data-efficient than BPE under *minimal* midtraining recipes — bare-completion training, no loss masking, no format diversity, no chat scaffolding. Under proper SFT, BPE almost entirely catches up. The byte advantage is real but specifically valuable when full instruction-tuning isn't an option."

Pending: run the ablation, confirm sft-mask closes the BPE gap, write up.

### Open questions for future runs (not blocking the blog)

- 300k cells for the two anchor models (early-byte, d24-BPE) — does byte saturate? Does BPE keep climbing or plateau?
- General-eval impact: cute_pt with these recipes might degrade CORE; the original "(4) mix in general text" idea (deferred) would test that.
- Apply same recipe-improvement approach to chat_sft? Probably orthogonal — chat_sft already has all 5 of these features built in.
- **From-scratch BPE on 120M CUTE-only tokens**: if cute_pt-at-300k destroys general capability anyway, does the 11×-ish pretraining contribute anything to the final CUTE score? Train a fresh d24 from random init on the 300k CUTE data (60M tokens × 2 epochs = 120M training tokens), eval CUTE, compare to `d24-cute-sft-mask-300000w`. Three outcomes worth distinguishing: (a) from-scratch matches/beats → pretraining contributed nothing of value; the comparison was always "char-task model vs general model with char-data tacked on". (b) From-scratch dramatically worse → pretraining provides essential scaffolding (Q/A format, English priors, instruction-following) that 60M narrow tokens can't bootstrap. (c) Mixed by subtask → pretraining specifically helps word-knowledge tasks (spell_inverse, orth) but not pure manipulation. ~25 min train + 15 min eval per cell on RTX 6000 PRO; under 2h for a 3-seed average.
- Confusion matrix for contains_char at 100k vs 300k: at 0.48 (chance on a 2-class task), we can't distinguish "lost capability but kept format" from "format collapse to one of Yes/No". Per-cell Yes/No prediction distribution + fraction of format-invalid predictions resolves it. Spot-checkable in minutes; would tell us *which* failure mode dominates the contains_char regression.

---

## 2026-03-24: Parameter-Golf Ideas Sweep (Negative)

Reviewed `openai/parameter-golf` for small/simple ideas that might transfer to nanochat pretraining without bloating the codebase. Cached notes are in `knowledge/parameter_golf.md`.

### Rationale

The parameter-golf leaderboard is a useful source of:

- tiny architecture tweaks
- short-run optimizer/schedule tricks
- Muon-related systems ideas

But much of that repo is optimized for a very different objective:

- fit in a 16MB artifact
- train in under 10 minutes on 8xH100
- evaluate on compression / bpb

So only a small subset of ideas looked worth trying in nanochat.

### Ideas Tried

**1. LeakyReLU(0.5)^2**
- Replaced `relu^2` in the MLP with `leaky_relu(x, 0.5)^2`
- **Result:** Slightly better per-step quality, but slightly slower. Net worse on wall clock.

**2. Partial RoPE**
- Applied rotary embeddings to only the first quarter of each head dimension
- **Result:** Slightly worse.

**3. LN Scale**
- Multiplied each block's normalized input by `1/sqrt(layer_idx+1)` before attention and MLP
- **Result:** Did not help.

**4. Orthogonal init**
- Switched the non-zero transformer matrices to orthogonal init while preserving zero-init output projections
- **Result:** Did not help.

**5. XSA (Exclusive Self Attention)**
- Implemented XSA on the deepest 3 non-VE layers only, so it projected against the plain `v` path rather than `v + VE`
- **Result:** Slightly better step quality but not wall clock. Not worth the extra compute in the hot attention path.

### Notes

- EMA/SWA had already been tried earlier (I skipped recording it) and did not help.
- Bigram hash embeddings had already been explored much earlier and did help somewhat, but the added parameters / VRAM / complexity were not justified at larger scale. See the Jan 27-28 entries above.

### Conclusion

This pass did not find any cheap parameter-golf transfer that clearly improves nanochat on the metric that matters: wall clock time to capability.

---

## 2026-03-04: Remove autocast, explicit dtype management, fp16 GradScaler

Replaced `torch.amp.autocast` throughout the codebase with explicit dtype management via a single `COMPUTE_DTYPE` global. Also added fp16 training support with GradScaler.

### Motivation

autocast is "magic we don't control" — it silently decides which ops run in which precision via internal allowlists. For this codebase, autocast was doing very little: the only thing it actually cast was `nn.Linear` weights from fp32 to bf16 for matmuls. `F.rms_norm`, `F.cross_entropy`, and Flash Attention all handle their own dtypes already. By making precision explicit, we gain fine-grained control (e.g. can experiment with fp32 norms) and eliminate an unnecessary layer of abstraction.

### What changed

**Core mechanism** (`nanochat/common.py`, `nanochat/gpt.py`):
- `COMPUTE_DTYPE` auto-detected from hardware: SM 80+ → bf16, pre-Ampere → fp32, CPU/MPS → fp32. Override via `NANOCHAT_DTYPE` env var.
- Custom `Linear(nn.Linear)` class that casts weights to match input dtype in forward: `F.linear(x, self.weight.to(dtype=x.dtype))`. This is the single mechanism that replaces autocast.
- Embeddings cast to `COMPUTE_DTYPE` at init (saves memory). Exception: fp16 keeps embeddings fp32 because GradScaler cannot unscale fp16 gradients.
- Embedding output explicitly cast to `COMPUTE_DTYPE` in `GPT.forward()` (no-op for bf16, active for fp16 path).
- RoPE cos/sin cache uses `COMPUTE_DTYPE` instead of hardcoded bf16.

**Autocast removal** (11 files):
- Deleted `--dtype` CLI flag, `ptdtype` variables, `autocast_ctx` definitions, and all `with autocast_ctx:` blocks from: `base_train.py`, `chat_sft.py`, `chat_rl.py`, `chat_cli.py`, `chat_eval.py`, `chat_web.py`, `base_eval.py`, `engine.py`, `bench_train_toks.py`, `test_e2e_pipeline.py`.

**fp16 + GradScaler** (`base_train.py`, `chat_sft.py`):
- `scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None`
- Backward: `scaler.scale(loss).backward()` vs plain `loss.backward()`
- After accumulation: `scaler.unscale_(optimizer)` → distributed inf-sync via `scaler._found_inf_per_device(optimizer)` all-reduced with `ReduceOp.MAX` → `scaler.step(optimizer)` → `scaler.update()`
- Zero overhead for bf16/fp32 paths (scaler is None, no branching inside kernels).

**FP8 fix** (`nanochat/fp8.py`, `base_train.py`):
- `Float8Linear.forward` explicitly casts input to `COMPUTE_DTYPE` (previously relied on autocast).
- `disable_fp8` context manager now creates our custom `Linear` (not vanilla `nn.Linear`) when swapping out Float8Linear during eval.

**Flash Attention** (`flash_attention.py`):
- FA3 Hopper kernels don't support fp16 or fp32, so `USE_FA3` (module-level constant, resolved once at import) returns False, falling back to SDPA.

---

## 2026-03-04: Dataset upgrade: FineWeb-EDU 100B → ClimbMix 400B

Switched the pretraining dataset from FineWeb-EDU 100B to ClimbMix 400B. This is by far the single biggest improvement to nanochat's GPT-2 speedrun time, bringing it down from **2 hours 46 minutes to 2 hours 1 minute** — a 27% reduction.

### What is ClimbMix?

ClimbMix 400B is a curated 400B-token pretraining mixture hosted at `karpathy/climbmix-400b-shuffle` on HuggingFace. It comes form [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix). It is a blend of high-quality web text, code, math, and other sources, designed to be a better general-purpose pretraining dataset than FineWeb-EDU alone.

### What changed

- **Dataset**: `karpathy/fineweb-edu-100b-shuffle` → `karpathy/climbmix-400b-shuffle` (up to 6543 shards available vs the previous 1823 data shards, allowing for longer training in the future)
- **Data directory**: `base_data/` → `base_data_climbmix/` (clean separation from legacy data)
- **Model depth**: d26 → d24. ClimbMix trains more efficiently, so a smaller model reaches GPT-2 capability
- **Shard count**: Only approx 150 data shards (~7B tokens) are now needed for GPT-2 capability
- **Eval tokens**: doubled from 40 to 80 batches for more stable validation loss estimates
- **Legacy fallback**: added a migration warning in `list_parquet_files()` that detects the old `base_data/` directory and falls back gracefully, so existing users see clear upgrade instructions on `git pull`

### Context

This is the sixth attempt at beating FineWeb-EDU on CORE score — the previous five all failed (see entries on 2026-02-17, 2026-02-10, 2026-01-12 below). ClimbMix is the first dataset to convincingly surpass it, and the margin is large enough to also shrink the model from d26 to d24.

---

## 2026-03-02: SoftCap tuning

Quick experiment to tune logit softcap on d24 scale. Tried 5..30. 5 was terrible, the rest of them were all about equal with the exception of 20, which was the best. Minor but solid improvement: val loss improved by ~1e-3 (0.716 -> 0.715). Setting as default.

## 2026-02-19: Mixture of Experts (negative)

Implemented a DeepSeekV3-style Mixture of Experts layer as a drop-in replacement for the dense MLP. The MoE branch works and improves per-step validation loss, but is not a net improvement on wall clock time due to MoE overhead (at least for our scale of interest of approx GPT-2 capability).

### Implementation

Follows DeepSeekV3 and using torchtitan as reference:

- **8 routed experts, top-2 routing** with sigmoid gating (not softmax)
- **1 shared expert** (dense MLP processing all tokens, following DeepSeekV3)
- **Auxiliary-loss-free load balancing** (DeepSeekV3's expert bias nudging)
- **Iso-FLOP sizing**: `expert_hidden_dim = round(4 * dim / (top_k + num_shared) / 128) * 128`, so active FLOPs per token match the dense MLP
- **`torch._grouped_mm`** for dispatching tokens to experts in a single kernel (instead of a Python for-loop)
- **3D expert weight tensors** `(num_experts, hidden, dim)` — Muon's Polar Express operates on the last two dims, so each expert is independently orthogonalized
- **Active parameter counting** for scaling laws (only `top_k + shared` experts, not all 8)

### What was easy

- The core MoE forward pass: router, sort tokens by expert, grouped matmul, scatter back. Conceptually clean.
- Shared expert: just an `nn.Linear` MLP that runs on all tokens alongside the routed path.
- 3D expert params + Muon: only required fixing `second_momentum_buffer` shape to preserve leading dims.
- Load balancing: DeepSeekV3's bias nudging is simple and effective (~10 lines).

### What was hard / ugly

- **`torch._grouped_mm` quirks**: requires bf16 (not fp32), column-major right operand, int32 cumulative offsets. The API is undocumented and only discoverable by trial and error.
- **Token count padding**: torchtitan pads each expert's token count to alignment multiples (8 for bf16) for better grouped_mm throughput. We implemented this with both a pure PyTorch approach and a copy of torchtitan's Triton kernel. Both compiled cleanly (0 graph breaks), but with ~65K tokens across 8 experts, each expert already gets ~8K tokens which is well-aligned. The padding overhead (gather/scatter) actually regressed MFU from 35% to 33%. Reverted.
- **FP8 + MoE**: `torch._grouped_mm` does NOT support FP8. There's a separate `torch._scaled_grouped_mm` API that requires per-row scaling (not per-tensor like our `Float8Linear`). The backward pass for weight gradients needs per-group column-wise scaling, which torchao implements with custom Triton kernels. We investigated thoroughly (see `dev/moe_fp8.md`) but did not implement — would require either depending on `torchao.prototype` (unstable) or writing ~200 lines of custom autograd + quantization code. Partial FP8 support exists: the shared expert's `nn.Linear` layers do get converted, but the routed experts (3D `nn.Parameter`) stay in bf16.

### Results

- d18: MFU dropped from ~46% to ~35% (the grouped_mm dispatch + token sorting overhead is significant)
- Per-step improvement in validation loss does not compensate for the throughput hit
- Net negative on wall clock time

### What remains (if revisited)

- **FP8 for routed experts**: Use `torch._scaled_grouped_mm` with a custom `_Float8GroupedMatmul` autograd function, with bf16 fallback for weight gradient (avoiding the per-group column-wise Triton kernels).

What's really needed is a fused "FlashMoE" kernel that handles routing + expert dispatch + matmul in one shot (like FlashAttention did for attention), with all the needed features. This doesn't exist yet. Rawdogging MoE with current PyTorch primitives is painful — lots of sorting, gathering, scattering, and layout wrangling around the actual compute.

### Verdict

MoE is not worth the trouble for nanochat right now. The code bloat is substantial (moe.py, router, shared expert, load balancing, optimizer fixes, FP8 gaps, active param counting) and the performance is worse wall-clock at our scale of interest. The fundamental issue is that the grouped_mm dispatch overhead eats the FLOP savings from sparsity, at least at our model scales and sequence lengths.

---

## 2026-02-17: Pretraining Data: FineWeb (negative)

Tried vanilla fineweb instead of fineweb-edu dataset. Significantly, shockingly worse results:

- d26 (GPT-2): CORE 0.2602 → 0.2241

This is the fifth failed attempt to beat pure FineWeb-EDU on CORE score.

---

## 2026-02-17: Pretraining Data Mixture Experiment (negative)

Tried [hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT](https://huggingface.co/datasets/hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT), a mixture of FinePDFs, DCLM, and FineWeb-EDU. Slightly worse on both model sizes tested:

- d26 (GPT-2): CORE 0.2602 → 0.2549
- d18: CORE 0.199 → 0.192

This is the fourth failed attempt to beat pure FineWeb-EDU on CORE score.

---

## 2026-02-16: SFT Script Upgrades

Brought `chat_sft.py` up to parity with `base_train.py` and tuned settings based on SFT sweeps.

Tuning:

- **Optimizer warm-start** (`--load-optimizer=1`, default on): loads pretrained momentum buffers via new `load_optimizer_state()` in `checkpoint_manager.py`. LRs are reset to fresh SFT values after load. Loading the optimizer works slightly better but not by too much.
- **LR schedule**: replaced "constant 80%, linear to 0" with warmup/constant/warmdown matching `base_train.py` (`--warmup-ratio`, `--warmdown-ratio`, `--init-lr-frac`, `--final-lr-frac`). Similar to pretraining, warmdown ratio of 0.5 worked the best. `--init-lr-frac` changed from 1.0 slightly lower to 0.8.
- **LR tuning**: attempted to tune all the individual LRs (e.g. does SFT prefer lower LR for embeddings? etc.) but all of this produced negative results.
- **Data mixture**: MMLU epochs 1→3, GSM8K epochs 2→4 (confirmed best from sweeps). Epoch counts now configurable via `--mmlu-epochs` / `--gsm8k-epochs`. Might remove these in the future though.

Quality of life, footguns, minor fixes:

- **Hyperparameter inheritance**: SFT now inherits batch sizes and LRs from the pretrained checkpoint metadata by default (CLI overrides still work). Also saved `total_batch_size` to `base_train.py` checkpoint metadata.
- **GC management**: disabled Python GC after step 1 to avoid ~500ms pauses (manual collect every 5000 steps), same as base pretraining.
- **ChatCORE eval**: periodic eval during SFT (`--chatcore-every=200`) across all 6 tasks, logged to wandb.
- **MFU**: uses `get_peak_flops()` for actual GPU instead of hardcoded H100 value.
- Removed `--dry-run` and `--dtype` flags. All ranks now participate in checkpoint save.

---

## 2026-02-05: Auto Batch Size Scaling

### Background

So far, the `--total-batch-size` was hardcoded to be `2**19 = 524,288` ~= 0.5M tokens. This was the optimal setting for d12, but when I tried to re-tune it for d26 (GPT-2), I noticed that the optimal was closer to `2**20 = 1,048,576` ~= 1M tokens. This is to be expected - larger models prefer a higher optimal total batch size. However, we have to make sure that all settings of `--depth` get their own optimal batch size calculated in some principled way. Here, I referenced the "Power Lines" paper from Cerebras ([arXiv:2505.13738](https://arxiv.org/abs/2505.13738)) for a lot of related experimentation. In particular, they found that **Bopt ∝ D^0.383** (where D is the number of training tokens, not the number of parameters!). So the idea is to tune the optimal batch size on d12, and then extrapolate it with this power law to bigger models. The 0.383 exponent means batch size grows slowly: 10× more tokens only justifies ~2.4× bigger batch. For nanochat's compute-optimal training (D ∝ N via `--target-param-data-ratio`), this means deeper models naturally want larger batches.

### Implementation

Added `--total-batch-size=-1` (now the default) to auto-compute optimal batch:

```python
get_scaling_params = lambda m: m.num_scaling_params()['transformer_matrices'] + m.num_scaling_params()['lm_head']
if args.total_batch_size == -1:
    D_REF = args.target_param_data_ratio * get_scaling_params(build_model_meta(12))
    B_REF = 2**19
    args.total_batch_size = 2 ** round(math.log2(B_REF * (target_tokens / D_REF) ** 0.383))
```

Reference point: d=12 model with B=2^19 (empirically validated). The reference is computed dynamically so that if the architecture changes (e.g., different `--aspect-ratio`), the math automatically adjusts. However, if the model actually does change too much, one would also want to re-tune the optimal batch size for d=12.

### Results

With this formula, we currently get:

| Depth | Scaling Params | Target Tokens | Auto Batch |
|-------|---------------|---------------|------------|
| d=8   | 42M           | 0.44B         | 2^18 = 262K |
| d=10-16 | 70M-235M    | 0.7B-2.5B     | 2^19 = 524K |
| d=18-26 | 324M-918M   | 3.4B-9.6B     | 2^20 = 1.05M |
| d=32-50 | 1.7B-6.2B   | 17.6B-65.6B   | 2^21 = 2.1M |

In particular, this matches empirical observations that d26 prefers ~2^20 while d12 prefers ~2^19.

### Code Cleanup

Also refactored model initialization to use `build_model_meta(depth)` helper and `dataclasses.asdict()` for cleaner config handling.

### Useful references

- [Bergsma et al., Power Laws for Batch Size, Model Size, and Training Horizon](https://arxiv.org/abs/2505.13738)
- [McCandlish et al., An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)
- [Brown et al., Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Merrill et al., The Batch Size–Critical Batch Size Myth](https://arxiv.org/abs/2505.23971)

### One more thing (batch size ramp)

Tried batch size ramping. The simplest implementation I could think of "tricks" the existing training loop by slicing each micro-batch into smaller pieces and calling optimizer.step() more frequently early in training (1/8 → 1/4 → 1/2 → full batch over the first x% of training, with sqrt LR scaling). Also required a torch.compile warmup phase to pre-compile all slice sizes and avoid recompilation spikes during training. While the idea is sound and small gains were observed, they weren't sufficient to justify the code complexity introduced (conditional slicing logic, warmup with state save/restore, etc.). Not merged for now.

---

## 2026-02-05: SwiGLU Activation (Negative Result)

Replaced ReLU² MLP activation with SwiGLU (inspired by [twitter](https://x.com/_xjdr/status/2019141521690567058)). SwiGLU uses three projections instead of two, so to match parameters and FLOPs we scale hidden_dim from 4× to 8/3×:

```python
# Old ReLU²: 2 matrices, 4x expansion
#   params: 2 × n × 4n = 8n²
#   flops:  2 × 2n × 4n = 16n² per token
self.c_fc   = Linear(n_embd, 4 * n_embd)
self.c_proj = Linear(4 * n_embd, n_embd)
x = c_proj(relu(c_fc(x)).square())

# New SwiGLU: 3 matrices, 8/3x expansion
#   params: 2 × n × (8n/3) + (8n/3) × n = 8n²  ✓ matches
#   flops:  3 × 2n × (8n/3) = 16n² per token   ✓ matches
hidden_dim = (8 * n_embd) // 3
self.w1 = Linear(n_embd, hidden_dim)  # gate
self.w2 = Linear(n_embd, hidden_dim)  # up
self.w3 = Linear(hidden_dim, n_embd)  # down
x = w3(silu(w1(x)) * w2(x))
```

Tested at both d12 and d24 (GPT-2 scale). Worse on all measures — step efficiency, wall clock time, and FLOPs. ReLU² remains superior for nanochat. **Not adopted.**

---

## 2026-02-03: Flip Muon MLP LR Multiplier (PR #492)

Tested flipping the shape-based LR heuristic in Muon from boosting tall matrices (input projections like `c_fc`) to boosting wide matrices (output projections like `c_proj`). The original code applies `max(1, rows/cols)^0.5`, giving ~2x LR to `c_fc`. The flipped version gives ~2x LR to `c_proj` instead, which aligns with classical fan-in/fan-out scaling conventions. This was proposed in [PR #492](https://github.com/karpathy/nanochat/pull/492) and showed improvements in modded-nanogpt.

**Result:** Quick d12 experiment: slightly worse **Not adopted.**

---

## 2026-02-03: Skip AdamW Every Other Step

Inspired by modded-nanogpt, tried stepping AdamW only on odd iterations while Muon steps every iteration. The idea is that small AdamW params (embeddings, scalars, gates) don't need updates as frequently as the large weight matrices, and skipping saves both compute and communication.

Added `skip_adamw` parameter to `MuonAdamW.step()` and `DistMuonAdamW.step()` plus a matching `zero_grad(skip_adamw=...)` to let AdamW gradients accumulate over 2 steps. Used `lr *= 2**-0.5` (sqrt scaling) to compensate for the 2x effective batch size on AdamW params.

**Result:** for nanochat d12, we see ~2% faster tok/s, but each step is slightly worse in loss. On net, when plotting against wall clock time, it's slightly worse. **Not adopted.**

---

## 2026-02-02: FP8 Training with torchao

Integrated FP8 training using `torchao.float8` to accelerate Linear layer matmuls on H100 GPUs.

### Background

FP8 (8-bit floating point) uses H100's FP8 tensor cores for ~2x theoretical matmul throughput. The tradeoff is quantization overhead: computing scales and casting tensors to/from FP8. Still, as an example torchtitan (Meta's distributed training framework) reports 25-28% speedups with FP8 for some of their experiments.

**Previous attempt (Jan 2026):** FP8 on just `lm_head` following modded-nanogpt with custom ops → 1% speedup, +2GB memory. Failed due to fragile torch.compile interaction. But this experiment was also done on ~d12 scale back then instead of the bigger model that gets GPT-2 capability of approx d24.

**This attempt:** Use torchao's `convert_to_float8_training()` on ALL Linear layers, increase model size to d24. The core snippet is:

```python
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
config = Float8LinearConfig.from_recipe_name("tensorwise")
convert_to_float8_training(model, config=config)
```

But in practice it's more involved (see base_train.py).

### Results

**Microbenchmark (d26 MLP, 65536x1664 @ 1664x6656):**

| Method | Forward | Fwd+Bwd | Speedup |
|--------|---------|---------|---------|
| BF16 + compile | 2.00ms | 4.79ms | 1.00x |
| FP8 rowwise + compile | 1.84ms | 4.55ms | 1.08x |
| FP8 tensorwise + compile | 1.45ms | 4.06ms | **1.38x** |
| FP8 rowwise (no compile) | 2.89ms | 21.86ms | 0.23x ❌ |

torch.compile is MANDATORY. Without it, FP8 is 4x slower due to unfused scaling ops.

**Full training (d26):**

| Config | tok/sec | vs baseline |
|--------|---------|-------------|
| BF16 baseline | 630K | 1.00x |
| FP8 rowwise | 564K | 0.90x ❌ |
| FP8 tensorwise | 740K | **1.17x** ✓ |

Memory usage also decreases quite a bit, by ~9GB (activations stored as FP8 instead of BF16).

Seeing 17% speedup is encouraging but we're still not done yet because each step is now in lower precision and less powerful individually, so to make up for the precision drop we have to train longer. Empirically, running some sweeps overnight on d24 scale, I saw that the actual speedup (when you match performance) is closer to 5%. It's possible that our LLMs at ~d24 scale are still too small to confidently enjoy the speedups that come from fp8 for bigger models.

### Key Learnings

For nanochat at approximate scale of interest (~GPT-2 capability, ~d24):

1. **Tensorwise >> Rowwise** - Rowwise computes per-row scales, overhead exceeds benefit. Tensorwise uses one scale per tensor.
2. **Filter small layers** - Layers with dims not divisible by 16 must be skipped (FP8 hardware requirement)
3. **Larger models benefit more** - d12 was still slower with FP8; d26+ shows gains. Therefore, in some depths there is a benefit to fp8 and in some there isn't. Keeping it configurable for now, passed in via kwargs and default off.
4. **The effective, capability-matched speedup is lower still** - because each step is of slightly lower precision/quality.

### Integration

Added `--fp8` flag to `base_train.py`, default recipe is "tensorwise", example of turning on:

```bash
torchrun --nproc_per_node=8 -m scripts.base_train --depth=24 --fp8
```

Uses tensorwise by default. Requires `torchao==0.15.0` (compatible with torch 2.9.1), which was added to dependencies.

**TLDR**: turning on fp8 for GPT-2 capability nanochat model gives approx +5% capability-matched speedup.

---

## 2026-01-29: Hyperball/MuonH Experiments (Negative Result)

Explored Hyperball optimization from [this post](https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd) (saved to `knowledge/muonh.md`). Constrains weights to sphere of radius R (initial norm): `W_{t+1} = R · Normalize(W_t - η·R · Normalize(u_t))`. Had to change a number of details in a branch, e.g. not use zero init for our projections (or the initial norm would be zero), keep track of the initial norm, adjust Muon -> MuonH for the update.

Experiments on d12:

| Experiment | Result |
|------------|--------|
| MuonH for matrix params | Worse than baseline |
| MuonH + LR sweep (2.5e-3 to 1e-2) | Still worse |
| Added learnable RMSNorm scales (paper says γ preserves expressivity) | Still worse |
| Various RMSNorm init tweaks, e.g. 0 at init to residual | Still worse |
| AdamH for lm_head (paper recommends this) | Broken - loss plateaus (see below) |
| AdamH + learnable output scales | Still worse |

Could not outperform the baseline implementation. The article doesn't go into too much detail on how AdamH is applied to `lm_head` exactly. The classifier layer has to be able to increase in magnitude to make more confident predictions over time. Tried a sensible version with added 0-D learnable scalar, and also with RMSNorms with per-channel learnable scalars both pre and post resnet blocks.

**Result:** This was not an out-of-the-box win for nanochat even with a mild attempt over a few hours at a bit of tuning and debugging. The idea itself is intuitively appealing. Might come back around later to try harder later.

---

## 2026-01-28: Reverted Bigram Hash Embeddings

Removed bigram embeddings (engram-lite) from the codebase. At larger scale (d25), the improvement was tiny and disappeared entirely when measured by wall clock time. It also bloated the VRAM used. The extra parameters and complexity aren't justified.

---

## 2026-01-27: Bigram Hash Embeddings (Engram-lite)

Explored N-gram memory modules inspired by the [DeepSeek Engram paper](https://arxiv.org/abs/2601.07372) and [modded-nanogpt PR #201](https://github.com/KellerJordan/modded-nanogpt/pull/201).

### Background

The Engram paper introduces "conditional memory" as a complement to MoE - using O(1) hash lookups to retrieve static N-gram patterns instead of reconstructing them through computation. Key insight: transformers waste early layers "simulating retrieval through computation" for patterns like named entities and formulaic phrases that could be simple table lookups.

### What We Tried

**1. Full Engram module with context-aware gating (paper design)**
```python
# Hash bigrams to retrieve embeddings, then gate with hidden state
e = embed(hash(prev_token, curr_token))
q = RMSNorm(h)           # hidden state as query
k = RMSNorm(W_k @ e)     # projected embedding as key
v = W_v @ e
α = sigmoid(q · k / √d)  # scalar gate per position
output = α * v
```
- Injected after block 1 (paper found early injection optimal)
- Slight improvement, but quite a bit of complexity added.

**2. Early-layer only injection**
- Only inject bigram signal in first 4 layers (where paper claims static pattern offloading helps most)
- **Result:** Actually hurt performance. The model seems to need uniform injection across all layers.

**3. Trigrams**
- Extended to hash both 2-grams and 3-grams, concatenating embeddings
- **Result:** No improvement over bigrams alone. Dilutes capacity from more frequent 2-gram patterns.

**4. Bigram-only with x0-style injection (modded-nanogpt engram-lite approach)**
- Simple hash: `(36313 * curr) XOR (27191 * prev) mod table_size`
- Zero-init embedding table, learned per-layer lambdas
- Add to residual at every layer: `x = resid_λ[i]*x + x0_λ[i]*x0 + bigram_λ[i]*x0_bigram`
- **Result:** This simple approach works and provides a consistent improvement.

TLDR The winning approach follows modded-nanogpt's "engram-lite", simply adding the following module and feeding its output into the residual branch (gated by a per-layer learnable \lambda) before every single block:

```python
class BigramEmbed(nn.Module):
    def __init__(self, vocab_size, embed_dim, table_multiplier=5):
        self.embed = nn.Embedding(vocab_size * table_multiplier, embed_dim)

    def forward(self, idx):
        h = (36313 * idx[:, 1:]) ^ (27191 * idx[:, :-1]) % (table_size - 1)
        return self.embed(h)
```

As for optimal hyperparameters:

- **Table size:** `vocab_size * 5` (~164K entries for 32K vocab). Swept a number of settings and 5 was optimal.
- **Injection:** Every layer via learned `bigram_lambdas` (init 0.1 was better than 0.0).
- **Normalization:** Also tried adding a `norm()` to the embeddings (mirroring the token embeddings), this was slightly worse.
- **Init:** Zero-init embedding, so starts as identity (tried small noisy init, it's worse)
- **Optimizer:** AdamW with same LR as token embeddings

### Key Learnings

1. **Gating didn't help at our scale.** The paper's context-aware gating mechanism (sigmoid dot-product gate) added parameters and complexity without improvement. modded-nanogpt found the same: "simple direct addition to the residual stream outperformed by a decent margin."

2. **Uniform injection beats early-only.** Despite the paper's finding that early layers benefit most, restricting injection to early layers hurt. The x0-style "add everywhere with learned lambda" pattern works better for our architecture/scale.

3. **Bigrams are sufficient.** Trigrams didn't help - the extra context doesn't pay for the diluted capacity.

4. **Scale matters.** The Engram paper's results are at 27B params with MoE. At our ~100M-1B scale, the simpler approach wins. The elaborate gating mechanism may become useful at larger scales where collision handling matters more.

### Parameters Added

For d12 model with `table_multiplier=5`:
- Bigram embedding: 32768 × 5 × 768 = ~126M params
- Per-layer lambdas: 12 scalars (negligible)

If you're keeping track, we now have *a lot* of parameters, a significant amount of them in embeddings (token embeddings, bigram embeddings, value embeddings). For example, for a d12 we now have:

```
Parameter counts:
wte                     : 25,165,824
bigram_embed            : 125,829,120
value_embeds            : 150,994,944
lm_head                 : 25,165,824
transformer_matrices    : 84,935,808
scalars                 : 36
total                   : 412,091,556
```

In other words, only about a quarter of parameters are now weight projections and the vast majority is embedding tables.

Still, on all axes (steps, wall clock time, flops), this somewhat parameter-bloated architecture beats the baseline and will now become the default.

After adding the engram-lite, I re-ran the scaling laws to determine the new optimal tokens:params ratio. I swept FLOPs in the range 1e18..1e19, exponentially strided in 4 settings (1e18, 2e18, 5e18, 1e19). I looked at a number of ways of determining the effective parameter count for the purposes of the scaling laws. The results looked like this:

```
Kaplan-style (all projections including lm_head and no embeddings)

Optimal configurations (from quadratic fits):
FLOPs        Eff Params      Tokens          Ratio      Val BPB
-----------------------------------------------------------------
1e+18        110,678,115     1,241,505,403   11.2       0.8972
2e+18        167,797,457     1,785,336,422   10.7       0.8616
5e+18        250,650,865     2,642,234,152   10.8       0.8293
1e+19        381,758,347     3,806,871,243   10.3       0.7999

N \propto C^0.54, D \propto C^0.49

Chinchilla-style (all parameters, period.)

Optimal configurations (from quadratic fits):
FLOPs        Eff Params      Tokens          Ratio      Val BPB
-----------------------------------------------------------------
1e+18        416,320,605     1,232,157,011   3.0        0.8974
2e+18        560,239,841     1,763,669,281   3.2        0.8616
5e+18        741,495,903     2,629,909,368   3.6        0.8291
1e+19        988,644,331     3,884,841,895   4.0        0.7999

N \propto C^0.37, D \propto C^0.50

Transformer-only-style (only the projections inside the transformer)

Optimal configurations (from quadratic fits):
FLOPs        Eff Params      Tokens          Ratio      Val BPB
-----------------------------------------------------------------
1e+18        80,259,665      1,315,639,547   17.2       0.8966
2e+18        131,488,566     1,864,134,141   14.5       0.8622
5e+18        220,985,474     2,595,328,843   12.1       0.8302
1e+19        401,213,504     3,328,704,512   8.5        0.7994

N \propto C^0.70, D \propto C^0.41
```

Clearly, the Kaplan-style ratios are most consistent and produce stable ~0.5 exponents for both params and tokens, meaning we can have a single fixed ratio of tokens:params for compute optimal models. This turns out to be about ~10.5, which now becomes the new default.

---

## 2026-01-19 to 2026-01-22: Optimizer Hyperparameter Sweep

Ran ~320 experiments across 6 rounds, scaling from d12→d16→d20 to find optimal optimizer hyperparameters. Added granular per-component control to `setup_optimizers()` — separate LRs and betas for embedding, unembedding, value_embeds, resid_lambdas, x0_lambdas, and Muon matrix params.

### What We Swept
- Learning rates for all 6 parameter groups
- Beta1/beta2 for all 5 AdamW groups
- Muon momentum (start/end), weight decay
- Hundreds of combinations (2-way, 3-way, 4-way, etc.)

### The Journey

**At d12**, found two independent improvement routes:
- **Route A:** emb_lr↑ (0.3→0.4), weight_decay↑ (0.1→0.15), matrix_lr↑ (0.02→0.025)
- **Route B:** x0_lr↓ (0.5→0.2), x0_beta1↑ (0.8→0.9+)

Both gave ~0.002 improvement, but combining them caused conflicts. Fine-tuning found wd=0.13, matrix_lr=0.027, emb_lr=0.38 helped slightly. Best d12 config: Route A + x0_beta1=0.95.

**At d16**, Route B became competitive with Route A. The routes still conflicted when combined.

**At d20** (target scale), everything changed:
- Fine-tuned values from d12 **actively hurt** performance
- Routes no longer conflicted
- Just `x0_beta1=0.96` alone captured nearly all the gains

### Final x0_beta1 Sweep at d20

| x0_beta1 | val/bpb | Δ vs baseline |
|----------|---------|---------------|
| **0.96** | **0.7971** | **-0.0007** |
| 0.94 | 0.7972 | -0.0006 |
| 0.90 | 0.7972 | -0.0006 |
| 0.97 | 0.7977 | -0.0001 |
| 0.98 | 0.8011 | +0.0033 💀 |

Flat plateau from 0.90-0.96, then sharp cliff at 0.97+.

### Key Learnings

1. **Hyperparameters are scale-dependent.** What works at d12 doesn't transfer to d20. The elaborate fine-tuning that won at d12 actively hurts at d20.

2. **Improvement magnitude shrinks with scale.** ~0.002 at d12 → ~0.0007 at d20. The baseline is already better-tuned for larger models.

3. **Sharp cliffs exist.** x0_beta1=0.98 is catastrophic while 0.96 is optimal.

4. **Don't over-tune on small proxies.** Validate at target scale before shipping.

### Final Recommendation

For production d20 runs, add one flag:
```
--x0-lambdas-beta1=0.96
```

Skip everything else discovered at smaller scales.

---

## 2026-01-18: More various experiments

- Tried Muon custom kernels for XXT and all the others. The improvement was there for targeted tests (~20%) but washed out completely to noise in an actual training run, especially because the Muon compute is split across all the workers. Abandoned due to complexity bloat.
- Fuse Q,K,V,O nn.Linear layers into a single QKVO Linear layer. ~Zero impact
- Tried the `sa_lambdas` that gate QKV and O. Slightly confused because of the use of rmsnorm, which erases the effect of any scalar multiplier. Helped a tiny bit (~1e-4 of loss), abandoned to control complexity.

---

## 2026-01-17: Various experiments

Modded-nanogpt uses [Value Embeddings](https://arxiv.org/abs/2410.17897) (VEs) in a funny U-shaped structure, 3 of them in total and with gates. I tried a large number of tweaks on this today:

- VEs at every layer, at alternating layers, U shaped, front and back. Alternating layers worked best, i.e. we end up with *a lot* more VEs than modded-nanogpt, at every other layer. It works better.
- Many parameters sharing ideas to reduce new parameter count, nothing here worked. All failed.
- Many ideas to reduce parameter count, the LLM hates all of them: low rank decompositions, projections. All failed.
- Gated yes or no and how much. Gate helps.

Long story short is that the models *love* Value Embeddings. It is a way to add a huge amount of capacity (parameters) to the model at almost zero cost of FLOPs, because these embeddings are simply added to the Values tensor. Any attempt to reduce the capacity of value embeddings (param sharing, low rank, projections) fail. The model wants many of them, and with all the capacity, and doing so wins across all x axes of steps, flops and wall clock. I re-ran the scaling laws and, because the models are now very parameter bloated, the optimal ratio has halved from 8 to 4! Way down lower than Chinchilla's 20 at this point.

Other experiments, looking at val/bpb as a function of all of steps, flops and wall clock time:

- Aspect ratio of 128 is worse than 64, I tried a sweep fixing FLOPs == 1e18 and 64 outperforms. The LLM prefers to be slightly thinner and longer.
- Head dim definitely prefers to be 128 instead of 64, i.e. fewer bigger heads
- Bunch of other random stuff like that.

Keeping all of this work on a private branch for now but hope to push shortly.

---

## 2026-01-17: Modded-nanogpt Ideas Sweep (Continued)

Continued testing ideas from modded-nanogpt.

| Idea | Result | Notes |
|------|--------|-------|
| Attention gates | No improvement | Per-head learnable gates on attention output. +1GB memory, decreased efficiency. |
| Batch size schedule | Abandoned | 8→16→24 with LR scaling. Made training script too bloated/complex, not worth cognitive overhead. |
| Value embeddings | Helps a lot | Experiments still ongoing, more on this later. |

---

## 2026-01-16: Flash Attention 3 Fallback to SDPA

Added automatic fallback from Flash Attention 3 to PyTorch's `scaled_dot_product_attention` (SDPA) for users without Hopper GPUs. This enables nanochat to run on older CUDA GPUs, CPU, and MPS (Apple Silicon).

### Implementation

Created `nanochat/flash_attention.py` - a unified interface that:
- Detects FA3 availability at import time (requires sm90+ / Hopper)
- Exports a `flash_attn` object matching FA3's API exactly (`flash_attn.flash_attn_func`, `flash_attn.flash_attn_with_kvcache`)
- Automatically routes to FA3 or SDPA based on hardware
- Handles tensor layout differences: FA3 uses (B, T, H, D), SDPA uses (B, H, T, D)
- Implements sliding window attention via explicit masks for SDPA
- Manages KV cache manually for SDPA (FA3 does it in-place)

### Changes to Existing Files

Changes to existing code were intentionally kept extremely minimal.

**gpt.py**: Only the import line changed and a comment

**engine.py**: Zero changes needed

**base_train.py**: Added status print and warnings:
- Prints whether FA3 or SDPA fallback is being used
- Warns about efficiency loss without FA3
- Warns about sliding window support if `--window-pattern` is not "L"

### Testing

Tests are split into two classes due to dtype/device constraints:

1. **TestFA3VsSDPA**: Comparison tests requiring Hopper GPU + bfloat16. Run both implementations on identical inputs and verify outputs match (max diff typically 0, at most ~0.004 for sliding window).

2. **TestSDPAOnly**: SDPA-only tests that run on any device with appropriate dtype. Verify forward pass, backward pass, and KV cache work correctly.

Added `_override_impl` mechanism for testing - can force 'fa3' or 'sdpa' to directly compare implementations.

### Notes

- SDPA fallback is significantly slower than FA3 especially in that it lacks the sliding window attention support
- Recommend `--window-pattern L` (full context) when using SDPA fallback

---

## 2026-01-16: Modded-nanogpt Ideas Sweep (Mostly Negative)

Tested several architectural ideas from modded-nanogpt to see if they transfer to nanochat. All of these did not help:

| Idea | Result | Notes |
|------|--------|-------|
| Half-truncated RoPE | No improvement | Only first half of head dims get RoPE (base 1024, linspace). Second half "stationary". |
| Asymmetric softcap | Slightly worse | `23 * sigmoid((x+5)/7.5)` vs our symmetric `15 * tanh(x/15)`. May only help with FP8. |
| Smear gate | Negligible | Blend each token with predecessor via learned gate. Tiny improvement not worth n_embd² params. |
| Backout | No improvement | Save activations at ~60% through network, subtract scaled version at end. |
| Skip connection | Slightly worse | Save at layer ~25%, add at layer ~50%. Also +2GB memory from storing activations. |

Value Embeddings do show promise. I need a more elaborate exploration of a few related ideas, which I leave for tomorrow.

---

## 2026-01-15: Olmo pretraining mix (Negative result)

I attempted to train on the Olmo 3 pretraining dataset [allenai/dolma3_mix-6T](https://huggingface.co/datasets/allenai/dolma3_mix-6T) instead of FineWeb-edu. I ran into a number of [errors and issues](https://huggingface.co/datasets/allenai/dolma3_mix-6T/discussions/2) trying to both download and process the dataset and then noticed some quality issues (e.g. some documents seem to be extremely short, like "5".). I managed to work around these with some sensible hacks (e.g. reject documents less than 100 characters in length) and tried to process the dataset exactly as FineWeb, re-trained the tokenizer and trained a d16 model. The CORE score decreased from 15.5 to 13.8, i.e. the result is quite a bit worse.

I am still looking to try the [DCLM dataset](https://arxiv.org/abs/2406.11794), which according to the paper should be better that FineWeb-edu. I do have some concerns that the same group both prepared the DCLM dataset *and* introduced the CORE score so I'm a bit hesitant in case there was some overfitting to CORE score adjacent data distribution.

Classifying as negative result and reverting back to FineWeb-edu for now.

---

## 2026-01-13: Varlen Attention (Negative Result)

Attempted to prevent attention from "leaking" across document boundaries using Flash Attention's `flash_attn_varlen_func`, similar to modded-nanogpt's approach.

### Background

With the BOS-aligned dataloader, multiple documents are packed into each row. Standard attention allows tokens to attend across document boundaries within a row. The hypothesis was that preventing this "leakage" via varlen attention might improve training.

### Approach: Compute cu_seqlens from inputs

- Find BOS positions: `(inputs.view(-1) == bos_token_id).nonzero()`
- Gotcha 1: Variable-length `cu_seqlens` caused torch.compile recompilation (25s/iter!) - fixed by padding to fixed size
- Gotcha 2: `nonzero()` inside compiled model hit recompile limit - fixed by moving computation outside compiled region

### Final Results (d16)

| Metric | Baseline | Varlen |
|--------|----------|--------|
| val_bpb | 0.85427 | 0.85407 |
| MFU | ~same | ~same |
| tok/sec | ~same | ~same |

Essentially identical. The 0.0002 bpb improvement is almost noise.

### Conclusion

Not worth the code complexity. The "leakage" across document boundaries within a row is not harmful - the model handles it fine. The BOS-aligned dataloader already provides the key benefit (every row starts with proper context). Not merging to master.

---

## 2026-01-13: BOS-Aligned Dataloader with Bin Packing

Redesigned the pretraining and midtraining dataloader to ensure every sequence starts with a BOS token, and explored bin-packing algorithms to minimize wasted tokens.

### Problem Statement

The original dataloader streams tokens into a flat buffer and reshapes into batches. This means some rows start mid-document (no BOS), which could confuse the model during training. We want every row to start with BOS and contain well-formed documents.

### Approach 1: Greedy-Crop BOS (Simple)

Each row is built independently:
- Start with a document (which has BOS prepended)
- Pack more documents until row is full
- If a document doesn't fit, **crop it** to fill remaining space (discard the rest)
- 100% utilization (no padding), but wastes cropped tokens

### Waste Analysis

Measured token waste empirically on real data (T=2048):
- **39.4% of tokens are cropped** (discarded when docs don't fit)
- **22.9% is the theoretical minimum** (tokens in docs longer than T+1 that can never fit)
- The extra ~16.5% comes from "unlucky" cropping when a long doc starts near the end of a row

### Bin Packing Algorithms Explored

| Algorithm | Util% | Crop% | Pad% | Notes |
|-----------|-------|-------|------|-------|
| Greedy-Crop (baseline) | 100% | 39.4% | 0% | Simple, no wasted compute |
| Greedy-Pad | 78% | 23.0% | 22% | Pads instead of crops - wastes compute |
| First-Fit Decreasing (FFD) | 99.7% | 23.0% | 0.3% | Near-optimal packing, minimal padding |
| **BestFit-Crop** | 100% | 34.6% | 0% | Smart cropping, no padding |

### BestFit-Crop Algorithm

A middle ground that maintains 100% utilization while reducing cropping:

1. Buffer N documents
2. For each row, greedily pick the **largest doc that fits entirely**
3. Repeat until nothing fits
4. When nothing fits, crop a doc to fill remaining space exactly

This avoids "unlucky" crops by searching the buffer for better-fitting documents.

**Results (T=2048):**
- Crop waste reduced from 39.4% → 34.6% (~12% relative improvement)
- Still achieves 100% utilization (no padding, every token trains)
- Slightly more rows than baseline (uses more documents per batch)

### Decision: Keep Two Implementations

1. Keep the original implementation which is very simple, efficient and has 100% token utilization in the batch (no padding with ignore tokens), but creates slightly more confusing token streams for the LLM because documents during training can start abruptly from the middle with no context. Note that this never happens at test time, where BOS is always present.

2. **`_bos_bestfit` (BestFit-Crop, new default)**: Slightly more complex but still keeps 100% token utilization in the batch (no padding), but at the cost of discarding documents when they don't fit. In practice, about 34% of tokens are discarded with this approach. This is ok because for most models we care about we have plenty of data without having to go to multiple epochs. One more subtle effect is that it does skew the data distribution a tiny bit because, reliably and necessarily, tokens at the tails of long documents will be discarded. However, this doesn't seem to impact actual downstream performance.

### Midtraining

The midtraining dataloader was also updated. Because conversations are on average a lot shorter than pretraining documents, only about 3.3% of tokens get cropped.

### NOTE: loss scale

Do note that switching to the BOS dataloader changes the validation loss and makes all previous experiments not comparable in absolute value of the loss, because we have a lot fewer "confusing" tokens in the train/val batches. All tokens can look back and find the BOS token and have the full context of that document to make predictions. Therefore, the loss appears lower but this is "fake" to some extent, and the expectation is that the vast majority of relative comparisons done so far would agree with those before and after this change.

---

## 2026-01-13: Number Token Split Pattern

Validated the `\p{N}{1,2}` pattern in `SPLIT_PATTERN` (tokenizer.py line 30), which I only guessed earlier and had a TODO for to validate. GPT-4 uses `\p{N}{1,3}` to group number sequences of up to 3 digits into tokens, but we suspected smaller vocab sizes benefit from grouping fewer digits per token.

**Results (d12, vocab=32K):**
| Pattern | val_bpb |
|---------|---------|
| `\p{N}{1,1}` | 0.969 |
| `\p{N}{1,2}` | **0.965** |
| `\p{N}{1,3}` | 0.972 |

**Conclusion:** `{1,2}` is optimal for vocab size 32K. Grouping 3 digits wastes tokens on rare 3-digit combinations; grouping 1 digit is too fine-grained and bloats token sequences. Keeping `{1,2}` as default.

---

## 2026-01-13: FP8 Training for lm_head

Attempted to use FP8 (8-bit floating point) for the lm_head layer to speed up the large vocab projection matmul. H100 GPUs have FP8 tensor cores that can theoretically provide ~2x speedup over BF16.

### Implementation Approaches Tried

**1. Dynamic Scaling (failed)**
- Compute `x.abs().max()` and `w.abs().max()` each forward to determine scales
- Problem: `.item()` calls cause graph breaks with torch.compile
- Tried `@torch._dynamo.allow_in_graph` pattern (like torchao.float8) - worked but no speedup
- Tried `torch.library.custom_op` with float scales - caused NaN gradients after first optimizer step
- Root cause: interaction between custom ops, dynamic scale computation, and torch.compile is fragile

**2. Static Scaling (partial success)**
- Pre-set scales at init time like modded-nanogpt: `x_scale=10/448, w_scale=0.1/448`
- `grad_scale` computed dynamically from batch size (safe since it's just `1/(B*T)/57344` due to the gradient expression of cross entropy). modded-nanogpt has a bug here probably because they set `grad_scale = 0.75/448`, but grads are in E5M2 so this should probably be `1/57344`, 1 being the amax of any individual element of cross entropy loss, and no normalization by B,T because they use sum reduction not mean reduction.
- Uses `torch.library.custom_op` with `@torch.compile` on inner kernels
- This works correctly - no NaNs, proper gradients

### Results (d12)

| Metric | BF16 Baseline | FP8 lm_head |
|--------|---------------|-------------|
| GPU Memory | 34 GB | 36 GB |
| tok/sec | baseline | ~1% faster |

### The Memory Mystery

FP8 *should* save memory since we store `x_f8` (1 byte) instead of `x` (2 bytes) for backward. But we see 2GB *increase*. Suspected causes:
- `torch.compile` on inner kernels creating extra buffers/specializations
- `torch._scaled_mm` internal workspace allocations
- Custom op registration machinery overhead

Tried saving original weight `w` (just a reference to parameter) instead of `w_f8` in backward, then re-quantizing on the spot during backward - didn't help. Still saw bump.

### Microbenchmark vs Reality

Raw microbenchmark showed promise:
- BF16 matmul: 16.95 ms
- FP8 matmul (static scales): 10.31 ms (1.64x faster)
- FP8 with dynamic scaling: 12.25 ms (1.38x faster)

But in full training, the ~1% tok/sec improvement doesn't justify the 2GB memory increase and the added code complexity and the need to tune scale factors for both x and w.

### Code Artifacts

See the branch `fp8_attempt_fail` for:

- `nanochat/fp8_static.py` - Static scaling implementation (working)
- `nanochat/fp8_dynamic.py` - Dynamic scaling implementation (torchao-style, working but slow)
- `gpt.py` imports `fp8_static.LinearFP8` and simply swaps it for `lm_head` in `gpt.py`.

### Open Questions

- Why does the custom op approach use more memory than vanilla BF16?
- Why is the bump in tok_per_sec so low? We should see ~1.6X speedup in both the forward pass and also (twice) in backward pass for the gradients. Granted, Amdahl's law is part of the solution because our vocab_size is only 32K so the final layer isn't a huge part of the profile but the expected speedup is still not fully realized.

**Conclusion:** Negative result for now. The implementation works correctly but provides marginal speedup with *increased* memory usage. I'm not understanding the torch.compile interaction here. The complexity of FP8 custom ops isn't justified for lm_head alone. TODO to study in more detail the way this is implemented in other libraries, e.g. torchao.

---

## 2026-01-12: Multi-Token Prediction (MTP)

Ported multi-token prediction from modded-nanogpt. Instead of predicting just the next token, predict the next n tokens at each position with weighted loss.

### Implementation

- Instead of calling the loss `n_predict` times, uses a fancy batched computation using `unfold` + `gather` + cross-entropy decomposition (`CE = logsumexp - logits[target]`)
- Schedule anneals from 3-token to 1-token prediction:
  - 0-33%: `[1.0, 0.5, 0.25→0]` (3rd token fades)
  - 33-67%: `[1.0, 0.5→0]` (2nd token fades)
  - 67-100%: `[1.0]` (standard next-token)
- Weights normalized to sum to 1

### Results (d12)

| Metric | Baseline | MTP |
|--------|----------|-----|
| GPU Memory | 34 GB | 47 GB |
| MFU | 41% | 40% |
| val/bpb (per step) | baseline | same/slightly worse |
| val/bpb (wall clock) | baseline | noticeably worse |

**Conclusion:** Negative result for nanochat. The extra memory and compute overhead from predicting multiple tokens doesn't pay off, in fact the results get worse. The auxiliary loss signal may help in other settings (larger models, different architectures?), but for our setup it's pure overhead at the moment.

---

## 2026-01-11: Sliding Window Attention

Added configurable sliding window attention, inspired by GPT-3's alternating short/long pattern.

**Pattern string configuration:**
- New `--window_pattern` CLI arg and `GPTConfig.window_pattern` field
- Pattern is tiled across layers (e.g., `SSSL` for 20 layers → `SSSLSSSLSSSLSSSLSSSL`)
- Final layer always forced to L (full context) regardless of pattern
- Short window = `sequence_len // 2`
- Long window = `sequence_len` (full context)
- All previous models so far have been simply `L` and checkpoint loading is modified accordingly to fill in this param for old models, see `_patch_missing_config_keys`

Quick experiments showed `SSSL` (every 4th layer is long) works well - provides a good balance between compute savings and model quality. This is now the default.

---

## 2026-01-11: Flash Attention 3 Integration

Replaced PyTorch's `scaled_dot_product_attention` (FA2) with Flash Attention 3 for training and inference.

### Changes Made

**1. FA3 via `kernels` package**
- Official FA3 is "beta" and requires building from source (painful)
- Using `kernels` package from HuggingFace Hub: `get_kernel('varunneal/flash-attention-3')`
- Loads pre-built wheels, works out of the box on H100

**2. Simplified attention code**
- FA3 uses `(B, T, H, D)` layout matching our projection output directly - no transpose needed
- Training: `flash_attn.flash_attn_func(q, k, v, causal=True)`
- Inference: `flash_attn.flash_attn_with_kvcache()` handles all cache cases in one call
- Removed 3 separate FA2 code paths (training, single-token, chunk inference)
- GQA handled automatically when n_kv_heads < n_heads

**3. Rewrote KVCache for FA3**
- Old format: `(num_layers, 2, B, H, T, D)` combined tensor
- New format: separate `k_cache` and `v_cache` of shape `(num_layers, B, T, H, D)`
- FA3 updates cache in-place during `flash_attn_with_kvcache`
- Position tracked via `cache_seqlens` tensor (int32, per batch element)
- Simpler API: `get_layer_cache()`, `advance()`, `reset()`, `prefill()`

### Results

- **~9% improvement in tok/sec** during training out of the box
- Benchmarks showed FA3 is 2x faster than FA2 at realistic training sizes (batch=32, seq=2048)
- FA3 supports sliding window via `window_size=(left, 0)`, which is huge and expected to give further improvements. This is ready to tune but keeping full context for now.

---

## 2026-01-11: Per-Layer Residual Scalars (x0 & resid lambdas)

Cherry-picked an idea from modded-nanogpt around learnable per-layer residual connections.

### Changes Made

**1. x0_lambdas (x0 residual connections)**
- Save initial normalized embedding as `x0` after `norm(wte(idx))`
- At each layer, blend x0 back in: `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
- Zero-initialized, so disabled at start; model learns which layers benefit from the shortcut
- Provides direct path from embedding to deep layers, helps preserve token information

**2. resid_lambdas (residual stream scaling)**
- Per-layer multiplicative scaling of the residual stream
- Initialized to 1.0 (neutral, standard transformer behavior)
- Allows model to learn to amplify/dampen residual at each layer

**3. DistAdamW small parameter handling**
- Added support for parameters with < 1024 elements (like the scalar lambdas)
- Small params use `all_reduce` instead of `reduce_scatter`/`all_gather`
- Fixes crash when param shape isn't divisible by world_size

### Key Finding: Different LR Sensitivity

The two scalar types need very different learning rates:
- **x0_lambdas (additive)**: Can use normal LR (~0.5). Adding a fraction of x0 is forgiving.
- **resid_lambdas (multiplicative)**: Needs ~100x smaller LR (~0.005). Multiplying the residual compounds through layers.

Implementation: `resid_params` gets `scalar_lr * 0.01`, `x0_params` gets full `scalar_lr`.

### Experiment Results

Swept `--scalar_lr` (controlling x0_lambdas) at multiple depths:

| Depth | Baseline (disabled) | Best scalar_lr | Best val_bpb | Δ bpb |
|-------|---------------------|----------------|--------------|-------|
| d8    | 1.0885              | 0.20           | 1.0782       | -0.0103 |
| d12   | 0.9770              | 0.60           | 0.9693       | -0.0077 |
| d16   | 0.9059              | 0.20           | 0.9002       | -0.0057 |
| d20   | 0.8565              | 0.10           | 0.8526       | -0.0039 |

**Observations:**
- Consistent improvement across all model sizes
- Optimal LR varies by depth; default of 0.5 is reasonable, but 0.6 is better for d12
- Adding resid_lambdas (with 0.01x LR) gives small additional improvement over x0 alone

### Meta Device Footgun

Important lesson: `__init__` runs in meta device context, so any tensor values set there are fake. Must initialize actual values in `init_weights()`. Added docstring warning to `__init__`.

### Summary

Added `--scalar_lr` (default 0.5) controlling learnable per-layer scalars. The formula `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` gives the model control over residual scaling and direct shortcuts to the initial embedding. Solid improvement with essentially no compute overhead.

---

## 2026-01-10: Muon Optimizer Upgrades & Cautious Weight Decay

Cherry-picked improvements from NorMuon (modded-nanogpt) into our simpler Muon implementation. Decided against using NorMuon directly due to hard-coded architecture assumptions (expects 32 params split 10 attn + 22 mlp), parameter labeling requirements, and complexity.

### Changes Made

**1. Polar Express Orthogonalization**
- Replaced Newton-Schulz iteration with "Polar Express Sign Method" from [arxiv.org/pdf/2505.16932](https://arxiv.org/pdf/2505.16932)
- Uses 5 different coefficient tuples (one per iteration) instead of fixed coefficients
- Both methods kept in code for easy comparison (`zeropower_via_polar_express` vs `zeropower_via_newtonschulz5`)
- **Result:** No dramatic/noticeable difference in training, but keeping the new Polar Express as default.

**2. NorMuon Variance Reduction**
- Added per-neuron/column adaptive learning rate from NorMuon ([arxiv.org/pdf/2510.05491](https://arxiv.org/pdf/2510.05491))
- Maintains `second_momentum_buffer` with shape `[rows, 1]` or `[1, cols]` (whichever is smaller)
- Normalizes updates based on running per-row/col variance estimate (beta2=0.95)
- Memory overhead: ~1/max(rows, cols) per param, negligible
- **Result:** Led to a very small improvement, kept and enabled by default.

**3. Cautious Weight Decay**
- Only decays weights where `update * weight >= 0` (same sign) from [arxiv.org/abs/2411.16085](https://arxiv.org/abs/2411.16085)
- Standard WD always pulls toward zero; cautious WD skips decay when gradient is pushing weight away from zero
- **Implementation note:** Had to inline the logic rather than use a separate `@torch.compile` function. Passing changing float values (like `weight_decay` during scheduling) as function arguments triggers recompilation. Reading from `group["weight_decay"]` inside the step avoids this.
- **Result:** Solid improvements, especially the cautious version was better than standard wd.
- Now defaults to ON for Muon via the `weight_decay` param. AdamW still has no weight decay and is hardcoded to 0 weight decay, might try to re-tune this later.

**4. Weight decay schedule**
- Added a linear schedule to weight decay that is default on from 1.0 to 0.0 (i.e. start with max weight decay in the beginning of training, then ramp to 0 by the end). Worked better than a static setting in experiments. (modded-nanogpt has the same schedule but it is implemented in a more confusing way by multiplying twice by the learning rate, which is already wired up to a decay schedule).

### Weight Decay Scaling Experiments

Swept weight decay values at d8, d12, d16, d20 to find optimal values and scaling law.

**Optimal Values Found:**
| Depth | Width (channels) | Optimal WD |
|-------|------------------|------------|
| d8    | 512              | ~0.40      |
| d12   | 768              | ~0.22      |
| d16   | 1024             | ~0.10      |
| d20   | 1280             | ~0.08      |

**Scaling Law:**
- Fit power law: `WD = k / channels^α` in log-log space
- Found α ≈ 1.97 (approximately 2), meaning WD ∝ 1/width²

**Practical Formula:**
```
WD_target = WD_reference × (d_reference / d_target)²
```
Example: If d12 optimal is 0.22, then d20 optimal ≈ 0.22 × (12/20)² ≈ 0.08

**Reference:** Moonlight paper uses fixed WD=0.1 for their 15B MoE model. Our experiments indicated a scaling law where the optimal WD changed with depth, so we go along with the empirical scaling law.

### Summary

Muon was changed to use Polar Express, added NorMuon variance reduction, and cautious weight decay with schedule that ramps linearly to zero. All of these changes follow modded-nanogpt repo, but all of them were also validated piece by piece to yield improvements in nanochat with the exception of the Polar Express change which was in the noise. This is default on and configurable with `--weight_decay`, using simply 0.2 and ∝ 1/width² scaling. The kwarg `--weight_decay` is therefore changing as of this change. It used to configure AdamW via standard weight decay and now it becomes exclusively used in Muon (AdamW is hardcoded to 0.0), and it is scaled based on depth.

---

## 2026-01-08: exp_grad_clip - Gradient Clipping

**Hypothesis:** Gradient clipping may be unnecessary overhead. Tested L2 norm clipping at various thresholds (0.25, 0.5, 1.0, 2.0) and elementwise clipping.

**Results:**
- No benefit at any scale tested (d12, d20)
- All variants within noise (~0.9827 val_bpb)
- Grad norm never exceeds 1.0 naturally, so clipping is always inactive
- Clipping adds ~2% time overhead from the all-reduce

**Bug Found:** Original implementation clipped local gradients before sync. Since this codebase doesn't use DDP (gradient sync is in the optimizers), each rank was clipping based on its own local norm. Fixed on the branch with proper distributed all-reduce.

**Observation:** modded-nanogpt does not appear to clip either right now.

**Summary:** Deleted all grad-clip code paths. The code naturally produces well-behaved gradients. This improves a bit of MFU because we don't have to calculate and sync grad norms.
