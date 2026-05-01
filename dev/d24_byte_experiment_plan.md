# d24-byte-l experiment plan

Written 2026-04-30. Working state-of-play for the d24-byte-l byte-LM
experiment, the in-progress extension run (`d24-byte-l-ext`), the
post-extension SFT, and deferred design decisions.

## The core question

Does a from-scratch byte model at speedrun scale (d24, ~685M params,
ratio=16) get a meaningful character-level advantage over the equivalent
BPE model (d24-stock) once both are SFT-tuned with matched data?

The hypothesis: **byte representation should help on character-manipulation
tasks** (spell, count letters, etc.) because byte-level addressing is native;
BPE has to do this through opaque token embeddings.

The detuned-SFT design (1k examples per format task instead of the default
50k-200k) is calibrated to **teach the format without saturating
performance**, so the underlying representation-driven advantage has
headroom to show up. Past 200k-row SFT both models hit ~95% on these tasks
and the byte advantage gets washed out.

## Current state (as of 2026-04-30)

### Base run: `d24-byte-l`
- Single GPU (NPROC=1), depth=24, byte_tokenizer, max_seq_len=8192,
  device_batch_size=8, target_param_data_ratio=16, window_pattern=L,
  warmdown_ratio=0.1, FP8 enabled.
- num_iterations auto-computed = ~20,800 from ratio=16 × 730M scaling
  params. Warmdown_start = ~18,720.
- Currently at step ~3,704 with LR breakpoints
  `3700:1.0,4000:0.8,5400:0.8,6000:0.7,6300:0.7,9000:0.55` already
  applied, then later breakpoints
  `9300:0.55,11500:0.55,21500:0.2` planned via the extension.
- BPB descent has been slowing on the 0.55 plateau; from step 9300+ the
  iterative-cooldown plan kicks in.
- Wandb run id: `v9hmpzdm` in project `nanochat`.

### The extension run: `d24-byte-l-ext`
- Output tag separate from `d24-byte-l` so original checkpoints are
  immutable.
- Seeded by manually copying `model_009337.pt + optim_009337_rank0.pt +
  meta_009337.json` into `base_checkpoints/d24-byte-l-ext/`.
- Schedule (`runs/extend_d24_byte_l.sh`):
  - Hold 0.55 from step 9337 (effective resume) to 11,500.
  - Linear decay 0.55 → 0.2 from step 11,500 to 21,500.
  - 10% warmdown 0.2 → 0.05 over the last 2,390 steps.
  - num_iterations = 23,900.
- New wandb run named `d24-byte-l-ext` (no WANDB_RUN_ID — fresh plot).
- Same eval cadence as base: val/bpb every 100, CORE every 100, sample
  every 100, eval_tokens=4M.

### Post-extension SFT: `d24-byte-l-sft`
- Run via `runs/sft_d24_byte_l.sh` (standalone) or as the chained final
  block of `runs/resume_d24_byte_l.sh`.
- DEVICE_BATCH_SIZE=4 (halved from base — bf16 SFT activation memory),
  max_seq_len=8192 (preserved — SmolTalk byte-length distribution has
  median 3,690 bytes, 67% > 2048).
- BF16 only (FP8 NOT enabled — no validated baseline for FP8 SFT in the
  codebase, deliberately keeping the variable controlled).
- Eval cadence: eval_every=100, eval_tokens=4M, chatcore_every=100.
- Output tag `d24-byte-l-sft` so SFT checkpoints don't clobber base.

### What is in the SFT mix
- `SmolTalk(split="train")` — 460K rows, dominant.
- `CustomJSON(identity_conversations.jsonl)` × 2 epochs.
- `CustomJSON(qamis_*)` × 2 epochs (3 generators, ~1280 rows total).
- `CustomJSON(pep827_conversations.jsonl)` × 2 epochs.
- `MMLU(auxiliary_train)` × 3 epochs.
- `GSM8K(train)` × 4 epochs.
- `SimpleSpelling(size=1000)`, `SpellingBee(size=1000)`,
  `Addition(size=1000)`, `Multiplication(size=1000)` — the **detuned**
  format-teaching at 1k each.

## How to launch (assuming the d24-byte-l base extension is what's next)

On the runpod with `~/.cache/nanochat/base_checkpoints/d24-byte-l/` populated:

```bash
# 1. Seed the new tag dir with the source step 9337 checkpoint
D=$NANOCHAT_BASE_DIR/base_checkpoints
mkdir -p $D/d24-byte-l-ext
cp $D/d24-byte-l/{model,meta}_009337.* \
   $D/d24-byte-l/optim_009337_rank0.pt \
   $D/d24-byte-l-ext/

# 2. Launch the extension under screen
screen -L -Logfile ~/d24-byte-l-ext.log -S extend bash runs/extend_d24_byte_l.sh

# 3. After extension completes, run SFT against d24-byte-l-ext base.
#    Either modify sft_d24_byte_l.sh to use MODEL_TAG=d24-byte-l-ext,
#    or write a sft_d24_byte_l_ext.sh sibling.
```

## Evaluation plan

Post-SFT, the relevant signal is **ChatCORE-with-SpellingBee**:
- SpellingBee tests "did 1k rows of format teaching land?"
- If post-SFT SpellingBee climbs noticeably from base, the format teaching
  worked.
- If it stays low, 1k wasn't enough.
- If it shoots to ~100%, you over-saturated and lost the format-vs-skill
  decoupling the experiment was designed to expose.

### Comparison target
The d24-stock (BPE) baseline run already exists and was SFT-trained with
the same task mix. Run the same ChatCORE eval on both d24-stock-sft and
d24-byte-l-sft. The headline question: **does d24-byte-l-sft beat
d24-stock-sft on SpellingBee, GSM8K, and other character-touching tasks
at matched 1k-row format teaching?**

### Not running
- **CUTE eval directly** — pre-SFT it returned 0% because d24-scale
  models can't ICL the 4-shot CUTE format. The SimpleSpelling format is
  different from CUTE's format, so post-SFT CUTE wouldn't be informative
  either without first adding CUTE-style SFT data to the mix (see
  Deferred Decisions below).

## Architectural changes landed this session

These are committed to master and don't need to be revisited:

- **Piecewise-linear LR breakpoints** (`a8aa6b6`):
  `--lr-breakpoints "step:lrm,step:lrm,..."` for WSD-style iterative
  cooldowns inside the stable phase. Validated against breakpoints
  outside [warmup, warmdown_start]. Forwarded through `runs/resume.py`.
  Schedule generalizes the existing warmdown to start from
  stable_end_lrm rather than hardcoded 1.0.
- **Atomic checkpoint writes** (`9625a1f`, `6a79108`, `832b385`):
  - `.tmp` + `os.replace` per file means partial writes never produce a
    corrupt named checkpoint.
  - `model_<step>.pt` written LAST, after meta + all optim shards (with
    a DDP barrier between). Existence of model_*.pt proves the
    checkpoint set is complete; `latest`-resolution code only sees
    complete checkpoints.
  - SIGINT blocked during save (kernel queues it, delivered after save
    completes). SIGTERM/SIGQUIT/SIGKILL stay unmasked as escape hatches.
- **chat_sft save/resume/extend support** (`0e2cfe1`, `c3d90ee`):
  Mirror of base_train. New flags `--save-every`, `--resume-from-step`,
  `--resume-from-tag`. SIGUSR1 saves+continues, SIGINT saves+exits,
  second SIGINT force-quits. Loop state + dataloader cursor saved in
  meta. Wandb run resumption via WANDB_RUN_ID/WANDB_RESUME=must env
  vars. Per-step train metrics now go to wandb (was every 10).
- **finetune.sh env knobs**: `EVAL_EVERY`, `EVAL_TOKENS`,
  `CHATCORE_EVERY`, `MAX_SEQ_LEN` (`19780cf`, `1144a35`).
- **Step-metric alignment** for chat_sft (`f95805b`) and chat_rl
  (`17135d4`): wandb logs use the canonical `step` field as x-axis,
  so train/val/eval curves overlay correctly.
- **Muon WD logging** (`fda967e`): `train/wd` shown alongside
  `train/lrm` in wandb.

## Deferred decisions

### Byte tokenizer redesign (256 → 264)
**Decided to defer.** The current scheme uses `0x00=BOS, 0x01=ESCAPE`
with chat specials encoded as `[ESCAPE, 0x02..0x09]`. A cleaner design
would give chat specials dedicated IDs 256-263 (vocab_size=264, padded
to 320). The conversion would pad three kinds of tensors (wte, lm_head,
value_embeds) plus optimizer state from row-256 to row-320, with
random small init for the new rows.

Reasons not to do it now:
- Mid-experiment tokenizer change adds a confounding variable.
- Escape mechanism only fires ~10× per conversation (chat specials);
  practical cost is tiny.
- Tensors are currently shaped (256, ...), not (320, ...) — an actual
  conversion script is needed (~80 LOC), not a one-line meta edit. The
  ergonomics aren't free.

If pad_vocab_size_to had been larger so the tensors were already
(320, ...), this would be a one-line meta change. Worth designing for
in any future byte-LM work.

### CUTE-style synthetic SFT data
**Skipping for this run.** Adding `ins_char`/`del_char`/`sub_char`/
`swap_char` tasks would test the byte-vs-BPE manipulation-task gap
directly (the catastrophic-BPE-ceiling subtasks per Bolmo paper).
Format collision risk with SimpleSpelling rules out the CUTE `spell`
subtask itself. Use a 50k+ word source minus the published 1k CUTE test
words to avoid contamination, à la Bolmo.

Worth doing in a follow-up run if the d24-byte-l-sft vs d24-stock-sft
comparison shows a representation-driven advantage on SpellingBee
specifically — that's the hint that scaling up to the manipulation
tasks would amplify the win.

### FP8 in SFT
**Deferred.** Karpathy's commit history shows FP8 base training was
validated to leaderboard quality, but no SFT-with-FP8 baseline exists
in the codebase. Adding it for d24-byte-l-sft would conflate the byte
representation, the detuned mix, and FP8 SFT as variables. Run BF16
SFT now; if you want FP8 SFT later, a 500-step head-to-head against
BF16 SFT would tell you if quality regresses.

## Useful references

- `dev/bolmo_adaptation_report.md` — full Bolmo analysis. Key
  load-bearing finding: ~75% of Bolmo's CUTE win is data, not byte
  architecture (the +5.7 byteification-specific delta over Olmo-CT).
- `dev/cute_benchmark_notes.md` — CUTE subtask breakdown, BPE catastrophic
  ceilings on manipulation tasks (`swap_char` 10%, `ins_char` 9%).
- `dev/byte_lm_literature_review.md` — broader byte-LM landscape.

## Things to remember for next session

- Both d24-byte-l and d24-byte-l-ext exist as separate output dirs;
  don't conflate them.
- Wandb run id `v9hmpzdm` is the original d24-byte-l. The extension is
  a new run named `d24-byte-l-ext`. SFT will be a new run in
  `nanochat-sft` project.
- The user is on a single-GPU runpod with ~95GB GPU memory (likely
  RTX 6000 Pro or similar; nanochat's `get_peak_flops` table has a
  `pro 6000 → 504e12` entry that the user added).
- `tea_debug.log` shows up on the runpod (some preinstalled tool
  creates it); harmless, gitignored.
- `runs/move_checkpoints.py` is the simple "move old checkpoints to
  archive, keep last N" version after the user reverted the more
  complex copy-all variant.
