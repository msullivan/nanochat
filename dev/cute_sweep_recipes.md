# CUTE midtraining sweep — recipes & invocation

End-to-end pipeline for sweeping CUTE-format midtraining across
(dataset size × base model × training recipe). Compares byte vs BPE
tokenizer at varying CUTE training data levels.

## TL;DR

```bash
# Baseline recipe (no demos, no loss masking, low LR) — fills results.csv
SKIP_DONE=1 bash dev/sweep_cute_pt.sh

# Both interventions: SFT-like LR schedule + loss masking on answer tokens
# Fills results_sft-mask.csv, uses cute_checkpoints/d24-cute-sft-mask-30000w/ etc.
SFT_STYLE=1 MASK_BEFORE="Answer: " bash dev/sweep_cute_pt.sh

# Mask only
MASK_BEFORE="Answer: " bash dev/sweep_cute_pt.sh   # → results_mask.csv

# SFT-style LR only
SFT_STYLE=1 bash dev/sweep_cute_pt.sh              # → results_sft.csv
```

Each recipe lands in its own `results_<recipe>.csv` and its own
`cute_checkpoints/<model>-cute-<recipe>-<size>w/` dir, so 4 sweeps don't
collide and `SKIP_DONE=1` works correctly per recipe.

## What each recipe means

| RECIPE   | SFT_STYLE | MASK_BEFORE | FT_LRM | WARMDOWN | WD   | Loss mask           |
|----------|-----------|-------------|--------|----------|------|---------------------|
| nodemos  | 0         | (empty)     | 0.05   | 10%      | 0.28 | none (loss on all)  |
| mask     | 0         | "Answer: "  | 0.05   | 10%      | 0.28 | answer-only         |
| sft      | 1         | (empty)     | 0.8    | 50%      | 0    | none                |
| sft-mask | 1         | "Answer: "  | 0.8    | 50%      | 0    | answer-only         |

- **SFT_STYLE=1** flips FT_LRM 0.05 → 0.8, WARMDOWN_FRAC 0.1 → 0.5, and
  weight-decay 0.28 → 0 (mirrors `chat_sft.py`'s `init_lr_frac=0.8`,
  `warmdown_ratio=0.5`, `weight_decay=0.0`).
- Override WD independently with `WEIGHT_DECAY=<value>` if you want to mix
  SFT-style LR with non-zero WD (or vice versa).
- Note: with the nodemos default `FT_LRM=0.05` and `final_lr_frac=0.05`,
  the "warmdown" is effectively a no-op (0.05 → 0.05). Real LR decay only
  happens under `SFT_STYLE=1`.
- **MASK_BEFORE="Answer: "** tells the dataloader to set targets to -1
  for all positions up to (and including) the tokenized form of
  `"Answer: "` within each sub-document. Mirrors SFT's
  assistant-only loss without requiring chat format.

## Default sweep matrix

- **SIZES**: `100000 70000 50000 30000 10000 3000` (largest first)
- **MODELS**: `d24-byte-l-early d24 d24-byte-l d24-byte-l-ext`
  (early and BPE first — they're the byte-vs-BPE anchors)
- 4 sizes × 4 models = 24 cells per recipe

## Common env-var overrides

```bash
# Just the anchor cells at the biggest sizes
MODELS="d24-byte-l-early d24" SIZES="100000 30000" \
    SFT_STYLE=1 MASK_BEFORE="Answer: " bash dev/sweep_cute_pt.sh

# Match training compute across models (otherwise BPE gets ~4× fewer steps)
BUDGET_MODE=compute SFT_STYLE=1 MASK_BEFORE="Answer: " \
    bash dev/sweep_cute_pt.sh

# Force regenerate data shards (e.g. after a gen_cute_pt_data fix)
FORCE_REGEN=1 bash dev/sweep_cute_pt.sh
```

## Where things live

- **Data**: `$NANOCHAT_BASE_DIR/cute_pt_data_nodemos_<SIZE>/{shard_00000,shard_00001}.parquet`
- **Checkpoints**: `$NANOCHAT_BASE_DIR/cute_checkpoints/<MODEL>-cute-<RECIPE>-<SIZE>w/`
- **Results CSV**: `$NANOCHAT_BASE_DIR/cute_sweep/results[_<RECIPE>].csv`
  (legacy `results.csv` is the `nodemos` recipe's data)
- **Wandb runs**: project `nanochat-cute`, run names = DST_TAGs

## One-cell direct invocation (no sweep)

If you want to validate a single (model, size, recipe) cell without running
the sweep driver:

```bash
MODEL=d24
SIZE=100000
RECIPE=sft-mask
DST_TAG="${MODEL}-cute-${RECIPE}-${SIZE}w"
SRC_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/${MODEL}"
DST_DIR="$NANOCHAT_BASE_DIR/cute_checkpoints/${DST_TAG}"

# 1. Seed dst dir from base
STEP=$(ls "$SRC_DIR"/model_*.pt | sed 's/.*model_0*\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
STEP6=$(printf "%06d" "$STEP")
mkdir -p "$DST_DIR"
cp "$SRC_DIR/model_${STEP6}.pt"        "$DST_DIR/"
cp "$SRC_DIR/meta_${STEP6}.json"       "$DST_DIR/"
cp "$SRC_DIR/optim_${STEP6}_rank0.pt"  "$DST_DIR/"

# 2. Train
NANOCHAT_DATA_DIR="cute_pt_data_nodemos_${SIZE}" \
MODEL_TAG="$DST_TAG" \
FT_STEPS=40 \
SFT_STYLE=1 \
MASK_BEFORE="Answer: " \
EVAL_EVERY=-1 CORE_METRIC_EVERY=-1 SAMPLE_EVERY=-1 \
bash runs/cute_pt.sh

# 3. Eval
PYTHONPATH=. .venv/bin/python -m scripts.cute_eval \
    --source cute --model-tag "$DST_TAG" \
    --mode completion --prompt-style zero \
    --subtasks char --max-problems 100
```

## Plotting

`dev/plot_cute_sweep.py` reads a results CSV and produces two PNGs
(per-subtask grid + mean curve). Defaults read from
`/tmp/cute_sweep_results.csv` and write to `/tmp/`.

```bash
# Plot the nodemos (legacy) recipe
.venv/bin/python dev/plot_cute_sweep.py \
    --input ~/.cache/nanochat/cute_sweep/results.csv \
    --outdir /tmp --tag cute_nodemos

# Plot a specific recipe
.venv/bin/python dev/plot_cute_sweep.py \
    --input ~/.cache/nanochat/cute_sweep/results_sft-mask.csv \
    --outdir /tmp --tag cute_sft-mask
```

For multi-recipe comparison overlays, load multiple `results_*.csv`
files and label series by recipe (not yet wired in).

## Pulling results from the remote training machine

```bash
scp genai:~/.cache/nanochat/cute_sweep/results.csv /tmp/cute_sweep_results.csv
scp genai:~/.cache/nanochat/cute_sweep/results_sft-mask.csv /tmp/  # etc
```

## Wandb cleanup

```bash
# Delete all runs in the nanochat-cute project (use carefully)
.venv/bin/python -c "
import wandb
api = wandb.Api()
for r in api.runs('nanochat-cute'):
    print(f'deleting {r.id} {r.name}')
    r.delete()
"
```
