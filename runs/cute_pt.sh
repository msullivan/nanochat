#!/bin/bash
set -ex

# Generic CUTE-format midtraining launcher. Works for any base model
# (d24-byte-l, d24-byte-l-early, d24-byte-l-ext, d24-stock, ...). Trains
# on a synthetic CUTE-format corpus so the model learns the 4-shot demo
# + Question + Answer format directly.
#
# Checkpoints land under cute_checkpoints/$MODEL_TAG/ (NOT base_checkpoints)
# so cute_pt outputs never clobber the source pretraining run. wandb runs
# go to the nanochat-cute project for the same reason.
#
# PREREQUISITES
# 1. Generate the synthetic corpus into $NANOCHAT_BASE_DIR/cute_pt_data/:
#    PYTHONPATH=. .venv/bin/python dev/gen_cute_pt_data.py \
#        --out-dir "$NANOCHAT_BASE_DIR/cute_pt_data" \
#        --num-words 1000
#    (defaults size for smoke testing; bump --num-words and FT_STEPS for
#    serious runs)
#
# 2. Seed the new model tag dir with a finished base checkpoint, e.g.:
#    SRC=base_checkpoints/d24-byte-l-ext
#    DST=cute_checkpoints/d24-byte-l-ext-cute
#    STEP=$(ls "$NANOCHAT_BASE_DIR/$SRC"/model_*.pt | sed 's/.*model_0*\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
#    mkdir -p "$NANOCHAT_BASE_DIR/$DST"
#    cp "$NANOCHAT_BASE_DIR/$SRC/model_$(printf %06d $STEP).pt"     "$NANOCHAT_BASE_DIR/$DST/"
#    cp "$NANOCHAT_BASE_DIR/$SRC/meta_$(printf %06d $STEP).json"    "$NANOCHAT_BASE_DIR/$DST/"
#    cp "$NANOCHAT_BASE_DIR/$SRC/optim_$(printf %06d $STEP)_rank0.pt" "$NANOCHAT_BASE_DIR/$DST/"
#
# 3. Run: MODEL_TAG=d24-byte-l-ext-cute bash runs/cute_pt.sh
#
# ENV KNOBS
#   MODEL_TAG       required -- destination tag (also the resume source).
#                   The script discovers the seed step + architecture from
#                   the latest checkpoint already in cute_checkpoints/$MODEL_TAG.
#   FT_STEPS        finetune step budget (default: 50, ~12 epochs over the
#                   default-size 1k-word corpus at total_batch_size 1MB)
#   FT_LRM          LR multiplier during the flat finetune phase (default: 0.05)
#   WANDB_PROJECT   override wandb project (default: nanochat-cute)
#
# LR SCHEDULE
# Holds at FT_LRM from the resumed step for 90% of FT_STEPS, then warmdown
# over the last 10% to final_lr_frac. lr-breakpoints anchor the stable
# phase flat at FT_LRM (no LR jump on resume regardless of where the seed
# checkpoint left off).

cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-cute_pt_data}"

MODEL_TAG="${MODEL_TAG:?MODEL_TAG is required (e.g. d24-byte-l-ext-cute)}"
FT_STEPS="${FT_STEPS:-50}"
# SFT_STYLE=1 swaps in an SFT-like recipe: 80% of the pretrain peak LR held
# flat then warmed down over the last 50% of FT_STEPS, AND weight decay=0
# (mirrors chat_sft.py, which explicitly sets weight_decay=0.0 on the theory
# that pretraining warmdown already brought WD to zero). The default cute_pt
# recipe instead uses a "barely nudge" 5% LR with 10% warmdown and inherits
# base_train's WD default.
SFT_STYLE="${SFT_STYLE:-0}"
if [ "$SFT_STYLE" = "1" ]; then
    FT_LRM="${FT_LRM:-0.8}"
    WARMDOWN_FRAC="${WARMDOWN_FRAC:-0.5}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
else
    FT_LRM="${FT_LRM:-0.05}"
    WARMDOWN_FRAC="${WARMDOWN_FRAC:-0.1}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.28}"  # base_train default
fi
# MASK_BEFORE: when set, loss-mask all training tokens before (and including)
# this substring in each sub-document. For cute_pt --no-demos docs, the
# correct marker is `Answer: "` (including the opening quote). NOT `Answer: `
# alone -- BPE tokenizes a trailing bare space differently in isolation vs in
# context (the in-context space merges with the following quote into ` "`),
# so the standalone tokenization wouldn't match in any real doc and every
# sub-doc would get fully masked. With the quote, the marker tokenizes to the
# same boundary token sequence in both contexts and lines up with eval's
# prompt-end. Byte tokenizer is unaffected (no merges; bytes always match).
# The dataloader prints a one-shot warning on the first batch if marker
# not-found rate is high, which is your tripwire if this rule changes.
MASK_BEFORE="${MASK_BEFORE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-nanochat-cute}"
# In-training eval cadences. Sweep driver disables these (sets to -1) since
# the only eval we care about per cute_pt run is the post-finetune CUTE
# benchmark, and the in-training CORE eval costs minutes per call.
EVAL_EVERY="${EVAL_EVERY:-25}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-50}"
SAMPLE_EVERY="${SAMPLE_EVERY:-50}"
CKPT_SUBDIR=cute_checkpoints
export NANOCHAT_REPORT_TAG="$MODEL_TAG"

CKPT_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/$CKPT_SUBDIR/$MODEL_TAG"
LATEST_META=$(ls "$CKPT_DIR"/meta_*.json 2>/dev/null | sort | tail -1)
if [ -z "$LATEST_META" ]; then
    echo "ERROR: no meta_*.json in $CKPT_DIR -- seed the dst dir first (see header)" >&2
    exit 1
fi

# We only need the seed step here for the LR-schedule math; base_train
# itself recovers the model architecture (depth, max_seq_len, window_pattern,
# byte_tokenizer, ...) from the seed meta on resume.
SEED_STEP=$(.venv/bin/python -c "import json; print(json.load(open('$LATEST_META'))['step'])")

NUM_ITERATIONS=$(( SEED_STEP + FT_STEPS ))
if [ "$FT_STEPS" -le 2 ]; then
    # Too few steps for a meaningful "stable + warmdown" split (the two LR
    # breakpoints would collide or invert). Just anchor LR flat at FT_LRM.
    WARMDOWN_RATIO=0
    LR_BREAKPOINTS="${SEED_STEP}:${FT_LRM}"
else
    WARMDOWN_TAIL_STEPS=$(.venv/bin/python -c "print(max(1, int(${WARMDOWN_FRAC} * ${FT_STEPS})))")
    WARMDOWN_RATIO=$(.venv/bin/python -c "print(${WARMDOWN_TAIL_STEPS} / ${NUM_ITERATIONS})")
    WARMDOWN_START=$(( NUM_ITERATIONS - WARMDOWN_TAIL_STEPS ))
    LR_BREAKPOINTS="${SEED_STEP}:${FT_LRM},$((WARMDOWN_START - 1)):${FT_LRM}"
fi

echo "=== cute_pt: tag=$MODEL_TAG seed_step=$SEED_STEP +ft_steps=$FT_STEPS sft_style=$SFT_STYLE"
echo "=== num_iterations=$NUM_ITERATIONS lr_mult=$FT_LRM warmdown_ratio=$WARMDOWN_RATIO breakpoints=$LR_BREAKPOINTS"
echo "=== weight_decay=$WEIGHT_DECAY mask_before=${MASK_BEFORE:-<none>}"
echo "=== checkpoint_subdir=$CKPT_SUBDIR wandb_project=$WANDB_PROJECT"

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --device-batch-size=8 \
    --num-iterations="$NUM_ITERATIONS" \
    --warmdown-ratio="$WARMDOWN_RATIO" \
    --lr-breakpoints="$LR_BREAKPOINTS" \
    --weight-decay="$WEIGHT_DECAY" \
    --save-every=50 \
    --resume-from-step=latest \
    --model-tag="$MODEL_TAG" \
    --checkpoint-subdir="$CKPT_SUBDIR" \
    --wandb-project="$WANDB_PROJECT" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens=1048576 \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --sample-every="$SAMPLE_EVERY" \
    --mask-before="$MASK_BEFORE" \
    --fp8 \
    --run="$MODEL_TAG"
