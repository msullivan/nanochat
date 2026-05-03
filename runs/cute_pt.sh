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
FT_LRM="${FT_LRM:-0.05}"
WANDB_PROJECT="${WANDB_PROJECT:-nanochat-cute}"
CKPT_SUBDIR=cute_checkpoints
export NANOCHAT_REPORT_TAG="$MODEL_TAG"

CKPT_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/$CKPT_SUBDIR/$MODEL_TAG"
LATEST_META=$(ls "$CKPT_DIR"/meta_*.json 2>/dev/null | sort | tail -1)
if [ -z "$LATEST_META" ]; then
    echo "ERROR: no meta_*.json in $CKPT_DIR -- seed the dst dir first (see header)" >&2
    exit 1
fi

# Discover seed step + architecture from the seed checkpoint's meta.
# base_train builds the model from CLI args (not meta), so DEPTH /
# MAX_SEQ_LEN / WINDOW_PATTERN / BYTE_TOKENIZER must match the saved model.
read SEED_STEP DEPTH MAX_SEQ_LEN WINDOW_PATTERN BYTE_TOK <<< $(.venv/bin/python -c "
import json
m = json.load(open('$LATEST_META'))
u = m.get('user_config', {})
mc = m.get('model_config', {})
step = m.get('step', '?')
depth = u.get('depth', mc.get('n_layer', 24))
msl = u.get('max_seq_len', mc.get('sequence_len', 8192))
wp = u.get('window_pattern', mc.get('window_pattern', 'L'))
bt = 'true' if (u.get('byte_tokenizer') or m.get('byte_tokenizer') or mc.get('vocab_size', 32768) <= 320) else 'false'
print(step, depth, msl, wp, bt)
")
echo "discovered: seed_step=$SEED_STEP depth=$DEPTH max_seq_len=$MAX_SEQ_LEN window_pattern=$WINDOW_PATTERN byte=$BYTE_TOK"

# Schedule math: warmdown over last 10% of FT_STEPS, flat at FT_LRM before that
NUM_ITERATIONS=$(( SEED_STEP + FT_STEPS ))
WARMDOWN_TAIL_STEPS=$(( FT_STEPS / 10 ))
[ "$WARMDOWN_TAIL_STEPS" -lt 1 ] && WARMDOWN_TAIL_STEPS=1
WARMDOWN_RATIO=$(.venv/bin/python -c "print(${WARMDOWN_TAIL_STEPS} / ${NUM_ITERATIONS})")
WARMDOWN_START=$(( NUM_ITERATIONS - WARMDOWN_TAIL_STEPS ))
LR_BREAKPOINTS="${SEED_STEP}:${FT_LRM},$((WARMDOWN_START - 1)):${FT_LRM}"

echo "=== cute_pt: tag=$MODEL_TAG seed_step=$SEED_STEP +ft_steps=$FT_STEPS"
echo "=== num_iterations=$NUM_ITERATIONS warmdown_ratio=$WARMDOWN_RATIO breakpoints=$LR_BREAKPOINTS"
echo "=== checkpoint_subdir=$CKPT_SUBDIR wandb_project=$WANDB_PROJECT"

BYTE_FLAG=""
[ "$BYTE_TOK" = "true" ] && BYTE_FLAG="--byte-tokenizer"

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth="$DEPTH" \
    $BYTE_FLAG \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size=8 \
    --num-iterations="$NUM_ITERATIONS" \
    --window-pattern="$WINDOW_PATTERN" \
    --warmdown-ratio="$WARMDOWN_RATIO" \
    --lr-breakpoints="$LR_BREAKPOINTS" \
    --save-every=50 \
    --resume-from-step=latest \
    --model-tag="$MODEL_TAG" \
    --checkpoint-subdir="$CKPT_SUBDIR" \
    --wandb-project="$WANDB_PROJECT" \
    --eval-every=25 \
    --eval-tokens=1048576 \
    --core-metric-every=50 \
    --sample-every=50 \
    --fp8 \
    --run="$MODEL_TAG"

torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- \
    --device-batch-size=8 \
    --source=cute \
    --model-tag="$MODEL_TAG"
