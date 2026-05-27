#!/bin/bash
set -ex

# Mixed CUTE + ClimbMix continued-pretraining launcher.
#
# Parallel to runs/cute_pt.sh, but instead of 100% CUTE the dataloader
# interleaves CUTE docs with the general pretraining mix at a configurable
# ratio. Goal: learn the CUTE Q/A format without destroying general
# capability (which 300k cute_pt cells empirically do; CORE went to ~0).
#
# Primary stream = ClimbMix (the default base_data_climbmix dir).
# Mix stream     = the CUTE synthetic corpus, passed via MIX_DATA_DIR.
# MIX_FRACTION   = probability per doc of drawing from the CUTE stream
#                  (0.10 / 0.20 / 0.50 are the sweep points).
#
# mask_before (loss-mask up through 'Answer: "') applies ONLY to mix-stream
# docs -- ClimbMix docs are never masked, so the general pretraining signal
# is preserved.
#
# Checkpoints land under cute_mix_checkpoints/$MODEL_TAG/ (not cute_checkpoints,
# so this experiment can't clobber cute_pt outputs). wandb runs go to
# nanochat-cute-mix.
#
# PREREQUISITES
# 1. Generate the synthetic CUTE corpus (or reuse one from the cute_pt sweep):
#    PYTHONPATH=. .venv/bin/python dev/gen_cute_pt_data.py \
#        --out-dir "$NANOCHAT_BASE_DIR/cute_pt_data_nodemos_30000" \
#        --num-words 30000 --no-demos --seed 0
#
# 2. Seed the dst dir with a finished base checkpoint, e.g.:
#    SRC=base_checkpoints/d24-byte-l-ext
#    DST=cute_mix_checkpoints/d24-byte-l-ext-mix10-30000w
#    STEP=$(ls "$NANOCHAT_BASE_DIR/$SRC"/model_*.pt | sed 's/.*model_0*\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
#    mkdir -p "$NANOCHAT_BASE_DIR/$DST"
#    cp "$NANOCHAT_BASE_DIR/$SRC/model_$(printf %06d $STEP).pt"      "$NANOCHAT_BASE_DIR/$DST/"
#    cp "$NANOCHAT_BASE_DIR/$SRC/meta_$(printf %06d $STEP).json"     "$NANOCHAT_BASE_DIR/$DST/"
#    cp "$NANOCHAT_BASE_DIR/$SRC/optim_$(printf %06d $STEP)_rank0.pt" "$NANOCHAT_BASE_DIR/$DST/"
#
# 3. Run:
#    MODEL_TAG=d24-byte-l-ext-mix10-30000w \
#    MIX_DATA_DIR=cute_pt_data_nodemos_30000 \
#    MIX_FRACTION=0.10 \
#    FT_STEPS=480 \
#    bash runs/cute_mix.sh
#
# ENV KNOBS
#   MODEL_TAG       required -- destination tag (also the resume source).
#   MIX_DATA_DIR    required -- CUTE data dir (relative to NANOCHAT_BASE_DIR
#                   or absolute).
#   MIX_FRACTION    required -- in (0, 1]. Probability per doc of CUTE stream.
#   FT_STEPS        required -- finetune step budget. The sweep driver
#                   computes this from (CUTE-tokens target / MIX_FRACTION).
#   FT_LRM          LR multiplier during flat finetune phase (default 0.05;
#                   matches cute_pt's "barely nudge" default).
#   WARMDOWN_FRAC   fraction of FT_STEPS used for warmdown (default 0.1).
#   WEIGHT_DECAY    (default 0.28, base_train default).
#   MASK_BEFORE     loss-mask marker; defaults to empty (no mask). Set to
#                   'Answer: "' to apply the cute_pt answer-only mask
#                   (applied only to mix-stream docs in this loader).
#   WANDB_PROJECT   (default nanochat-cute-mix).
#
# LR SCHEDULE: same shape as cute_pt -- flat at FT_LRM from the resumed step,
# then linear warmdown over the last WARMDOWN_FRAC of FT_STEPS.

cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
# IMPORTANT: do NOT override NANOCHAT_DATA_DIR here. The primary stream needs
# to point at ClimbMix (the default base_data_climbmix). The mix stream is
# threaded via --mix-data-dir.

MODEL_TAG="${MODEL_TAG:?MODEL_TAG is required (e.g. d24-byte-l-ext-mix10-30000w)}"
MIX_DATA_DIR="${MIX_DATA_DIR:?MIX_DATA_DIR is required (e.g. cute_pt_data_nodemos_30000)}"
MIX_FRACTION="${MIX_FRACTION:?MIX_FRACTION is required (e.g. 0.10)}"
FT_STEPS="${FT_STEPS:?FT_STEPS is required}"
FT_LRM="${FT_LRM:-0.05}"
WARMDOWN_FRAC="${WARMDOWN_FRAC:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.28}"
MASK_BEFORE="${MASK_BEFORE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-nanochat-cute-mix}"
EVAL_EVERY="${EVAL_EVERY:-25}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-50}"
SAMPLE_EVERY="${SAMPLE_EVERY:-50}"
CKPT_SUBDIR=cute_mix_checkpoints
export NANOCHAT_REPORT_TAG="$MODEL_TAG"

CKPT_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/$CKPT_SUBDIR/$MODEL_TAG"
LATEST_META=$(ls "$CKPT_DIR"/meta_*.json 2>/dev/null | sort | tail -1)
if [ -z "$LATEST_META" ]; then
    echo "ERROR: no meta_*.json in $CKPT_DIR -- seed the dst dir first (see header)" >&2
    exit 1
fi

SEED_STEP=$(.venv/bin/python -c "import json; print(json.load(open('$LATEST_META'))['step'])")
NUM_ITERATIONS=$(( SEED_STEP + FT_STEPS ))
if [ "$FT_STEPS" -le 2 ]; then
    WARMDOWN_RATIO=0
    LR_BREAKPOINTS="${SEED_STEP}:${FT_LRM}"
else
    WARMDOWN_TAIL_STEPS=$(.venv/bin/python -c "print(max(1, int(${WARMDOWN_FRAC} * ${FT_STEPS})))")
    WARMDOWN_RATIO=$(.venv/bin/python -c "print(${WARMDOWN_TAIL_STEPS} / ${NUM_ITERATIONS})")
    WARMDOWN_START=$(( NUM_ITERATIONS - WARMDOWN_TAIL_STEPS ))
    LR_BREAKPOINTS="${SEED_STEP}:${FT_LRM},$((WARMDOWN_START - 1)):${FT_LRM}"
fi

echo "=== cute_mix: tag=$MODEL_TAG seed_step=$SEED_STEP +ft_steps=$FT_STEPS"
echo "=== mix_data_dir=$MIX_DATA_DIR mix_fraction=$MIX_FRACTION"
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
    --mix-data-dir="$MIX_DATA_DIR" \
    --mix-fraction="$MIX_FRACTION" \
    --fp8 \
    --run="$MODEL_TAG"
