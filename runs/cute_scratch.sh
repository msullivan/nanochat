#!/bin/bash
set -ex

# From-scratch byte-LM training on a CUTE-format corpus only.
#
# Parallel to runs/cute_pt.sh but trains from random init instead of
# resuming from a base checkpoint. Uses the standard pretraining LR
# schedule (warmup -> stable -> warmdown), NOT cute_pt's "barely nudge"
# midtraining schedule -- the question this experiment answers is "what
# can a byte model learn from CUTE data alone, no pretraining at all?"
# so it should be trained the way you'd actually train one from scratch.
#
# Checkpoints land under scratch_checkpoints/$MODEL_TAG/ so they don't
# collide with base_checkpoints/ or cute_checkpoints/. wandb runs go to
# the nanochat-scratch project for the same reason.
#
# PREREQUISITES
# 1. Generate the CUTE data (or reuse an existing cute_pt_data_nodemos_<size> dir):
#    PYTHONPATH=. .venv/bin/python dev/gen_cute_pt_data.py \
#        --out-dir "$NANOCHAT_BASE_DIR/cute_pt_data_nodemos_100000" \
#        --num-words 100000 --no-demos --seed 0
#
# 2. Run: MODEL_TAG=d24-byte-scratch-100000w NUM_ITERATIONS=160 \
#         NANOCHAT_DATA_DIR=cute_pt_data_nodemos_100000 \
#         bash runs/cute_scratch.sh
#
# ENV KNOBS
#   MODEL_TAG       required -- destination tag under scratch_checkpoints/
#   NUM_ITERATIONS  total training steps (default: 200). Includes warmup.
#                   Cells with <200 steps don't even complete LR warmup
#                   (default warmup_steps=40); pick this based on dataset
#                   size so you can actually train.
#   DEPTH           model depth (default: 24, matches our existing byte models)
#   BATCH           device-batch-size (default: 8, single-GPU friendly)
#   WANDB_PROJECT   override wandb project (default: nanochat-cute-scratch)
#   FP8             "1" enables --fp8 (default off; only meaningful on
#                   H100+, and we're on Blackwell here)

cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:?NANOCHAT_DATA_DIR required (e.g. cute_pt_data_nodemos_100000)}"

MODEL_TAG="${MODEL_TAG:?MODEL_TAG is required (e.g. d24-byte-scratch-100000w)}"
NUM_ITERATIONS="${NUM_ITERATIONS:-200}"
DEPTH="${DEPTH:-24}"
BATCH="${BATCH:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-nanochat-cute-scratch}"
FP8="${FP8:-0}"
EVAL_EVERY="${EVAL_EVERY:-250}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"

export NANOCHAT_REPORT_TAG="$MODEL_TAG"

echo "=== cute_scratch: tag=$MODEL_TAG depth=$DEPTH iters=$NUM_ITERATIONS data=$NANOCHAT_DATA_DIR ==="

FP8_ARG=""
[ "$FP8" = "1" ] && FP8_ARG="--fp8"

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth="$DEPTH" \
    --byte-tokenizer \
    --device-batch-size="$BATCH" \
    --num-iterations="$NUM_ITERATIONS" \
    --save-every=-1 \
    --model-tag="$MODEL_TAG" \
    --checkpoint-subdir=scratch_checkpoints \
    --wandb-project="$WANDB_PROJECT" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens=1048576 \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --sample-every="$SAMPLE_EVERY" \
    $FP8_ARG \
    --run="$MODEL_TAG"
