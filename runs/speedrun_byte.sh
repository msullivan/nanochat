#!/bin/bash
set -e

# Byte-level training run. Defaults to d12 with a text-matched data budget so
# the byte model sees roughly the same number of *characters* a BPE d12 baseline
# would at --target-param-data-ratio=8. See the "why ratio=120" comment below.
#
# Example launches:
#   bash runs/speedrun_byte.sh
#   DEPTH=24 bash runs/speedrun_byte.sh
#   WANDB_RUN=d12-byte screen -L -Logfile runs/speedrun_byte.log -S byte bash runs/speedrun_byte.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra "${UV_EXTRA:-gpu}"
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tags: keep byte runs separate from BPE runs in checkpoints and report dirs.
DEPTH="${DEPTH:-12}"
MODEL_TAG="${MODEL_TAG:-d${DEPTH}-byte}"
export NANOCHAT_REPORT_TAG="$MODEL_TAG"

# Knobs (override via env):
#   TARGET_DATA_RATIO: tokens-per-param. The in-code scaling_params is
#     transformer_matrices + lm_head, so byte d12 scaling params ~= 85M vs BPE
#     d12 ~= 110M. A BPE d12 run at speedrun ratio=8 sees ~4.2B chars of text;
#     matching that with bytes would want ratio ~= 50. We use 32 here to stay
#     in a more Chinchilla-ish training regime and keep runs shorter.
#   MAX_SEQ_LEN: byte seq=8192 gives ~8 KB of text per sequence, roughly
#     matching the context BPE at seq=2048 sees (~4.8 chars/token). Costs
#     ~27% more FLOPs/tok vs seq=2048.
TARGET_DATA_RATIO="${TARGET_DATA_RATIO:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# SFT is skipped by default: small byte models (d12) often aren't coherent enough
# after pretraining for SFT to land as anything but chat-format mimicry. Set
# RUN_SFT=1 once base_eval shows BPB/CORE indicating real learning.
RUN_SFT="${RUN_SFT:-0}"

echo "=== byte run: depth=$DEPTH tag=$MODEL_TAG ratio=$TARGET_DATA_RATIO seq_len=$MAX_SEQ_LEN ==="

# -----------------------------------------------------------------------------
# Fresh report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Dataset (no tokenizer training in byte mode)
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# -----------------------------------------------------------------------------
# Base training (byte tokenizer, no FP8: the lm_head matmul is 128x smaller and
# fp8 adds overhead without much win at this vocab; flip back on via NO_FP8=0).
if [ -n "$NO_FP8" ] && [ "$NO_FP8" = "0" ]; then FP8_ARG="--fp8"; else FP8_ARG=""; fi

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --byte-tokenizer \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --target-param-data-ratio="$TARGET_DATA_RATIO" \
    ${WINDOW_PATTERN:+--window-pattern=$WINDOW_PATTERN} \
    --model-tag="$MODEL_TAG" \
    $FP8_ARG \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    -g "$MODEL_TAG"

# -----------------------------------------------------------------------------
# SFT (optional -- set RUN_SFT=1 to enable)
if [ "$RUN_SFT" = "1" ]; then
    if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
        curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
            https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    fi

    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --model-tag="$MODEL_TAG" \
        --output-tag="$MODEL_TAG" \
        --run="$WANDB_RUN"

    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
        -i sft \
        -g "$MODEL_TAG"
else
    echo "Skipping SFT (set RUN_SFT=1 to enable)."
fi

# -----------------------------------------------------------------------------
python -m nanochat.report generate
