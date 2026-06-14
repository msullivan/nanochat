#!/bin/bash
set -ex

# GRPO RL on GSM8K for the byte chat model, starting from the annealed SFT
# checkpoint. Single-GPU (genai Blackwell).
#
# Key difference from the stock (BPE) recipe: --max-new-tokens is in BYTES, so
# the BPE default of 256 is far too short for GSM8K chain-of-thought + calculator
# tool calls. We use 1024 (~the byte-equivalent of the BPE 256-token default).
#
# Usage:
#   WANDB_RUN=byte-rl-v1 bash runs/chat_rl_byte.sh
#   screen -L -Logfile ~/byte-rl.log -S byte-rl bash runs/chat_rl_byte.sh

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

MODEL_TAG="${MODEL_TAG:-d24-byte-l-ext-chat-anneal-lr0.3}"
WANDB_RUN="${WANDB_RUN:-dummy}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
EXAMPLES_PER_STEP="${EXAMPLES_PER_STEP:-16}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
EVAL_EVERY="${EVAL_EVERY:-30}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-200}"
SAVE_EVERY="${SAVE_EVERY:-30}"

source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# RL saves to chatrl_checkpoints/$MODEL_TAG (output tag = model tag).
python -m scripts.chat_rl \
    --model-tag="$MODEL_TAG" \
    --max-new-tokens="$MAX_NEW_TOKENS" \
    --num-samples="$NUM_SAMPLES" \
    --examples-per-step="$EXAMPLES_PER_STEP" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --eval-every="$EVAL_EVERY" \
    --eval-examples="$EVAL_EXAMPLES" \
    --save-every="$SAVE_EVERY" \
    --run="$WANDB_RUN"

echo "=== done. RL checkpoint: chatrl_checkpoints/$MODEL_TAG"
