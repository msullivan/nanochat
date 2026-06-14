#!/bin/bash
set -ex

# Matched BPE post-training: SFT + anneal, using the SAME recipe as the byte
# model (chat_sft_byte.sh + chat_anneal_byte.sh) for a controlled comparison.
#
# The byte model's post-training was:
#   1. vocab conversion 256->265  (N/A for BPE -- already has chat specials)
#   2. SFT: 1 epoch, current mixture (SmolTalk, CuteChat, MMLU×3, GSM8K×4,
#      SpellingBee, Addition, Multiplication, identity, qamis, pep827)
#   3. Anneal: 300 steps (byte) / 100 steps (BPE) to match text volume
#      (1 BPE token ≈ 4-5 bytes), 50% CUTE, init_lr_frac=0.3, warmdown=0.95
#
# This script runs steps 2+3 for the BPE d24 base, then final evals on both.
# The original d24 SFT used an OLDER recipe (no CuteChat, no arithmetic),
# so we redo it from base to match.
#
# Usage:
#   WANDB_RUN=bpe-matched-v1 bash runs/chat_sft_bpe_matched.sh
#   screen -L -Logfile ~/bpe-matched.log -S bpe-matched bash runs/chat_sft_bpe_matched.sh

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

BASE_TAG="${BASE_TAG:-d24}"
SFT_TAG="${SFT_TAG:-d24-sft-matched}"
ANNEAL_TAG="${ANNEAL_TAG:-${SFT_TAG}-anneal}"
WANDB_RUN="${WANDB_RUN:-dummy}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
SAVE_EVERY="${SAVE_EVERY:-100}"
export NANOCHAT_REPORT_TAG="$SFT_TAG"

source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# --- Identity data ---
IDENTITY_DST="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
IDENTITY_FILE="${IDENTITY_FILE:-$(pwd)/nanochat_byte_identity__google_gemini-3-flash-preview.jsonl}"
if [ ! -f "$IDENTITY_FILE" ]; then
    echo "ERROR: identity file not found: $IDENTITY_FILE" >&2
    exit 1
fi
cp "$IDENTITY_FILE" "$IDENTITY_DST"

# --- Stage 1: SFT (matching byte recipe) ---
echo "=== SFT: $BASE_TAG -> $SFT_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$BASE_TAG" \
    --output-tag="$SFT_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --load-optimizer=1 \
    --save-every="$SAVE_EVERY" \
    --run="${WANDB_RUN}-sft"

# --- Stage 2: Anneal (matching byte recipe) ---
echo "=== Anneal: $SFT_TAG -> $ANNEAL_TAG"
export NANOCHAT_REPORT_TAG="$ANNEAL_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$BASE_TAG" \
    --resume-from-tag="$SFT_TAG" \
    --resume-from-step=latest \
    --anneal \
    --output-tag="$ANNEAL_TAG" \
    --cute-fraction=0.5 \
    --num-iterations=100 \
    --init-lr-frac=0.3 \
    --warmup-ratio=0.05 \
    --warmdown-ratio=0.95 \
    --final-lr-frac=0.0 \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --save-every=10 \
    --eval-every=10 \
    --chatcore-every=10 \
    --cute-every=10 \
    --cute-max-problems=50 \
    --chatcore-max-cat=200 \
    --chatcore-max-sample=24 \
    --run="${WANDB_RUN}-anneal"

# --- Stage 3: Final evals on both SFT and anneal ---
echo "=== Final evals: SFT ($SFT_TAG)"
export NANOCHAT_REPORT_TAG="$SFT_TAG"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft -g "$SFT_TAG"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.cute_eval -- \
    -i sft -g "$SFT_TAG" --mode chat --no-prefill --prompt-style bare --subtasks char

echo "=== Final evals: anneal ($ANNEAL_TAG)"
export NANOCHAT_REPORT_TAG="$ANNEAL_TAG"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft -g "$ANNEAL_TAG"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.cute_eval -- \
    -i sft -g "$ANNEAL_TAG" --mode chat --no-prefill --prompt-style bare --subtasks char

echo "=== done. SFT: chatsft_checkpoints/$SFT_TAG | Anneal: chatsft_checkpoints/$ANNEAL_TAG"
