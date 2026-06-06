#!/bin/bash
set -ex

# Anneal / consolidation phase for the byte chat model.
#
# Loads the finished SFT checkpoint's WEIGHTS (via chat_sft --anneal, which skips
# loop-state restore), re-warms the LR to a modest peak, and anneals it back to 0
# on a CUTE-HEAVY mix (~50% CUTE, matching the cute_mix recipe). The point: CUTE
# washes out in normal SFT because it's sparse during the LR->0 tail; here CUTE is
# dense during a fresh low-LR tail, so it's the last thing the model consolidates
# and nothing after erodes it. General data stays in the mix (the other ~50%) so
# chat/identity/MMLU survive -- do NOT go pure CUTE (that's the cute_pt lesson).
#
# Run AFTER runs/chat_sft_byte.sh has produced chatsft_checkpoints/$SFT_TAG.
#
# Example:
#   WANDB_RUN=byte-anneal-v1 bash runs/chat_anneal_byte.sh
#   INIT_LR_FRAC=0.2 ANNEAL_STEPS=400 WANDB_RUN=byte-anneal-lr02 bash runs/chat_anneal_byte.sh
#
# ENV KNOBS
#   CHAT_BASE_TAG  converted 265-vocab base (for arch + LR inheritance);
#                  default d24-byte-l-ext-chatbase
#   SFT_TAG        SFT checkpoint to load weights from; default d24-byte-l-ext-chat
#   OUTPUT_TAG     anneal output dir; default ${SFT_TAG}-anneal
#   CUTE_FRACTION  CUTE share of the anneal mixture (default 0.5)
#   ANNEAL_STEPS   length of the anneal phase (default 300)
#   INIT_LR_FRAC   re-warm peak as a fraction of base LR (default 0.3; the SFT ran
#                  at 0.8 -- keep this LOWER. THE knob to sweep; too high forgets
#                  general, too low won't consolidate CUTE).
#   WARMUP_RATIO   fraction of steps to ramp up to peak (default 0.05)
#   WARMDOWN_RATIO fraction of steps to anneal down (default 0.95 -> ends at 0)
#   SAVE_EVERY / EVAL_EVERY / CHATCORE_EVERY / CUTE_EVERY  cadence (default 30 each)
#   CUTE_MAX_PROBLEMS (50) / CHATCORE_MAX_CAT (200) / CHATCORE_MAX_SAMPLE (24)
#                  smaller-than-default eval sizes so frequent evals stay cheap
#   WANDB_RUN      wandb run name ("dummy" disables; default dummy)
#   NPROC_PER_NODE / DEVICE_BATCH_SIZE  (default 1 / 4; bf16 SFT/anneal OOMs at 8)
#   IDENTITY_FILE  identity .jsonl to stage (default committed byte identity)

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

CHAT_BASE_TAG="${CHAT_BASE_TAG:-d24-byte-l-ext-chatbase}"
SFT_TAG="${SFT_TAG:-d24-byte-l-ext-chat}"
OUTPUT_TAG="${OUTPUT_TAG:-${SFT_TAG}-anneal}"
CUTE_FRACTION="${CUTE_FRACTION:-0.5}"
ANNEAL_STEPS="${ANNEAL_STEPS:-300}"
INIT_LR_FRAC="${INIT_LR_FRAC:-0.3}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.95}"
SAVE_EVERY="${SAVE_EVERY:-30}"
# Frequent but SMALL evals (the anneal is short; we want fine-grained curves
# without each eval dominating). EVERY=30 -> ~10 points over 300 steps.
EVAL_EVERY="${EVAL_EVERY:-30}"
CHATCORE_EVERY="${CHATCORE_EVERY:-30}"
CUTE_EVERY="${CUTE_EVERY:-30}"
CUTE_MAX_PROBLEMS="${CUTE_MAX_PROBLEMS:-50}"      # per CUTE subtask (was 100)
CHATCORE_MAX_CAT="${CHATCORE_MAX_CAT:-200}"       # cap MMLU/ARC categorical (was -1 = full 14k)
CHATCORE_MAX_SAMPLE="${CHATCORE_MAX_SAMPLE:-24}"  # per generative ChatCORE task
WANDB_RUN="${WANDB_RUN:-dummy}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
export NANOCHAT_REPORT_TAG="$OUTPUT_TAG"

source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

CKPT_ROOT="$NANOCHAT_BASE_DIR/chatsft_checkpoints"
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$CHAT_BASE_TAG" ]; then
    echo "ERROR: converted base $CHAT_BASE_TAG not found (run chat_sft_byte.sh first)" >&2; exit 1
fi
if [ -z "$(ls "$CKPT_ROOT/$SFT_TAG"/model_*.pt 2>/dev/null)" ]; then
    echo "ERROR: no SFT checkpoint in $CKPT_ROOT/$SFT_TAG to anneal from" >&2; exit 1
fi

# Identity data (the anneal mix still contains identity/qamis/pep tasks).
IDENTITY_DST="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
IDENTITY_FILE="${IDENTITY_FILE:-$(pwd)/nanochat_byte_identity__google_gemini-3-flash-preview.jsonl}"
if [ -f "$IDENTITY_FILE" ]; then
    cp "$IDENTITY_FILE" "$IDENTITY_DST"
elif [ ! -f "$IDENTITY_DST" ]; then
    echo "ERROR: identity file not found: $IDENTITY_FILE" >&2; exit 1
fi

echo "=== anneal: from SFT '$SFT_TAG' -> '$OUTPUT_TAG' | cute_frac=$CUTE_FRACTION steps=$ANNEAL_STEPS init_lr_frac=$INIT_LR_FRAC"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$CHAT_BASE_TAG" \
    --resume-from-tag="$SFT_TAG" \
    --resume-from-step=latest \
    --anneal \
    --output-tag="$OUTPUT_TAG" \
    --cute-fraction="$CUTE_FRACTION" \
    --num-iterations="$ANNEAL_STEPS" \
    --init-lr-frac="$INIT_LR_FRAC" \
    --warmup-ratio="$WARMUP_RATIO" \
    --warmdown-ratio="$WARMDOWN_RATIO" \
    --final-lr-frac=0.0 \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --save-every="$SAVE_EVERY" \
    --eval-every="$EVAL_EVERY" \
    --chatcore-every="$CHATCORE_EVERY" \
    --cute-every="$CUTE_EVERY" \
    --cute-max-problems="$CUTE_MAX_PROBLEMS" \
    --chatcore-max-cat="$CHATCORE_MAX_CAT" \
    --chatcore-max-sample="$CHATCORE_MAX_SAMPLE" \
    --run="$WANDB_RUN"

# Final evals on the annealed model.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft -g "$OUTPUT_TAG"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.cute_eval -- \
    -i sft -g "$OUTPUT_TAG" --mode chat --no-prefill --prompt-style bare --subtasks char

echo "=== anneal done. checkpoint: chatsft_checkpoints/$OUTPUT_TAG"
