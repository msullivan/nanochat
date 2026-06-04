#!/bin/bash
set -ex

# Chat SFT for the byte model, starting FROM THE BASE checkpoint (not a
# cute-finetuned one): the SFT mixture already teaches CUTE in chat format via
# CuteChat, so one clean SFT stage covers general chat + char-level skills.
#
# Pipeline:
#   1. Convert the 256-vocab base byte checkpoint -> 265-vocab (chat specials
#      don't exist in the old vocab) into base_checkpoints/$CHAT_BASE_TAG.
#   2. SFT from that converted base -> chatsft_checkpoints/$OUTPUT_TAG, with the
#      live CUTE eval (cute/* in wandb) and ChatCORE.
#   3. Final chat-mode CUTE eval (zero-shot, no prefill) + ChatCORE eval.
#
# Tuned for genai (single RTX 6000). Override NPROC_PER_NODE for multi-GPU, but
# see the LOAD_OPTIMIZER note below -- the converted optimizer is a rank0-only
# shard, so multi-GPU resume of the warm-started optimizer is unsupported.
#
# Example:
#   WANDB_RUN=byte-chat-v1 bash runs/chat_sft_byte.sh
#   BASE_TAG=d24-byte-l-ext IDENTITY_FILE=$NANOCHAT_BASE_DIR/nanochat_byte_identity__gemma.jsonl \
#     WANDB_RUN=byte-chat-v1 bash runs/chat_sft_byte.sh
#   screen -L -Logfile runs/chat_sft_byte.log -S bytesft bash runs/chat_sft_byte.sh
#
# ENV KNOBS
#   BASE_TAG          256-vocab base byte checkpoint to start from (default d24-byte-l-ext)
#   CHAT_BASE_TAG     converted 265-vocab base tag (default ${BASE_TAG}-chatbase)
#   OUTPUT_TAG        SFT output tag (default ${BASE_TAG}-chat)
#   WANDB_RUN         wandb run name; "dummy" disables logging (default dummy)
#   NPROC_PER_NODE    GPUs (default 1)
#   DEVICE_BATCH_SIZE per-device MICRO-batch (default 8, matching the pretrain that
#                     already fit on this card; grad-accum keeps the effective batch
#                     fixed, so this only trades VRAM vs speed. Lower if it OOMs.)
#   MAX_SEQ_LEN       cap context (default: inherit pretrain 8192; try 2048-4096 if OOM)
#   NUM_ITERATIONS    steps (default -1 = one full epoch over the mixture)
#   LOAD_OPTIMIZER    warm-start optim from the converted base (default 1; set 0 for multi-GPU)
#   CUTE_SIZE         CuteChat examples per subtask in the SFT mix (default chat_sft's 2000)
#   CUTE_EVERY        steps between in-training CUTE evals (default chat_sft's 200)
#   CUTE_MAX_PROBLEMS problems per CUTE subtask in the eval (default chat_sft's 32)
#   IDENTITY_FILE     path to identity .jsonl (copied into identity_conversations.jsonl).
#                     Default: the committed nanochat-byte identity. Falls back to the
#                     generic S3 file (BPE-nanochat -- wrong here) only if neither exists.

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

BASE_TAG="${BASE_TAG:-d24-byte-l-ext}"
CHAT_BASE_TAG="${CHAT_BASE_TAG:-${BASE_TAG}-chatbase}"
OUTPUT_TAG="${OUTPUT_TAG:-${BASE_TAG}-chat}"
WANDB_RUN="${WANDB_RUN:-dummy}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-1}"
export NANOCHAT_REPORT_TAG="$OUTPUT_TAG"

source .venv/bin/activate

# Fail fast if CUDA isn't actually visible (genai occasionally flakes and would
# otherwise silently fall back to a useless float32-on-CPU run).
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

CKPT_ROOT="$NANOCHAT_BASE_DIR/base_checkpoints"
SRC_DIR="$CKPT_ROOT/$BASE_TAG"
DST_DIR="$CKPT_ROOT/$CHAT_BASE_TAG"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: base checkpoint $SRC_DIR not found" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. Convert 256 -> 265 vocab (once). Skip if already converted.
if [ -z "$(ls "$DST_DIR"/model_*.pt 2>/dev/null)" ]; then
    STEP=$(ls "$SRC_DIR"/model_*.pt | sed 's/.*model_0*\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
    echo "=== converting $BASE_TAG step $STEP (vocab 256 -> 265) into $CHAT_BASE_TAG"
    python dev/convert_byte_tokenizer_unescaped.py \
        --src "$SRC_DIR" --step "$STEP" --dst "$DST_DIR" --ranks 1
else
    echo "=== converted base already present at $DST_DIR; skipping conversion"
fi

# -----------------------------------------------------------------------------
# 2. Identity data. chat_sft reads $NANOCHAT_BASE_DIR/identity_conversations.jsonl,
# so we copy the byte identity (committed in the repo) into that path. Default is
# the generated nanochat-byte identity; override IDENTITY_FILE to use another.
IDENTITY_DST="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
IDENTITY_FILE="${IDENTITY_FILE:-$(pwd)/nanochat_byte_identity__google_gemini-3-flash-preview.jsonl}"
if [ -f "$IDENTITY_FILE" ]; then
    echo "=== using identity data $IDENTITY_FILE"
    cp "$IDENTITY_FILE" "$IDENTITY_DST"
elif [ ! -f "$IDENTITY_DST" ]; then
    echo "!! WARNING: IDENTITY_FILE $IDENTITY_FILE not found and no $IDENTITY_DST present."
    echo "!! Downloading the generic identity_conversations.jsonl -- it describes the"
    echo "!! BPE nanochat and is WRONG for the byte model. Generate a byte identity with"
    echo "!!   python dev/gen_identity_data.py --gateway-url ... --num 1000"
    echo "!! and re-run with IDENTITY_FILE=<that file>."
    curl -L -o "$IDENTITY_DST" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# -----------------------------------------------------------------------------
# 3. Optional passthrough flags
SFT_ARGS=()
[ -n "$MAX_SEQ_LEN" ]        && SFT_ARGS+=(--max-seq-len="$MAX_SEQ_LEN")
[ -n "$NUM_ITERATIONS" ]     && SFT_ARGS+=(--num-iterations="$NUM_ITERATIONS")
[ -n "$CUTE_SIZE" ]          && SFT_ARGS+=(--cute-size="$CUTE_SIZE")
[ -n "$CUTE_EVERY" ]         && SFT_ARGS+=(--cute-every="$CUTE_EVERY")
[ -n "$CUTE_MAX_PROBLEMS" ]  && SFT_ARGS+=(--cute-max-problems="$CUTE_MAX_PROBLEMS")

if [ "$NPROC_PER_NODE" -gt 1 ] && [ "$LOAD_OPTIMIZER" = "1" ]; then
    echo "!! WARNING: NPROC_PER_NODE>1 with LOAD_OPTIMIZER=1, but the converted optimizer"
    echo "!! is a single rank0 shard. Set LOAD_OPTIMIZER=0 for multi-GPU." >&2
fi

# -----------------------------------------------------------------------------
# 4. SFT
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$CHAT_BASE_TAG" \
    --output-tag="$OUTPUT_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --load-optimizer="$LOAD_OPTIMIZER" \
    "${SFT_ARGS[@]}" \
    --run="$WANDB_RUN"

# -----------------------------------------------------------------------------
# 5. Final evals: ChatCORE + chat-mode CUTE (zero-shot, no prefill -- matches how
#    the chat model was trained to answer).
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft -g "$OUTPUT_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.cute_eval -- \
    -i sft -g "$OUTPUT_TAG" --mode chat --no-prefill --prompt-style zero --subtasks char

echo "=== done. SFT checkpoint: chatsft_checkpoints/$OUTPUT_TAG"
