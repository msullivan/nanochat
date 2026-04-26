#!/bin/bash
set -e

# Resume a base_train run from a checkpoint, either to anneal the LR/WD/momentum
# down to zero (MODE=anneal) or to extend training to a longer total horizon
# (MODE=extend). Inherits model config (depth, byte_tokenizer, max_seq_len,
# device_batch_size, target_param_data_ratio, window_pattern, fp8) from the
# source run's saved meta, so you only need to specify the deltas.
#
# Examples:
#   # Anneal: WARMDOWN_RATIO defaults to 0.1, anneal length derived so the
#   # warmdown is the last WARMDOWN_RATIO fraction of the (FROM_STEP + anneal)
#   # total. With FROM_STEP=4500, WARMDOWN_RATIO=0.1 -> anneal 500 steps.
#   SOURCE_TAG=d24-byte FROM_STEP=4500 MODE=anneal bash runs/resume_byte.sh
#
#   # Extend: continue from FROM_STEP to NEW_TOTAL with the last WARMDOWN_RATIO
#   # of NEW_TOTAL spent in warmdown.
#   SOURCE_TAG=d24-byte FROM_STEP=4500 MODE=extend NEW_TOTAL=10000 \
#       bash runs/resume_byte.sh

: "${SOURCE_TAG:?SOURCE_TAG is required}"
: "${FROM_STEP:?FROM_STEP is required}"
: "${MODE:?MODE is required: anneal | extend}"

NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
SOURCE_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$SOURCE_TAG"
FROM_STEP_PADDED=$(printf '%06d' "$FROM_STEP")
META_FILE="$SOURCE_DIR/meta_${FROM_STEP_PADDED}.json"

if [ ! -f "$META_FILE" ]; then
    echo "Source meta not found: $META_FILE" >&2
    echo "Available checkpoints in $SOURCE_DIR:" >&2
    ls "$SOURCE_DIR" 2>&1 | grep '^model_' | head -20 >&2 || true
    exit 1
fi

# -----------------------------------------------------------------------------
# Inherit model config from the source meta
DEPTH=$(jq -r '.user_config.depth' "$META_FILE")
BYTE_TOKENIZER=$(jq -r '.user_config.byte_tokenizer' "$META_FILE")
MAX_SEQ_LEN=$(jq -r '.user_config.max_seq_len' "$META_FILE")
DEVICE_BATCH_SIZE=$(jq -r '.user_config.device_batch_size' "$META_FILE")
TARGET_DATA_RATIO=$(jq -r '.user_config.target_param_data_ratio' "$META_FILE")
WINDOW_PATTERN=$(jq -r '.user_config.window_pattern' "$META_FILE")
FP8=$(jq -r '.user_config.fp8' "$META_FILE")

# -----------------------------------------------------------------------------
# Defaults overridable via env
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SAVE_EVERY="${SAVE_EVERY:-500}"

# -----------------------------------------------------------------------------
# Mode-specific computation
case "$MODE" in
    anneal)
        # ANNEAL_LEN such that WARMDOWN_RATIO is the warmdown fraction of (FROM_STEP + ANNEAL_LEN)
        ANNEAL_LEN=$(python3 -c "print(round($WARMDOWN_RATIO * $FROM_STEP / (1 - $WARMDOWN_RATIO)))")
        NUM_ITERATIONS=$((FROM_STEP + ANNEAL_LEN))
        # Recompute effective warmdown_ratio after rounding so warmdown_start lands exactly on FROM_STEP
        EFFECTIVE_RATIO=$(python3 -c "print($ANNEAL_LEN / $NUM_ITERATIONS)")
        DEFAULT_TAG="${SOURCE_TAG}-anneal-from-${FROM_STEP}"
        ;;
    extend)
        : "${NEW_TOTAL:?NEW_TOTAL is required for MODE=extend}"
        if [ "$NEW_TOTAL" -le "$FROM_STEP" ]; then
            echo "NEW_TOTAL ($NEW_TOTAL) must be > FROM_STEP ($FROM_STEP)" >&2
            exit 1
        fi
        NUM_ITERATIONS=$NEW_TOTAL
        EFFECTIVE_RATIO=$WARMDOWN_RATIO
        DEFAULT_TAG="${SOURCE_TAG}-ext-${NEW_TOTAL}"
        ;;
    *)
        echo "Unknown MODE: $MODE (expected: anneal | extend)" >&2
        exit 1
        ;;
esac

OUTPUT_TAG="${OUTPUT_TAG:-$DEFAULT_TAG}"
WANDB_RUN="${WANDB_RUN:-$OUTPUT_TAG}"
export NANOCHAT_REPORT_TAG="$OUTPUT_TAG"

echo "=== resume: source=$SOURCE_TAG@$FROM_STEP mode=$MODE -> tag=$OUTPUT_TAG ==="
echo "    inherited: depth=$DEPTH ratio=$TARGET_DATA_RATIO seq_len=$MAX_SEQ_LEN"
echo "               batch=$DEVICE_BATCH_SIZE window=$WINDOW_PATTERN byte=$BYTE_TOKENIZER fp8=$FP8"
echo "    schedule:  num_iterations=$NUM_ITERATIONS warmdown_ratio=$EFFECTIVE_RATIO save_every=$SAVE_EVERY"

# -----------------------------------------------------------------------------
# Build pass-through flag list for store_true bools
EXTRA_FLAGS=()
[ "$BYTE_TOKENIZER" = "true" ] && EXTRA_FLAGS+=(--byte-tokenizer)
[ "$FP8" = "true" ] && EXTRA_FLAGS+=(--fp8)

source .venv/bin/activate

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --target-param-data-ratio="$TARGET_DATA_RATIO" \
    --num-iterations="$NUM_ITERATIONS" \
    --warmdown-ratio="$EFFECTIVE_RATIO" \
    --window-pattern="$WINDOW_PATTERN" \
    --save-every="$SAVE_EVERY" \
    --resume-from-step="$FROM_STEP" \
    --resume-from-tag="$SOURCE_TAG" \
    --model-tag="$OUTPUT_TAG" \
    "${EXTRA_FLAGS[@]}" \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$OUTPUT_TAG"
