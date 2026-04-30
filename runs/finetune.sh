#!/bin/bash
set -ex

# Run SFT + eval starting from a pretrained base model.
# Assumes base_checkpoints/ already exists in $NANOCHAT_BASE_DIR (i.e. base_train.py has been run).
#
# Example launches:
#   bash runs/finetune.sh
#   WANDB_RUN=sft-d24 bash runs/finetune.sh
#   MODEL_TAG=d24-byte OUTPUT_TAG=d24-byte-cute bash runs/finetune.sh
#   screen -L -Logfile runs/finetune.log -S finetune bash runs/finetune.sh
#
# Env knobs:
#   MODEL_TAG    base checkpoint to load from (default: chat_sft.py picks d{depth})
#   OUTPUT_TAG   SFT checkpoint dir to write to (default: same as MODEL_TAG)
#   WANDB_RUN    wandb run name; "dummy" disables logging (default: dummy)

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv (no-op if already set up)
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
# Sanity check: base model must already be present
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints" ]; then
    echo "ERROR: $NANOCHAT_BASE_DIR/base_checkpoints/ not found."
    echo "       Run runs/speedrun.sh (or scripts.base_train) first to produce a pretrained model."
    exit 1
fi

# -----------------------------------------------------------------------------
# SFT data: identity_conversations.jsonl lives under $NANOCHAT_BASE_DIR.
# qamis_* and pep827_conversations.jsonl are committed in the repo root and referenced from there.
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "Downloading identity_conversations.jsonl..."
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# -----------------------------------------------------------------------------
# Optional: write SFT checkpoint under a custom dir name (so re-running with a
# different mixture doesn't clobber an earlier run). Eval picks up the same tag.
SFT_EXTRA_ARGS=()
EVAL_EXTRA_ARGS=()
if [ -n "$MODEL_TAG" ]; then
    SFT_EXTRA_ARGS+=(--model-tag="$MODEL_TAG")
fi
if [ -n "$OUTPUT_TAG" ]; then
    SFT_EXTRA_ARGS+=(--output-tag="$OUTPUT_TAG")
    EVAL_EXTRA_ARGS+=(-g "$OUTPUT_TAG")
    # Also scope the report dir so SFT + eval section files live in report/<tag>/
    export NANOCHAT_REPORT_TAG="$OUTPUT_TAG"
fi
# Optional eval-cadence overrides (forwarded to chat_sft if set):
#   EVAL_EVERY        steps between val-bpb evals (default 200)
#   EVAL_TOKENS       val tokens per eval (default 40*524288 = 20.9M)
#   CHATCORE_EVERY    steps between ChatCORE evals (default 200)
#   MAX_SEQ_LEN       max context length (default: inherit from pretrain meta)
if [ -n "$EVAL_EVERY" ]; then
    SFT_EXTRA_ARGS+=(--eval-every="$EVAL_EVERY")
fi
if [ -n "$EVAL_TOKENS" ]; then
    SFT_EXTRA_ARGS+=(--eval-tokens="$EVAL_TOKENS")
fi
if [ -n "$CHATCORE_EVERY" ]; then
    SFT_EXTRA_ARGS+=(--chatcore-every="$CHATCORE_EVERY")
fi
if [ -n "$MAX_SEQ_LEN" ]; then
    SFT_EXTRA_ARGS+=(--max-seq-len="$MAX_SEQ_LEN")
fi

# -----------------------------------------------------------------------------
# SFT + eval
torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-8} -m scripts.chat_sft -- \
    --device-batch-size=${DEVICE_BATCH_SIZE:-16} \
    --load-optimizer=${LOAD_OPTIMIZER:-1} \
    "${SFT_EXTRA_ARGS[@]}" \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-8} -m scripts.chat_eval -- -i sft "${EVAL_EXTRA_ARGS[@]}"
