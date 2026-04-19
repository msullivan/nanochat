#!/bin/bash
set -e

# Copy the trained checkpoints + tokenizer from the remote pod back to the local machine.
# Skips optimizer shards (multi-GB, not needed for inference).
#
# Env knobs (with defaults matching the current git-sync remote):
#   REMOTE_USER=root
#   REMOTE_HOST=big-boy
#   REMOTE_PORT=15090
#   REMOTE_BASE_DIR=/workspace/nanochat-cache
#   LOCAL_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}
#   FETCH_BASE=0   # set to 1 to also pull base_checkpoints/ (large)

REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-big-boy}"
REMOTE_PORT="${REMOTE_PORT:-15090}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/workspace/nanochat-cache}"
LOCAL_BASE_DIR="${LOCAL_BASE_DIR:-${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}}"
FETCH_BASE="${FETCH_BASE:-0}"

SSH_CMD="ssh -p $REMOTE_PORT"
REMOTE="$REMOTE_USER@$REMOTE_HOST"

mkdir -p "$LOCAL_BASE_DIR/chatsft_checkpoints" "$LOCAL_BASE_DIR/tokenizer"

echo "Fetching SFT checkpoints -> $LOCAL_BASE_DIR/chatsft_checkpoints/"
rsync -avP -e "$SSH_CMD" \
    --exclude 'optim_*.pt' \
    "$REMOTE:$REMOTE_BASE_DIR/chatsft_checkpoints/" \
    "$LOCAL_BASE_DIR/chatsft_checkpoints/"

echo "Fetching tokenizer -> $LOCAL_BASE_DIR/tokenizer/"
rsync -avP -e "$SSH_CMD" \
    "$REMOTE:$REMOTE_BASE_DIR/tokenizer/" \
    "$LOCAL_BASE_DIR/tokenizer/"

if [ "$FETCH_BASE" = "1" ]; then
    mkdir -p "$LOCAL_BASE_DIR/base_checkpoints"
    echo "Fetching base checkpoints -> $LOCAL_BASE_DIR/base_checkpoints/"
    rsync -avP -e "$SSH_CMD" \
        --exclude 'optim_*.pt' \
        "$REMOTE:$REMOTE_BASE_DIR/base_checkpoints/" \
        "$LOCAL_BASE_DIR/base_checkpoints/"
fi

echo "Done. Try: python -m scripts.chat_cli"
