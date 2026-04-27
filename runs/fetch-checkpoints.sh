#!/bin/bash
set -e

# Copy the trained checkpoints + tokenizer from the remote pod back to the local machine.
# Skips optimizer shards (multi-GB, not needed for inference).
#
# Usage:
#   ./fetch-checkpoints.sh             # fetch all model tags
#   ./fetch-checkpoints.sh d24-byte    # fetch only the d24-byte model tag
#
# Host/port/user default to the git-sync remote (`git config remote.sync.url`)
# if one is configured, e.g. ssh://root@big-boy:16804/... → host=big-boy, port=16804.
# User falls back to "root" and port to 22 when not set by either the remote URL
# or an env-var override.
#
# Env knobs (override in any combination):
#   REMOTE_USER        default: user from sync URL, else "root"
#   REMOTE_HOST        default: host from sync URL, else "big-boy"
#   REMOTE_PORT        default: port from sync URL, else 22
#   REMOTE_BASE_DIR    default: /workspace/nanochat-cache
#   LOCAL_BASE_DIR     default: $NANOCHAT_BASE_DIR or ~/.cache/nanochat
#   FETCH_BASE=1       also pull base_checkpoints/ (large)

MODEL_TAG="${1:-}"

# Parse the git-sync remote URL (ssh://[user@]host[:port][/path]) if present.
SYNC_URL="$(git config --get remote.sync.url 2>/dev/null || true)"
SYNC_USER=""
SYNC_HOST=""
SYNC_PORT=""
if [[ "$SYNC_URL" =~ ^ssh://(([^@]+)@)?([^:/]+)(:([0-9]+))?(/.*)?$ ]]; then
    SYNC_USER="${BASH_REMATCH[2]}"
    SYNC_HOST="${BASH_REMATCH[3]}"
    SYNC_PORT="${BASH_REMATCH[5]}"
fi

REMOTE_USER="${REMOTE_USER:-${SYNC_USER:-root}}"
REMOTE_HOST="${REMOTE_HOST:-${SYNC_HOST:-big-boy}}"
REMOTE_PORT="${REMOTE_PORT:-${SYNC_PORT:-22}}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/workspace/nanochat-cache}"
LOCAL_BASE_DIR="${LOCAL_BASE_DIR:-${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}}"
FETCH_BASE="${FETCH_BASE:-0}"

SSH_CMD="ssh -p $REMOTE_PORT"
REMOTE="$REMOTE_USER@$REMOTE_HOST"

mkdir -p "$LOCAL_BASE_DIR/chatsft_checkpoints" "$LOCAL_BASE_DIR/tokenizer" "$LOCAL_BASE_DIR/report"

if [ -n "$MODEL_TAG" ]; then
    echo "Fetching only model tag: $MODEL_TAG"
    SFT_SRC="$REMOTE_BASE_DIR/chatsft_checkpoints/$MODEL_TAG/"
    SFT_DST="$LOCAL_BASE_DIR/chatsft_checkpoints/$MODEL_TAG/"
    BASE_SRC="$REMOTE_BASE_DIR/base_checkpoints/$MODEL_TAG/"
    BASE_DST="$LOCAL_BASE_DIR/base_checkpoints/$MODEL_TAG/"
    mkdir -p "$SFT_DST"
else
    SFT_SRC="$REMOTE_BASE_DIR/chatsft_checkpoints/"
    SFT_DST="$LOCAL_BASE_DIR/chatsft_checkpoints/"
    BASE_SRC="$REMOTE_BASE_DIR/base_checkpoints/"
    BASE_DST="$LOCAL_BASE_DIR/base_checkpoints/"
fi

echo "Fetching SFT checkpoints -> $SFT_DST"
rsync -avP -e "$SSH_CMD" \
    --exclude 'optim_*.pt' \
    "$REMOTE:$SFT_SRC" \
    "$SFT_DST"

echo "Fetching tokenizer -> $LOCAL_BASE_DIR/tokenizer/"
rsync -avP -e "$SSH_CMD" \
    "$REMOTE:$REMOTE_BASE_DIR/tokenizer/" \
    "$LOCAL_BASE_DIR/tokenizer/"

echo "Fetching report -> $LOCAL_BASE_DIR/report/"
rsync -avP -e "$SSH_CMD" \
    "$REMOTE:$REMOTE_BASE_DIR/report/" \
    "$LOCAL_BASE_DIR/report/"

if [ "$FETCH_BASE" = "1" ]; then
    mkdir -p "$BASE_DST"
    echo "Fetching base checkpoints -> $BASE_DST"
    rsync -avP -e "$SSH_CMD" \
        --exclude 'optim_*.pt' \
        "$REMOTE:$BASE_SRC" \
        "$BASE_DST"
fi

echo "Done. Try: python -m scripts.chat_cli"
