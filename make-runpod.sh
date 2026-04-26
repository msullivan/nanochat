#!/bin/bash -e

# Create a Runpod H100 pod attached to the shared nanochat network volume,
# poll until SSH info is available, and write the full pod descriptor to
# .pods/$NAME.json. Prints the ssh command on completion.
#
# Usage: ./make-runpod.sh <name> <gpu-count>

NAME="$1"
GPUS="$2"

if [ -z "$NAME" ] || [ -z "$GPUS" ]; then
    echo "Usage: $0 <name> <gpu-count>" >&2
    exit 1
fi

VOL_ID=ij2krw25ls
TEMPLATE_ID=runpod-ubuntu-2204
GPU_TYPE="NVIDIA H100 80GB HBM3"
DISK_GB=25

mkdir -p .pods

CREATE_JSON=".pods/$NAME-create.json"
POD_JSON=".pods/$NAME.json"

runpodctl pod create \
    --template-id "$TEMPLATE_ID" \
    --gpu-id "$GPU_TYPE" \
    --container-disk-in-gb "$DISK_GB" \
    --network-volume-id "$VOL_ID" \
    --gpu-count "$GPUS" \
    --name "$NAME" \
    | tee "$CREATE_JSON"

POD_ID=$(jq -r .id "$CREATE_JSON")
if [ -z "$POD_ID" ] || [ "$POD_ID" = "null" ]; then
    echo "Failed to extract pod id from $CREATE_JSON" >&2
    exit 1
fi
echo "Pod id: $POD_ID"

echo "Waiting for SSH info..."
for i in $(seq 1 60); do
    runpodctl pod get "$POD_ID" > "$POD_JSON"
    IP=$(jq -r '.ssh.ip // empty' "$POD_JSON")
    PORT=$(jq -r '.ssh.port // empty' "$POD_JSON")
    if [ -n "$IP" ] && [ -n "$PORT" ]; then
        break
    fi
    sleep 5
done

if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "Timed out waiting for SSH info. Last pod state in $POD_JSON" >&2
    exit 1
fi

SSH_CMD=$(jq -r '.ssh.ssh_command' "$POD_JSON")
SYNC_URL="ssh://root@$IP:$PORT/root/nanochat"
git remote set-url sync "$SYNC_URL" 2>/dev/null || git remote add sync "$SYNC_URL"

if [ ! -f "$HOME/.wandb-key" ]; then
    echo "~/.wandb-key not found; aborting before remote setup" >&2
    exit 1
fi
WANDB_KEY=$(< "$HOME/.wandb-key")

echo "Running setup on pod (apt + uv + clone + wandb login)..."
{
    printf 'export WANDB_KEY=%q\n' "$WANDB_KEY"
    cat setup-runpod.sh
} | ssh -p "$PORT" "root@$IP" bash -s

echo
echo "Pod ready."
echo "  SSH:  $SSH_CMD"
echo "  sync: $SYNC_URL"
