#!/bin/bash
set -e

apt-get update
apt-get install -y rsync screen python3-dev
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# uv installer drops into ~/.local/bin and adds to ~/.bashrc, but ~/.bashrc
# usually short-circuits in non-interactive shells. Source uv's env file
# directly so PATH picks up uv whether we're interactive or piped via ssh.
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"

[ -d nanochat ] || git clone https://github.com/msullivan/nanochat.git

if [ -n "$WANDB_KEY" ]; then
    uvx wandb login "$WANDB_KEY"
else
    uvx wandb login
fi
