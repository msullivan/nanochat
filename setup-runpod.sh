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

# Persist bash history on the network volume so it survives pod rebuilds.
# (setup-runpod.sh re-runs on each new pod so re-appending to the ephemeral
# ~/.bashrc each time is fine; the grep makes it idempotent within a pod.)
touch /workspace/.bash_history
if ! grep -qF 'HISTFILE=/workspace/.bash_history' "$HOME/.bashrc" 2>/dev/null; then
    cat >> "$HOME/.bashrc" <<'EOF'

# nanochat: persist bash history on /workspace so it survives pod rebuilds
export HISTFILE=/workspace/.bash_history
export HISTSIZE=10000
export HISTFILESIZE=20000
shopt -s histappend
# Append history after each command so ungraceful shell deaths don't lose it.
PROMPT_COMMAND="history -a${PROMPT_COMMAND:+; $PROMPT_COMMAND}"
EOF
fi

if [ -n "$WANDB_KEY" ]; then
    uvx wandb login "$WANDB_KEY"
else
    uvx wandb login
fi
