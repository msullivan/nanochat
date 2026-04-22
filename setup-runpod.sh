#!/bin/bash

apt-get update
apt-get install -y rsync screen python3-dev
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
. ~/.bashrc
git clone https://github.com/msullivan/nanochat.git
uvx wandb login
