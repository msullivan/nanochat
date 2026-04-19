#!/bin/bash

apt-get update
apt-get install rsync screen
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
. ~/.bashrc
git clone https://github.com/msullivan/nanochat.git
uvx wandb login
apt-get install python3-dev
