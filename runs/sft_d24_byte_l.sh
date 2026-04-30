#!/bin/bash
set -ex

# SFT-only invocation for the d24-byte-l base. Use when the base is already
# trained and you just want to run SFT (e.g., the base-train phase of
# resume_d24_byte_l.sh already completed but the chained SFT didn't get to run,
# or you want to re-run SFT against the same base with a different mix).
#
# Wrap in screen at call site, e.g.:
#   screen -L -Logfile ~/d24-byte-l-sft.log -S sft bash runs/sft_d24_byte_l.sh
#
# Env knobs:
#   SFT_WANDB_RUN  override the wandb run name (default: d24-byte-l-sft)
#   OUTPUT_TAG     override the SFT output tag (default: d24-byte-l-sft)

cd "$(dirname "$0")/.."

# Strip any inherited WANDB_RUN_ID / WANDB_RESUME from a prior base-train
# session — SFT is a fresh run in the nanochat-sft project.
unset WANDB_RUN_ID WANDB_RESUME

NPROC_PER_NODE=1 \
DEVICE_BATCH_SIZE=8 \
MAX_SEQ_LEN=2048 \
MODEL_TAG=d24-byte-l \
OUTPUT_TAG="${OUTPUT_TAG:-d24-byte-l-sft}" \
WANDB_RUN="${SFT_WANDB_RUN:-d24-byte-l-sft}" \
EVAL_EVERY=100 \
EVAL_TOKENS=4194304 \
CHATCORE_EVERY=100 \
bash runs/finetune.sh
