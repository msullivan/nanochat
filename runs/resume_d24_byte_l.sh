#!/bin/bash
set -e

# One-off resume for the d24-byte-l run launched as:
#   MODEL_TAG=d24-byte-l NPROC_PER_NODE=1 DEPTH=24 DEVICE_BATCH_SIZE=8 \
#   WARMDOWN_RATIO=0.1 WINDOW_PATTERN=L TARGET_DATA_RATIO=16 SAVE_EVERY=200 \
#   bash runs/speedrun_byte.sh
#
# Picks up at step 200 (first SAVE_EVERY checkpoint), with denser eval cadences:
#   - val BPB every 100 steps on a smaller eval set (4M tokens)
#   - CORE every 500 steps
#   - sample every 200 steps
#
# Wrap in screen at call site, e.g.:
#   screen -L -Logfile ~/d24-byte-l-resume.log -S train bash runs/resume_d24_byte_l.sh
#
# To append onto the original wandb run instead of starting a new one, set
# WANDB_RUN_ID to the original run's id (last path component of the wandb URL):
#   WANDB_RUN_ID=abc12345 bash runs/resume_d24_byte_l.sh

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NANOCHAT_REPORT_TAG=d24-byte-l

# wandb continues the original run if WANDB_RUN_ID is set; new run otherwise.
if [ -n "$WANDB_RUN_ID" ]; then
    export WANDB_RESUME=must
    WANDB_FLAG="--run=d24-byte-l"   # name is ignored when resuming by id
else
    WANDB_FLAG="--run=d24-byte-l-resumed"
fi

source .venv/bin/activate

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=24 \
    --byte-tokenizer \
    --max-seq-len=8192 \
    --device-batch-size=8 \
    --target-param-data-ratio=16 \
    --window-pattern=L \
    --warmdown-ratio=0.1 \
    --save-every=100 \
    --resume-from-step=400 \
    --model-tag=d24-byte-l \
    --eval-every=100 \
    --eval-tokens=4194304 \
    --core-metric-every=500 \
    --sample-every=200 \
    --fp8 \
    $WANDB_FLAG

torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- \
    --device-batch-size=8 \
    --model-tag=d24-byte-l
