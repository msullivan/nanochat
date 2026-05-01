#!/bin/bash
set -ex

# Extend the d24-byte-l base run from a chosen mid-stable-phase checkpoint
# under a NEW model tag (d24-byte-l-ext) so the original run's checkpoints
# stay intact and we can compare the two trajectories cleanly.
#
# PREREQUISITE: copy the seed checkpoint into the new tag's dir first, e.g.:
#   D=$NANOCHAT_BASE_DIR/base_checkpoints
#   mkdir -p $D/d24-byte-l-ext
#   cp $D/d24-byte-l/{model,meta}_009337.* \
#      $D/d24-byte-l/optim_009337_rank0.pt \
#      $D/d24-byte-l-ext/
# After copying, --resume-from-step=latest under d24-byte-l-ext picks it up.
#
# Schedule (per WSD discussion):
#   - Resume from step 9337 of d24-byte-l (copied into d24-byte-l-ext above),
#     just past where the LR=0.55 plateau began. The 9300:0.55 breakpoint
#     anchor stays in place so the schedule reads lrm=0.55 from step 9337
#     onward (linear interp between two same-value anchors = flat).
#   - Hold flat at 0.55 from step 9337 to 11500 (~2.2k steps to confirm
#     plateau is real before further cooldown).
#   - Linear decay 0.55 -> 0.2 from step 11500 to 21500 (10k step ramp).
#   - Standard 10% warmdown 0.2 -> final_lr_frac=0.05 over the last 2390
#     steps (warmdown_start = 23900 - round(0.1 * 23900) = 21510, so the
#     21500:0.2 breakpoint lands just inside the stable-phase window).
#
# This is a NEW wandb run (no WANDB_RUN_ID/RESUME); separate plot, separate
# checkpoints under base_checkpoints/d24-byte-l-ext/. Old d24-byte-l
# checkpoints stay where they are -- safe to abort and try a different
# schedule by re-running this script with different --lr-breakpoints.
#
# Wrap in screen at call site:
#   screen -L -Logfile ~/d24-byte-l-ext.log -S extend bash runs/extend_d24_byte_l.sh

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NANOCHAT_REPORT_TAG=d24-byte-l-ext

# Strip any inherited wandb-resume env vars from a prior session in this shell.
unset WANDB_RUN_ID WANDB_RESUME

source .venv/bin/activate

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=24 \
    --byte-tokenizer \
    --max-seq-len=8192 \
    --device-batch-size=8 \
    --target-param-data-ratio=16 \
    --num-iterations=23900 \
    --window-pattern=L \
    --warmdown-ratio=0.1 \
    --lr-breakpoints="9300:0.55,11500:0.55,21500:0.2" \
    --save-every=100 \
    --resume-from-step=latest \
    --model-tag=d24-byte-l-ext \
    --eval-every=100 \
    --eval-tokens=4194304 \
    --core-metric-every=100 \
    --sample-every=100 \
    --fp8 \
    --run=d24-byte-l-ext

torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- \
    --device-batch-size=8 \
    --model-tag=d24-byte-l-ext
