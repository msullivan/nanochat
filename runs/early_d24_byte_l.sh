#!/bin/bash
set -ex

# Early-stop the d24-byte-l base run from step 4000 with a short warmdown
# to step 5500, under a NEW model tag (d24-byte-l-early) so the original
# run's checkpoints stay intact and we can compare the two trajectories
# cleanly. Mirror of extend_d24_byte_l.sh but ENDING the run early instead
# of extending it.
#
# At step 4000 the original d24-byte-l run had lrm=0.8 (the breakpoints
# were 3700:1.0,4000:0.8,5400:0.8,6000:0.7,...). We anchor the new schedule
# with breakpoint 4000:0.8 so the lrm reads 0.8 at warmdown_start, then
# warmdown takes over and brings lrm linearly to final_lr_frac (default
# 0.05) over the remaining 1500 steps.
#
# PREREQUISITE: copy the seed checkpoint into the new tag's dir first, e.g.:
#   D=$NANOCHAT_BASE_DIR/base_checkpoints
#   mkdir -p $D/d24-byte-l-early
#   cp $D/d24-byte-l/{model,meta}_004000.* \
#      $D/d24-byte-l/optim_004000_rank0.pt \
#      $D/d24-byte-l-early/
# After copying, --resume-from-step=latest under d24-byte-l-early picks it up.
#
# Schedule:
#   - Resume from step 4000 of d24-byte-l (copied into d24-byte-l-early).
#   - Warmdown lrm 0.8 -> final_lr_frac from step 4000 to step 5500.
#   - num_iterations = 5500, warmdown_ratio = 1500/5500 ≈ 0.272727 so that
#     warmdown_start_iter = 5500 - round(0.272727 * 5500) = 5500 - 1500 = 4000.
#
# This is a NEW wandb run (no WANDB_RUN_ID/RESUME); separate plot, separate
# checkpoints under base_checkpoints/d24-byte-l-early/. Old d24-byte-l
# checkpoints stay where they are.
#
# Wrap in screen at call site:
#   screen -L -Logfile ~/d24-byte-l-early.log -S early bash runs/early_d24_byte_l.sh

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NANOCHAT_REPORT_TAG=d24-byte-l-early

# Strip any inherited wandb-resume env vars from a prior session in this shell.
unset WANDB_RUN_ID WANDB_RESUME

source .venv/bin/activate

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=24 \
    --byte-tokenizer \
    --max-seq-len=8192 \
    --device-batch-size=8 \
    --target-param-data-ratio=16 \
    --num-iterations=5500 \
    --window-pattern=L \
    --warmdown-ratio=0.272727 \
    --lr-breakpoints="4000:0.8" \
    --save-every=100 \
    --resume-from-step=latest \
    --model-tag=d24-byte-l-early \
    --eval-every=100 \
    --eval-tokens=4194304 \
    --core-metric-every=100 \
    --sample-every=100 \
    --fp8 \
    --run=d24-byte-l-early

torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- \
    --device-batch-size=8 \
    --model-tag=d24-byte-l-early
