#!/bin/bash
set -ex

# CUTE-format completion-style continued-pretraining finetune for the byte
# d24 model. Trains on a synthetic corpus of CUTE-format documents (4-shot
# demo + Question + Answer in quotes) so the model learns the format and
# character-manipulation operations directly, with no chat tokenization,
# loss masking, or python-tool short-circuit.
#
# Cheaper than chat SFT because:
#   - FP8 stays on (validated for base training)
#   - No per-example chat rendering or loss-mask construction
#   - Best-fit packing into max_seq_len byte sequences
#   - Smaller total token budget (~200 MB synthetic data)
#
# PREREQUISITES
# 1. Generate the synthetic corpus into $NANOCHAT_BASE_DIR/cute_pt_data/
#    .venv/bin/python dev/gen_cute_pt_data.py \
#        --out-dir "$NANOCHAT_BASE_DIR/cute_pt_data" \
#        --num-words 1000
#    (1000 words * 8 subtasks = ~8k docs * ~500 bytes = ~4 MB. At
#    total_batch_size=256K tokens / max_seq_len=8192 = 32 sequences/step,
#    that's ~16 steps for one full pass over the corpus. The default
#    FT_STEPS below is sized for ~12 epochs over this small corpus --
#    bump --num-words and FT_STEPS together for serious training.)
#
# 2. Seed the new model tag dir with a finished base checkpoint, e.g. the
#    end of the d24-byte-l-ext run:
#    D="$NANOCHAT_BASE_DIR/base_checkpoints"
#    SRC=d24-byte-l-ext  STEP=23900  DST=d24-byte-l-cute
#    mkdir -p "$D/$DST"
#    cp "$D/$SRC/model_$(printf %06d $STEP).pt"     "$D/$DST/"
#    cp "$D/$SRC/meta_$(printf %06d $STEP).json"    "$D/$DST/"
#    cp "$D/$SRC/optim_$(printf %06d $STEP)_rank0.pt" "$D/$DST/"
#
# 3. Run: bash runs/cute_pt_d24_byte_l.sh
#
# LR SCHEDULE
# Default schedule is conservative: hold at lrm=0.05 for the bulk of the
# finetune and warmdown to final_lr_frac. Override via env if you want
# different. The relevant variables:
#   RESUME_STEP   - the step number of the seed checkpoint
#   FT_STEPS      - how many finetune steps to add on top
#   FT_LRM        - LR multiplier during the flat finetune phase
# These wire into NUM_ITERATIONS = RESUME_STEP + FT_STEPS, and a single
# breakpoint at RESUME_STEP:FT_LRM that pins the stable phase. Warmdown_ratio
# is sized so the last 10% of FT_STEPS is the warmdown, leaving 90% flat.

cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
# Redirect base_train's dataloader away from climbmix to our synthetic shards
# (relative to NANOCHAT_BASE_DIR; can also be absolute).
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-cute_pt_data}"
export NANOCHAT_REPORT_TAG=d24-byte-l-cute
unset WANDB_RUN_ID WANDB_RESUME

RESUME_STEP="${RESUME_STEP:-23900}"
FT_STEPS="${FT_STEPS:-200}"
FT_LRM="${FT_LRM:-0.05}"

# warmdown over the LAST 10% of the finetune (= last 0.1*FT_STEPS steps),
# expressed as a fraction of NUM_ITERATIONS so base_train's existing math
# locks in the right warmdown_start.
NUM_ITERATIONS=$(( RESUME_STEP + FT_STEPS ))
WARMDOWN_TAIL_STEPS=$(( FT_STEPS / 10 ))
# warmdown_start_iter = NUM_ITERATIONS - round(WARMDOWN_RATIO * NUM_ITERATIONS).
# Choose WARMDOWN_RATIO so that WARMDOWN_RATIO * NUM_ITERATIONS == WARMDOWN_TAIL_STEPS.
WARMDOWN_RATIO=$(.venv/bin/python -c "print(${WARMDOWN_TAIL_STEPS} / ${NUM_ITERATIONS})")

# Anchor the stable phase at FT_LRM starting at RESUME_STEP, plus a second
# anchor just before warmdown_start so the stable phase reads flat.
WARMDOWN_START=$(( NUM_ITERATIONS - WARMDOWN_TAIL_STEPS ))
LR_BREAKPOINTS="${RESUME_STEP}:${FT_LRM},$((WARMDOWN_START - 1)):${FT_LRM}"

echo "=== cute_pt: resume @ step $RESUME_STEP, +$FT_STEPS finetune steps "
echo "=== num_iterations=$NUM_ITERATIONS warmdown_ratio=$WARMDOWN_RATIO"
echo "=== lr_breakpoints=$LR_BREAKPOINTS"

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=24 \
    --byte-tokenizer \
    --max-seq-len=8192 \
    --device-batch-size=8 \
    --num-iterations="$NUM_ITERATIONS" \
    --window-pattern=L \
    --warmdown-ratio="$WARMDOWN_RATIO" \
    --lr-breakpoints="$LR_BREAKPOINTS" \
    --save-every=50 \
    --resume-from-step=latest \
    --model-tag=d24-byte-l-cute \
    --eval-every=25 \
    --eval-tokens=1048576 \
    --core-metric-every=50 \
    --sample-every=50 \
    --fp8 \
    --run=d24-byte-l-cute

torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- \
    --device-batch-size=8 \
    --model-tag=d24-byte-l-cute
