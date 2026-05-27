#!/bin/bash
# Sweep CUTE-only training from random init across dataset sizes.
#
# Companion to dev/sweep_cute_pt.sh -- same SIZES, same CUTE data, same
# evaluation -- but trains a fresh byte-LM from random init instead of
# resuming from a pretrained base. Answers: "what does pretraining buy
# for CUTE specifically? Could you skip it entirely?"
#
# Pairs with the existing cute_pt sweep so plots can overlay
# (pretrained + finetuned) vs (from-scratch) at matched data sizes.
# Wandb runs go to the nanochat-cute-scratch project.
#
# Outer loop: dataset size. For each size: regenerate (or reuse) the CUTE
# no-demos data → train d24-byte from scratch for N_EPOCHS over it → run
# zero-shot CUTE eval → append to results_scratch.csv.
#
# ENV KNOBS
#   SIZES        word counts for --num-words (default matches the existing
#                cute_pt sweep so curves overlay cleanly)
#   N_EPOCHS     epochs over each dataset (default: 2). Matches cute_pt
#                so each cell sees the same CUTE-data exposure as the
#                corresponding cute_pt cell did.
#   DEPTH        model depth (default: 24)
#   BATCH        device-batch-size (default: 8)
#   MIN_STEPS    floor on NUM_ITERATIONS (default: 80). Stops the very
#                small sizes from being "1 step" runs that don't even
#                complete the default 40-step LR warmup. Smaller cells
#                will be uninformative but tell you where the floor is.
#   SKIP_DONE    "1" (default) to skip (size) pairs already in the CSV
#   FORCE_REGEN  "1" to regenerate data shards even if present
#   RESULT_CSV   results path (default: $NANOCHAT_BASE_DIR/cute_sweep/results_scratch.csv)
#   PROMPT_STYLE eval prompt style: zero (default) or fewshot
#   EVAL_MAX     --max-problems for cute_eval (default: 100; -1 = all)
#   TPW_BYTE     tokens-per-word for the byte model (default: 800)
#   TOKENS_PER_STEP  default 1048576 (1MB), matches d24 device-batch=8 single-GPU

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.
export OMP_NUM_THREADS=1

SIZES="${SIZES:-30000 100000 300000}"
N_EPOCHS="${N_EPOCHS:-2}"
DEPTH="${DEPTH:-24}"
BATCH="${BATCH:-8}"
MIN_STEPS="${MIN_STEPS:-80}"
SKIP_DONE="${SKIP_DONE:-1}"
PROMPT_STYLE="${PROMPT_STYLE:-zero}"
EVAL_MAX="${EVAL_MAX:-100}"
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
RESULT_CSV="${RESULT_CSV:-$BASE_DIR/cute_sweep/results_scratch.csv}"
TPW_BYTE="${TPW_BYTE:-800}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-1048576}"

mkdir -p "$(dirname "$RESULT_CSV")"
if [ ! -s "$RESULT_CSV" ]; then
    echo "model,size,subtask,accuracy,n_passed,n_total,ft_steps" > "$RESULT_CSV"
fi
echo "=== RESULT_CSV: $RESULT_CSV ==="

EVAL_SUBTASKS="${SUBTASKS:-char}"

# Consistent model "name" used in the CSV. Lets plot scripts identify
# from-scratch rows distinct from cute_pt rows (which use d24-byte-l-* etc).
SCRATCH_MODEL_NAME="d24-byte-scratch"

already_done() {
    [ "$SKIP_DONE" = "1" ] || return 1
    local size="$1"
    local n
    n=$(awk -F, -v m="$SCRATCH_MODEL_NAME" -v s="$size" 'NR>1 && $1==m && $2==s {c++} END {print c+0}' "$RESULT_CSV")
    [ "$n" -ge 8 ]
}

for SIZE in $SIZES; do
    echo "=============================================================="
    echo "=== SIZE=$SIZE (from-scratch) ==="
    echo "=============================================================="

    if already_done "$SIZE"; then
        echo "--- SKIP (already in $RESULT_CSV): size=$SIZE ---"
        continue
    fi

    DATA_DIR="$BASE_DIR/cute_pt_data_nodemos_${SIZE}"
    if [ "$FORCE_REGEN" = "1" ] || [ ! -f "$DATA_DIR/shard_00000.parquet" ]; then
        echo "--- generating no-demos data: $DATA_DIR ($SIZE words) ---"
        .venv/bin/python dev/gen_cute_pt_data.py \
            --out-dir "$DATA_DIR" \
            --num-words "$SIZE" \
            --no-demos \
            --seed 0
    else
        echo "--- data exists: $DATA_DIR ---"
    fi

    DATASET_TOKENS=$((SIZE * TPW_BYTE))
    STEPS_PER_EPOCH=$(( (DATASET_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
    NUM_ITERATIONS=$(( N_EPOCHS * STEPS_PER_EPOCH ))
    [ "$NUM_ITERATIONS" -lt "$MIN_STEPS" ] && NUM_ITERATIONS=$MIN_STEPS

    DST_TAG="${SCRATCH_MODEL_NAME}-${SIZE}w"
    echo "--- TRAIN (from scratch): size=$SIZE num_iterations=$NUM_ITERATIONS tag=$DST_TAG ---"

    NANOCHAT_DATA_DIR="$(basename "$DATA_DIR")" \
        MODEL_TAG="$DST_TAG" \
        NUM_ITERATIONS="$NUM_ITERATIONS" \
        DEPTH="$DEPTH" \
        BATCH="$BATCH" \
        EVAL_EVERY=-1 \
        CORE_METRIC_EVERY=-1 \
        SAMPLE_EVERY=-1 \
        bash runs/cute_scratch.sh

    echo "--- EVAL: size=$SIZE prompt=$PROMPT_STYLE ---"
    EVAL_ARGS=( -m scripts.cute_eval
                --source scratch
                --model-tag "$DST_TAG"
                --mode completion
                --prompt-style "$PROMPT_STYLE"
                --subtasks "$EVAL_SUBTASKS" )
    [ "$EVAL_MAX" != "-1" ] && EVAL_ARGS+=( --max-problems "$EVAL_MAX" )

    EVAL_OUT=$(mktemp)
    .venv/bin/python "${EVAL_ARGS[@]}" 2>&1 | tee "$EVAL_OUT"

    # Append per-subtask rows. "model" column gets SCRATCH_MODEL_NAME so plots
    # can identify these rows distinctly from cute_pt rows (which use the
    # base-model tag in the model column).
    awk -v m="$SCRATCH_MODEL_NAME" -v s="$SIZE" -v ft="$NUM_ITERATIONS" '
        /^[a-z_]+: [0-9]+\/[0-9]+ \([0-9.]+%\)/ {
            sub(":", "", $1)
            split($2, np, "/")
            acc = np[1] / np[2]
            printf "%s,%s,%s,%.6f,%d,%d,%d\n", m, s, $1, acc, np[1], np[2], ft
        }
    ' "$EVAL_OUT" >> "$RESULT_CSV"
    rm -f "$EVAL_OUT"
done

echo "=============================================================="
echo "=== DONE. From-scratch results in $RESULT_CSV ==="
echo "=== Overlay with cute_pt sweep results to compare. ==="
echo "=============================================================="
