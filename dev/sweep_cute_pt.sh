#!/bin/bash
# Sweep CUTE midtraining across (dataset size × base model).
#
# Outer loop: dataset size (--num-words). Inner loop: model.
# For each (size, model): regenerate no-demos data → seed cute_checkpoints
# from the base ckpt → run cute_pt for N_EPOCHS over the dataset → run
# zero-shot CUTE eval → append results to a CSV.
#
# Outer-loop-on-size means partial sweeps still produce complete cross-model
# comparisons at every size you've reached, even if you ctrl-C halfway.
#
# ENV KNOBS
#   SIZES        word counts for --num-words (default: "30000 10000 3000 1000",
#                largest first so you see whether the experiment is producing
#                signal at all before burning time on small-data cells; even at
#                1000 the model wasn't scoring on spell at FT_STEPS=2)
#   MODELS       base model tags (default: "d24-byte-l-early d24-byte-l d24-byte-l-ext d24");
#                "d24" is the stock BPE base — directory is just "d24", not "d24-stock"
#   N_EPOCHS     epochs over each dataset (default: 2)
#   FT_LRM       LR multiplier during finetune (default: 0.05)
#   BATCH        device-batch-size for cute_pt (default: 8)
#   MIN_STEPS    floor on FT_STEPS (default: 1). The default-batch (1MB) is larger
#                than the entire dataset for sizes ≲ a few thousand words, so a
#                meaningful "N_EPOCHS=2" can be 1 step at the small end. A higher
#                floor like 50 would silently override the epoch scaling and turn
#                the sweep into a constant-compute one. Raise this if you want
#                that framing instead.
#   FORCE_REGEN  set to 1 to regenerate data shards even if present
#   SKIP_DONE    set to 1 to skip (size, model) pairs that already have a results CSV row
#   RESULT_CSV   results path (default: $NANOCHAT_BASE_DIR/cute_sweep/results.csv)
#   PROMPT_STYLE eval prompt style: zero (default) or fewshot
#   EVAL_MAX     --max-problems for cute_eval (default: 100; -1 = all). Eval
#                is single-prompt sequential on one GPU so each eval costs
#                ~1ms × max_problems × n_subtasks of wall clock; 100 is the
#                sweet spot for sweep curves where shape > precision.

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.
export OMP_NUM_THREADS=1

SIZES="${SIZES:-30000 10000 3000 1000}"
MODELS="${MODELS:-d24-byte-l-early d24-byte-l d24-byte-l-ext d24}"
N_EPOCHS="${N_EPOCHS:-2}"
FT_LRM="${FT_LRM:-0.05}"
BATCH="${BATCH:-8}"
MIN_STEPS="${MIN_STEPS:-1}"
PROMPT_STYLE="${PROMPT_STYLE:-zero}"
EVAL_MAX="${EVAL_MAX:-100}"
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
RESULT_CSV="${RESULT_CSV:-$BASE_DIR/cute_sweep/results.csv}"

mkdir -p "$(dirname "$RESULT_CSV")"
if [ ! -s "$RESULT_CSV" ]; then
    echo "model,size,subtask,accuracy,n_passed,n_total,ft_steps" > "$RESULT_CSV"
fi

# Subtask set for byte models' structural advantage (and what cute_pt
# generates). Override via SUBTASKS env var if you want full 14.
EVAL_SUBTASKS="${SUBTASKS:-char}"

# Auto-detect: bytes per example for no-demos varies a bit by subtask;
# eyeball ~100 tokens per example for byte tokenizer, ~25 for BPE. 8
# examples per word in the generator. Used only to scale FT_STEPS.
tokens_per_word_for() {
    case "$1" in
        *byte*) echo $((8 * 100)) ;;
        *)      echo $((8 * 25)) ;;
    esac
}

# Per-step training tokens for d24-class models is 1MB (1048576) by default
# auto-compute. Override via TOKENS_PER_STEP if a different model size.
TOKENS_PER_STEP="${TOKENS_PER_STEP:-1048576}"

already_done() {
    [ "$SKIP_DONE" = "1" ] || return 1
    local model="$1" size="$2"
    # Look for any row matching this (model, size) — if all 8 char-level
    # subtasks are present we consider it done.
    local n
    n=$(awk -F, -v m="$model" -v s="$size" 'NR>1 && $1==m && $2==s {c++} END {print c+0}' "$RESULT_CSV")
    [ "$n" -ge 8 ]
}

for SIZE in $SIZES; do
    echo "=============================================================="
    echo "=== SIZE=$SIZE ==="
    echo "=============================================================="

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

    for MODEL in $MODELS; do
        if already_done "$MODEL" "$SIZE"; then
            echo "--- SKIP (already in $RESULT_CSV): model=$MODEL size=$SIZE ---"
            continue
        fi

        DST_TAG="${MODEL}-cute-nodemos-${SIZE}w"
        DST_DIR="$BASE_DIR/cute_checkpoints/$DST_TAG"
        SRC_DIR="$BASE_DIR/base_checkpoints/$MODEL"

        # Seed dst dir from latest base checkpoint if not already seeded.
        # cute_pt.sh expects model_*.pt + meta_*.json + optim_*_rank0.pt to
        # be present so it can resume.
        if [ ! -f "$DST_DIR"/meta_*.json 2>/dev/null ] && ! ls "$DST_DIR"/meta_*.json >/dev/null 2>&1; then
            echo "--- seeding $DST_DIR from $SRC_DIR ---"
            mkdir -p "$DST_DIR"
            STEP=$(ls "$SRC_DIR"/model_*.pt 2>/dev/null | sed 's/.*model_0*\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
            if [ -z "$STEP" ]; then
                echo "ERROR: no model_*.pt in $SRC_DIR" >&2
                continue
            fi
            STEP6=$(printf "%06d" "$STEP")
            cp "$SRC_DIR/model_${STEP6}.pt"        "$DST_DIR/"
            cp "$SRC_DIR/meta_${STEP6}.json"       "$DST_DIR/"
            cp "$SRC_DIR/optim_${STEP6}_rank0.pt"  "$DST_DIR/"
        fi

        # FT_STEPS: enough for ~N_EPOCHS over the dataset.
        TPW=$(tokens_per_word_for "$MODEL")
        DATASET_TOKENS=$((SIZE * TPW))
        STEPS_PER_EPOCH=$(( (DATASET_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
        FT_STEPS=$(( N_EPOCHS * STEPS_PER_EPOCH ))
        [ "$FT_STEPS" -lt "$MIN_STEPS" ] && FT_STEPS=$MIN_STEPS

        echo "--- TRAIN: model=$MODEL size=$SIZE ft_steps=$FT_STEPS lrm=$FT_LRM ---"
        # Disable in-training evals; the only eval we care about per iteration
        # is the post-finetune CUTE benchmark run a few lines below.
        NANOCHAT_DATA_DIR="$(basename "$DATA_DIR")" \
            MODEL_TAG="$DST_TAG" \
            FT_STEPS="$FT_STEPS" \
            FT_LRM="$FT_LRM" \
            EVAL_EVERY=-1 \
            CORE_METRIC_EVERY=-1 \
            SAMPLE_EVERY=-1 \
            bash runs/cute_pt.sh

        echo "--- EVAL: model=$MODEL size=$SIZE prompt=$PROMPT_STYLE ---"
        EVAL_ARGS=( -m scripts.cute_eval
                    --source cute
                    --model-tag "$DST_TAG"
                    --mode completion
                    --prompt-style "$PROMPT_STYLE"
                    --subtasks "$EVAL_SUBTASKS" )
        [ "$EVAL_MAX" != "-1" ] && EVAL_ARGS+=( --max-problems "$EVAL_MAX" )

        # Capture per-subtask accuracy lines (format: "subtask: N/M (P%)").
        EVAL_OUT=$(mktemp)
        .venv/bin/python "${EVAL_ARGS[@]}" 2>&1 | tee "$EVAL_OUT"

        # Append rows to the CSV: model,size,subtask,acc,n,total,ft_steps
        awk -v m="$MODEL" -v s="$SIZE" -v ft="$FT_STEPS" '
            /^[a-z_]+: [0-9]+\/[0-9]+ \([0-9.]+%\)/ {
                sub(":", "", $1)
                split($2, np, "/")
                acc = np[1] / np[2]
                printf "%s,%s,%s,%.6f,%d,%d,%d\n", m, s, $1, acc, np[1], np[2], ft
            }
        ' "$EVAL_OUT" >> "$RESULT_CSV"
        rm -f "$EVAL_OUT"
    done
done

echo "=============================================================="
echo "=== DONE. Results in $RESULT_CSV ==="
echo "=============================================================="
