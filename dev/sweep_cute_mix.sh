#!/bin/bash
# Sweep mixed CUTE + ClimbMix midtraining over (cute_size × mix_fraction).
#
# Companion to dev/sweep_cute_pt.sh. Same CUTE sizes parametrization
# (so cells overlay onto existing plots), but the model trains on a
# Bernoulli mix of CUTE docs and the general ClimbMix pretraining text,
# with mask_before applied ONLY to the CUTE half. Tests whether mixing
# preserves general capability (CORE) while still teaching CUTE format.
#
# ENV KNOBS
#   SIZES         CUTE word counts (default matches cute_pt sweep).
#   MIX_FRACTIONS Bernoulli probabilities of drawing the CUTE stream per
#                 batch (default: "0.10 0.20 0.50").
#   MODEL         base model tag (default: d24-byte-l-ext). Sweep is
#                 single-model -- the byte-vs-bpe axis can be added by
#                 running the sweep multiple times with MODEL overridden.
#   N_EPOCHS      epochs over the CUTE data per cell (default: 2). Used
#                 only to compute FT_STEPS; the model also sees N_EPOCHS *
#                 (1/MIX_FRACTION - 1) "epochs" of ClimbMix from the
#                 primary stream.
#   FT_LRM        forwarded (default 0.05; cute_mix.sh's default).
#   BATCH         forwarded device-batch-size (default 8).
#   MIN_STEPS     floor on FT_STEPS (default 80). Cells below this don't
#                 even complete LR warmup.
#   MASK_BEFORE   forwarded (default 'Answer: "'; matches cute_pt).
#   FORCE_REGEN   set 1 to regenerate CUTE shards even if present.
#   SKIP_DONE     set 1 (default) to skip cells already in the CSV.
#   RESULT_CSV    default: $NANOCHAT_BASE_DIR/cute_sweep/results_mix.csv
#   PROMPT_STYLE  eval prompt style (default: zero).
#   EVAL_MAX      --max-problems for cute_eval (default: 100; -1 = all).
#   TPW_BYTE      tokens-per-word for byte model (default: 800).
#   TOKENS_PER_STEP  default 1048576 (1MB), matches d24 device-batch=8.
#
# Per-cell flow:
#   1. Ensure CUTE shards exist for this SIZE.
#   2. Seed cute_mix_checkpoints/$DST_TAG from base_checkpoints/$MODEL.
#   3. Run cute_mix.sh with FT_STEPS sized so the model sees
#      ~N_EPOCHS * SIZE*TPW_BYTE CUTE tokens (FT_STEPS scales with
#      1/MIX_FRACTION).
#   4. cute_eval --source mix → append per-subtask rows to RESULT_CSV.
#
# Cells are tagged "${MODEL}-mix${MIX_PCT}-${SIZE}w" in the CSV so plot
# scripts can split by fraction.

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.
export OMP_NUM_THREADS=1

SIZES="${SIZES:-30000 100000 50000 3000 10000 300000 1000 5000}"
MIX_FRACTIONS="${MIX_FRACTIONS:-0.10 0.20 0.50}"
MODEL="${MODEL:-d24-byte-l-ext}"
N_EPOCHS="${N_EPOCHS:-2}"
FT_LRM="${FT_LRM:-0.05}"
BATCH="${BATCH:-8}"
MIN_STEPS="${MIN_STEPS:-80}"
MASK_BEFORE="${MASK_BEFORE:-Answer: \"}"
SKIP_DONE="${SKIP_DONE:-1}"
PROMPT_STYLE="${PROMPT_STYLE:-zero}"
EVAL_MAX="${EVAL_MAX:-100}"
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
RESULT_CSV="${RESULT_CSV:-$BASE_DIR/cute_sweep/results_mix.csv}"
TPW_BYTE="${TPW_BYTE:-800}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-1048576}"

mkdir -p "$(dirname "$RESULT_CSV")"
if [ ! -s "$RESULT_CSV" ]; then
    echo "model,size,subtask,accuracy,n_passed,n_total,ft_steps" > "$RESULT_CSV"
fi
echo "=== RESULT_CSV: $RESULT_CSV ==="

EVAL_SUBTASKS="${SUBTASKS:-char}"

already_done() {
    [ "$SKIP_DONE" = "1" ] || return 1
    local model="$1" size="$2"
    local n
    n=$(awk -F, -v m="$model" -v s="$size" 'NR>1 && $1==m && $2==s {c++} END {print c+0}' "$RESULT_CSV")
    [ "$n" -ge 8 ]
}

# Format mix fraction as integer percent for the model tag (0.10 -> 10).
fraction_pct() {
    .venv/bin/python -c "print(int(round(float('$1') * 100)))"
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

    for MIX in $MIX_FRACTIONS; do
        MIX_PCT=$(fraction_pct "$MIX")
        DST_TAG="${MODEL}-mix${MIX_PCT}-${SIZE}w"

        if already_done "$DST_TAG" "$SIZE"; then
            echo "--- SKIP (already in $RESULT_CSV): $DST_TAG size=$SIZE ---"
            continue
        fi

        DST_DIR="$BASE_DIR/cute_mix_checkpoints/$DST_TAG"
        SRC_DIR="$BASE_DIR/base_checkpoints/$MODEL"

        # Seed dst dir from latest base checkpoint if not already seeded.
        # cute_mix.sh expects model_*.pt + meta_*.json + optim_*_rank0.pt
        # to be present so it can resume.
        if ! ls "$DST_DIR"/meta_*.json >/dev/null 2>&1; then
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

        # FT_STEPS: enough for ~N_EPOCHS of CUTE data given the mix fraction.
        # CUTE_TOKENS_TARGET = SIZE * TPW_BYTE * N_EPOCHS (same exposure as
        # cute_pt 100% sweep at this size). Total steps = target / (per_step
        # tokens * mix_fraction), since only mix_fraction of each step is
        # CUTE. So 10% mix needs ~10x more total steps than the cute_pt cell.
        CUTE_TOKENS_TARGET=$((SIZE * TPW_BYTE * N_EPOCHS))
        FT_STEPS=$(.venv/bin/python -c "
import math
print(max($MIN_STEPS, math.ceil($CUTE_TOKENS_TARGET / ($TOKENS_PER_STEP * $MIX))))
")

        echo "--- TRAIN: tag=$DST_TAG size=$SIZE mix=$MIX (=${MIX_PCT}%) ft_steps=$FT_STEPS ---"

        MIX_DATA_DIR="$(basename "$DATA_DIR")" \
            MODEL_TAG="$DST_TAG" \
            MIX_FRACTION="$MIX" \
            FT_STEPS="$FT_STEPS" \
            FT_LRM="$FT_LRM" \
            MASK_BEFORE="$MASK_BEFORE" \
            EVAL_EVERY=-1 \
            CORE_METRIC_EVERY=-1 \
            SAMPLE_EVERY=-1 \
            bash runs/cute_mix.sh

        echo "--- EVAL: tag=$DST_TAG size=$SIZE prompt=$PROMPT_STYLE ---"
        EVAL_ARGS=( -m scripts.cute_eval
                    --source mix
                    --model-tag "$DST_TAG"
                    --mode completion
                    --prompt-style "$PROMPT_STYLE"
                    --subtasks "$EVAL_SUBTASKS" )
        [ "$EVAL_MAX" != "-1" ] && EVAL_ARGS+=( --max-problems "$EVAL_MAX" )

        EVAL_OUT=$(mktemp)
        .venv/bin/python "${EVAL_ARGS[@]}" 2>&1 | tee "$EVAL_OUT"

        # Append per-subtask rows. "model" column uses the cell tag so plots
        # can split by mix fraction.
        awk -v m="$DST_TAG" -v s="$SIZE" -v ft="$FT_STEPS" '
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
echo "=== DONE. Mixed-recipe results in $RESULT_CSV ==="
echo "=== Overlay with results.csv (cute_pt) to compare. ==="
echo "=============================================================="
