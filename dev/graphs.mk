# Regenerate all CUTE-project figures into dev/graphs/.
#
#   make -f dev/graphs.mk            # build everything
#   make -f dev/graphs.mk curve      # one target
#   make -f dev/graphs.mk clean
#
# Run from the repo root. Some targets read local result CSVs; the learning-
# curve and dip targets pull from wandb live; the CORE target scp's base_eval
# CSVs off genai first (override host with GENAI=...).
#
# NOTE: the underlying numbers come from training runs, not from this Makefile;
# these targets only re-render the plots from already-collected data.

PY      := .venv/bin/python3
OUT     := dev/graphs
GENAI   := genai

# local result CSVs (snapshots committed in the repo root)
SFTMASK := results_sft-mask.csv
CURVECSV := dev/curve_data.csv

.PHONY: all clean curve dip sweep mix core coredata
all: curve dip sweep mix core

$(OUT):
	mkdir -p $(OUT)

# Every script writes <OUT>/png/<name>.png and <OUT>/svg/<name>.svg via
# dev/plotsave.py.

# 1. Byte vs BPE matched learning curves (reads dev/curve_data.csv)
curve: | $(OUT)
	$(PY) dev/plot_byte_vs_bpe_curve.py --csv $(CURVECSV) --outdir $(OUT)

# 2. spell_inverse transient dip (pulls dip-probe + curve runs from wandb)
dip: | $(OUT)
	$(PY) dev/plot_dip.py --outdir $(OUT)

# 3. CUTE sweep: per-subtask grid + mean (reads sft-mask results CSV)
sweep: | $(OUT)
	$(PY) dev/plot_cute_sweep.py --input $(SFTMASK) --outdir $(OUT) --tag cute_sweep_sftmask

# 4. mix vs cute_pt bar chart (reads sft-mask results CSV + baked-in mix numbers)
mix: | $(OUT)
	$(PY) dev/plot_mix_vs_cute_pt.py --input $(SFTMASK) --outdir $(OUT)

# 5. CORE base-eval. Mirror genai's base_eval/<tag>/ layout into
# $(OUT)/coredata/<tag>/ so plot_core derives model identity from the subdir
# name (not a step-number lookup). -r preserves the per-model dirs.
coredata:
	mkdir -p $(OUT)/coredata
	scp -qr $(GENAI):'~/.cache/nanochat/base_eval/d24-byte-l-early' $(OUT)/coredata/
	scp -qr $(GENAI):'~/.cache/nanochat/base_eval/d24-byte-l'       $(OUT)/coredata/
	scp -qr $(GENAI):'~/.cache/nanochat/base_eval/d24-byte-l-ext'   $(OUT)/coredata/
	scp -qr $(GENAI):'~/.cache/nanochat/base_eval/d24'              $(OUT)/coredata/

core: coredata | $(OUT)
	$(PY) dev/plot_core.py --indir $(OUT)/coredata --outdir $(OUT)

clean:
	rm -rf $(OUT)/png $(OUT)/svg $(OUT)/coredata
	rm -f $(OUT)/*.png $(OUT)/*.svg
