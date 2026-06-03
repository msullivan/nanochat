"""
Plot CUTE sweep results from a results CSV produced by dev/sweep_cute_pt.sh.

Two figures are produced:
1. 8-panel grid: per-subtask accuracy vs training-words, one line per model
2. mean across char subtasks vs training-words, one line per model

Usage:
    python dev/plot_cute_sweep.py \
        --input ~/.cache/nanochat/cute_sweep/results.csv \
        --outdir /tmp

The input CSV is expected to have columns: model, size, subtask, accuracy.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter
from plotsave import save_fig

# Model order: most-pretrained byte -> least-pretrained byte -> BPE stock.
# Palette is colorblind-friendly: a sequential blue ramp for the three byte
# variants (light = less pretrained, dark = more pretrained) plus Wong-orange
# for BPE -- distinguishable by hue from any blue under all common forms of
# colorblindness (deuteranopia, protanopia, tritanopia). Each model also gets
# a distinct marker shape so the plot remains readable in B&W print and for
# anyone who sees the blues collapse together.
MODEL_ORDER = ["d24-byte-l-ext", "d24-byte-l", "d24-byte-l-early", "d24"]
MODEL_COLORS = {
    "d24-byte-l-ext":   "#08306b",   # very dark blue (most pretrained byte)
    "d24-byte-l":       "#2171b5",   # medium blue
    "d24-byte-l-early": "#6baed6",   # light blue (least pretrained byte)
    "d24":              "#e69f00",   # Wong orange (BPE stock)
}
MODEL_MARKERS = {
    "d24-byte-l-ext":   "^",   # triangle
    "d24-byte-l":       "s",   # square
    "d24-byte-l-early": "o",   # circle
    "d24":              "D",   # diamond
}
# Display labels for the legend; the underlying CSV still uses "d24" for the
# BPE stock model since that's the on-disk checkpoint directory name.
MODEL_LABELS = {
    "d24": "d24-bpe",
}
SUBTASKS = ["spell", "spell_inverse", "contains_char", "orth",
            "ins_char", "del_char", "sub_char", "swap_char"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/tmp/cute_sweep_results.csv",
                        help="path to results CSV (model,size,subtask,accuracy)")
    parser.add_argument("--outdir", default="/tmp",
                        help="directory to write PNGs into")
    parser.add_argument("--tag", default="cute_sweep",
                        help="prefix for the two output PNG filenames")
    args = parser.parse_args()

    df = pd.read_csv(os.path.expanduser(args.input))
    os.makedirs(args.outdir, exist_ok=True)

    # Figure 1: 8-panel grid showing per-subtask accuracy across (size, model).
    # Layout strategy: reserve explicit vertical space at top via tight_layout
    # rect for suptitle + legend (constrained_layout's "outside" legend mode
    # silently eats the suptitle), and only render axis labels on the outer
    # edges so the inner panels aren't repeating "training words"/"accuracy".
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=True)
    for i, (ax, subtask) in enumerate(zip(axes.flat, SUBTASKS)):
        for model in MODEL_ORDER:
            sub = df[(df["subtask"] == subtask) & (df["model"] == model)].sort_values("size")
            if len(sub) == 0:
                continue
            ax.plot(sub["size"], sub["accuracy"],
                    marker=MODEL_MARKERS[model], linestyle="-",
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS.get(model, model),
                    markersize=8, linewidth=2)
        ax.set_xscale("log")
        ax.set_title(subtask, fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        row, col = divmod(i, 4)
        if row == axes.shape[0] - 1:
            ax.set_xlabel("training words")
        if col == 0:
            ax.set_ylabel("accuracy")
        if subtask in ("contains_char", "orth"):
            ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7,
                       label="chance (binary task)" if subtask == "contains_char" else None)
        # Show plain integer tick labels (5×10^4 style is hard to scan)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Collect handles from every subplot so the "chance" line (only on the
    # binary-task subplots) makes it into the figure-level legend.
    seen = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    # Reserve top ~10% of figure for suptitle (top 3%) + legend strip (next 5%).
    fig.suptitle("CUTE accuracy by subtask × dataset size × model "
                 "(zero-shot eval, no-demos training)",
                 fontsize=15, y=0.985)
    fig.legend(seen.values(), seen.keys(),
               loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=len(seen), frameon=True, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, args.outdir, f"{args.tag}_per_subtask", dpi=120)

    # Figure 2: mean across char subtasks, one line per model.
    fig2, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    mean_by_cell = df.groupby(["model", "size"])["accuracy"].mean().reset_index()
    for model in MODEL_ORDER:
        sub = mean_by_cell[mean_by_cell["model"] == model].sort_values("size")
        if len(sub) == 0:
            continue
        ax2.plot(sub["size"], sub["accuracy"],
                 marker=MODEL_MARKERS[model], linestyle="-",
                 color=MODEL_COLORS[model],
                 label=MODEL_LABELS.get(model, model),
                 markersize=10, linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("training words")
    ax2.set_ylabel("mean accuracy (char subtasks)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("CUTE mean accuracy: model × dataset size")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.legend(loc="upper left")
    save_fig(fig2, args.outdir, f"{args.tag}_mean", dpi=120)

    print("\n=== mean per cell (char subtasks) ===")
    pivot = mean_by_cell.pivot(index="model", columns="size", values="accuracy")
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
