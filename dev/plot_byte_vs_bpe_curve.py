"""Faceted byte-vs-BPE CUTE learning curves, read from dev/curve_data.csv.

Matched recipe (mix50 @ 300k words, LR0.05, WD0.28); only the tokenizer
differs. Run dev/fetch_curve_data.py first to (re)generate the CSV.
One panel per CUTE subtask + the mean + CORE; x = finetune step (log).

Usage:
    python dev/plot_byte_vs_bpe_curve.py [--csv PATH] [--out PATH]
Defaults: --csv dev/curve_data.csv, --out byte_vs_bpe_curve.png (cwd).
"""
import argparse
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", default="dev/curve_data.csv",
                    help="input CSV from fetch_curve_data.py (default: dev/curve_data.csv)")
parser.add_argument("--out", default="byte_vs_bpe_curve.png",
                    help="output image path (default: byte_vs_bpe_curve.png in cwd)")
args = parser.parse_args()

CSV = args.csv
# Match the styling in plot_cute_sweep.py ("CUTE accuracy by subtask × dataset
# size × model") so colors/markers/labels are consistent across CUTE figures:
#   d24-byte-l-ext: dark blue  #08306b, triangle "^"  -> "d24-byte-l-ext"
#   d24 (BPE stock): Wong orange #e69f00, diamond "D" -> "d24-bpe"
MODEL_STYLE = {
    "byte": dict(color="#08306b", marker="^", label="d24-byte-l-ext"),
    "bpe":  dict(color="#e69f00", marker="D", label="d24-bpe"),
}
# Both curve runs used total_batch_size = 1,048,576 tokens/step (confirmed in
# the training logs). x-axis is finetune tokens = ft_step * this. Equal token
# budget for both; on this CUTE data BPE tokens are ~3.0x denser than bytes
# (2.97 vs 1.00 chars/token, measured on 500 docs of shard 0 of
# cute_pt_data_nodemos_300000), so at matched tokens the byte run saw ~3x
# fewer underlying characters.
TOKENS_PER_STEP = 1_048_576

# Layout: 2x5 grid filled row-major. The 8 subtasks fill the left 4 columns
# (top row + bottom row); the right column holds the two summary metrics,
# MEAN (top) above CORE (bottom).
PANELS = [
    ("cute/spell", "spell"),
    ("cute/spell_inverse", "spell_inverse"),
    ("cute/contains_char", "contains_char"),
    ("cute/orth", "orth"),
    ("cute/mean", "MEAN (8 subtasks)"),
    ("cute/ins_char", "ins_char"),
    ("cute/del_char", "del_char"),
    ("cute/sub_char", "sub_char"),
    ("cute/swap_char", "swap_char"),
    ("core_metric", "CORE (in-train, 100/task)"),
]

# data[model][metric] -> list of (ft_tokens, value) sorted by tokens
data = defaultdict(lambda: defaultdict(list))
with open(CSV) as f:
    for r in csv.DictReader(f):
        tokens = int(r["ft_step"]) * TOKENS_PER_STEP
        data[r["model"]][r["metric"]].append((tokens, float(r["value"])))
for m in data:
    for k in data[m]:
        data[m][k].sort()

# 6 columns: 4 subtask cols + a narrow empty spacer + the summary col. The
# spacer gives the divider line a gutter to live in so it doesn't cross any
# tick labels / xlabels. Top band reserved for suptitle + boxed legend.
fig = plt.figure(figsize=(20, 8))
gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.22, 1],
                      top=0.81, bottom=0.085, left=0.035, right=0.99,
                      hspace=0.30, wspace=0.06)
SUBTASK_COLS = [0, 1, 2, 3]   # grid columns that hold the 8 subtasks
SUMMARY_COL = 5               # grid column for mean (row0) / CORE (row1)

# Map each panel to a (row, gridcol): subtasks fill the 4 left cols across
# both rows; mean -> (0, summary), CORE -> (1, summary).
slots = {}
sub = [p for p in PANELS if p[0] not in ("cute/mean", "core_metric")]
for i, p in enumerate(sub):          # 8 subtasks, row-major over 2x4 left block
    slots[(i // 4, SUBTASK_COLS[i % 4])] = p
slots[(0, SUMMARY_COL)] = ("cute/mean", "mean (8 subtasks)")
slots[(1, SUMMARY_COL)] = ("core_metric", "CORE (in-train, 100/task)")

mean_ax = None
for (row, col), (metric, title) in slots.items():
    ax = fig.add_subplot(gs[row, col])
    for model in ("byte", "bpe"):
        st = MODEL_STYLE[model]
        pts = data[model].get(metric, [])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=st["marker"], color=st["color"], lw=2, ms=7, label=st["label"])
    ax.set_xscale("log")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    # 0.5 chance line on the binary tasks, matching plot_cute_sweep.py.
    if metric in ("cute/contains_char", "cute/orth"):
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7,
                   label="chance (binary task)" if metric == "cute/contains_char" else None)
    # Labels/ticklabels only on edges. The summary column is visually detached
    # (spacer + divider) so it gets its own y-label; within each block, x-label
    # only on the bottom row, y-label only on the left edge.
    is_summary = (col == SUMMARY_COL)
    is_left_edge = (col == 0) or is_summary
    is_bottom = (row == 1)
    if is_bottom:
        ax.set_xlabel("finetune tokens (log)")
    else:
        ax.tick_params(labelbottom=False)
    if is_left_edge:
        ax.set_ylabel("accuracy")
    else:
        ax.tick_params(labelleft=False)
    if metric == "cute/mean":
        mean_ax = ax

# Boxed legend in the reserved top band, between suptitle and panels. Collect
# handles across panels so the binary-task "chance" line is included.
seen = {}
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(l, h)
fig.legend(seen.values(), seen.keys(), loc="center", ncol=len(seen),
           frameon=True, fontsize=12, bbox_to_anchor=(0.5, 0.87))

# Divider in the spacer gutter between the subtask block and the summary
# column. constrained_layout finalizes positions only after a draw, so resolve
# layout first, then drop a figure-level vline in the middle of the empty gap.
fig.canvas.draw()
# midpoint of the gutter = between last subtask col's right edge and summary col's left edge
last_sub = [ax for ax in fig.axes if ax.get_subplotspec().colspan.start == 3][0]
x_div = (last_sub.get_position().x1 + mean_ax.get_position().x0) / 2
# span the divider over the actual vertical extent of the panels (bottom of
# the bottom row to top of the top row), read after layout resolves.
y_bot = min(ax.get_position().y0 for ax in fig.axes)
y_top = max(ax.get_position().y1 for ax in fig.axes)
fig.add_artist(plt.Line2D([x_div, x_div], [y_bot, y_top],
                          color="0.6", lw=1.2, ls="--",
                          transform=fig.transFigure))

fig.suptitle(
    "Byte vs BPE — CUTE learning curves, matched recipe "
    "(50% mix of ClimbMix + synthetic CUTE, 300k words, LR 0.05, WD 0.28)\n"
    "On this data BPE tokens are 3.0x denser than bytes.",
    fontsize=14, y=0.97)
plt.savefig(args.out, dpi=130, bbox_inches="tight")
print(f"saved {args.out}")
