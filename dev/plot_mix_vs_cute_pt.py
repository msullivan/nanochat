"""Compare d24-byte-l-ext at 30k CUTE words: cute_pt (sft-mask) vs cute_mix at three ratios."""
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

SUBTASKS = ["spell", "spell_inverse", "contains_char", "orth",
            "ins_char", "del_char", "sub_char", "swap_char"]

# Local CSVs have the cute_pt 30k row; cute_mix rows pulled from genai earlier.
CUTE_MIX = {
    "mix10": {"spell": 1.00, "spell_inverse": 1.00, "contains_char": 1.00, "orth": 0.74,
              "ins_char": 1.00, "del_char": 0.99, "sub_char": 1.00, "swap_char": 1.00},
    "mix20": {"spell": 0.99, "spell_inverse": 1.00, "contains_char": 0.98, "orth": 0.77,
              "ins_char": 1.00, "del_char": 1.00, "sub_char": 1.00, "swap_char": 1.00},
    "mix50": {"spell": 1.00, "spell_inverse": 0.96, "contains_char": 0.61, "orth": 0.73,
              "ins_char": 0.99, "del_char": 0.97, "sub_char": 1.00, "swap_char": 1.00},
}
CORES = {"mix10": 0.250, "mix20": 0.243, "mix50": 0.242}
BASE_CORE = 0.237

def read_30k_row(path, model):
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["model"] == model and r["size"] == "30000":
                out[r["subtask"]] = float(r["accuracy"])
    return out

cute_pt = read_30k_row("results_sft-mask.csv", "d24-byte-l-ext")

recipes = [("cute_pt (sft-mask)", cute_pt, None),
           ("mix10 (10% CUTE)", CUTE_MIX["mix10"], CORES["mix10"]),
           ("mix20 (20% CUTE)", CUTE_MIX["mix20"], CORES["mix20"]),
           ("mix50 (50% CUTE)", CUTE_MIX["mix50"], CORES["mix50"])]

# Colorblind palette (Wong)
colors = ["#0072B2", "#009E73", "#F0E442", "#D55E00"]

fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
x = np.arange(len(SUBTASKS))
width = 0.2
for i, (name, scores, core) in enumerate(recipes):
    vals = [scores.get(s, 0) for s in SUBTASKS]
    label = name if core is None else f"{name}  (CORE={core:.3f})"
    ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(SUBTASKS, rotation=20, ha="right")
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="yes/no random baseline")
ax.set_title(f"d24-byte-l-ext @ 30k CUTE words: cute_pt vs cute_mix\n(base d24-byte-l-ext CORE = {BASE_CORE:.3f})")
ax.legend(loc="lower left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.savefig("/tmp/mix_vs_cute_pt.png", dpi=140)
print("saved /tmp/mix_vs_cute_pt.png")
