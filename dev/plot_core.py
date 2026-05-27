"""
Plot CORE eval results across our four d24 base models.

Reads /tmp/base_model_*.csv (one CSV per model, copied from
genai:~/.cache/nanochat/base_eval/). The CSV filename uses step count
which we map back to model identity per Sully's tip:
  - 5500  -> byte-l-early (byte, shortest pretrain)
  - 5568  -> d24-bpe      (BPE)
  - 10374 -> byte-l       (byte, full speedrun)
  - 23900 -> byte-l-ext   (byte, extended)
"""
import os
import matplotlib.pyplot as plt
import pandas as pd


# Step count -> (display label, color, marker). Same palette as cute plots.
STEP_TO_MODEL = {
    5500:  ("d24-byte-l-early", "#6baed6", "o"),
    10374: ("d24-byte-l",       "#2171b5", "s"),
    23900: ("d24-byte-l-ext",   "#08306b", "^"),
    5568:  ("d24-bpe",          "#e69f00", "D"),
}

# Order in which to display models in plots
MODEL_ORDER = ["d24-byte-l-ext", "d24-byte-l", "d24-byte-l-early", "d24-bpe"]


def load_all(csv_dir="/tmp"):
    """Read all base_model_*.csv files and return per-model task DataFrames."""
    by_model = {}  # label -> DataFrame(task, accuracy, centered)
    for fname in sorted(os.listdir(csv_dir)):
        if not (fname.startswith("base_model_") and fname.endswith(".csv")):
            continue
        step = int(fname[len("base_model_"):-len(".csv")])
        if step not in STEP_TO_MODEL:
            print(f"  skipping {fname}: unknown step {step}")
            continue
        label, _, _ = STEP_TO_MODEL[step]
        df = pd.read_csv(
            os.path.join(csv_dir, fname),
            skipinitialspace=True,
            names=["task", "accuracy", "centered"],
            header=0,
        )
        # Clean whitespace
        df["task"] = df["task"].str.strip()
        by_model[label] = df
    return by_model


def main():
    by_model = load_all("/tmp")
    if not by_model:
        raise RuntimeError("no CSVs found in /tmp/base_model_*.csv")

    print(f"loaded {len(by_model)} models: {sorted(by_model.keys())}")

    # ---------- Figure 1: CORE score across models ----------
    core_rows = []
    for label, df in by_model.items():
        core = df[df["task"] == "CORE"]["centered"].iloc[0]
        core_rows.append((label, core))

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    labels_ordered = [m for m in MODEL_ORDER if m in by_model]
    cores = [dict(core_rows)[m] for m in labels_ordered]
    colors = [STEP_TO_MODEL[step][1] for step in STEP_TO_MODEL
              for _ in [None] if STEP_TO_MODEL[step][0] in labels_ordered]
    # Sort colors to match labels_ordered
    label_to_color = {STEP_TO_MODEL[s][0]: STEP_TO_MODEL[s][1] for s in STEP_TO_MODEL}
    colors = [label_to_color[m] for m in labels_ordered]
    bars = ax.bar(labels_ordered, cores, color=colors)
    for bar, v in zip(bars, cores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("CORE score (centered, mean across 22 tasks)")
    ax.set_title("CORE: base-eval across the four d24 base models")
    ax.set_ylim(0, max(cores) * 1.12)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig("/tmp/core_summary.png", dpi=120, bbox_inches="tight")
    print("saved /tmp/core_summary.png")
    print("\n=== CORE scores ===")
    for label in labels_ordered:
        print(f"  {label:20} {dict(core_rows)[label]:.4f}")

    # ---------- Figure 2: per-task centered accuracy, horizontal grouped bars ----------
    # Per-task pivot: rows = tasks (excluding CORE), columns = models
    all_tasks = list(by_model[labels_ordered[0]]["task"])
    all_tasks = [t for t in all_tasks if t != "CORE"]
    n_tasks = len(all_tasks)
    n_models = len(labels_ordered)

    fig, ax = plt.subplots(figsize=(11, max(8, 0.32 * n_tasks)), constrained_layout=True)
    bar_h = 0.8 / n_models
    y_positions = list(range(n_tasks))
    for i, label in enumerate(labels_ordered):
        df = by_model[label]
        df_indexed = df.set_index("task")
        vals = [df_indexed.loc[t, "centered"] for t in all_tasks]
        offsets = [y - (n_models - 1) * bar_h / 2 + i * bar_h for y in y_positions]
        ax.barh(offsets, vals, height=bar_h, label=label, color=label_to_color[label])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_tasks)
    ax.invert_yaxis()
    ax.set_xlabel("centered accuracy (0 = random baseline)")
    ax.set_title("CORE per-task centered accuracy across d24 base models")
    ax.axvline(0, color="gray", linewidth=0.7, alpha=0.7)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.savefig("/tmp/core_per_task.png", dpi=120, bbox_inches="tight")
    print("saved /tmp/core_per_task.png")


if __name__ == "__main__":
    main()
