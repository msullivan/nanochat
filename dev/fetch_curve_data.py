"""Download matched byte-vs-BPE CUTE learning-curve data from wandb to CSV.

Two runs, identical recipe (mix50 @ 300k words, LR0.05, WD0.28), only the
tokenizer differs:
  BPE : d24-mix50-300000w-curve
  BYTE: d24-byte-l-ext-mix50-300000w-curve

In-training evals: full 8-subtask CUTE @ 100 problems + CORE @ 100/task,
log-spaced ft_steps. Writes one tidy CSV: model,ft_step,metric,value.
"""
import csv
import wandb

PROJECT = "msullivan-emarhavil-heavy-industries/nanochat-cute-mix"
RUNS = {
    "bpe":        "d24-mix50-300000w-curve",
    "byte":       "d24-byte-l-ext-mix50-300000w-curve",
    "byte-early": "d24-byte-l-early-mix50-300000w-curve",
}
SUBTASKS = ["spell", "spell_inverse", "contains_char", "orth",
            "ins_char", "del_char", "sub_char", "swap_char"]
METRICS = ["cute/mean"] + [f"cute/{s}" for s in SUBTASKS] + ["core_metric"]

OUT = "dev/curve_data.csv"


def pick_run(api, name):
    """Pick the finished run of this name that actually has cute/mean data."""
    cands = [r for r in api.runs(PROJECT, order="-created_at") if r.name == name]
    for r in cands:
        h = r.history(keys=["ft_step", "cute/mean"], samples=5000)
        if len(h) and "cute/mean" in h.columns and h["cute/mean"].notna().any():
            return r
    raise SystemExit(f"no run with data found for {name!r} (candidates: {[(c.state) for c in cands]})")


def main():
    api = wandb.Api()
    rows = []
    for label, name in RUNS.items():
        r = pick_run(api, name)
        print(f"{label}: {name}  id={r.id}  state={r.state}")
        # one history pull per metric so a NaN in one doesn't drop a whole row
        for m in METRICS:
            h = r.history(keys=["ft_step", m], samples=5000)
            if m not in h.columns:
                print(f"  WARN: {m} absent")
                continue
            for _, row in h.iterrows():
                ft, v = row.get("ft_step"), row.get(m)
                if ft == ft and v == v:  # both non-NaN
                    rows.append((label, name, int(ft), m, float(v)))
    rows.sort(key=lambda x: (x[0], x[3], x[2]))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "run_name", "ft_step", "metric", "value"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
