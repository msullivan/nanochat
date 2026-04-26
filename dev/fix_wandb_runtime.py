"""
One-shot fix-up: my dev/retag_wandb_step_metric.py script accidentally
inflated wandb's `_runtime` on every run by re-init'ing each one (each
resume session counts as more runtime).

This script restores `_runtime` from the `total_training_time` field that
base_train logs, which is the actual training wall-clock and was unaffected.

Usage:
    uv run python dev/fix_wandb_runtime.py [--entity ENTITY] [--project nanochat] [--dry-run]
"""
import argparse
import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default=None)
    p.add_argument("--project", default="nanochat")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = list(api.runs(path))
    if args.limit:
        runs = runs[:args.limit]
    print(f"Found {len(runs)} runs in {path}")

    for r in runs:
        # scan_history is cheaper than history() for one column.
        max_ttt = None
        for row in r.scan_history(keys=["total_training_time"]):
            v = row.get("total_training_time")
            if v is not None and (max_ttt is None or v > max_ttt):
                max_ttt = v
        current = r.summary.get("_runtime")
        if max_ttt is None:
            print(f"  {r.id} {r.name}: no total_training_time logged, skipping (current _runtime={current})")
            continue
        delta = (current - max_ttt) if current is not None else None
        print(f"  {r.id} {r.name}: _runtime {current} -> {max_ttt:.1f}  (drop {delta:.1f}s)" if delta is not None
              else f"  {r.id} {r.name}: _runtime -> {max_ttt:.1f}")
        if args.dry_run:
            continue
        r.summary["_runtime"] = max_ttt
        r.summary.update()


if __name__ == "__main__":
    main()
