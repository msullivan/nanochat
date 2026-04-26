"""
One-shot: tag every existing wandb run in the nanochat project with
`step_metric="step"` so dashboards plot against training step instead of
wandb's internal call counter.

Usage:
    uv run python dev/retag_wandb_step_metric.py [--entity ENTITY] [--project nanochat] [--dry-run]

Requires wandb to already be logged in (`wandb login` or WANDB_API_KEY env).
Defaults --entity to your wandb default if not specified.
"""
import argparse
import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default=None, help="wandb entity (default: your username)")
    p.add_argument("--project", default="nanochat")
    p.add_argument("--dry-run", action="store_true", help="list runs, don't modify")
    p.add_argument("--limit", type=int, default=None, help="cap number of runs (debug)")
    args = p.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = list(api.runs(path))
    if args.limit:
        runs = runs[:args.limit]
    print(f"Found {len(runs)} runs in {path}")

    for r in runs:
        print(f"  {r.id}  {r.name}  state={r.state}")
        if args.dry_run:
            continue
        # Resume the run (just to define metrics; logs nothing else).
        with wandb.init(id=r.id, project=args.project, entity=args.entity,
                        resume="must", reinit=True) as run:
            run.define_metric("step")
            run.define_metric("*", step_metric="step")


if __name__ == "__main__":
    main()
