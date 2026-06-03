"""Tally total compute (FLOPs + wall-clock) across the nanochat-cute* wandb
projects. base_train logs total_training_flops and total_training_time; every
run also has _runtime (wall-clock seconds) in its summary.

Writes a per-run table + totals to stdout. Pass --projects to override.
"""
import argparse
import wandb

ENTITY = "msullivan-emarhavil-heavy-industries"
DEFAULT_PROJECTS = ["nanochat-cute-mix", "nanochat-cute", "nanochat-cute-scratch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS)
    args = ap.parse_args()

    api = wandb.Api()
    rows = []
    for proj in args.projects:
        try:
            runs = api.runs(f"{ENTITY}/{proj}")
        except Exception as e:
            print(f"# skip {proj}: {e}")
            continue
        for r in runs:
            s = r.summary
            rows.append({
                "project": proj,
                "name": r.name,
                "state": r.state,
                "runtime_s": float(s.get("_runtime") or 0.0),
                "flops": float(s.get("total_training_flops") or 0.0),
                "train_s": float(s.get("total_training_time") or 0.0),
            })

    rows.sort(key=lambda x: -x["runtime_s"])
    print(f"{'project':22} {'run':40} {'state':9} {'wall_h':>8} {'train_h':>8} {'PFLOPs':>10}")
    tot_rt = tot_fl = tot_tr = 0.0
    for x in rows:
        tot_rt += x["runtime_s"]; tot_fl += x["flops"]; tot_tr += x["train_s"]
        print(f"{x['project']:22} {x['name'][:40]:40} {x['state']:9} "
              f"{x['runtime_s']/3600:8.2f} {x['train_s']/3600:8.2f} {x['flops']/1e15:10.2f}")
    print("-" * 100)
    print(f"{'TOTAL':22} {len(rows)} runs{'':33} {'':9} "
          f"{tot_rt/3600:8.2f} {tot_tr/3600:8.2f} {tot_fl/1e15:10.2f}")
    print()
    print(f"total wall-clock : {tot_rt/3600:.1f} h  ({tot_rt/86400:.2f} days)")
    print(f"total train-time : {tot_tr/3600:.1f} h  (excludes eval/idle)")
    print(f"total FLOPs      : {tot_fl:.3e}  ({tot_fl/1e15:.1f} PFLOP, {tot_fl/1e18:.3f} EFLOP)")


if __name__ == "__main__":
    main()
