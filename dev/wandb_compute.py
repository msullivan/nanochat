"""Compute usage report across wandb projects.

Breaks runs down by project (and optionally by host / GPU config), reporting
wall-clock hours and GPU-hours.

Two metadata gotchas this handles explicitly:

  1. metadata['gpu_count'] is the number of GPUs on the MACHINE, not the
     number the run used. On a 2-GPU box where we run --nproc_per_node=1,
     runtime * gpu_count double-counts. So GPUs-actually-used is resolved per
     host via GPUS_USED below (override as needed), falling back to gpu_count.

  2. metadata['gpu'] can be mis-detected (e.g. the genai box, an RTX 6000 PRO
     Blackwell, reports as "RTX 3090"). We surface a HOST_LABEL mapping for a
     human-meaningful provider/GPU name rather than trusting the string.

Wall-clock (_runtime) is the only fully trustworthy field and is reported
alongside GPU-hours.

Usage:
    python dev/wandb_compute.py                      # all nanochat* projects
    python dev/wandb_compute.py --projects nanochat nanochat-cute
    python dev/wandb_compute.py --by host            # group by host instead
    python dev/wandb_compute.py --per-run            # full per-run table
    python dev/wandb_compute.py --entity NAME --match nanochat
"""
import argparse
from collections import defaultdict

import wandb

ENTITY = "msullivan-emarhavil-heavy-industries"

# Actual GPUs used per host (run-level), since metadata gpu_count is machine-level.
# genai: friend's single-GPU box (RTX 6000 PRO Blackwell), always nproc=1.
# Container-hash hosts are Runpod torchrun jobs that DID use all machine GPUs,
# so for those gpu_count is correct -> leave them to the fallback.
GPUS_USED = {
    "genai.nerdnet.org": 1,
}
# Human-meaningful host -> "provider / gpu" label (metadata gpu string is unreliable).
HOST_LABEL = {
    "genai.nerdnet.org": "friend's box (RTX 6000 PRO Blackwell, 1 GPU)",
    "enki": "local (no GPU / utility runs)",
}


def gpus_used(host, gpu_count):
    if host in GPUS_USED:
        return GPUS_USED[host]
    return gpu_count or 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--projects", nargs="+", default=None,
                    help="explicit project names (default: all matching --match)")
    ap.add_argument("--match", default="nanochat",
                    help="substring projects must contain (default: nanochat)")
    ap.add_argument("--by", default="project", choices=["project", "host", "gpu"],
                    help="primary grouping for the summary table")
    ap.add_argument("--per-run", action="store_true", help="also print every run")
    args = ap.parse_args()

    api = wandb.Api()
    if args.projects:
        proj_names = args.projects
    else:
        proj_names = [p.name for p in api.projects(args.entity)
                      if args.match.lower() in p.name.lower()]

    runs = []
    for proj in proj_names:
        for r in api.runs(f"{args.entity}/{proj}", per_page=500):
            m = r.metadata or {}
            host = m.get("host") or "?"
            gpu_count = int(m.get("gpu_count") or 0)
            used = gpus_used(host, gpu_count)
            rt_h = float(r.summary.get("_runtime") or 0.0) / 3600.0
            runs.append({
                "project": proj, "name": r.name, "state": r.state,
                "host": host, "gpu": m.get("gpu") or "?",
                "machine_gpus": gpu_count, "gpus_used": used,
                "wall_h": rt_h, "gpu_h": rt_h * used,
            })

    keyfn = {"project": lambda x: x["project"],
             "host": lambda x: x["host"],
             "gpu": lambda x: f"{x['gpu']} x{x['machine_gpus']}"}[args.by]

    agg = defaultdict(lambda: [0, 0.0, 0.0])  # key -> [n, wall_h, gpu_h]
    for x in runs:
        a = agg[keyfn(x)]
        a[0] += 1; a[1] += x["wall_h"]; a[2] += x["gpu_h"]

    if args.per_run:
        runs.sort(key=lambda x: -x["gpu_h"])
        print(f"{'project':22} {'run':38} {'host':20} {'used':>4} {'wall_h':>8} {'gpu_h':>8}")
        for x in runs:
            print(f"{x['project']:22} {x['name'][:38]:38} {x['host'][:20]:20} "
                  f"{x['gpus_used']:4d} {x['wall_h']:8.2f} {x['gpu_h']:8.2f}")
        print()

    print(f"=== usage by {args.by} ===")
    print(f"{args.by:30} {'runs':>5} {'wall_h':>9} {'GPU_h':>9}")
    tn = tw = tg = 0
    for k in sorted(agg, key=lambda x: -agg[x][2]):
        a = agg[k]
        print(f"{str(k)[:30]:30} {a[0]:5d} {a[1]:9.1f} {a[2]:9.1f}")
        tn += a[0]; tw += a[1]; tg += a[2]
    print("-" * 56)
    print(f"{'TOTAL':30} {tn:5d} {tw:9.1f} {tg:9.1f}")
    print(f"\nwall-clock: {tw:.1f} h ({tw/24:.1f} days)")
    print(f"GPU-hours : {tg:.1f} h ({tg/24:.1f} GPU-days)")

    # host legend for anything we have a label for
    hosts = {x["host"] for x in runs}
    labeled = [(h, HOST_LABEL[h]) for h in sorted(hosts) if h in HOST_LABEL]
    if labeled:
        print("\nhost legend:")
        for h, lbl in labeled:
            print(f"  {h:22} {lbl}")
    unlabeled = sorted(h for h in hosts if h not in HOST_LABEL and h != "?")
    if unlabeled:
        print(f"  unlabeled hosts (likely Runpod containers): {', '.join(unlabeled)}")


if __name__ == "__main__":
    main()
