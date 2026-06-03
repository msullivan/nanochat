"""spell_inverse transient-dip figure: dense dip-probe run overlaid on the
byte-l-early curve run. Pulls both from wandb live.
  --outdir DIR   writes <DIR>/{png,svg}/dip_curve.* (default: /tmp)"""
import os
import csv, json
import argparse
import wandb
import matplotlib.pyplot as plt
from plotsave import save_fig

ENT = "msullivan-emarhavil-heavy-industries/nanochat-cute-mix"
TPS = 1_048_576
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--outdir", default="/tmp")
OUTDIR = _ap.parse_args().outdir

def pull(name, metrics):
    api = wandb.Api()
    cands = [r for r in api.runs(ENT, order="-created_at") if r.name == name]
    run = None
    for c in cands:
        h = c.history(keys=["ft_step", metrics[0]], samples=3000)
        if len(h) and metrics[0] in h.columns and h[metrics[0]].notna().any():
            run = c; break
    out = {}
    for m in metrics:
        h = run.history(keys=["ft_step", m], samples=3000)
        out[m] = {int(r["ft_step"]): float(r[m]) for _, r in h.iterrows()
                  if r.get("ft_step") == r.get("ft_step") and r.get(m) == r.get(m)}
    return out

METRICS = ["cute/spell", "cute/spell_inverse"]
dip = pull("d24-byte-l-early-dipprobe", METRICS)
curve = pull("d24-byte-l-early-mix50-300000w-curve", METRICS)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for ax, m, title in zip(axes, METRICS, ["spell", "spell_inverse"]):
    for series, color, marker, lbl in [
        (curve, "#6baed6", "o", "byte-l-early (curve run, sparse)"),
        (dip,   "#08306b", "x", "byte-l-early (dip-probe, dense ft16-32)"),
    ]:
        d = series[m]
        xs = sorted(d)
        ax.plot([x * TPS for x in xs], [d[x] for x in xs],
                marker=marker, color=color, lw=1.5, ms=6, label=lbl)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("finetune tokens (log)")
    ax.set_ylabel("accuracy")
axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle("byte-l-early: spell_inverse transient dip, resolved by dense dip-probe\n"
             "(forward-spell's spaced-output reflex transiently contaminates inverse-spell; recovers by ~ft64)",
             fontsize=12)
save_fig(fig, OUTDIR, "dip_curve", dpi=130)
