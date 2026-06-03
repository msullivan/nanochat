import wandb, json
api = wandb.Api()
ent = "msullivan-emarhavil-heavy-industries"
cands = [x for x in api.runs(f"{ent}/nanochat-cute-mix", order="-created_at")[:12]
         if x.name == "d24-byte-l-early-dipprobe"]
r = None
for c in cands:
    h = c.history(keys=["ft_step", "cute/spell_inverse"], samples=2000)
    if len(h) and "cute/spell_inverse" in h.columns and h["cute/spell_inverse"].notna().any():
        r = c
        break
if r is None:
    print(json.dumps({"found": False}))
    raise SystemExit
def col(name):
    h = r.history(keys=["ft_step", name], samples=2000)
    return {int(row["ft_step"]): float(row[name]) for _, row in h.iterrows()
            if row.get("ft_step") == row.get("ft_step") and row.get(name) == row.get(name)}
si = col("cute/spell_inverse")
sp = col("cute/spell")
# trough = ft with min spell_inverse (ignore ft<8 warmup where everything is ~0)
cand = {k: v for k, v in si.items() if k >= 8}
trough_ft = min(cand, key=cand.get)
print(json.dumps({
    "found": True, "state": r.state,
    "spell_inverse": si, "spell": sp,
    "trough_ft": trough_ft, "trough_val": si[trough_ft],
    "trough_abs_step": 5500 + trough_ft,
}, indent=0))
