import wandb
api = wandb.Api()
ent = "msullivan-emarhavil-heavy-industries"
def util(r):
    try: h=r.history(stream="events", samples=5000)
    except Exception: return {}
    out={}
    for c in h.columns:
        if c.startswith("system.gpu.") and c.endswith(".gpu"):
            i=c.split(".")[2]
            s=h[c].dropna()
            if len(s): out[i]=(round(float(s.mean())), round(float(s.max())))
    return out
for proj in ["nanochat","nanochat-cute-mix"]:
    for r in api.runs(f"{ent}/{proj}", per_page=500):
        m=r.metadata or {}
        if m.get("host")!="genai.nerdnet.org": continue
        rt=float(r.summary.get("_runtime") or 0)/3600
        if rt<0.5: continue
        u=util(r)  # idx -> (mean, max)
        d0=u.get('0',('-','-')); d1=u.get('1',('-','-'))
        print("%-32s wall=%5.1fh  3090(dev0) mean/max=%s/%s   6000(dev1) mean/max=%s/%s"%(
            r.name[:32], rt, d0[0],d0[1], d1[0],d1[1]))
