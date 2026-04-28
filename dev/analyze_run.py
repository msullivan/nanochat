"""Pull BPB / loss curves for the in-progress d24-byte-l run and analyze
descent rate. The d12 baseline runs in this project are too short / partial to
serve as comparisons, so this focuses on the d24-byte-l curve in isolation
plus its closest finished sibling for shape reference.

Usage: uv run python dev/analyze_run.py
"""
import math
import wandb

ENTITY = "msullivan-emarhavil-heavy-industries"
PROJECT = "nanochat"
RUN_ID = "v9hmpzdm"  # d24-byte-l


def fetch(run_id):
    api = wandb.Api()
    r = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    cfg = dict(r.config)
    rows = list(r.scan_history())
    return r, cfg, rows


def col(rows, k):
    return [(row.get("step"), row.get(k)) for row in rows
            if row.get("step") is not None and row.get(k) is not None]


def slope_log_x(xs, ys):
    if len(xs) < 2:
        return None
    lx = [math.log10(max(x, 1)) for x in xs]
    n = len(xs)
    mx = sum(lx) / n
    my = sum(ys) / n
    num = sum((lx[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((lx[i] - mx) ** 2 for i in range(n))
    return num / den if den > 0 else None


def slope_linear(xs, ys):
    if len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    return num / den if den > 0 else None


def smooth(pts, win=20):
    """Trailing window mean of the y-values."""
    out = []
    ys_buf = []
    for x, y in pts:
        ys_buf.append(y)
        if len(ys_buf) > win:
            ys_buf.pop(0)
        out.append((x, sum(ys_buf) / len(ys_buf)))
    return out


def ascii_plot(series, width=72, height=20, ylabel="bpb", x_log=False, y_label_fmt="{:.4f}"):
    all_pts = [(x, y) for _, pts in series for x, y in pts]
    if not all_pts:
        print("(no data)")
        return
    xs_raw = [x for x, _ in all_pts]
    ys = [y for _, y in all_pts]
    if x_log:
        xs_t = [math.log10(max(x, 1)) for x in xs_raw]
    else:
        xs_t = xs_raw
    xmin, xmax = min(xs_t), max(xs_t)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1e-9

    grid = [[" "] * width for _ in range(height)]
    chars = ["o", "x", "*", "+", "."]

    for i, (_, pts) in enumerate(series):
        ch = chars[i % len(chars)]
        for x, y in pts:
            xt = math.log10(max(x, 1)) if x_log else x
            col = int((xt - xmin) / (xmax - xmin) * (width - 1))
            row = height - 1 - int((y - ymin) / (ymax - ymin) * (height - 1))
            if 0 <= row < height and 0 <= col < width:
                grid[row][col] = ch

    print(f"  y: {ylabel}  x: {'log10(step)' if x_log else 'step'}")
    for i, row in enumerate(grid):
        if i == 0:
            label = f"{y_label_fmt.format(ymax):>9} |"
        elif i == height - 1:
            label = f"{y_label_fmt.format(ymin):>9} |"
        else:
            label = "          |"
        print(label + "".join(row))
    print(" " * 10 + "+" + "-" * width)
    if x_log:
        x0, x1 = 10**xmin, 10**xmax
        print(" " * 11 + f"{x0:<{width//2}.0f}{x1:>{width//2}.0f}")
    else:
        print(" " * 11 + f"{xmin:<{width//2}.0f}{xmax:>{width//2}.0f}")
    print()
    for i, (label, _) in enumerate(series):
        print(f"  {chars[i % len(chars)]} = {label}")
    print()


def windowed_slopes(pts, window_steps=400):
    """For each non-overlapping window of `window_steps`, compute BPB drop."""
    if not pts:
        return []
    pts = sorted(pts)
    out = []
    cur_start = pts[0][0]
    cur = []
    for s, b in pts:
        if s - cur_start >= window_steps and cur:
            xs = [p[0] for p in cur]
            ys = [p[1] for p in cur]
            slope = slope_linear(xs, ys)
            out.append((cur_start, cur[-1][0], ys[0], ys[-1],
                        ys[-1] - ys[0], slope))
            cur_start = s
            cur = []
        cur.append((s, b))
    if cur:
        xs = [p[0] for p in cur]
        ys = [p[1] for p in cur]
        slope = slope_linear(xs, ys) if len(xs) > 1 else None
        out.append((cur_start, cur[-1][0], ys[0], ys[-1],
                    ys[-1] - ys[0], slope))
    return out


def main():
    print(f"fetching {RUN_ID}…")
    r, cfg, rows = fetch(RUN_ID)
    print(f"  state: {r.state}")
    print(f"  display name: {r.name}")

    bpb_pts = sorted(set(col(rows, "val/bpb")))
    loss_pts = sorted(col(rows, "train/loss"))
    core_pts = sorted(col(rows, "core_metric"))
    lrm_pts = sorted(col(rows, "train/lrm"))
    last_step = max(s for s, _ in loss_pts)

    print()
    print("=" * 80)
    print("CONFIG (relevant fields)")
    print("=" * 80)
    for k in ["depth", "byte_tokenizer", "window_pattern", "target_param_data_ratio",
              "num_iterations", "warmdown_ratio", "weight_decay", "max_seq_len",
              "device_batch_size", "fp8"]:
        if k in cfg:
            print(f"   {k:30s} = {cfg[k]}")

    print()
    print("=" * 80)
    print("PROGRESS")
    print("=" * 80)
    print(f"   latest step:       {last_step}")
    print(f"   #BPB points:       {len(bpb_pts)}  (latest: {bpb_pts[-1] if bpb_pts else None})")
    print(f"   #loss points:      {len(loss_pts)}")
    print(f"   #CORE points:      {len(core_pts)}")
    if lrm_pts:
        last_lrm = lrm_pts[-1][1]
        print(f"   train/lrm:         {last_lrm:.4f}  (1.0 = stable phase, <1 = warmdown)")

    print()
    print("=" * 80)
    print("BPB BY WINDOW (drop and slope per 400-step window)")
    print("=" * 80)
    print(f"   {'start':>6}  {'end':>6}  {'bpb_start':>9}  {'bpb_end':>9}  "
          f"{'drop':>8}  {'slope (Δbpb/step)':>18}")
    for start, end, b0, b1, d, slope in windowed_slopes(bpb_pts, window_steps=400):
        slope_s = f"{slope:+.6f}" if slope is not None else "    n/a"
        print(f"   {start:>6}  {end:>6}  {b0:>9.4f}  {b1:>9.4f}  "
              f"{d:>+8.4f}  {slope_s:>18}")

    print()
    print("=" * 80)
    print("BPB SLOPE INTERPRETATION")
    print("=" * 80)
    if len(bpb_pts) >= 8:
        # Drop initial warmup steps (BPB > 2 is clearly warmup)
        stable = [(s, b) for s, b in bpb_pts if b < 2 and s >= 200]
        xs = [s for s, _ in stable]
        ys = [b for _, b in stable]
        s_log = slope_log_x(xs, ys)
        # Recent half
        mid = len(stable) // 2
        s_log_recent = slope_log_x([s for s, _ in stable[mid:]],
                                    [b for _, b in stable[mid:]])
        s_log_early = slope_log_x([s for s, _ in stable[:mid]],
                                   [b for _, b in stable[:mid]])
        print(f"   stable-phase BPB slope vs log10(step):")
        print(f"     full stable range (step >=200):   {s_log:+.4f} per decade")
        print(f"     early half:                       {s_log_early:+.4f} per decade")
        print(f"     recent half:                      {s_log_recent:+.4f} per decade")
        print()
        print("   In stable LR, loss should fall ~linearly in log(tokens). For a")
        print("   healthy run on the right side of compute-optimal, recent ≈ early.")
        print(f"   recent/early ratio = {s_log_recent / s_log_early:.2f}")
        print("   <1 means slowing down (descent rate flattening)")
        print("   ~1 means on track")

    print()
    print("=" * 80)
    print("LOSS BY WINDOW (mean over 200-step windows)")
    print("=" * 80)
    if loss_pts:
        # bin loss by 200-step windows
        bins = {}
        for s, l in loss_pts:
            b = (s // 200) * 200
            bins.setdefault(b, []).append(l)
        print(f"   {'window':>10}  {'n':>5}  {'mean_loss':>10}  {'min':>10}  {'max':>10}")
        for b in sorted(bins):
            ls = bins[b]
            print(f"   {b:>5}-{b+199:<4}  {len(ls):>5}  {sum(ls)/len(ls):>10.4f}  "
                  f"{min(ls):>10.4f}  {max(ls):>10.4f}")

    print()
    print("=" * 80)
    print("CORE METRIC PROGRESSION")
    print("=" * 80)
    for s, c in core_pts:
        print(f"   step {s:>5}  CORE {c:.4f}")

    print()
    print("=" * 80)
    print("BPB PLOT (stable phase only, log-x)")
    print("=" * 80)
    stable_bpb = [(s, b) for s, b in bpb_pts if b < 2 and s >= 100]
    ascii_plot([("d24-byte-l val/bpb", stable_bpb)], x_log=True,
               ylabel="val/bpb (zoomed)")

    print("=" * 80)
    print("LOSS PLOT (smoothed, last 1500 steps)")
    print("=" * 80)
    recent_loss = [(s, l) for s, l in loss_pts if s >= max(0, last_step - 1500)]
    smooth_loss = smooth(recent_loss, win=50)
    ascii_plot([("train/loss (50-step trailing mean)", smooth_loss)],
               x_log=False, ylabel="train/loss")


if __name__ == "__main__":
    main()
