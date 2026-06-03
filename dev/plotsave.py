"""Shared figure-saving helper for the dev/ plot scripts.

save_fig(fig, outdir, name) writes both:
    <outdir>/png/<name>.png
    <outdir>/svg/<name>.svg
creating the png/ and svg/ subdirs as needed. Keeps every plot script writing
to the same place in the same two formats (see dev/graphs.mk).
"""
import os


def save_fig(fig, outdir, name, dpi=130):
    paths = []
    for fmt in ("png", "svg"):
        d = os.path.join(outdir, fmt)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{name}.{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"saved {p}")
        paths.append(p)
    return paths
