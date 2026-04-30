"""Move all but the most recent N checkpoints from a nanochat checkpoint dir
to a target directory.

A "checkpoint" is the set of files for a given training step:
    model_{step:06d}.pt
    meta_{step:06d}.json
    optim_{step:06d}_rank{rank:d}.pt   (one per rank)

Default is dry-run; pass --execute to actually copy/delete.
"""
import argparse
import os
import re
import shutil
import sys
from collections import defaultdict


STEP_RE = re.compile(r"^(?:model|meta|optim)_(\d+)(?:_rank\d+)?\.(?:pt|json)$")


def group_by_step(checkpoint_dir):
    by_step = defaultdict(list)
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            continue
        m = STEP_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        by_step[step].append(name)
    return by_step


def human_size(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PiB"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="nanochat checkpoint directory (contains model_*.pt files)")
    p.add_argument("target", help="directory to move old checkpoints into")
    p.add_argument("-k", "--keep", type=int, default=3, help="number of most recent checkpoints to keep (default: 3)")
    p.add_argument("--execute", action="store_true", help="actually perform the operation (default: dry-run)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--copy-only", action="store_true", help="copy to target but do not delete from source")
    mode.add_argument("--prune-only", action="store_true", help="delete from source but do not copy to target")
    args = p.parse_args()

    if not os.path.isdir(args.source):
        sys.exit(f"error: source is not a directory: {args.source}")
    if args.keep < 0:
        sys.exit(f"error: --keep must be >= 0, got {args.keep}")

    by_step = group_by_step(args.source)
    if not by_step:
        sys.exit(f"error: no checkpoint files (model_*.pt / meta_*.json / optim_*.pt) found in {args.source}")

    steps_sorted = sorted(by_step.keys())
    keep_steps = set(steps_sorted[-args.keep:]) if args.keep > 0 else set()
    move_steps = [s for s in steps_sorted if s not in keep_steps]

    do_copy = not args.prune_only
    do_delete = not args.copy_only
    verb = "MOVE" if (do_copy and do_delete) else ("COPY" if do_copy else "PRUNE")
    prefix = "[dry-run] " if not args.execute else ""

    print(f"{prefix}source: {args.source}")
    print(f"{prefix}target: {args.target}")
    print(f"{prefix}found {len(steps_sorted)} checkpoints: {steps_sorted}")
    print(f"{prefix}keeping {sorted(keep_steps)}; will {verb.lower()} {move_steps}")

    if not move_steps:
        print(f"{prefix}nothing to do")
        return

    if args.execute and do_copy:
        os.makedirs(args.target, exist_ok=True)

    total_bytes = 0
    for step in move_steps:
        files = sorted(by_step[step])
        step_bytes = 0
        for name in files:
            src = os.path.join(args.source, name)
            dst = os.path.join(args.target, name)
            size = os.path.getsize(src)
            step_bytes += size
            print(f"{prefix}  {verb} {name} ({human_size(size)})")
            if not args.execute:
                continue
            if do_copy:
                if os.path.exists(dst):
                    if os.path.getsize(dst) != size:
                        sys.exit(f"error: {dst} exists with different size; aborting")
                    print(f"{prefix}    target already present with matching size, skipping copy")
                else:
                    tmp = dst + ".part"
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
            if do_delete:
                os.remove(src)
        total_bytes += step_bytes
        print(f"{prefix}  step {step}: {human_size(step_bytes)}")

    print(f"{prefix}total: {human_size(total_bytes)} across {len(move_steps)} checkpoints")
    if not args.execute:
        print("(dry-run; pass --execute to apply)")


if __name__ == "__main__":
    main()
