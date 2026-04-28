"""
Resume a base_train run from a checkpoint, either to anneal the LR/WD/momentum
to zero (--mode anneal) or to extend training to a longer total horizon
(--mode extend). Inherits depth, byte_tokenizer, max_seq_len, device_batch_size,
target_param_data_ratio, window_pattern, and fp8 from the source run's saved
meta so you only specify the deltas.

Each argument also reads from a corresponding env var (uppercased), so this
script is drop-in usable from the same env-var calling pattern as
runs/speedrun_byte.sh. CLI flags take precedence over env vars.

Examples:
    # Anneal (CLI):
    uv run python runs/resume.py --source-tag d24-byte --from-step 4500 \\
        --mode anneal

    # Anneal (env-var style):
    SOURCE_TAG=d24-byte FROM_STEP=4500 MODE=anneal uv run python runs/resume.py

    # Extend:
    uv run python runs/resume.py --source-tag d24-byte --from-step 4500 \\
        --mode extend --new-total 10000

See dev/anneal_workflow.md for the design and gotchas.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def env_arg(name, type_=str, default=None, required=False):
    """argparse kwargs with env var as fallback. Precedence: CLI > env > default.

    - If env var is set, parsed-and-typed value becomes the default.
    - If env unset and `default` is given, that's used.
    - If env unset, no default, and required=True, argparse will require the flag.
    """
    if name in os.environ:
        return {"default": type_(os.environ[name]), "type": type_}
    if default is not None:
        return {"default": default, "type": type_}
    if required:
        return {"required": True, "type": type_}
    return {"default": None, "type": type_}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source-tag", **env_arg("SOURCE_TAG", required=True),
                   help="MODEL_TAG of the source run (env: SOURCE_TAG)")
    p.add_argument("--from-step", **env_arg("FROM_STEP", required=True),
                   help="checkpoint step to resume from, or 'latest' (env: FROM_STEP)")
    p.add_argument("--mode", choices=["anneal", "extend"],
                   **env_arg("MODE", required=True),
                   help="anneal: pure warmdown from FROM_STEP. extend: continue then warmdown. (env: MODE)")
    p.add_argument("--new-total", **env_arg("NEW_TOTAL", type_=int),
                   help="(extend only) new total iteration count (env: NEW_TOTAL)")
    p.add_argument("--warmdown-ratio", **env_arg("WARMDOWN_RATIO", type_=float, default=0.1),
                   help="warmdown fraction of total iterations (env: WARMDOWN_RATIO)")
    p.add_argument("--lr-breakpoints", **env_arg("LR_BREAKPOINTS", default=""),
                   help="piecewise-linear LR breakpoints during the stable phase, "
                        "e.g. '4000:1.0,4200:0.8' (env: LR_BREAKPOINTS). Forwarded to base_train.")
    p.add_argument("--save-every", **env_arg("SAVE_EVERY", type_=int, default=500),
                   help="checkpoint cadence (env: SAVE_EVERY)")
    p.add_argument("--nproc", **env_arg("NPROC_PER_NODE", type_=int, default=8),
                   help="GPUs per node (env: NPROC_PER_NODE)")
    p.add_argument("--output-tag", **env_arg("OUTPUT_TAG"),
                   help="output MODEL_TAG; default derived from mode+source (env: OUTPUT_TAG)")
    p.add_argument("--wandb-run", **env_arg("WANDB_RUN"),
                   help="wandb run name; defaults to output tag (env: WANDB_RUN)")
    p.add_argument("--skip-eval", action="store_true",
                   help="skip the base_eval pass after training")
    args = p.parse_args()

    if args.mode == "extend" and args.new_total is None:
        p.error("--new-total (or NEW_TOTAL env) is required when --mode extend")

    base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat")))
    source_dir = base_dir / "base_checkpoints" / args.source_tag

    # Resolve --from-step latest by scanning the source dir
    if args.from_step == "latest":
        ckpts = sorted(source_dir.glob("model_*.pt"))
        if not ckpts:
            print(f"No checkpoints found in {source_dir}", file=sys.stderr)
            sys.exit(1)
        args.from_step = int(ckpts[-1].name.removeprefix("model_").removesuffix(".pt"))
        print(f"Resolved --from-step latest -> {args.from_step}", file=sys.stderr)
    else:
        args.from_step = int(args.from_step)

    if args.mode == "extend" and args.new_total <= args.from_step:
        p.error(f"--new-total ({args.new_total}) must be > --from-step ({args.from_step})")

    meta_file = source_dir / f"meta_{args.from_step:06d}.json"
    if not meta_file.exists():
        print(f"Source meta not found: {meta_file}", file=sys.stderr)
        for ckpt in sorted(source_dir.glob("model_*.pt"))[:20]:
            print(f"  available: {ckpt.name}", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(meta_file.read_text())
    src = meta["user_config"]

    # Schedule math
    if args.mode == "anneal":
        # Pick anneal_len so warmdown_ratio is the warmdown fraction of (from_step + anneal_len).
        anneal_len = round(args.warmdown_ratio * args.from_step / (1 - args.warmdown_ratio))
        num_iterations = args.from_step + anneal_len
        # Recompute effective ratio after rounding so warmdown_start lands exactly on from_step.
        effective_ratio = anneal_len / num_iterations
        default_tag = f"{args.source_tag}-anneal-from-{args.from_step}"
    else:  # extend
        num_iterations = args.new_total
        effective_ratio = args.warmdown_ratio
        default_tag = f"{args.source_tag}-ext-{args.new_total}"

    output_tag = args.output_tag or default_tag
    wandb_run = args.wandb_run or output_tag

    print(f"=== resume: source={args.source_tag}@{args.from_step} mode={args.mode} -> tag={output_tag} ===")
    print(f"    inherited: depth={src['depth']} ratio={src['target_param_data_ratio']} "
          f"seq_len={src['max_seq_len']} batch={src['device_batch_size']} "
          f"window={src['window_pattern']} byte={src['byte_tokenizer']} fp8={src['fp8']}")
    print(f"    schedule:  num_iterations={num_iterations} warmdown_ratio={effective_ratio:.6f} "
          f"save_every={args.save_every}")

    train_cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={args.nproc}",
        "-m", "scripts.base_train", "--",
        f"--depth={src['depth']}",
        f"--max-seq-len={src['max_seq_len']}",
        f"--device-batch-size={src['device_batch_size']}",
        f"--target-param-data-ratio={src['target_param_data_ratio']}",
        f"--num-iterations={num_iterations}",
        f"--warmdown-ratio={effective_ratio}",
        f"--window-pattern={src['window_pattern']}",
        f"--save-every={args.save_every}",
        f"--resume-from-step={args.from_step}",
        f"--resume-from-tag={args.source_tag}",
        f"--model-tag={output_tag}",
        f"--run={wandb_run}",
    ]
    if src["byte_tokenizer"]:
        train_cmd.append("--byte-tokenizer")
    if src["fp8"]:
        train_cmd.append("--fp8")
    if args.lr_breakpoints:
        train_cmd.append(f"--lr-breakpoints={args.lr_breakpoints}")

    env = os.environ.copy()
    env["NANOCHAT_REPORT_TAG"] = output_tag
    subprocess.run(train_cmd, check=True, env=env)

    if not args.skip_eval:
        eval_cmd = [
            "torchrun", "--standalone", f"--nproc_per_node={args.nproc}",
            "-m", "scripts.base_eval", "--",
            f"--device-batch-size={src['device_batch_size']}",
            f"--model-tag={output_tag}",
        ]
        subprocess.run(eval_cmd, check=True, env=env)


if __name__ == "__main__":
    main()
