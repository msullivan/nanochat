"""
Merge an N-rank DDP optimizer checkpoint (optim_<step>_rank{0..N-1}.pt) into a
single-rank file (optim_<step>_rank0.pt) so the run can be resumed on a
machine with fewer GPUs (typically 1).

What's actually sharded by nanochat's DistMuonAdamW (nanochat/optim.py):

    AdamW group, 'small' params (numel < 1024):
        all_reduce on grad, state replicated identically on every rank.
        -> take rank-0's tensor verbatim.

    AdamW group, 'large' params (numel >= 1024):
        reduce_scatter on grad, each rank stores exp_avg / exp_avg_sq of
        shape (p.shape[0]/world_size, *rest). Rank N owns rows
        N*rank_size .. (N+1)*rank_size.
        -> concatenate along dim 0.

    Muon group:
        Each group has K matrices of identical shape, divided across ranks
        in chunks of ceil(K/world_size) (last chunk zero-padded). State is
        stored on the group's FIRST param: momentum_buffer of shape
        (chunk_size, *p.shape), and a similarly-shaped second_momentum_buffer.
        -> concatenate along dim 0, then truncate to K rows (drops padding).

Detection without needing the model file: for any per-param tensor in the
optimizer state, compare across shards. If all shards have the same value
the tensor is replicated; otherwise it is sharded along dim 0.

After merge, the destination machine reconstructs the optimizer with
world_size=1, which expects:
    AdamW large param state: (full_first_dim, *rest)  -- matches concat
    Muon group state:        (K, *p.shape)            -- matches concat-then-truncate
    everything else:         identical to single-rank zeros_like(p) shape

USAGE
    uv run python dev/merge_optim_shards.py \\
        --src $NANOCHAT_BASE_DIR/base_checkpoints/d24-byte-l-ext \\
        --step 23900 \\
        --world-size 8

    By default the merged shard is written back into --src as
    optim_<step>_rank0.pt (with the original rank0 backed up as
    optim_<step>_rank0.pt.pre-merge unless --in-place-no-backup).
"""

import argparse
import copy
import os
import shutil
import torch


def merge_shards(shards):
    """Merge a list of per-rank optimizer state dicts into a single rank-0
    state dict, using rank-0's skeleton (param_groups, scalar fields,
    replicated tensors). See module docstring for what gets concatenated
    vs taken verbatim."""
    world_size = len(shards)
    out = copy.deepcopy(shards[0])

    pid_to_group = {}
    for g in out["param_groups"]:
        for pid in g["params"]:
            pid_to_group[pid] = g

    n_concat = 0
    n_replicated = 0
    n_truncated = 0
    for pid, pstate in out["state"].items():
        group = pid_to_group[pid]
        for key, t in list(pstate.items()):
            if not isinstance(t, torch.Tensor):
                continue  # 'step' counter, etc -- already correct from rank 0
            shard_tensors = [shards[r]["state"][pid][key] for r in range(world_size)]
            if all(torch.equal(shard_tensors[0], st) for st in shard_tensors[1:]):
                n_replicated += 1
                continue  # replicated; rank-0 copy is fine
            merged = torch.cat(shard_tensors, dim=0)
            if group.get("kind") == "muon":
                k = len(group["params"])
                if merged.shape[0] != k:
                    merged = merged[:k]
                    n_truncated += 1
            pstate[key] = merged
            n_concat += 1
    return out, n_concat, n_replicated, n_truncated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="checkpoint dir holding optim_<step>_rank{0..N-1}.pt")
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int, help="number of source ranks (e.g. 8)")
    parser.add_argument("--dst", default=None, help="destination dir (defaults to --src)")
    parser.add_argument("--in-place-no-backup", action="store_true",
                        help="overwrite optim_<step>_rank0.pt without backing up the original")
    args = parser.parse_args()

    src = args.src
    dst = args.dst or src
    step_str = f"{args.step:06d}"

    paths = [os.path.join(src, f"optim_{step_str}_rank{r}.pt") for r in range(args.world_size)]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    print(f"loading {args.world_size} shards from {src} step={step_str}...")
    shards = [torch.load(p, map_location="cpu") for p in paths]

    out, n_concat, n_replicated, n_truncated = merge_shards(shards)
    print(f"  concatenated tensors:    {n_concat}")
    print(f"  replicated tensors:      {n_replicated}  (took rank 0's copy)")
    print(f"  muon-padding truncated:  {n_truncated}")

    os.makedirs(dst, exist_ok=True)
    dst_path = os.path.join(dst, f"optim_{step_str}_rank0.pt")
    if dst == src and os.path.exists(dst_path) and not args.in_place_no_backup:
        backup = dst_path + ".pre-merge"
        if not os.path.exists(backup):
            shutil.copy2(dst_path, backup)
            print(f"  backed up original rank-0 shard to {backup}")
    tmp = dst_path + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, dst_path)
    sz = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"wrote {dst_path}  ({sz:.1f} MB)")
    print()
    print("To resume on a 1-GPU machine, point base_train at this dir with --resume-from-step={}".format(args.step))
    print("(model_<step>.pt and meta_<step>.json are unchanged -- already single-rank)")


if __name__ == "__main__":
    main()
