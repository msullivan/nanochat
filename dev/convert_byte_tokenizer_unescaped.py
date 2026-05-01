"""
One-shot conversion: byte-tokenizer checkpoint, OLD escape scheme -> NEW dedicated-id scheme.

OLD: vocab_size=256. 0x00=BOS, 0x01=ESCAPE, chat specials encoded as
     [ESCAPE, 0x02..0x09]. Embedding tensors are (256, D).
NEW: vocab_size=265 (256 raw bytes + 9 specials at IDs 256..264). The model
     pads vocab to 320 (next multiple of 64). Embedding tensors are (320, D).

Row mapping (so the model behaves nearly identically right after conversion,
without any training):
    NEW rows 0..255  <- OLD rows 0..255           (verbatim; preserves all byte semantics)
    NEW row  256     <- OLD row  0   (BOS)
    NEW rows 257..264 <- OLD rows 2..9 (the chat-special second-bytes; same order
                                         as SPECIAL_TOKENS in byte_tokenizer.py)
    NEW rows 265..319 <- zeros        (vocab padding; unused, sliced off in forward())

This same structure is applied to the model state (wte, lm_head, value_embeds.<i>)
AND to the matching optimizer state slots (exp_avg, exp_avg_sq) -- those are AdamW
groups, so per-param state shape == param shape (modulo DDP sharding, see notes).

Usage (NPROC=1 checkpoints only -- see notes at bottom for DDP shards):

    uv run python dev/convert_byte_tokenizer_unescaped.py \\
        --src $NANOCHAT_BASE_DIR/base_checkpoints/d24-byte-l \\
        --step 9337 \\
        --dst $NANOCHAT_BASE_DIR/base_checkpoints/d24-byte-l-u
"""

import argparse
import json
import os
import shutil
import torch

OLD_VOCAB = 256
NEW_VOCAB = 265
PADDED_VOCAB = 320  # next multiple of 64 from 265

# OLD escape-second-byte values for the 8 chat specials, in NEW-id order:
# new ID 257 (user_start)      <- old escape-byte 0x02
# new ID 258 (user_end)        <- old escape-byte 0x03
# ...
# new ID 264 (output_end)      <- old escape-byte 0x09
OLD_CHAT_SECONDS = [0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]

# These are the names of vocab-coupled tensors in the model state dict.
# wte / lm_head are always present; value_embeds keys are layer-indexed.
def is_vocab_tensor(name, value):
    if not isinstance(value, torch.Tensor):
        return False
    if value.ndim < 2 or value.shape[0] != OLD_VOCAB:
        return False
    return (
        name == "transformer.wte.weight"
        or name == "lm_head.weight"
        or name.startswith("value_embeds.")
    )


def pad_vocab_tensor(t):
    """Take an OLD (256, D) tensor and return a NEW (320, D) tensor with the
    row mapping documented at the top of this file."""
    assert t.shape[0] == OLD_VOCAB, f"expected first dim {OLD_VOCAB}, got {t.shape}"
    out = torch.zeros((PADDED_VOCAB, *t.shape[1:]), dtype=t.dtype, device=t.device)
    out[:OLD_VOCAB] = t                     # bytes 0..255 unchanged
    out[256] = t[0]                          # NEW BOS <- OLD BOS (row 0x00)
    for new_offset, old_row in enumerate(OLD_CHAT_SECONDS):
        out[257 + new_offset] = t[old_row]   # NEW chat-special <- OLD escape-second-byte
    # rows 265..319 stay zero (padding -- the model crops to vocab_size=265 in forward())
    return out


def convert_model(src_path, dst_path):
    print(f"[model] loading {src_path}")
    state = torch.load(src_path, map_location="cpu")
    converted = {}
    for name, value in state.items():
        if is_vocab_tensor(name, value):
            new_value = pad_vocab_tensor(value)
            print(f"  pad {name}: {tuple(value.shape)} -> {tuple(new_value.shape)}  dtype={value.dtype}")
            converted[name] = new_value
        else:
            converted[name] = value
    print(f"[model] writing {dst_path}")
    tmp = dst_path + ".tmp"
    torch.save(converted, tmp)
    os.replace(tmp, dst_path)


def convert_optim(src_path, dst_path):
    """The AdamW optimizer state is a dict of the form:
        {'state': {param_id: {'step': ..., 'exp_avg': T, 'exp_avg_sq': T}, ...},
         'param_groups': [...]}
    Any tensor inside state[*] whose shape[0] == OLD_VOCAB and whose ndim >= 2
    is a vocab-coupled buffer (exp_avg / exp_avg_sq for wte / lm_head /
    value_embeds) and gets the same row-padding as the model tensors.

    Note on DDP: under DDP, AdamW state for "large" params is sliced to
    (shape[0]/world_size, *rest) on each rank. This script assumes
    NPROC=1 -- see the docstring for guidance on multi-rank conversion.
    """
    print(f"[optim] loading {src_path}")
    state = torch.load(src_path, map_location="cpu")
    if "state" not in state:
        # Some older save schemas dump optimizer state differently; just bail.
        raise RuntimeError(f"unexpected optimizer state schema; top-level keys: {list(state.keys())}")
    n_padded = 0
    for pid, pstate in state["state"].items():
        for key, value in list(pstate.items()):
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == OLD_VOCAB:
                new_value = pad_vocab_tensor(value)
                print(f"  pad state[{pid}][{key}]: {tuple(value.shape)} -> {tuple(new_value.shape)}  dtype={value.dtype}")
                pstate[key] = new_value
                n_padded += 1
    print(f"[optim] padded {n_padded} tensor(s); writing {dst_path}")
    tmp = dst_path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, dst_path)


def convert_meta(src_path, dst_path):
    print(f"[meta] loading {src_path}")
    with open(src_path) as f:
        meta = json.load(f)
    if "model_config" not in meta:
        raise RuntimeError(f"meta file missing model_config: {src_path}")
    old_vs = meta["model_config"].get("vocab_size")
    if old_vs != OLD_VOCAB:
        raise RuntimeError(f"meta vocab_size is {old_vs}, expected {OLD_VOCAB} -- not a byte-tokenizer checkpoint?")
    if not meta.get("byte_tokenizer", False):
        print("  WARN: meta does not have byte_tokenizer=True; proceeding anyway")
    meta["model_config"]["vocab_size"] = NEW_VOCAB
    meta.setdefault("byte_tokenizer", True)
    print(f"[meta] vocab_size {OLD_VOCAB} -> {NEW_VOCAB}; writing {dst_path}")
    tmp = dst_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, dst_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source checkpoint dir, e.g. base_checkpoints/d24-byte-l")
    parser.add_argument("--step", required=True, type=int, help="step to convert")
    parser.add_argument("--dst", required=True, help="destination dir; will be created if missing")
    parser.add_argument("--ranks", type=int, default=1, help="number of optim shards (rank files) to convert")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    step_str = f"{args.step:06d}"

    src_model = os.path.join(args.src, f"model_{step_str}.pt")
    dst_model = os.path.join(args.dst, f"model_{step_str}.pt")
    src_meta = os.path.join(args.src, f"meta_{step_str}.json")
    dst_meta = os.path.join(args.dst, f"meta_{step_str}.json")

    if not os.path.exists(src_model):
        raise FileNotFoundError(src_model)
    if not os.path.exists(src_meta):
        raise FileNotFoundError(src_meta)

    # 1) Optim shards (write first; mirrors save_checkpoint ordering -- model file
    #    written LAST is the "checkpoint complete" marker).
    if args.ranks > 1:
        print(f"WARN: --ranks > 1 detected. Only AdamW slice rows 0..255/world_size are present\n"
              f"      on each shard, so this script's pad logic does NOT correctly produce a\n"
              f"      fresh-DDP-compatible optimizer file. Re-shard manually if you need DDP\n"
              f"      resume; for now, the simplest path is to run the resume on world_size=1\n"
              f"      with the rank0 shard only.")
    for rank in range(args.ranks):
        src_optim = os.path.join(args.src, f"optim_{step_str}_rank{rank}.pt")
        dst_optim = os.path.join(args.dst, f"optim_{step_str}_rank{rank}.pt")
        if os.path.exists(src_optim):
            convert_optim(src_optim, dst_optim)
        else:
            print(f"[optim] no shard at {src_optim}; skipping rank {rank}")

    # 2) Meta
    convert_meta(src_meta, dst_meta)

    # 3) Model (LAST -- existence is the checkpoint-complete marker)
    convert_model(src_model, dst_model)

    print(f"\nDone. New checkpoint at {args.dst} step {step_str}.")
    print("Resume with --resume-from-step={} pointing at the new dir/tag.".format(args.step))


if __name__ == "__main__":
    main()
