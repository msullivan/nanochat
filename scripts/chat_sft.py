"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import gc
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import signal
import time
import wandb
import torch
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_token_bytes
from nanochat.byte_tokenizer import ByteTokenizer, get_byte_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state, load_checkpoint, find_last_step
from nanochat.loss_eval import evaluate_bpb
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval
from scripts.cute_eval import run_cute_subtask

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SpellingBee
from tasks.arithmetic import Addition, Multiplication
from tasks.cute import CUTE, CUTE_CHAR_LEVEL
from tasks.cute_chat import CuteChat

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--output-tag", type=str, default=None, help="output tag for SFT checkpoint dir (defaults to --model-tag, then to d{depth})")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes (default: inherit from pretrained checkpoint)
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain)")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain)")
# Optimization (default: inherit from pretrained checkpoint)
parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: inherit from pretrain)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=100, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=2*524288, help="number of tokens to evaluate val loss on (2*524288 = 1.05M, matching the cute_* finetune runs)")
parser.add_argument("--chatcore-every", type=int, default=100, help="evaluate ChatCORE metric every N steps (-1 = disable)")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task for ChatCORE")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task for ChatCORE")
parser.add_argument("--cute-every", type=int, default=100, help="evaluate CUTE char-level accuracy (chat mode, held-out leukas/cute) every N steps (-1 = disable). Logged separately as cute/* -- NOT folded into chatcore_metric.")
parser.add_argument("--cute-max-problems", type=int, default=100, help="max problems per CUTE subtask in the in-training eval")
# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="number of epochs of MMLU in training mixture (teaches Multiple Choice)")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use)")
parser.add_argument("--cute-size", type=int, default=4000, help="examples per CUTE char-level subtask in the mixture (x8 subtasks). Synthetic chat-style char ops; eval words excluded. 0 disables.")
parser.add_argument("--cute-fraction", type=float, default=None, help="if set (0,1), auto-size CuteChat so CUTE is this fraction of the whole mixture (overrides --cute-size). Used for the anneal phase, e.g. 0.5.")
# Save / resume (mirrors base_train.py's pattern)
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--resume-from-step", type=str, default="-1", help="resume SFT from this step (-1 = disable, 'latest' = highest step in resume dir)")
parser.add_argument("--resume-from-tag", type=str, default=None, help="SFT tag to resume from (default: same as --output-tag); use this to resume one run into a fresh output dir")
parser.add_argument("--anneal", action="store_true", help="anneal/consolidation phase: load WEIGHTS (and optim momentum) from the --resume-from-* checkpoint but start a FRESH LR schedule + dataloader at step 0 (instead of continuing). Lets you re-warm the LR and anneal back down on a CUTE-heavy mix. Pair with --cute-fraction, a short --num-iterations, a modest --init-lr-frac, --warmdown-ratio=1.0, --final-lr-frac=0.")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)
# Use the training step we log as the canonical x-axis for all metrics, instead
# of wandb's internal call counter. Lets us log on a non-uniform cadence (e.g.
# eval every 200 steps + train every 10) without misaligning train vs val vs
# chatcore curves.
wandb_run.define_metric("step")
wandb_run.define_metric("*", step_metric="step")

# Flash Attention status
if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.")

# Resume-detection. We always need to load the base first to get the model
# architecture and tokenizer; if resuming, we then overwrite the model weights
# (and optimizer + loop state below) with the in-progress SFT checkpoint.
resuming = args.resume_from_step != "-1"
# Anneal mode reuses the resume path to load WEIGHTS+optim from an SFT checkpoint,
# but starts a FRESH schedule (step 0, fresh dataloader) so we can re-warm the LR
# and anneal down on a CUTE-heavy mix. Loop/dataloader state is NOT restored.
if args.anneal:
    assert resuming, "--anneal requires --resume-from-step (and usually --resume-from-tag) to load weights from"
restore_loop_state = resuming and not args.anneal

# Load the model and tokenizer (from base; SFT weights applied later if resuming)
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
if isinstance(tokenizer, ByteTokenizer):
    token_bytes = get_byte_token_bytes(device=device)
else:
    token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
# Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
# pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
# restore our fresh SFT LRs after loading.
# Skip when resuming from an in-progress SFT checkpoint -- we'll load that
# optimizer state below and don't want to also load the base's optim first.
base_dir = get_base_dir()
if args.load_optimizer and not resuming:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

# If resuming an in-progress SFT run: overwrite the model weights and optimizer
# state with the SFT checkpoint. Loop state and dataloader state are pulled out
# below from the same meta_data.
output_dirname = args.output_tag or args.model_tag or f"d{depth}"
checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
sft_meta_data = None
if resuming:
    resume_dirname = args.resume_from_tag if args.resume_from_tag else output_dirname
    resume_dir = os.path.join(base_dir, "chatsft_checkpoints", resume_dirname)
    if args.resume_from_step == "latest":
        args.resume_from_step = find_last_step(resume_dir)
    else:
        args.resume_from_step = int(args.resume_from_step)
    print0(f"Resuming SFT from step {args.resume_from_step} (source: {resume_dir})")
    sft_model_data, sft_optim_data, sft_meta_data = load_checkpoint(resume_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    # assign=False (copy into existing params) is REQUIRED here: the optimizer was
    # already built above referencing these Parameter objects. assign=True swaps in
    # new Parameter objects, orphaning the optimizer -> grads land on the new params
    # while the optimizer holds the old ones (grad=None at step). Same shapes/dtypes
    # (same arch), so an in-place copy is safe.
    orig_model.load_state_dict(sft_model_data, strict=True, assign=False)
    del sft_model_data
    if sft_optim_data is not None:
        # Save initial_lr from the freshly-built optimizer (set above by setup_optimizer)
        # since load_state_dict will overwrite param_group metadata.
        fresh_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(sft_optim_data)
        del sft_optim_data
        # Restore the schedule-anchor LR; the schedule will then re-apply lrm
        # against the saved-step's progress on the first iteration.
        for group, lr in zip(optimizer.param_groups, fresh_lrs):
            group["lr"] = lr
        print0("Loaded SFT optimizer state from resume checkpoint")
else:
    args.resume_from_step = -1  # normalize for downstream comparisons

# GradScaler for fp16 training (bf16/fp32 don't need it)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# Override the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# SFT data mixture and DataLoader
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
qamis_files = [
    "qamis_conversations__haiku.jsonl",                           # 500 rows, cleanest grounding
    "qamis_conversations__google_gemini-3-flash-preview.jsonl",   # 498 rows
    "qamis_conversations__openai_gpt-5-mini.jsonl",               # 282 rows (partial run)
]
qamis_filepaths = [os.path.join(repo_root, f) for f in qamis_files]
pep827_filepath = os.path.join(repo_root, "pep827_conversations.jsonl")  # 952 rows
non_cute_tasks = [
    SmolTalk(split="train"), # 460K rows of general conversations
    CustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
    CustomJSON(filepath=identity_conversations_filepath), # 2 epochs of these
    *[CustomJSON(filepath=fp) for fp in qamis_filepaths], # ~1280 rows of Qamis D&D setting (across 3 gen models)
    *[CustomJSON(filepath=fp) for fp in qamis_filepaths], # 2 epochs
    CustomJSON(filepath=pep827_filepath), # 952 rows of PEP 827 Q&A
    CustomJSON(filepath=pep827_filepath), # 2 epochs
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch

    # SimpleSpelling dropped -- superseded by CuteChat's 'spell' subtask below.
    SpellingBee(size=5000, split="train"), # 5K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
    Addition(size=10000, split="train"), # 10K rows of Addition (mostly 2-term, some 3/4/5-term)
    Multiplication(size=10000, split="train"), # 10K rows of Multiplication (small direct, larger by partial products)
]

# Chat-style CUTE char-level tasks (synthetic; eval words excluded). One per
# subtask so each can be sized/weighted independently. Terse quoted answers,
# mixed phrasing bank (edit tasks/cute_chat.TEMPLATES). Size is either a fixed
# per-subtask --cute-size, or auto-computed from --cute-fraction to make CUTE
# that fraction of the whole mixture (the anneal phase uses ~0.5).
if args.cute_fraction is not None:
    assert 0.0 < args.cute_fraction < 1.0, "--cute-fraction must be in (0, 1)"
    general_len = sum(len(t) for t in non_cute_tasks)
    cute_total = round(general_len * args.cute_fraction / (1.0 - args.cute_fraction))
    cute_per_subtask = max(1, cute_total // len(CUTE_CHAR_LEVEL))
    print0(f"--cute-fraction={args.cute_fraction}: general={general_len:,} rows -> CuteChat {cute_per_subtask:,}/subtask x{len(CUTE_CHAR_LEVEL)} = {cute_per_subtask*len(CUTE_CHAR_LEVEL):,} CUTE rows")
else:
    cute_per_subtask = args.cute_size
cute_tasks = ([CuteChat(subtask=st, size=cute_per_subtask, split="train") for st in CUTE_CHAR_LEVEL]
              if cute_per_subtask > 0 else [])

train_tasks = non_cute_tasks + cute_tasks
train_dataset = TaskMixture(train_tasks)
print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs}, CUTE {cute_per_subtask*len(CUTE_CHAR_LEVEL) if cute_tasks else 0:,})")
val_dataset = TaskMixture([
    SmolTalk(split="test"), # 24K rows in test set
    MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
    GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
]) # total: 24K + 5.2K + 0.42K ~= 29.6K rows
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = False # we will toggle this to True when we reach the end of the training dataset
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
current_epoch = 1 # track epoch for logging
# Shared mutable state-dict for the train dataloader, updated after each yield.
# Captured into save_checkpoint's meta and restored on resume so we don't replay
# already-consumed conversations.
dataloader_state = {"cursor": ddp_rank, "consumed": ddp_rank, "epoch": 1}
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: list of (token_ids, loss_mask) tuples
    conv_buffer = []
    if split == "train":
        cursor = dataloader_state["cursor"]
        consumed = dataloader_state["consumed"]
        epoch = dataloader_state["epoch"]
    else:
        cursor = ddp_rank
        consumed = ddp_rank
        epoch = 1
    it = 0  # iteration counter

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
                # Note: last_step is now triggered based on consumption, not fetching

    while True:
        rows = []
        mask_rows = []
        row_lengths = []  # Track actual content length (excluding padding) for each row
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                # Ensure buffer has conversations
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size  # Track actual consumption
                else:
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    content_len = len(row)
                    row.extend([bos_token] * remaining)  # Pad with BOS tokens
                    mask_row.extend([0] * remaining)
                    padded = True
                    break  # Row is now full (with padding)

            # Track content length: full row if no padding, otherwise the length before padding
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # Stopping condition to respect num_iterations, if given. NOTE: `it` counts
        # microbatches (dataloader yields), while num_iterations counts OPTIMIZER
        # steps -- so the budget is num_iterations * grad_accum_steps microbatches.
        it += 1
        if args.num_iterations > 0 and it >= args.num_iterations * grad_accum_steps and split == "train":
            last_step = True

        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / (args.num_iterations * grad_accum_steps)
            else:
                approx_progress = consumed / dataset_size
            # Trigger last_step when we've consumed enough (instead of when cursor wraps)
            if consumed >= dataset_size:
                last_step = True

        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        # Apply the loss mask from render_conversation (mask=1 for assistant completions,
        # mask=0 for user prompts, BOS, special tokens, tool outputs). mask[1:] aligns
        # with targets (shifted by 1). Unmasked positions get -1 (ignore_index).
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        # Mask out padding positions in targets (set to -1 = ignore_index)
        # For each row, positions >= (content_length - 1) in targets should be masked
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1

        # Publish the current state for the train loader so save_checkpoint can
        # snapshot it. Val loader doesn't need to (it's rebuilt fresh per eval).
        if split == "train":
            dataloader_state["cursor"] = cursor
            dataloader_state["consumed"] = consumed
            dataloader_state["epoch"] = epoch

        yield inputs, targets

# Restore dataloader state if resuming (but NOT when annealing -- fresh schedule).
# Captured snapshot from the saved meta; the generator reads these on its first iteration.
if restore_loop_state and sft_meta_data is not None:
    saved_dl = sft_meta_data.get("dataloader_state", None)
    if saved_dl is not None:
        dataloader_state.update(saved_dl)
        print0(f"Restored dataloader state: {saved_dl}")

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0 # will go from 0 to 1 over the course of the epoch

# Learning rate schedule (linear warmup, constant, linear warmdown)
# Same shape as base_train but uses progress (0→1) instead of absolute step counts,
# because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
x, y = next(train_loader) # prefetch the very first batch of data
ema_beta = 0.9 # EMA decay factor
val_bpb = None
if restore_loop_state and sft_meta_data is not None:
    step = sft_meta_data["step"]
    val_bpb = sft_meta_data.get("val_bpb")
    loop_state = sft_meta_data.get("loop_state", {})
    min_val_bpb = loop_state.get("min_val_bpb", float("inf"))
    smooth_train_loss = loop_state.get("smooth_train_loss", 0)
    total_training_time = loop_state.get("total_training_time", 0)
    progress = loop_state.get("progress", 0)
    print0(f"Restored loop state: step={step}, min_val_bpb={min_val_bpb:.4f}, total_time={total_training_time/60:.2f}m")
else:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0
    total_training_time = 0

# Graceful save/exit signaling. Two triggers, both checked at the top of the
# main loop:
#   - SIGUSR1 (kill -USR1 <pid>) -> save checkpoint and continue training.
#   - SIGINT (Ctrl-C, OR wandb dashboard "Stop Run") -> save and exit cleanly.
# Multi-rank: SIGUSR1 must be sent to every rank manually; SIGINT is usually
# forwarded by torchrun. A second SIGINT restores default handling (force-quit
# escape hatch if the save itself hangs).
signal_save_requested = False
signal_exit_requested = False
def _save_handler(signum, frame):
    global signal_save_requested
    signal_save_requested = True
    print(f"[rank {ddp_rank}] received SIGUSR1, will save checkpoint at next loop iteration", flush=True)
def _save_and_exit_handler(signum, frame):
    global signal_save_requested, signal_exit_requested
    if signal_exit_requested:
        print(f"[rank {ddp_rank}] second SIGINT received, exiting immediately", flush=True)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt
    signal_save_requested = True
    signal_exit_requested = True
    print(f"[rank {ddp_rank}] received SIGINT (Ctrl-C / wandb Stop), will save and exit at next loop iteration", flush=True)
signal.signal(signal.SIGUSR1, _save_handler)
signal.signal(signal.SIGINT, _save_and_exit_handler)

while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # Multi-rank consensus on exit: if any rank received SIGINT and didn't have
    # it forwarded, broadcast the flag so all ranks save+exit together.
    if is_ddp_initialized():
        flag = torch.tensor([1 if signal_exit_requested else 0], device=device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        if flag.item():
            signal_save_requested = True
            signal_exit_requested = True

    # Save checkpoint: at the end of the run, every save_every steps (excluding
    # the resume step), or whenever SIGUSR1 / SIGINT was requested.
    if last_step or signal_save_requested or (args.save_every > 0 and step > 0 and step != args.resume_from_step and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,
                "byte_tokenizer": isinstance(tokenizer, ByteTokenizer),
                "dataloader_state": dict(dataloader_state),
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "progress": progress,
                },
            },
            rank=ddp_rank,
        )
        signal_save_requested = False

    # If we were asked to exit, do so now that the checkpoint is on disk;
    # skip the (potentially long) eval/sample blocks at this step.
    if signal_exit_requested:
        print0(f"Exiting after save at step {step} (SIGINT / wandb Stop requested)")
        break

    # once in a while: evaluate the val bpb (all ranks participate)
    if step != args.resume_from_step and (last_step or (args.eval_every > 0 and step % args.eval_every == 0)):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: generative evals (all ranks participate). Use the original
    # uncompiled model because the inputs keep changing shape. ChatCORE and CUTE
    # share one Engine so the decode cudagraph is only captured once per eval.
    def _eval_due(every):
        return every > 0 and step != args.resume_from_step and (last_step or (step > 0 and step % every == 0))
    do_chatcore = _eval_due(args.chatcore_every)
    do_cute = _eval_due(args.cute_every)
    if do_chatcore or do_cute:
        model.eval()
        engine = Engine(orig_model, tokenizer)

    chatcore_results = {}
    if do_chatcore:
        all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
        baseline_accuracies = {
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
            'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
        }
        task_results = {}
        for task_name in all_tasks:
            limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
            max_problems = None if limit < 0 else limit  # -1 means no limit
            acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                batch_size=args.device_batch_size, max_problems=max_problems)
            task_results[task_name] = acc
            print0(f"  {task_name}: {100*acc:.2f}%")
        # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
        def centered_mean(tasks):
            return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)
        chatcore = centered_mean(all_tasks)
        chatcore_cat = centered_mean(categorical_tasks)
        print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "chatcore_metric": chatcore,
            "chatcore_cat": chatcore_cat,
            **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
        })

    # CUTE char-level accuracy on the held-out leukas/cute set, chat mode. Logged
    # under cute/* and deliberately kept OUT of chatcore_metric (it would swamp
    # the 6-task average and break comparability with the standard metric).
    # Zero-shot and NO prefill: a chat-tuned model answers the user turn directly,
    # so we don't carry the paper's 4-shot demos or jam `Answer: "` into the prompt.
    if do_cute:
        cute_results = {}
        for subtask in CUTE_CHAR_LEVEL:
            task = CUTE(subtask=subtask, mode="chat", prefill=False, prompt_style="bare")
            num_lenient, num_strict, total = run_cute_subtask(task, tokenizer, engine, max_new_tokens=64,
                                                              max_problems=args.cute_max_problems)
            acc = num_lenient / total if total > 0 else 0.0  # in-training tracks lenient
            cute_results[subtask] = acc
            print0(f"  CUTE/{subtask}: {100*acc:.2f}%")
        cute_mean = sum(cute_results.values()) / len(cute_results)
        print0(f"Step {step:05d} | CUTE mean (chat): {100*cute_mean:.2f}%")
        wandb_run.log({
            "step": step,
            "cute/mean": cute_mean,
            **{f"cute/{subtask}": acc for subtask, acc in cute_results.items()},
        })

    if do_chatcore or do_cute:
        model.train()

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        # Progress for the LR schedule. With an explicit step budget, use the
        # OPTIMIZER-step fraction (approx_progress from the loader is in microbatch
        # units -- it/num_iterations overshoots by grad_accum_steps). Otherwise
        # (full-epoch, num_iterations=-1) fall back to the data-consumed progress.
        if args.num_iterations > 0:
            progress = step / args.num_iterations
        else:
            progress = max(progress, approx_progress) # only increase progress monotonically
    # step the optimizer
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")
    wandb_run.log({
        "step": step,
        "total_training_flops": flops_so_far,
        "total_training_time": total_training_time,
        "train/loss": debiased_smooth_loss,
        "train/lrm": lrm,
        "train/dt": dt,
        "train/tok_per_sec": tok_per_sec,
        "train/mfu": mfu,
        "train/epoch": current_epoch,
    })

    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    if step == 1:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # freeze all currently surviving objects and exclude them from GC
        gc.disable() # disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="SFT", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of iterations": step,
        "DDP world size": ddp_world_size,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
