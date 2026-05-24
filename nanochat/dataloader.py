"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    mask_before=None,
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    # Per-document loss masking. If mask_before is given, every sub-document in
    # a packed row has its targets set to -1 (ignore_index) up to and including
    # the tokenized form of mask_before. Lets cute_pt-style runs train only on
    # the answer portion of each Q/A document, mirroring SFT's assistant-only
    # loss mask. Sub-docs that don't contain the marker (e.g., cropped tails)
    # have their entire target region masked.
    mask_before_ids = None
    if mask_before:
        mask_before_ids = tokenizer.encode(mask_before)
        if not mask_before_ids:
            raise ValueError(f"mask_before {mask_before!r} encoded to empty token sequence")
    # Sanity-check counters: BPE tokenizes `mask_before` *standalone*, but the
    # marker is searched for in the *in-context* tokenization of packed docs.
    # If BPE merges differ across those two settings (e.g. trailing space gets
    # absorbed into the next token), the marker never matches and the whole
    # sub-doc gets masked → zero training signal. Surface a loud one-shot
    # warning on the first batch if the marker-not-found rate is high.
    _marker_check_done = False

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # Per-doc loss masking: for each row, find sub-document boundaries
        # (BOS tokens) and the mask_before marker within each sub-doc, then
        # set cpu_targets positions that "predict" the prompt region to -1.
        # Applied AFTER the inputs/targets split so we don't corrupt inputs.
        if mask_before_ids is not None:
            _mb_len = len(mask_before_ids)
            _n_found = 0
            _n_subdocs = 0
            for ri in range(B):
                row_list = row_buffer[ri].tolist()
                bos_positions = [i for i, t in enumerate(row_list) if t == bos_token]
                bos_positions.append(row_capacity)  # sentinel for last sub-doc end
                for k in range(len(bos_positions) - 1):
                    s = bos_positions[k]
                    e = bos_positions[k + 1]  # exclusive
                    # Locate mask_before_ids within row_list[s:e]
                    marker_pos = -1
                    limit = e - _mb_len + 1
                    for j in range(s, limit):
                        if row_list[j:j + _mb_len] == mask_before_ids:
                            marker_pos = j
                            break
                    _n_subdocs += 1
                    if marker_pos >= 0:
                        _n_found += 1
                    # Prompt region in row coordinates: [s, mask_end).
                    # mask_end = marker_pos + _mb_len if found, else e (mask
                    # the whole sub-doc -- happens for cropped tails that
                    # don't contain the marker).
                    mask_end = marker_pos + _mb_len if marker_pos >= 0 else e
                    # target index t corresponds to row_buffer[ri, t+1].
                    # Prompt input positions [s, mask_end) → target indices
                    # [s-1, mask_end-1), clamped to >=0 for the first sub-doc.
                    t_start = max(s - 1, 0)
                    t_end = mask_end - 1  # exclusive
                    if t_end > t_start:
                        cpu_targets[ri, t_start:t_end] = -1
            if not _marker_check_done:
                _marker_check_done = True
                _rate = _n_found / max(1, _n_subdocs)
                if _rate < 0.5:
                    print(f"!!! WARNING: mask_before marker {mask_before!r} -> "
                          f"{mask_before_ids} found in only {_n_found}/{_n_subdocs} "
                          f"({_rate:.0%}) sub-docs of first batch. Likely "
                          f"BPE in-context vs standalone tokenization mismatch "
                          f"(e.g. trailing space merging with following token). "
                          f"Most/all training targets will be masked to -1 -> "
                          f"zero gradient signal. Check that mask_before "
                          f"tokenizes the same way in real packed docs.")
                else:
                    print(f"[dataloader] mask_before marker found in "
                          f"{_n_found}/{_n_subdocs} ({_rate:.0%}) sub-docs of first batch")

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
