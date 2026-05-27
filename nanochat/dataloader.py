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

API shape:
   - `_document_batches(...)` -> iterator of (text_batch, info), one source's docs.
   - `attach_mask(stream, mask_before_ids)` -> wraps a stream so each yield
     carries a per-batch mask_before_ids (or None = no mask for these docs).
   - `merge_streams(streams_with_weights, rng)` -> Bernoulli-merges several such
     streams. The merged stream's items carry whichever source's mask_before_ids
     came along.
   - `tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, B, T,
     doc_stream, ...)` -> the single packer. Takes any stream of
     (text_batch, info, mask_before_ids) yields; doesn't care if the underlying
     source is one parquet dir or a merged mix of several.
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=None):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).

    data_dir overrides the default (NANOCHAT_DATA_DIR / climbmix). Used by callers
    that want a stream from a non-default location (e.g. the mix stream in a
    merged-source training run).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files(data_dir=data_dir)
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


def attach_mask(stream, mask_before_ids):
    """
    Wraps a (text_batch, info) stream so each yield additionally carries the
    per-batch mask_before_ids (a list of token IDs, or None for no mask).
    All docs produced by this stream share the same mask config -- the packer
    later uses it to mark which sub-docs to apply loss-masking to.
    """
    for text_batch, info in stream:
        yield text_batch, info, mask_before_ids


def merge_streams(streams_with_weights, rng):
    """
    Merge several (text_batch, info, mask_before_ids) streams by per-batch
    Bernoulli sampling. Each yield is whichever item the picked source
    produced next, so the consumed batch's mask_before_ids comes along
    naturally. info is wrapped as {source_index: info} so the consumer can
    track per-source position for resume; last-seen info per source is
    accumulated, not overwritten on a non-selected source.

    streams_with_weights: list of (stream, weight). Weights are normalised
    internally so they don't have to sum to 1.
    """
    streams = [s for s, _ in streams_with_weights]
    weights = [w for _, w in streams_with_weights]
    total = sum(weights)
    assert total > 0, "merge_streams: weights sum to 0"
    cum = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)
    last_infos = [None] * len(streams)
    while True:
        r = rng.random()
        # linear scan -- list is tiny (2 streams in practice)
        i = 0
        while i < len(cum) - 1 and r > cum[i]:
            i += 1
        text_batch, info, mask_before_ids = next(streams[i])
        last_infos[i] = info
        merged_info = {str(j): inf for j, inf in enumerate(last_infos)}
        yield text_batch, merged_info, mask_before_ids


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, doc_stream,
    tokenizer_threads=4,
    device="cuda",
    buffer_size=1000,
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

    doc_stream: iterator yielding (text_batch, info, mask_before_ids) tuples.
       - text_batch: list[str] of raw documents to tokenize+pack.
       - info: opaque per-batch state (passed back out in state_dict so
         the caller can resume).
       - mask_before_ids: list[int] | None. If non-None, every doc in this
         batch will have its loss-targets masked up to and including the
         first occurrence of this subsequence in the packed row. None = no
         masking for this batch's docs (e.g. general pretraining text in a
         mixed run).

    Each yielded doc carries its batch's mask_before_ids through the packer
    so a single merged stream can interleave masked and unmasked docs.
    """
    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()
    # Each buffered doc is (tokens, mask_before_ids). mask_before_ids may be None.
    doc_buffer = []
    last_info = None

    # Sanity-check counters: BPE tokenizes `mask_before` *standalone*, but the
    # marker is searched for in the *in-context* tokenization of packed docs.
    # If BPE merges differ across those two settings (e.g. trailing space gets
    # absorbed into the next token), the marker never matches and the whole
    # sub-doc gets masked -> zero training signal. Surface a loud one-shot
    # warning on the first batch where any maskable docs were present.
    _marker_check_done = False

    def refill_buffer():
        nonlocal last_info
        text_batch, info, mask_before_ids = next(doc_stream)
        last_info = info
        token_lists = tokenizer.encode(text_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append((tokens, mask_before_ids))

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        # Per-row placement records: (start_pos, end_pos, mask_before_ids). The
        # mask pass uses these instead of re-scanning for BOS so each sub-doc
        # gets exactly its origin batch's mask config.
        placements = [[] for _ in range(B)]

        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely (origin-agnostic best-fit).
                best_idx = -1
                best_len = 0
                for i, (doc, _mb) in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc, mb = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    placements[row_idx].append((pos, pos + doc_len, mb))
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining.
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i][0]))
                    doc, mb = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    placements[row_idx].append((pos, pos + remaining, mb))
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # Per-doc loss masking. Each placement carries its batch's
        # mask_before_ids; None placements are left untouched. For maskable
        # placements: find the marker subsequence within [s, e). If found,
        # mask targets up to and including the marker. If not found (e.g. a
        # cropped tail), mask the entire sub-doc.
        _n_found = 0
        _n_maskable_subdocs = 0
        for ri in range(B):
            row_list = None  # lazy: only materialize per row when needed
            for (s, e, mb) in placements[ri]:
                if mb is None:
                    continue
                if row_list is None:
                    row_list = row_buffer[ri].tolist()
                _n_maskable_subdocs += 1
                _mb_len = len(mb)
                marker_pos = -1
                limit = e - _mb_len + 1
                for j in range(s, limit):
                    if row_list[j:j + _mb_len] == mb:
                        marker_pos = j
                        break
                if marker_pos >= 0:
                    _n_found += 1
                mask_end = marker_pos + _mb_len if marker_pos >= 0 else e
                t_start = max(s - 1, 0)
                t_end = mask_end - 1
                if t_end > t_start:
                    cpu_targets[ri, t_start:t_end] = -1
        if not _marker_check_done and _n_maskable_subdocs > 0:
            _marker_check_done = True
            _rate = _n_found / _n_maskable_subdocs
            if _rate < 0.5:
                print(f"!!! WARNING: mask_before marker found in only "
                      f"{_n_found}/{_n_maskable_subdocs} ({_rate:.0%}) "
                      f"maskable sub-docs of first batch. Likely BPE in-context "
                      f"vs standalone tokenization mismatch. Most maskable "
                      f"targets will be -1 -> zero gradient signal on those docs.")
            else:
                print(f"[dataloader] mask_before marker found in "
                      f"{_n_found}/{_n_maskable_subdocs} ({_rate:.0%}) "
                      f"maskable sub-docs of first batch")

        # state_dict is whatever info the stream last gave us. For a single
        # _document_batches stream this is (pq_idx, rg_idx, epoch); for a
        # merged stream it's a dict keyed by source index. We also surface
        # the legacy top-level pq_idx/rg_idx/epoch keys when present so
        # base_train's progress-line logging keeps working without a format
        # check (mirroring whichever source produced the most recent batch).
        state_dict = {"info": last_info}
        if isinstance(last_info, tuple) and len(last_info) == 3:
            state_dict["pq_idx"] = last_info[0]
            state_dict["rg_idx"] = last_info[1]
            state_dict["epoch"] = last_info[2]
        elif isinstance(last_info, dict):
            # merged stream: surface source 0's position as the top-level summary
            primary_info = last_info.get("0")
            if isinstance(primary_info, tuple) and len(primary_info) == 3:
                state_dict["pq_idx"] = primary_info[0]
                state_dict["rg_idx"] = primary_info[1]
                state_dict["epoch"] = primary_info[2]

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets


def build_single_stream(tokenizer, split, resume_state_dict, mask_before=None,
                        tokenizer_batch_size=128, data_dir=None):
    """
    Convenience: build a (text_batch, info, mask_before_ids) stream from one
    parquet source. Tokenizes mask_before once. Use this when you have a
    single data dir; for multi-source mixes use merge_streams over several
    of these.
    """
    mask_before_ids = None
    if mask_before:
        mask_before_ids = tokenizer.encode(mask_before)
        if not mask_before_ids:
            raise ValueError(f"mask_before {mask_before!r} encoded to empty token sequence")
    base = _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=data_dir)
    return attach_mask(base, mask_before_ids)
