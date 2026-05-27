"""
Distributed dataloader for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

API shape:
   - `_document_batches(...)` -> iterator of (text_batch, info), one source's docs.
   - `attach_mask(stream, mask_before_ids)` -> wraps a stream so each yield
     carries a per-batch mask_before_ids (or None = no mask for these docs).
   - `build_single_stream(...)` -> convenience: parquet source -> stream with
     a fixed mask config.
   - `tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, B, T, streams_with_weights, ...)` -> the packer. Takes a
     list of (stream, weight) tuples; for each row, Bernoulli-picks one
     source and packs the entire row from that source's buffer with
     best-fit. Per-row source selection is the standard production pattern
     for dataset mixing -- mask config is uniform per row, best-fit works
     within-source where doc sizes are comparable, and the realized mix
     ratio exactly matches the weights.
"""

import random
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
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
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
    All docs produced by this stream share the same mask config.
    """
    for text_batch, info in stream:
        yield text_batch, info, mask_before_ids


def build_single_stream(tokenizer, split, resume_state_dict, mask_before=None,
                        tokenizer_batch_size=128, data_dir=None):
    """
    Convenience: build a (text_batch, info, mask_before_ids) stream from one
    parquet source. Tokenizes mask_before once.
    """
    mask_before_ids = None
    if mask_before:
        mask_before_ids = tokenizer.encode(mask_before)
        if not mask_before_ids:
            raise ValueError(f"mask_before {mask_before!r} encoded to empty token sequence")
    base = _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=data_dir)
    return attach_mask(base, mask_before_ids)


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, streams_with_weights,
    tokenizer_threads=4,
    device="cuda",
    buffer_size=1000,
    rng_seed=0,
):
    """
    BOS-aligned dataloader with per-row source selection + within-source best-fit.

    streams_with_weights: list of (stream, weight). Each stream yields
        (text_batch, info, mask_before_ids). Weights normalized internally.
        For single-source use, pass [(stream, 1.0)].

    For each row:
      1. Bernoulli-pick a source by weight.
      2. Pack the row entirely from that source's buffer using best-fit
         (largest-doc-that-fits, crop shortest when nothing fits).
      3. Row's sub-docs all carry the picked source's mask_before_ids.

    Why per-row and not within-row: when mixing sources with very different
    average doc sizes (CUTE Q/A ~80 bytes vs ClimbMix web text ~5000 bytes),
    within-row best-fit systematically picks the larger source and the
    smaller one ends up as crop fodder at row tails -- with mask_before set,
    the marker is chopped off and masking signal is destroyed. Per-row
    selection makes each row internally consistent: best-fit works fine on
    similarly-sized docs, and mask_before is applied uniformly.

    Properties:
    - Every row starts with BOS
    - 100% utilization (no padding)
    - Realized mix ratio matches weights (one Bernoulli per row × B rows
      per batch -> ratio converges quickly)

    state_dict carries per-source position info + RNG state for approximate
    resume. Legacy single-stream state dicts (no "sources" key) are ignored.
    """
    assert len(streams_with_weights) > 0, "need at least one (stream, weight)"
    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()

    streams = [s for s, _ in streams_with_weights]
    weights = [float(w) for _, w in streams_with_weights]
    total_w = sum(weights)
    assert total_w > 0, "weights sum to 0"
    cum = []
    acc = 0.0
    for w in weights:
        acc += w / total_w
        cum.append(acc)

    n_sources = len(streams)
    source_buffers = [[] for _ in range(n_sources)]
    source_last_infos = [None] * n_sources

    rng = random.Random(rng_seed)

    _marker_check_done = False

    def refill(src_idx):
        text_batch, info, mask_before_ids = next(streams[src_idx])
        source_last_infos[src_idx] = info
        token_lists = tokenizer.encode(text_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            source_buffers[src_idx].append((tokens, mask_before_ids))

    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    def pick_source():
        r = rng.random()
        i = 0
        while i < len(cum) - 1 and r > cum[i]:
            i += 1
        return i

    while True:
        # Per-row placements: (start_pos, end_pos, mask_before_ids). Used by
        # the mask pass and (in single-stream mode) also serves as the BOS
        # boundary record without needing to re-scan.
        placements = [[] for _ in range(B)]

        for row_idx in range(B):
            src_idx = pick_source()
            buf = source_buffers[src_idx]

            pos = 0
            while pos < row_capacity:
                while len(buf) < buffer_size:
                    refill(src_idx)

                remaining = row_capacity - pos

                # Best-fit within this row's chosen source.
                best_idx = -1
                best_len = 0
                for i, (doc, _mb) in enumerate(buf):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc, mb = buf.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    placements[row_idx].append((pos, pos + doc_len, mb))
                    pos += doc_len
                else:
                    shortest_idx = min(range(len(buf)), key=lambda i: len(buf[i][0]))
                    doc, mb = buf.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    placements[row_idx].append((pos, pos + remaining, mb))
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # Per-doc loss masking. Each placement carries its source's
        # mask_before_ids; None placements are left untouched. For maskable
        # placements: find the marker subsequence within [s, e). If found,
        # mask targets up to and including the marker. If not found (e.g. a
        # cropped tail), mask the entire sub-doc.
        _n_found = 0
        _n_maskable_subdocs = 0
        for ri in range(B):
            row_list = None
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

        # state_dict carries per-source position info. For single-source the
        # consumer can still use the legacy top-level pq_idx/rg_idx/epoch via
        # the convenience copy below (mirrors source 0).
        state_dict = {
            "sources": [
                {"pq_idx": info[0], "rg_idx": info[1], "epoch": info[2]}
                if isinstance(info, tuple) and len(info) == 3 else {"info": info}
                for info in source_last_infos
            ],
            "rng_state": rng.getstate(),
        }
        s0 = source_last_infos[0]
        if isinstance(s0, tuple) and len(s0) == 3:
            state_dict["pq_idx"] = s0[0]
            state_dict["rg_idx"] = s0[1]
            state_dict["epoch"] = s0[2]

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
