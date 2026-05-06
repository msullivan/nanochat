"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import random

from jinja2 import Template
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


def _build_item_rows(idx, data, tokenizer, model, task_meta):
    """Render prompts + tokenize for a single item, returning (tokens, start, end, gold).
    Tokens are truncated to model.max_seq_len if needed (with index adjustment).
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Truncate to model.max_seq_len; shift indices down accordingly.
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0
                assert e - num_to_crop >= 0
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    return tokens, start_idxs, end_idxs, item.get('gold')


@torch.no_grad()
def evaluate_examples(indices, model, tokenizer, data, device, task_meta):
    """Evaluate a chunk of examples in a single padded forward.

    All candidate sequences across all items are concatenated into one batch,
    right-padded to the longest sequence, and run through the model in one
    pass. Per-item scoring (loss-min for MC/schema, argmax-equality for LM)
    happens after the forward via per-row (start, end) indices and a row->item
    mapping. Returns a list[bool] aligned with `indices`.
    """
    if not indices:
        return []

    task_type = task_meta['task_type']

    all_tokens, all_start, all_end = [], [], []
    item_row_ranges = []   # (row_start, row_end_exclusive) per item
    item_golds = []        # one entry per item
    for idx in indices:
        tokens, start_idxs, end_idxs, gold = _build_item_rows(idx, data, tokenizer, model, task_meta)
        row_start = len(all_tokens)
        all_tokens.extend(tokens)
        all_start.extend(start_idxs)
        all_end.extend(end_idxs)
        item_row_ranges.append((row_start, len(all_tokens)))
        item_golds.append(gold)

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(all_tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, input_ids)

    results = []
    for sub_i, (rs, re) in enumerate(item_row_ranges):
        if task_type == 'language_modeling':
            assert re - rs == 1, "LM task is expected to have exactly one row per item"
            si, ei = all_start[rs], all_end[rs]
            predicted_tokens = predictions[rs, si - 1:ei - 1]
            actual_tokens = input_ids[rs, si:ei]
            is_correct = torch.all(predicted_tokens == actual_tokens).item()
        elif task_type in ('multiple_choice', 'schema'):
            mean_losses = [losses[r, all_start[r] - 1:all_end[r] - 1].mean().item()
                           for r in range(rs, re)]
            pred_idx = mean_losses.index(min(mean_losses))
            is_correct = pred_idx == item_golds[sub_i]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        results.append(is_correct)
    return results


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Single-example shim around evaluate_examples (back-compat)."""
    return evaluate_examples([idx], model, tokenizer, data, device, task_meta)[0]


def evaluate_task(model, tokenizer, data, device, task_meta, batch_items=8):
    """
    Evaluate one task across many examples. Distributes examples across ranks
    (if torchrun) and processes each rank's slice in chunks of `batch_items`
    items per forward (see evaluate_examples). batch_items=1 reproduces the
    old per-example behavior. Memory scales roughly with batch_items × T_max
    × vocab_size, so byte models can afford a much larger value than BPE.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    rank_indices = list(range(rank, len(data), world_size))
    for chunk_start in range(0, len(rank_indices), batch_items):
        chunk = rank_indices[chunk_start:chunk_start + batch_items]
        results = evaluate_examples(chunk, model, tokenizer, data, device, task_meta)
        for idx, is_correct in zip(chunk, results):
            correct[idx] = float(is_correct)
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()
