"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type, print0, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache(nn.Module):
    """
    KV Cache as an nn.Module with all state held in registered buffers.

    Why nn.Module: when this cache is attached to the model (via
    GPT.setup_caches), its buffers are part of the model's state. PyTorch's
    torch.compile mode="reduce-overhead" then treats cache mutations as
    state-of-the-graph rather than as mutations to external inputs (the
    latter blocks cudagraph capture). Matches the pattern Meta's gpt-fast
    reference uses for fast PyTorch-native autoregressive decode.

    All position-related book-keeping is external: callers pass input_pos
    (the absolute positions to write into) to model.forward and the cache
    is updated via fancy indexing. No internal cache_seqlens / .item() ->
    no host syncs in the captured region.

    Layout: (n_layers, B, T_max, H_kv, D), matching the FA3-friendly
    (B, T, H, D) per-layer slice that flash_attn_with_kvcache wants.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, dtype, n_embd):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.n_embd = n_embd
        # Register buffers so they're tracked as module state. Get device
        # from the parent module's parameters at forward time (buffers
        # auto-move when the module is .to(device)'d).
        self.register_buffer("k_cache",
            torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, dtype=dtype),
            persistent=False)
        self.register_buffer("v_cache",
            torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, dtype=dtype),
            persistent=False)
        # Smear's "prev embedding" (model-level state, not per-layer).
        self.register_buffer("prev_embedding",
            torch.zeros(batch_size, 1, n_embd, dtype=dtype),
            persistent=False)
        # Bigram-VE's "prev token id" (model-level state).
        self.register_buffer("prev_token_id",
            torch.zeros(batch_size, dtype=torch.long),
            persistent=False)

    def reset(self):
        """Reset transient state at the start of a new generation.

        k_cache/v_cache don't need clearing -- they'll be overwritten by
        prefill at positions [0, prompt_len). prev_embedding and prev_token_id
        are read on the first forward before they're written by prefill,
        so zero them defensively (prefill will overwrite the last-position
        slot anyway).
        """
        self.prev_embedding.zero_()
        self.prev_token_id.zero_()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer, use_cuda_graphs=True):
        """
        use_cuda_graphs: if True (default on CUDA), wrap the per-decode-step forward
        in torch.compile(mode="reduce-overhead"). That fuses kernels (Inductor) AND
        captures them into a CUDA graph (cudagraph_trees) -- removing both GPU work
        and per-step launch overhead. On d24 this measured ~1.35x faster than a
        hand-rolled CUDA graph over eager (unfused) kernels, with far less code, and
        the same compiled step serves single-stream (B=1) and batched (B, +mask)
        decode -- the mask is just a forward input cudagraph_trees copies in.

        Decode is a fixed shape so it's graph-friendly; prefill stays eager (variable
        prompt length would force per-shape recompiles). reduce-overhead must be the
        ONLY graphing mechanism on the model -- mixing a manual torch.cuda.CUDAGraph
        with cudagraph_trees on the same cache poisons both (catastrophic re-record).
        """
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
        self.use_cuda_graphs = use_cuda_graphs and torch.cuda.is_available()

        def _decode_step(ids, input_pos, attention_mask=None):
            # Single decode step -> last-position logits (B, vocab). Compiled with
            # reduce-overhead on CUDA; plain eager otherwise (CPU / disabled).
            return self.model.forward(ids, input_pos=input_pos, attention_mask=attention_mask)[:, -1, :]

        if self.use_cuda_graphs:
            self._decode = torch.compile(_decode_step, mode="reduce-overhead", dynamic=False)
        else:
            self._decode = _decode_step

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Single prefill + decode loop with kv cache attached to the model.

        The model's kv_cache is set up once via model.setup_caches() (idempotent
        across calls; resets transient state). input_pos is maintained
        externally and incremented per decode step -- no host syncs inside
        forward, which means torch.compile mode="reduce-overhead" can capture
        the decode forward into a cudagraph automatically.
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        # NOTE: cuda -> bfloat16 and everything else -> float32 is a repo-wide
        # assumption; encoded here to allocate the kv cache at the right dtype.
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Special tokens for tool-use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        def _as_list(tok):
            return [tok] if isinstance(tok, int) else list(tok)
        def _tail_matches(tokens, special):
            s = _as_list(special)
            return len(tokens) >= len(s) and tokens[-len(s):] == s
        def _force_special(state, special):
            state.forced_tokens.extend(_as_list(special))

        # 1) Set up the KV cache as model state. Standardize on the model's
        # full sequence_len so the cache is allocated once on the first call
        # and reused across all subsequent generate()s -- the captured
        # cudagraph references this cache's tensor addresses, so any
        # reallocation between calls invalidates it and forces recapture.
        # The trade is a bigger cache for short-prompt evals (~24MB at d24)
        # but constant addresses across prompts.
        max_seq_length = self.model.config.sequence_len
        self.model.setup_caches(batch_size=num_samples, max_seq_length=max_seq_length, dtype=dtype)

        # 2) Prefill at batch=num_samples (prompt duplicated across batch dim).
        # For num_samples=1 (cute_eval) no waste; for num_samples>1 we trade a
        # bit of redundant prefill compute for a simpler graph-capturable
        # decode path (single cache, no batch-1 -> batch-N copy).
        prompt_len = len(tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=device).expand(num_samples, -1).contiguous()
        input_pos = torch.arange(prompt_len, device=device, dtype=torch.long)
        logits = self.model.forward(ids, input_pos=input_pos)[:, -1, :]  # (num_samples, vocab_size)

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Decode loop. self._decode is the (compiled) single-step forward;
        # cudagraph_trees manages the static graph buffers internally, so we just
        # hand it fresh tensors each step.
        cur_pos = prompt_len
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            # Sample from the current logits
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # State machine for forced tokens, special-token detection, tool use
            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if _tail_matches(state.current_tokens, assistant_end) or next_token == bos:
                    state.completed = True
                if _tail_matches(state.current_tokens, python_start):
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif _tail_matches(state.current_tokens, python_end) and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            _force_special(state, output_start)
                            state.forced_tokens.extend(result_tokens)
                            _force_special(state, output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            # Feed the next tokens to the (compiled) decode step.
            ids_step = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self._decode(ids_step, torch.tensor([cur_pos], device=device, dtype=torch.long))
            cur_pos += 1

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        assistant_end_list = [assistant_end] if isinstance(assistant_end, int) else list(assistant_end)
        ae_len = len(assistant_end_list)
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    results[i].append(token)
                    masks[i].append(mask)
                    # Check for end conditions
                    if token == bos or results[i][-ae_len:] == assistant_end_list:
                        # Strip the terminal tokens
                        results[i] = results[i][:-ae_len] if results[i][-ae_len:] == assistant_end_list else results[i][:-1]
                        masks[i] = masks[i][:-ae_len] if len(masks[i]) >= ae_len else masks[i][:-1]
                        completed[i] = True
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks

    @torch.inference_mode()
    def generate_batched(self, prompts, max_tokens, temperature=0.0, top_k=None, seed=42):
        """Batched generation over DISTINCT prompts via left-padding + KV cache.

        prompts: list[list[int]]. Returns list[list[int]] of generated tokens per
        prompt (prompt and the terminal eos token excluded).

        Prompts are left-padded to a common length so every row's next-token
        position lines up; a per-row key-padding mask (passed to the model as
        attention_mask) stops queries from attending to the pad columns, and the
        smear boundary is handled inside GPT.forward. Decoding is batched: one
        (compiled) forward per step over all rows, with per-row EOS stopping. The
        same self._decode serves single-stream and batched -- the mask is just an
        input the compiled cudagraph copies in.

        Eval-oriented: no tool-use / forced-token state machine (use generate()
        for that).
        """
        assert isinstance(prompts, list) and all(isinstance(p, list) for p in prompts)
        device = self.model.get_device()
        dtype = COMPUTE_DTYPE  # honor NANOCHAT_DTYPE (cache must match activation dtype)
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device); rng.manual_seed(seed)

        # assistant_end may be a single id (unescaped tokenizer) or a multi-token
        # escape sequence (legacy tokenizer) -> tail-match like generate_batch.
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        ae_list = [assistant_end] if isinstance(assistant_end, int) else list(assistant_end)
        ae_len = len(ae_list)
        bos = self.tokenizer.get_bos_token_id()

        B = len(prompts)
        L = max(len(p) for p in prompts)
        # Size the cache TIGHT (prompt + generation), not the full sequence_len.
        # The kv-cache attention runs SDPA over the whole allocated cache each step;
        # eager (no cudagraph) that makes an 8192-wide cache pathologically slow for
        # short generations. A tight cache keeps the per-step attention small.
        max_seq_length = min(L + max_tokens, self.model.config.sequence_len)
        self.model.setup_caches(batch_size=B, max_seq_length=max_seq_length, dtype=dtype)
        self.model.kv_cache.reset()

        # Left-pad prompts; build the full-cache validity mask (B, max_seq_length).
        # Pad with BOS (id is harmless: those columns are masked out everywhere).
        pad_id = bos if isinstance(bos, int) else 0
        ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        needs_mask = any(len(p) < L for p in prompts)
        cache_valid = torch.ones(B, max_seq_length, dtype=torch.bool, device=device) if needs_mask else None
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
            if needs_mask and len(p) < L:
                cache_valid[i, :L - len(p)] = False  # left-pad columns are invalid keys

        # Prefill (eager). logits[:, -1] is the last (rightmost = always real,
        # left-pad) position.
        input_pos = torch.arange(L, device=device, dtype=torch.long)
        logits = self.model.forward(ids, input_pos=input_pos, attention_mask=cache_valid)[:, -1, :]

        cur = L
        out = [[] for _ in range(B)]
        done = [False] * B
        num_generated = 0
        while num_generated < max_tokens and not all(done):
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            toks = next_ids[:, 0].tolist()
            for i in range(B):
                if done[i]:
                    continue
                out[i].append(toks[i])
                if toks[i] == bos:
                    out[i] = out[i][:-1]; done[i] = True            # strip bos
                elif out[i][-ae_len:] == ae_list:
                    out[i] = out[i][:-ae_len]; done[i] = True        # strip assistant_end
            num_generated += 1
            if all(done):
                break
            # (compiled) batched decode step; cache_valid (or None) is the key-padding mask.
            logits = self._decode(next_ids, torch.tensor([cur], device=device, dtype=torch.long), cache_valid)
            cur += 1
        return out


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
