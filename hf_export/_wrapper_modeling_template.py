"""HuggingFace wrapper for the nanochat byte model (GENERATED -- do not edit).

This file is emitted by hf_export/generate_hf_model.py. It is a THIN wrapper:
all architecture math lives in the copied gpt.py (the single source of truth,
verbatim from nanochat). This class only:
  - builds a GPTConfig from the HF NanochatConfig,
  - calls GPT.forward for the no-cache (training/eval) path, and
  - bridges HF generate to GPT's native KV-cache (setup_caches + input_pos).

Because nothing here re-expresses the forward, architecture changes in nanochat's
gpt.py flow through on the next `generate_hf_model.py` run with no hand-editing.
"""

import torch
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_nanochat import NanochatConfig
    from .gpt import GPT, GPTConfig
except ImportError:  # local (non-package) import
    from configuration_nanochat import NanochatConfig
    from gpt import GPT, GPTConfig


def _gpt_config_from(hf):
    return GPTConfig(
        sequence_len=hf.sequence_len,
        vocab_size=hf.vocab_size,
        n_layer=hf.n_layer,
        n_head=hf.n_head,
        n_kv_head=hf.n_kv_head,
        n_embd=hf.n_embd,
        window_pattern=hf.window_pattern,
        bigram_value_embeds=hf.bigram_value_embeds,
        disable_value_embeds=hf.disable_value_embeds,
    )


class NanochatForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanochatConfig
    base_model_prefix = "gpt"
    main_input_name = "input_ids"
    _no_split_modules = ["Block"]
    _supports_attention_backend = False

    def __init__(self, config):
        super().__init__(config)
        self.gpt = GPT(_gpt_config_from(config))
        # Rotary tables: GPT computes them in __init__, but under from_pretrained's
        # meta-device construction that yields uninitialized memory. Rebuild lazily
        # on the first real forward (and on device change).
        self._rotary_ready = False
        self.post_init()

    # --- embeddings plumbing -------------------------------------------------
    def get_input_embeddings(self):
        return self.gpt.transformer.wte

    def set_input_embeddings(self, value):
        self.gpt.transformer.wte = value

    def get_output_embeddings(self):
        return self.gpt.lm_head

    def _ensure_rotary(self, device):
        if self._rotary_ready and self.gpt.cos.device == device:
            return
        head_dim = self.gpt.config.n_embd // self.gpt.config.n_head
        cos, sin = self.gpt._precompute_rotary_embeddings(self.gpt.rotary_seq_len, head_dim, device=device)
        self.gpt.cos, self.gpt.sin = cos, sin
        self._rotary_ready = True

    # --- forward -------------------------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, labels=None, past_key_values=None,
                use_cache=False, cache_position=None, **kwargs):
        # attention_mask is accepted for HF-generate compatibility; this wrapper
        # assumes unpadded, left-aligned sequences (single-stream / equal-length
        # batches), so the mask is not consumed.
        self._ensure_rotary(input_ids.device)

        if labels is not None or not use_cache:
            # No-cache full forward (training / eval / parity). GPT.forward
            # returns softcapped logits sliced to vocab_size.
            logits = self.gpt(input_ids)
            loss = None
            if labels is not None:
                sl = logits[:, :-1, :].contiguous()
                lab = labels[:, 1:].contiguous()
                loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1), ignore_index=-1)
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

        # Cached decode path: drive GPT's native KV-cache via input_pos. Prefill
        # is detected by cache_position starting at 0 (HF sets cache_position even
        # though it also pre-creates its own DynamicCache, which we ignore -- the
        # real cache state lives on self.gpt.kv_cache).
        B, T = input_ids.shape
        dtype = self.gpt.transformer.wte.weight.dtype
        if cache_position is None:
            cache_position = torch.arange(T, device=input_ids.device)
        if int(cache_position[0]) == 0:
            self.gpt.setup_caches(B, self.config.sequence_len, dtype)
            self.gpt.kv_cache.reset()
        logits = self.gpt(input_ids, input_pos=cache_position)
        # Sentinel: truthy so HF's loop knows a cache is live (state lives on self.gpt).
        return CausalLMOutputWithPast(logits=logits, past_key_values=True)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      cache_position=None, attention_mask=None, **kwargs):
        # cache_position tells us absolute positions of the tokens to process.
        # Prefill (starts at 0): feed the whole prompt. Decode: feed only the new
        # token(s). GPT keeps its own KVCache, so we ignore HF's past object.
        if cache_position is None:
            cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        if int(cache_position[0]) == 0:
            model_in = input_ids
        else:
            model_in = input_ids[:, -cache_position.shape[0]:]
        return {
            "input_ids": model_in,
            "past_key_values": past_key_values,
            "use_cache": True,
            "cache_position": cache_position,
        }
