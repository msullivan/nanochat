"""HuggingFace config for the nanochat custom GPT (byte-tokenizer variant).

Mirrors nanochat/gpt.py::GPTConfig plus the few extra constants that live as
module-level globals or hard-coded literals in the reference forward (rope base,
logit softcap, vocab padding, value-embed gate width / scale, qk scale).
"""

from transformers.configuration_utils import PretrainedConfig


class NanochatConfig(PretrainedConfig):
    model_type = "nanochat"

    def __init__(
        self,
        sequence_len: int = 8192,
        vocab_size: int = 265,
        n_layer: int = 24,
        n_head: int = 12,
        n_kv_head: int = 12,
        n_embd: int = 1536,
        window_pattern: str = "L",
        bigram_value_embeds: bool = False,
        disable_value_embeds: bool = False,
        pad_vocab_size_to: int = 64,
        rope_base: float = 100000.0,
        logit_softcap: float = 15.0,
        qk_scale: float = 1.2,
        ve_gate_channels: int = 12,
        ve_gate_scale: float = 3.0,
        smear_gate_channels: int = 24,
        rotary_overcompute: int = 10,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern
        self.bigram_value_embeds = bigram_value_embeds
        self.disable_value_embeds = disable_value_embeds
        self.pad_vocab_size_to = pad_vocab_size_to
        self.rope_base = rope_base
        self.logit_softcap = logit_softcap
        self.qk_scale = qk_scale
        self.ve_gate_channels = ve_gate_channels
        self.ve_gate_scale = ve_gate_scale
        self.smear_gate_channels = smear_gate_channels
        self.rotary_overcompute = rotary_overcompute
        # HF-standard aliases used by various utilities / GenerationMixin.
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.max_position_embeddings = sequence_len
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def padded_vocab_size(self) -> int:
        p = self.pad_vocab_size_to
        return ((self.vocab_size + p - 1) // p) * p
