"""HuggingFace byte-level tokenizer for the nanochat byte model.

Vocab = 256 raw bytes (ids 0..255) + 9 special tokens (ids 256..264), matching
nanochat/byte_tokenizer.py (the canonical 265-wide scheme). Text is encoded as
its raw UTF-8 bytes; each byte is one token. Special tokens get dedicated ids.

Byte tokens are represented internally with the GPT-2 "bytes_to_unicode" trick
(every byte -> a printable, reversible unicode char) so they round-trip through
HF's str-based token machinery. convert_tokens_to_string maps them back to bytes
and UTF-8-decodes with errors="replace" (matching the reference decode).
"""

from transformers.tokenization_utils import PreTrainedTokenizer, AddedToken


# Fixed id order -- baked into the model's embedding rows; do not renumber.
SPECIAL_TOKENS = [
    "<|bos|>",              # 256
    "<|user_start|>",       # 257
    "<|user_end|>",         # 258
    "<|assistant_start|>",  # 259
    "<|assistant_end|>",    # 260
    "<|python_start|>",     # 261
    "<|python_end|>",       # 262
    "<|output_start|>",     # 263
    "<|output_end|>",       # 264
]


def bytes_to_unicode():
    """GPT-2's reversible byte<->printable-unicode map (256 entries)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class NanochatByteTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {}

    def __init__(self, **kwargs):
        # byte <-> unicode-char maps, and char <-> id over the 256 byte range
        self.byte_encoder = bytes_to_unicode()           # byte(int) -> char
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}  # char -> byte(int)
        self._id_to_byte_char = {i: self.byte_encoder[i] for i in range(256)}
        self._byte_char_to_id = {c: i for i, c in self._id_to_byte_char.items()}

        # Strip any role-token kwargs a reload might pass; we pin the ids below
        # so they cannot perturb the 256..264 ordering.
        for k in ("bos_token", "eos_token", "pad_token", "unk_token", "additional_special_tokens"):
            kwargs.pop(k, None)
        super().__init__(**kwargs)

        # Add the 9 specials IN ORDER first, so ids land at exactly 256..264
        # (baked into the model's embedding rows). Only then assign roles --
        # otherwise HF adds bos/eos/pad before the ordered list and scrambles ids.
        if self.convert_tokens_to_ids("<|bos|>") is None or self.convert_tokens_to_ids("<|bos|>") == self.unk_token_id:
            self.add_special_tokens(
                {"additional_special_tokens": [AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS]}
            )
        self.add_special_tokens(
            {"bos_token": "<|bos|>", "eos_token": "<|assistant_end|>", "pad_token": "<|bos|>"}
        )

    @property
    def vocab_size(self):
        return 256  # base vocab (byte tokens); specials are added tokens 256..264

    def get_vocab(self):
        vocab = {self._id_to_byte_char[i]: i for i in range(256)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        # Each UTF-8 byte becomes one (printable) byte-token char.
        return [self.byte_encoder[b] for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token):
        if token in self._byte_char_to_id:
            return self._byte_char_to_id[token]
        # specials are handled by HF's added_tokens_encoder, but guard anyway
        return self.added_tokens_encoder.get(token)

    def _convert_id_to_token(self, index):
        if index in self._id_to_byte_char:
            return self._id_to_byte_char[index]
        return self.added_tokens_decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        # tokens here are byte-token chars only (HF strips specials out first).
        bs = bytearray(self.byte_decoder[c] for tok in tokens for c in tok)
        return bs.decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()  # vocab is fully procedural; nothing on disk to write
