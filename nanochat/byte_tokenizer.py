"""
Byte-level tokenizer: vocab size 265 (= 256 raw bytes + 9 specials).

Bytes 0x00..0xff occupy IDs 0..255 untouched. Specials get dedicated IDs
above the byte range, so no escaping is needed on encode/decode.

Legacy compatibility (legacy_vocab=256):
    Old byte checkpoints were trained with a 256-wide vocab where BOS lived at
    row 0 (and the other specials were 2-byte escape sequences that base models
    never used). Such models only ever consume raw bytes + <|bos|>. Passing
    legacy_vocab=256 makes this class *present* that old 256-wide ID space at
    its boundaries: BOS is remapped 256 -> 0 (256 % 256 == 0), raw bytes are
    identity, get_vocab_size()/get_bos_token_id() report the old values, and
    decode/byte-counts use the old space. This lets an unmodified old (256, D)
    checkpoint load and eval bit-for-bit -- no weight conversion needed. It is
    base-model only: the chat specials (257..264) have no home in 256 and raise
    if requested, so chat checkpoints must use the conversion script instead.
"""

import copy
import torch

# Special token IDs sit above the byte range. Order is fixed; do NOT renumber
# without writing a checkpoint conversion -- these IDs are baked into the
# embedding rows of every saved model.
SPECIAL_TOKENS = {
    "<|bos|>":              256,
    "<|user_start|>":       257,
    "<|user_end|>":         258,
    "<|assistant_start|>":  259,
    "<|assistant_end|>":    260,
    "<|python_start|>":     261,
    "<|python_end|>":       262,
    "<|output_start|>":     263,
    "<|output_end|>":       264,
}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

VOCAB_SIZE = 256 + len(SPECIAL_TOKENS)  # 265
BOS = SPECIAL_TOKENS["<|bos|>"]


class ByteTokenizer:

    def __init__(self, legacy_vocab=None):
        # legacy_vocab=256 -> impersonate the old 256-wide scheme (base only).
        # None -> canonical 265/320 scheme.
        assert legacy_vocab is None or legacy_vocab == 256, \
            f"legacy_vocab must be None or 256, got {legacy_vocab}"
        self.legacy_vocab = legacy_vocab

    def get_vocab_size(self):
        return self.legacy_vocab if self.legacy_vocab is not None else VOCAB_SIZE

    def get_bos_token_id(self):
        # 256 % 256 == 0: old BOS lived at row 0.
        return BOS % self.legacy_vocab if self.legacy_vocab is not None else BOS

    def get_special_tokens(self):
        return set(SPECIAL_TOKENS.keys())

    def encode_special(self, text):
        tid = SPECIAL_TOKENS[text]
        if self.legacy_vocab is not None:
            # Base models only ever use <|bos|>; the chat specials have no slot
            # in the 256-wide vocab. Fail loud rather than silently mod a chat
            # special down into a junk byte ID.
            if text != "<|bos|>":
                raise ValueError(
                    f"legacy_vocab byte tokenizer supports only <|bos|> among specials, "
                    f"got {text!r}. Legacy mode is base-model only; convert chat checkpoints."
                )
            return tid % self.legacy_vocab  # 256 -> 0
        return tid

    def encode(self, text, prepend=None, append=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t, prepend=prepend, append=append) for t in text]

        ids = []
        if prepend is not None:
            ids.append(prepend if isinstance(prepend, int) else self.encode_special(prepend))
        ids.extend(text.encode("utf-8"))
        if append is not None:
            ids.append(append if isinstance(append, int) else self.encode_special(append))
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        out = bytearray()
        if self.legacy_vocab is not None:
            # Old space: id 0 = BOS, ids 1..255 = raw bytes. (Literal NUL bytes
            # never occur in base text, so the 0/BOS overlap is moot.)
            for i in ids:
                if i == 0:
                    out.extend(b"<|bos|>")
                elif i < 256:
                    out.append(i)
            return out.decode("utf-8", errors="replace")
        for i in ids:
            if i < 256:
                out.append(i)
            elif i in ID_TO_SPECIAL:
                out.extend(ID_TO_SPECIAL[i].encode())
            # ids beyond the vocab (e.g. padding rows) are silently dropped
        return out.decode("utf-8", errors="replace")

    def id_to_token(self, id):
        if self.legacy_vocab is not None:
            if id == 0:
                return "<|bos|>"
            return chr(id) if 32 <= id < 127 else f"<0x{id:02X}>"
        if id in ID_TO_SPECIAL:
            return ID_TO_SPECIAL[id]
        if id < 256:
            return chr(id) if 32 <= id < 127 else f"<0x{id:02X}>"
        return f"<id_{id}>"

    def render_conversation(self, conversation, max_tokens=2048):
        assert self.legacy_vocab is None, \
            "render_conversation needs the chat specials; legacy_vocab mode is base-model only"
        ids, mask = [], []

        def add(toks, m):
            if isinstance(toks, int):
                toks = [toks]
            ids.extend(toks)
            mask.extend([m] * len(toks))

        # Merge system message into first user message
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            msgs = conversation["messages"]
            msgs[1]["content"] = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
            msgs = msgs[1:]
        else:
            msgs = conversation["messages"]

        add(BOS, 0)
        for i, msg in enumerate(msgs):
            assert msg["role"] == ("user" if i % 2 == 0 else "assistant")
            content = msg["content"]

            if msg["role"] == "user":
                add(SPECIAL_TOKENS["<|user_start|>"], 0)
                add(self.encode(content), 0)
                add(SPECIAL_TOKENS["<|user_end|>"], 0)
            else:
                add(SPECIAL_TOKENS["<|assistant_start|>"], 0)
                if isinstance(content, str):
                    add(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        v = self.encode(part["text"])
                        if part["type"] == "text":
                            add(v, 1)
                        elif part["type"] == "python":
                            add(SPECIAL_TOKENS["<|python_start|>"], 1)
                            add(v, 1)
                            add(SPECIAL_TOKENS["<|python_end|>"], 1)
                        elif part["type"] == "python_output":
                            add(SPECIAL_TOKENS["<|output_start|>"], 0)
                            add(v, 0)
                            add(SPECIAL_TOKENS["<|output_end|>"], 0)
                add(SPECIAL_TOKENS["<|assistant_end|>"], 1)

        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, conversation):
        conversation = copy.deepcopy(conversation)
        conversation["messages"].pop()
        ids, _ = self.render_conversation(conversation)
        ids.append(SPECIAL_TOKENS["<|assistant_start|>"])
        return ids

    def save(self, tokenizer_dir):
        import os
        os.makedirs(tokenizer_dir, exist_ok=True)
        with open(os.path.join(tokenizer_dir, "byte_tokenizer.marker"), "w") as f:
            f.write("byte_tokenizer\n")
        tb = get_byte_token_bytes()
        with open(os.path.join(tokenizer_dir, "token_bytes.pt"), "wb") as f:
            torch.save(tb, f)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        return cls()


def get_byte_token_bytes(device="cpu", legacy_vocab=None):
    """Token byte counts for BPB calculation. 1 for raw-byte IDs, 0 for
    special-token IDs. Length matches the tokenizer's vocab size.

    legacy_vocab=256 returns the old-space vector: id 0 = BOS (0 bytes),
    ids 1..255 = 1 byte each.
    """
    if legacy_vocab is not None:
        tb = torch.ones(legacy_vocab, dtype=torch.int32, device=device)
        tb[0] = 0  # id 0 = BOS in the legacy scheme
        return tb
    tb = torch.zeros(VOCAB_SIZE, dtype=torch.int32, device=device)
    tb[:256] = 1
    return tb
