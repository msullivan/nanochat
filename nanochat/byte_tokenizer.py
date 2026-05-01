"""
Byte-level tokenizer: vocab size 265 (= 256 raw bytes + 9 specials).

Bytes 0x00..0xff occupy IDs 0..255 untouched. Specials get dedicated IDs
above the byte range, so no escaping is needed on encode/decode.
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

    def get_vocab_size(self):
        return VOCAB_SIZE

    def get_bos_token_id(self):
        return BOS

    def get_special_tokens(self):
        return set(SPECIAL_TOKENS.keys())

    def encode_special(self, text):
        return SPECIAL_TOKENS[text]

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
        for i in ids:
            if i < 256:
                out.append(i)
            elif i in ID_TO_SPECIAL:
                out.extend(ID_TO_SPECIAL[i].encode())
            # ids beyond the vocab (e.g. padding rows) are silently dropped
        return out.decode("utf-8", errors="replace")

    def id_to_token(self, id):
        if id in ID_TO_SPECIAL:
            return ID_TO_SPECIAL[id]
        if id < 256:
            return chr(id) if 32 <= id < 127 else f"<0x{id:02X}>"
        return f"<id_{id}>"

    def render_conversation(self, conversation, max_tokens=2048):
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


def get_byte_token_bytes(device="cpu"):
    """Token byte counts for BPB calculation. 1 for raw-byte IDs (0..255),
    0 for the special-token IDs above. Length matches vocab_size."""
    tb = torch.zeros(VOCAB_SIZE, dtype=torch.int32, device=device)
    tb[:256] = 1
    return tb
