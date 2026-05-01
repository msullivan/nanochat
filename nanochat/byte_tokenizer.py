"""
Byte-level tokenizer: vocab size 256, no merges.

\x00 = BOS. \x01 = escape prefix for literal \x00/\x01 and special tokens:
  \x01\x00 = literal \x00, \x01\x01 = literal \x01,
  \x01\x02..\x01\x09 = user_start..output_end
"""

import copy
import torch

BOS = 0x00
ESCAPE = 0x01

# Special token name -> escape second byte (None = BOS itself)
SPECIAL_TOKENS = {
    "<|bos|>": None,
    "<|user_start|>": 0x02,
    "<|user_end|>": 0x03,
    "<|assistant_start|>": 0x04,
    "<|assistant_end|>": 0x05,
    "<|python_start|>": 0x06,
    "<|python_end|>": 0x07,
    "<|output_start|>": 0x08,
    "<|output_end|>": 0x09,
}
ESCAPE_TO_NAME = {v: k for k, v in SPECIAL_TOKENS.items() if v is not None}


class ByteTokenizer:

    def get_vocab_size(self):
        return 256

    def get_bos_token_id(self):
        return BOS

    def get_special_tokens(self):
        return set(SPECIAL_TOKENS.keys())

    def encode_special(self, text):
        if text == "<|bos|>":
            return BOS
        return [ESCAPE, SPECIAL_TOKENS[text]]

    def encode_special_list(self, text):
        if text == "<|bos|>":
            return [BOS]
        return [ESCAPE, SPECIAL_TOKENS[text]]

    def encode(self, text, prepend=None, append=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t, prepend=prepend, append=append) for t in text]

        ids = []

        # prepend
        if prepend is not None:
            tok = prepend if isinstance(prepend, (int, list)) else self.encode_special(prepend)
            if isinstance(tok, int):
                ids.append(tok)
            else:
                ids.extend(tok)

        # text as UTF-8 bytes, escaping \x00 and \x01
        for b in text.encode("utf-8"):
            if b <= ESCAPE:
                ids.extend([ESCAPE, b])
            else:
                ids.append(b)

        # append
        if append is not None:
            tok = append if isinstance(append, (int, list)) else self.encode_special(append)
            if isinstance(tok, int):
                ids.append(tok)
            else:
                ids.extend(tok)

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        out = bytearray()
        i = 0
        while i < len(ids):
            if ids[i] == ESCAPE and i + 1 < len(ids):
                n = ids[i + 1]
                if n in ESCAPE_TO_NAME:
                    out.extend(ESCAPE_TO_NAME[n].encode())
                else:
                    out.append(n)
                i += 2
            elif ids[i] == BOS:
                out.extend(b"<|bos|>")
                i += 1
            else:
                out.append(ids[i])
                i += 1
        return out.decode("utf-8", errors="replace")

    def id_to_token(self, id):
        if id == BOS:
            return "<|bos|>"
        if id == ESCAPE:
            return "<ESC>"
        return chr(id) if 32 <= id < 127 else f"<0x{id:02X}>"

    def render_conversation(self, conversation, max_tokens=2048):
        ids, mask = [], []

        def add(toks, m):
            if isinstance(toks, int):
                toks = [toks]
            ids.extend(toks)
            mask.extend([m] * len(toks))

        def special(name):
            return [ESCAPE, SPECIAL_TOKENS[name]]

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
                add(special("<|user_start|>"), 0)
                add(self.encode(content), 0)
                add(special("<|user_end|>"), 0)
            else:
                add(special("<|assistant_start|>"), 0)
                if isinstance(content, str):
                    add(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        v = self.encode(part["text"])
                        if part["type"] == "text":
                            add(v, 1)
                        elif part["type"] == "python":
                            add(special("<|python_start|>"), 1)
                            add(v, 1)
                            add(special("<|python_end|>"), 1)
                        elif part["type"] == "python_output":
                            add(special("<|output_start|>"), 0)
                            add(v, 0)
                            add(special("<|output_end|>"), 0)
                add(special("<|assistant_end|>"), 1)

        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, conversation):
        conversation = copy.deepcopy(conversation)
        conversation["messages"].pop()
        ids, _ = self.render_conversation(conversation)
        ids.extend([ESCAPE, SPECIAL_TOKENS["<|assistant_start|>"]])
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
    """Token byte counts for BPB calculation. 1 for all bytes except BOS and ESCAPE."""
    tb = torch.ones(256, dtype=torch.int32, device=device)
    tb[BOS] = 0
    tb[ESCAPE] = 0
    return tb
