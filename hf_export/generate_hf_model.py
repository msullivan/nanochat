"""Generate a standalone HuggingFace artifact from a nanochat checkpoint.

The architecture math is NOT reimplemented: this copies nanochat/gpt.py verbatim
(only rewriting its imports) so gpt.py stays the single source of truth. A thin
generated wrapper (modeling_nanochat.py) wraps GPT for the HF interface.

Output dir is self-contained -- loads with just `transformers` + `torch`:
    AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True)

Usage:
    python hf_export/generate_hf_model.py \
        --src ~/.cache/nanochat/chatsft_checkpoints/d24-byte-l-ext-chat \
        --step latest --dst ~/hf_models/nanochat-byte --dtype bfloat16 --verify

Base (pretrained-only) checkpoints: add --base (no chat template, EOS=<|bos|>).
The 265-vocab converted form is expected (e.g. d24-byte-l-ext-u), not the
legacy 256-wide scheme.
"""

import os
import json
import shutil
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (filename, [(old, new, must_exist), ...]) import rewrites applied to copied source.
GPT_REWRITES = [
    ("from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE",
     "from ._runtime import get_dist_info, print0, COMPUTE_DTYPE"),
    ("from nanochat.optim import MuonAdamW, DistMuonAdamW",
     "MuonAdamW = DistMuonAdamW = None  # optim unused at inference (export stub)"),
    ("from nanochat.flash_attention import flash_attn",
     "from .flash_attention import flash_attn"),
    ("from nanochat.engine import KVCache",
     "from ._runtime import KVCache"),
]
FLASH_REWRITES = [
    ("from nanochat.common import COMPUTE_DTYPE",
     "from ._runtime import COMPUTE_DTYPE"),
]


def _rewrite(src_path, rewrites):
    text = open(src_path).read()
    for entry in rewrites:
        old, new = entry[0], entry[1]
        required = entry[2] if len(entry) > 2 else True
        if old not in text:
            assert not required, f"expected import not found in {src_path!r}: {old!r}\n(generator out of sync with nanochat source)"
            continue
        text = text.replace(old, new)
    # Hard guard: no executable nanochat package import may survive. Skip lines
    # inside triple-quoted docstrings (which may mention `from nanochat...` as
    # usage examples) by tracking doc-string state.
    in_doc = False
    for line in text.splitlines():
        if not in_doc:
            s = line.strip()
            assert not (s.startswith("from nanochat") or s.startswith("import nanochat")), \
                f"unrewritten nanochat import in {src_path!r}: {s!r}"
        if (line.count('"""') + line.count("'''")) % 2 == 1:
            in_doc = not in_doc
    return text


def _resolve_step(src, step):
    if step != "latest":
        return int(step)
    steps = [int(f[6:-3]) for f in os.listdir(src) if f.startswith("model_") and f.endswith(".pt")]
    if not steps:
        raise FileNotFoundError(f"no model_*.pt in {src}")
    return max(steps)


def generate(src, step, dst, dtype=torch.float32, base=False):
    step_i = _resolve_step(src, step)
    model_data = torch.load(os.path.join(src, f"model_{step_i:06d}.pt"), map_location="cpu")
    meta = json.load(open(os.path.join(src, f"meta_{step_i:06d}.json")))
    mc = dict(meta["model_config"])
    mc.setdefault("window_pattern", "L")
    assert mc["vocab_size"] == 265, (
        f"expected a 265-vocab checkpoint, got {mc['vocab_size']}; legacy 256-wide "
        f"base checkpoints must be converted first (dev/convert_byte_tokenizer_unescaped.py)")

    os.makedirs(dst, exist_ok=True)

    # 1. Copy gpt.py + flash_attention.py with rewritten imports (single source of arch math).
    open(os.path.join(dst, "gpt.py"), "w").write(_rewrite(os.path.join(ROOT, "nanochat", "gpt.py"), GPT_REWRITES))
    open(os.path.join(dst, "flash_attention.py"), "w").write(
        _rewrite(os.path.join(ROOT, "nanochat", "flash_attention.py"), FLASH_REWRITES))

    # 2. Copy static templates (runtime shim, wrapper, config/tokenizer code,
    #    and -- for chat models -- the chat template).
    shutil.copy(os.path.join(HERE, "_runtime_template.py"), os.path.join(dst, "_runtime.py"))
    shutil.copy(os.path.join(HERE, "_wrapper_modeling_template.py"), os.path.join(dst, "modeling_nanochat.py"))
    shutil.copy(os.path.join(HERE, "configuration_nanochat.py"), os.path.join(dst, "configuration_nanochat.py"))
    shutil.copy(os.path.join(HERE, "tokenization_nanochat.py"), os.path.join(dst, "tokenization_nanochat.py"))
    if not base:
        shutil.copy(os.path.join(HERE, "chat_template.jinja"), os.path.join(dst, "chat_template.jinja"))

    # 3. Config with auto_map for trust_remote_code loading.
    import sys
    sys.path.insert(0, HERE)
    from configuration_nanochat import NanochatConfig
    config = NanochatConfig(**mc)
    config.architectures = ["NanochatForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_nanochat.NanochatConfig",
        "AutoModelForCausalLM": "modeling_nanochat.NanochatForCausalLM",
    }
    config.save_pretrained(dst)

    # 4. Weights: strip torch.compile prefix, drop non-persistent rotary buffers,
    #    cast, and prefix with "gpt." to match the wrapper's submodule.
    from safetensors.torch import save_file
    sd = {}
    for k, v in model_data.items():
        k = k.removeprefix("_orig_mod.")
        if k.endswith(".cos") or k.endswith(".sin") or k == "cos" or k == "sin":
            continue
        if v.is_floating_point():
            v = v.to(dtype)
        sd[f"gpt.{k}"] = v.contiguous().clone()
    save_file(sd, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})

    # 5. Tokenizer + chat template (register_for_auto_class writes the auto_map so
    #    AutoTokenizer(trust_remote_code) picks our custom class instead of a fast one).
    from tokenization_nanochat import NanochatByteTokenizer
    NanochatByteTokenizer.register_for_auto_class("AutoTokenizer")
    tok = NanochatByteTokenizer(base_model=base)
    if not base:
        tok.chat_template = open(os.path.join(HERE, "chat_template.jinja")).read()
    tok.save_pretrained(dst)

    kind = "base" if base else "chat"
    print(f"generated standalone HF artifact: {src} (step {step_i}) -> {dst}  [{kind}, dtype={dtype}, vocab={config.vocab_size}]")
    return dst, step_i, mc


def verify(dst, src, step_i, mc, seqlen=64, batch=2, base=False, dtype=torch.float32):
    """Bit-exact logits parity vs the reference nanochat GPT + a generate smoke test.

    Comparison runs in fp32 on both sides. For a non-fp32 export the reference
    weights are first round-tripped through the export dtype, so the comparison
    is still exact (it checks the port, not the quantization)."""
    os.environ.setdefault("NANOCHAT_DTYPE", "float32")
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import torch as _t
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nanochat.checkpoint_manager import build_model

    ref, _tok, _meta = build_model(checkpoint_dir=src, step=step_i, device=_t.device("cpu"), phase="eval")
    ref = ref.to(_t.float32)
    if dtype != _t.float32:
        for p in ref.parameters():
            p.data = p.data.to(dtype).to(_t.float32)
    m = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True, dtype=_t.float32).eval()
    # tokenizer must load via AutoTokenizer (custom slow class via auto_map) and round-trip
    tk = AutoTokenizer.from_pretrained(dst, trust_remote_code=True)
    _s = 'Héllo 世界!'
    tok_ok = tk.decode(tk.encode(_s, add_special_tokens=False), skip_special_tokens=True) == _s
    expected_eos = "<|bos|>" if base else "<|assistant_end|>"
    tok_ok = tok_ok and tk.eos_token == expected_eos and (tk.chat_template is None) == base
    print(f"  tokenizer: AutoTokenizer load + round-trip + roles (eos={tk.eos_token}): {tok_ok}")

    _t.manual_seed(0)
    vocab = mc["vocab_size"]
    idx = _t.randint(0, vocab, (batch, seqlen))
    with _t.no_grad():
        rl = ref(idx)
        hl = m(input_ids=idx).logits
    max_abs = (rl - hl).abs().max().item()
    argmax = (rl.argmax(-1) == hl.argmax(-1)).float().mean().item()
    print(f"  parity: max|diff|={max_abs:.3e}  argmax={argmax*100:.4f}%")

    # cached generate vs no-cache greedy
    ids = _t.tensor([[0] + list(b"The ")])
    gen_c = m.generate(ids, attention_mask=_t.ones_like(ids), max_new_tokens=24, do_sample=False,
                       use_cache=True, pad_token_id=0)
    gen_n = m.generate(ids, attention_mask=_t.ones_like(ids), max_new_tokens=24, do_sample=False,
                       use_cache=False, pad_token_id=0)
    same = _t.equal(gen_c, gen_n)
    print(f"  generate: cache==no-cache: {same}")
    ok = max_abs == 0.0 and argmax == 1.0 and same and tok_ok
    print("  VERIFY OK" if ok else "  VERIFY FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--step", default="latest")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--base", action="store_true",
                    help="base (pretrained-only) artifact: no chat template, EOS=<|bos|>")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    dst, step_i, mc = generate(args.src, args.step, args.dst, dtype=dtype, base=args.base)
    if args.verify:
        ok = verify(dst, args.src, step_i, mc, base=args.base, dtype=dtype)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
