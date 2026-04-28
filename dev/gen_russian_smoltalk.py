"""Rewrite SmolTalk assistant turns into Russian-accented English via `claude -p`.

For each SmolTalk conversation, leaves the user turns alone but rewrites every
assistant turn through Haiku with a custom system prompt that applies grammatical
Russian-accent transformations (article drop, copula drop, simpler tenses).
Output JSONL is in CustomJSON format (one JSON array of {role, content} per
line, alternating user/assistant starting with user).

This is a stylistic SFT experiment: can the model learn a coherent register
shift from a few thousand examples and apply it to held-out chat turns. We
deliberately don't apply orthographic substitutions (e.g. v/w swap) because
they look gimmicky on the page even when they're phonologically real.

Examples:

    # Default: 1000 conversations via `claude -p` on Haiku, 8 workers
    uv run python dev/gen_russian_smoltalk.py --num 1000 --workers 8

    # Smoke test: 3 conversations, sequential, save metadata
    uv run python dev/gen_russian_smoltalk.py --num 3 --workers 1 --save-metadata

    # Resume from a known offset (useful if a previous run died)
    uv run python dev/gen_russian_smoltalk.py --num 5000 --start-idx 1000 --append
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Make sibling imports work when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from dev.llm_client import chat_completion


CLAUDE_MODEL = "haiku"


def get_base_dir():
    # Inlined from nanochat.common to avoid pulling in torch on machines that
    # only run this generator (no GPU needed for `claude -p`).
    base = os.environ.get("NANOCHAT_BASE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    os.makedirs(base, exist_ok=True)
    return base

SYSTEM_PROMPT = """\
You are a text-transformation utility. Your only job is to rewrite English text into Russian-accented English. You are not a chatbot, assistant, or coding helper -- you take text in, produce transformed text out, nothing else.

Apply these transformations consistently:

1. DROP ARTICLES. Russian has no articles. Omit "a", "an", and "the" wherever it sounds plausible. Examples: "the dog ran" -> "dog ran"; "I bought a book" -> "I bought book".

2. DROP THE COPULA. In present-tense statements, "is" and "are" can be dropped where Russian would omit them: "He is a teacher" -> "He teacher"; "They are happy" -> "They happy". Don't do it everywhere; do it sometimes.

3. SIMPLIFY VERB FORMS. Prefer simple tenses over progressive/perfect where possible: "I am going" -> "I go"; "I have been working" -> "I work" or "I working".

4. PRESERVE STRUCTURE AND CONTENT. Keep all the original meaning, factual information, lists, code blocks, headers, and formatting. Do NOT translate code or change variable names. Do NOT alter spelling. Do NOT summarize, expand, add commentary, ask questions, refuse, or comment on the task. Just transform the natural-language prose grammatically.

OUTPUT FORMAT: Output ONLY the rewritten text. No preface ("Here is..."), no quotes around the output, no apologies, no explanation, no JSON wrapping. Just the rewritten version, ready to use as a drop-in replacement for the input.\
"""


def rewrite_assistant_text(text, model=CLAUDE_MODEL):
    """Send `text` through claude -p with the Russian-accent system prompt."""
    if not text or not text.strip():
        return text
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    result = chat_completion(messages, model=model, backend="claude")
    return result["choices"][0]["message"]["content"]


def normalize_smoltalk_messages(raw_messages):
    """Drop the optional system message and return the user/assistant alternation.

    Returns None if the conversation doesn't fit our SFT shape (must alternate
    user, assistant, user, ... starting with user, content must be strings).
    """
    if not raw_messages:
        return None
    msgs = raw_messages[1:] if raw_messages[0]["role"] == "system" else list(raw_messages)
    if len(msgs) < 2:
        return None
    for i, m in enumerate(msgs):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            return None
        if not isinstance(m["content"], str):
            return None
    return msgs


def rewrite_conversation(idx, messages, model=CLAUDE_MODEL):
    """Rewrite every assistant turn in the conversation. User turns are passed through."""
    out = []
    for m in messages:
        if m["role"] == "assistant":
            new_content = rewrite_assistant_text(m["content"], model=model)
            out.append({"role": "assistant", "content": new_content})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return idx, out


def main():
    parser = argparse.ArgumentParser(description="Rewrite SmolTalk assistant turns with a Russian accent")
    parser.add_argument("--num", type=int, default=1000, help="Number of conversations to rewrite")
    parser.add_argument("--start-idx", type=int, default=0, help="SmolTalk row index to start from (for resume)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent claude -p calls")
    parser.add_argument("--model", type=str, default=CLAUDE_MODEL, help='Claude model id ("haiku", "sonnet", or full id)')
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path (default: $NANOCHAT_BASE_DIR/russian_conversations.jsonl)")
    parser.add_argument("--append", action="store_true", help="Append to output instead of clearing it")
    parser.add_argument("--save-metadata", action="store_true", help='Include {"meta": {...}} alongside the message array (will use {"messages":..., "meta":...} object form, NOT the bare-array CustomJSON form)')
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="SmolTalk split to source from")
    args = parser.parse_args()

    output_file = args.output or os.path.join(get_base_dir(), "russian_conversations.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not args.append and os.path.exists(output_file):
        os.remove(output_file)

    print(f"Loading SmolTalk ({args.split} split)...")
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=args.split)
    total_rows = len(ds)
    print(f"SmolTalk {args.split}: {total_rows:,} rows")

    end_idx = min(args.start_idx + args.num, total_rows)
    indices = list(range(args.start_idx, end_idx))
    print(f"Rewriting rows {args.start_idx}..{end_idx} ({len(indices)} conversations) with model={args.model!r}")
    print(f"Output: {output_file}")

    write_lock = threading.Lock()
    counts = {"done": 0, "skipped": 0, "errors": 0}
    t0 = time.time()

    def submit(idx):
        raw = ds[idx]["messages"]
        msgs = normalize_smoltalk_messages(raw)
        if msgs is None:
            return idx, None, "skip"
        try:
            _, rewritten = rewrite_conversation(idx, msgs, model=args.model)
            return idx, rewritten, "ok"
        except Exception as e:
            return idx, None, f"err:{type(e).__name__}:{e}"

    with open(output_file, "a", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(submit, idx): idx for idx in indices}
        for fut in as_completed(futures):
            idx, rewritten, status = fut.result()
            if status == "ok":
                if args.save_metadata:
                    record = {"messages": rewritten, "meta": {"smoltalk_idx": idx, "model": args.model}}
                else:
                    record = rewritten  # CustomJSON format: bare array per line
                with write_lock:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                counts["done"] += 1
            elif status == "skip":
                counts["skipped"] += 1
            else:
                counts["errors"] += 1
                print(f"  [idx={idx}] {status}", file=sys.stderr)

            n = counts["done"] + counts["skipped"] + counts["errors"]
            if n % 10 == 0 or n == len(indices):
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed > 0 else 0
                print(f"  {n}/{len(indices)} | done={counts['done']} skip={counts['skipped']} err={counts['errors']} | {rate:.2f} convo/s", flush=True)

    elapsed = time.time() - t0
    print(f"\nFinal: {counts['done']} written, {counts['skipped']} skipped, {counts['errors']} errors in {elapsed:.1f}s")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
