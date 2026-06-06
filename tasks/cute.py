"""
CUTE benchmark: Edman, Schmid, Fraser, EMNLP 2024 (arXiv:2409.15452).

14 subtasks probing character-level token understanding. Each subtask is a 4-shot
ICL prompt; answer is a short string. The HuggingFace mirror (leukas/cute) ships
prompts pre-rendered with the 4-shot template baked in, ending with `Question: ...`
and *no* `Answer: "` opener (the paper's evaluation appends that as a prefill).

We replicate the paper's prefill methodology by appending `Answer: "` to the prompt
before generation, and parse up to the next `"`.

Two prompt modes:
  - completion: feed the raw prompt as text (via tokenizer.encode), continue.
                Works for base or SFT checkpoints. Matches the paper's methodology.
  - chat:       wrap the prompt as a single user message via render_for_completion,
                let the assistant generate. Only valid for SFT/RL checkpoints.
                Apples-to-apples vs published numbers from chat-tuned models.
"""

from datasets import load_dataset
from tasks.common import Task

CUTE_SUBTASKS = [
    "spell", "spell_inverse",
    "contains_char", "contains_word",
    "orth", "sem",
    "ins_char", "ins_word",
    "del_char", "del_word",
    "sub_char", "sub_word",
    "swap_char", "swap_word",
]

# The subset where a byte model is structurally well-positioned to beat BPE
# (BPE ceilings are dismal; see dev/cute_benchmark_notes.md).
CUTE_CHAR_LEVEL = [
    "spell", "spell_inverse",
    "contains_char",
    "orth",
    "ins_char", "del_char", "sub_char", "swap_char",
]

ANSWER_PREFILL = '\n\nAnswer: "'


def extract_cute_answer(completion, prefilled=True):
    """
    Two modes:
      - prefilled=True: prompt ended with `Answer: "`, so completion starts inside
        the quoted answer. Take everything before the first `"`.
      - prefilled=False: model emits the whole `Answer: "..."` itself. Take the
        first quoted substring.

    In either mode, fall back to the stripped first line if the expected quote
    structure isn't present.
    """
    if completion is None:
        return None
    if prefilled:
        end = completion.find('"')
        if end >= 0:
            return completion[:end].strip()
        return completion.split("\n", 1)[0].strip()
    else:
        start = completion.find('"')
        if start < 0:
            return completion.split("\n", 1)[0].strip()
        end = completion.find('"', start + 1)
        if end < 0:
            return completion[start + 1:].split("\n", 1)[0].strip()
        return completion[start + 1:end].strip()


def _answer_trailing_clean(completion, prefilled):
    """True iff the model said just the answer, no trailing nonsense -- everything
    after the answer region is whitespace. Handles both the quoted canonical form
    (`"ans"`) and a bare single-line answer. Used by strict eval together with the
    engine's turn-ended (assistant_end) signal."""
    if completion is None:
        return False
    # Locate the end of the answer region, then require the remainder be whitespace.
    if prefilled:
        # completion starts INSIDE the answer; it ends at the first closing quote,
        # or (no quote) at the end of the first line.
        end = completion.find('"')
        tail = completion[end + 1:] if end >= 0 else completion.split("\n", 1)[1] if "\n" in completion else ""
    else:
        start = completion.find('"')
        if start >= 0:
            end = completion.find('"', start + 1)
            if end < 0:
                return False  # opened a quote but never closed it -> messy
            tail = completion[end + 1:]
        else:
            # bare single-line answer (no quotes): clean iff there's no second line
            tail = completion.split("\n", 1)[1] if "\n" in completion else ""
    return tail.strip() == ""


class CUTE(Task):

    def __init__(self, subtask, mode="completion", prefill=True, prompt_style="fewshot", **kwargs):
        super().__init__(**kwargs)
        assert subtask in CUTE_SUBTASKS, f"Unknown CUTE subtask: {subtask}"
        assert mode in ("completion", "chat"), f"mode must be completion|chat, got {mode}"
        assert prompt_style in ("fewshot", "zero", "bare"), f"prompt_style must be fewshot|zero|bare, got {prompt_style}"
        self.subtask = subtask
        self.mode = mode
        self.prefill = prefill
        self.prompt_style = prompt_style
        self.ds = load_dataset("leukas/cute", split=subtask)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["prompt"]
        if self.prompt_style in ("zero", "bare"):
            # The 4-shot demo block is fixed per subtask. Strip it: keep only
            # `Question: ...` onward. Pair with gen_cute_pt_data --no-demos
            # so the training surface form matches.
            idx = prompt.find("Question:")
            if idx >= 0:
                prompt = prompt[idx:]
            if self.prompt_style == "bare":
                # Also drop the "Question:" label itself -- a chat user wouldn't
                # type it. Matches CuteChat's canonical phrasings (no label).
                prompt = prompt[len("Question:"):].lstrip()
        prompt = prompt + (ANSWER_PREFILL if self.prefill else "")
        answer = row["answer"]

        if self.mode == "chat":
            # Wrap as a one-turn user message; the standard render_for_completion
            # will pop the assistant message and append <|assistant_start|>.
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            return {"messages": messages, "answer": answer, "prompt_text": prompt}
        else:
            return {"prompt_text": prompt, "answer": answer}

    def evaluate(self, conversation, assistant_response, finished=None, strict=False):
        gold = conversation["answer"]
        pred = extract_cute_answer(assistant_response, prefilled=self.prefill)
        correct = (pred == gold)
        if not strict:
            return int(correct)
        # Strict: the answer must be right AND the model must have (a) ENDED its turn
        # (stopped on assistant_end, not run out at max_tokens) and (b) said nothing
        # after the answer (no trailing nonsense). finished comes from the engine.
        ended = True if finished is None else bool(finished)
        clean = _answer_trailing_clean(assistant_response, self.prefill)
        return int(correct and ended and clean)


if __name__ == "__main__":
    # Preview a few examples from each subtask.
    for subtask in ["spell", "ins_char", "swap_char", "contains_char"]:
        print("=" * 100)
        print(f"=== {subtask} ===")
        task = CUTE(subtask=subtask, mode="completion")
        for i in range(2):
            ex = task[i]
            print("-" * 100)
            print(f"PROMPT:\n{ex['prompt_text']}")
            print(f"ANSWER: {ex['answer']!r}")
