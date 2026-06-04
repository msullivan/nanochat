"""
Chat-style CUTE training tasks (SFT).

The CUTE benchmark (leukas/cute) is character-level: spell a word, swap two
letters, count/contains a char, etc. A byte tokenizer is structurally well
positioned to learn these, so we over-represent them in SFT to teach the chat
model the capability -- the same idea as SpellingBee, generalized to all 8
char-level CUTE subtasks.

This generates *synthetic* training conversations. It does NOT touch the
leukas/cute eval data as training text; on the contrary it loads every CUTE
split to collect the eval target words and excludes them from the training word
pool (see _excluded_words), so there's no test contamination. Held-out eval is
the real benchmark via `scripts/cute_eval.py --mode chat`.

Design (per the SFT prep decisions):
  - terse answers: the assistant replies with just the quoted answer in CUTE
    surface form (e.g. spell -> "t h e", contains_char -> "Yes"), parseable by
    tasks.cute.extract_cute_answer(prefilled=False).
  - single mixed template bank per subtask (TEMPLATES below) -- this is the
    "vary how the question is asked" axis; edit/extend these to experiment.
  - all 8 char-level subtasks.

Preview:
    python -m tasks.cute_chat
"""

import string
import random

from tasks.common import Task
from tasks.cute import extract_cute_answer, CUTE_CHAR_LEVEL
from nanochat.common import download_file_with_lock

# 370K English words; same source SpellingBee uses.
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
TEST_RANDOM_SEED_OFFSET = 10_000_000

# Demo words baked into the published CUTE 4-shot prefixes; never use as targets.
DEMO_WORDS = {"alphabet", "hello", "zebra", "tongue", "bold", "cold", "brave",
              "the", "and", "cat", "dog"}

# -----------------------------------------------------------------------------
# Question template banks -- ONE mixed bank per subtask. This is the knob for
# experimenting with how questions are phrased: just edit/add entries. Field
# placeholders per subtask:
#   spell:          {word}
#   spell_inverse:  {spaced}              (space-separated letters; answer=word)
#   contains_char:  {letter} {word}
#   orth:           {word} {a} {b}        (a,b candidates in randomized order)
#   ins_char:       {x} {y} {word}        (insert x after every y)
#   del_char:       {letter} {word}
#   sub_char:       {x} {y} {word}        (replace every x with y)
#   swap_char:      {x} {y} {word}
TEMPLATES = {
    "spell": [
        "Spell out the word {word}",
        "How do you spell {word}",
        "Spell {word} letter by letter",
        "Write {word} one letter at a time",
        "Give me the letters of {word}",
        "What are the letters in {word}",
        "Spell {word} out with spaces between the letters",
        "Break {word} into individual letters",
        "List the letters of {word} in order",
        "Spell the word {word}",
    ],
    "spell_inverse": [
        "What word is spelled by {spaced}",
        "Combine these letters into a word: {spaced}",
        "Read this spelled-out word: {spaced}",
        "{spaced} spells what word",
        "Join these letters into a single word: {spaced}",
        "Which word do the letters {spaced} make",
        "Unspell {spaced}",
        "Write {spaced} as one word with no spaces",
        "What word is this: {spaced}",
    ],
    "contains_char": [
        "Is there a {letter} in {word}",
        "Does {word} contain the letter {letter}",
        "Is {letter} in {word}",
        "Does the word {word} have a {letter} in it",
        "Can you find a {letter} in {word}",
        "Is the letter {letter} present in {word}",
        "Does {word} include {letter}",
        "Tell me whether {word} contains {letter}",
        "Is there any {letter} in the word {word}",
    ],
    "orth": [
        "Which is closer to {word}: {a} or {b}",
        "Between {a} and {b}, which is more similar to {word}",
        "Which of {a} or {b} is a smaller edit from {word}",
        "Which word is more like {word}, {a} or {b}",
        "Closer in spelling to {word}: {a} or {b}",
        "Which one is nearer to {word} by edit distance, {a} or {b}",
        "{a} or {b} -- which more closely resembles {word}",
    ],
    "ins_char": [
        "Add a {x} after every {y} in {word}",
        "Insert {x} following each {y} in {word}",
        "Put a {x} after each {y} in {word}",
        "In {word}, place a {x} after every {y}",
        "Append {x} after every occurrence of {y} in {word}",
        "After each {y} in {word}, add a {x}",
    ],
    "del_char": [
        "Delete every {letter} from {word}",
        "Remove all the {letter} in {word}",
        "Take out every {letter} in {word}",
        "Drop all {letter} from {word}",
        "Erase every {letter} in {word}",
        "What is {word} with all the {letter} removed",
        "Strip the {letter} out of {word}",
    ],
    "sub_char": [
        "Replace every {x} with {y} in {word}",
        "Substitute {y} for every {x} in {word}",
        "Change all {x} to {y} in {word}",
        "Swap out every {x} for a {y} in {word}",
        "In {word}, turn every {x} into a {y}",
        "Replace each {x} in {word} with {y}",
    ],
    "swap_char": [
        "Swap every {x} and {y} in {word}",
        "Exchange the {x} and {y} in {word}",
        "In {word}, switch every {x} with {y} and vice versa",
        "Trade every {x} for {y} and every {y} for {x} in {word}",
        "Swap the letters {x} and {y} throughout {word}",
    ],
}

# -----------------------------------------------------------------------------
# Per-subtask operations. Each returns (fields_dict, answer) or None to skip.
# fields_dict holds the raw (unwrapped) substitution values for the templates.

def _swap_chars(s, x, y):
    return "".join(y if c == x else x if c == y else c for c in s)


def op_spell(word, rng, pool):
    return {"word": word}, " ".join(word)


def op_spell_inverse(word, rng, pool):
    return {"spaced": " ".join(word)}, word


def op_contains_char(word, rng, pool):
    if rng.random() < 0.5 and len(set(word)) < 26:
        letter = rng.choice([c for c in string.ascii_lowercase if c not in word])
    else:
        letter = rng.choice(list(set(word)))
    ans = "Yes" if letter in word else "No"
    return {"letter": letter, "word": word}, ans


def op_ins_char(word, rng, pool):
    y = rng.choice(list(set(word)))
    x = rng.choice(string.ascii_lowercase)
    answer = "".join(c + x if c == y else c for c in word)
    return {"x": x, "y": y, "word": word}, answer


def op_del_char(word, rng, pool):
    # avoid the degenerate all-one-letter word -> empty answer
    for _ in range(4):
        letter = rng.choice(list(set(word)))
        answer = word.replace(letter, "")
        if answer:
            return {"letter": letter, "word": word}, answer
    return None


def op_sub_char(word, rng, pool):
    x = rng.choice(list(set(word)))
    y = rng.choice([c for c in string.ascii_lowercase if c != x])
    answer = word.replace(x, y)
    return {"x": x, "y": y, "word": word}, answer


def op_swap_char(word, rng, pool):
    letters = list(set(word))
    if len(letters) < 2:
        return None
    x, y = rng.sample(letters, 2)
    return {"x": x, "y": y, "word": word}, _swap_chars(word, x, y)


def _levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def op_orth(word, rng, pool):
    # near = one-substitution edit; far = random pool word; verify near closer.
    for _ in range(8):
        i = rng.randrange(len(word))
        new_c = rng.choice([c for c in string.ascii_lowercase if c != word[i]])
        near = word[:i] + new_c + word[i + 1:]
        far = rng.choice(pool)
        if near in (far, word) or far == word:
            continue
        if _levenshtein(word, near) < _levenshtein(word, far):
            a, b = (near, far) if rng.random() < 0.5 else (far, near)
            return {"word": word, "a": a, "b": b}, near
    return None


OPS = {
    "spell": op_spell,
    "spell_inverse": op_spell_inverse,
    "contains_char": op_contains_char,
    "orth": op_orth,
    "ins_char": op_ins_char,
    "del_char": op_del_char,
    "sub_char": op_sub_char,
    "swap_char": op_swap_char,
}

# -----------------------------------------------------------------------------
# Word pool, loaded once and shared across all subtask instances.

_word_pool = None


def _get_word_pool():
    global _word_pool
    if _word_pool is None:
        path = download_file_with_lock(WORD_LIST_URL, WORD_LIST_URL.split("/")[-1])
        with open(path, encoding="utf-8") as f:
            words = [w.strip() for w in f]
        words = [w for w in words if 3 <= len(w) <= 12 and w.isalpha() and w.islower()]
        excluded = _excluded_words()
        _word_pool = [w for w in words if w not in excluded]
    return _word_pool


def _excluded_words():
    """Every quoted string in every CUTE eval Question line, plus the demo
    words -- off-limits as training targets to avoid test contamination."""
    import re
    from datasets import load_dataset
    excluded = set(DEMO_WORDS)
    all_subtasks = ['spell', 'spell_inverse', 'contains_char', 'contains_word',
                    'orth', 'sem', 'ins_char', 'ins_word', 'del_char',
                    'del_word', 'sub_char', 'sub_word', 'swap_char', 'swap_word']
    for s in all_subtasks:
        ds = load_dataset("leukas/cute", split=s)
        for row in ds:
            q = row["prompt"].split("Question:")[-1]
            for q_str in re.findall(r'"\s*([^"]*?)\s*"', q):
                excluded.add(q_str)
    return excluded


def _wrap(s, rng):
    """Randomly quote a value (none / ' / ") for surface-form variety."""
    q = rng.choice(["", "'", '"'])
    return f"{q}{s}{q}"


class CuteChat(Task):
    """Synthetic chat-style CUTE questions for one char-level subtask."""

    def __init__(self, subtask, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert subtask in CUTE_CHAR_LEVEL, f"subtask must be one of {CUTE_CHAR_LEVEL}, got {subtask}"
        assert split in ("train", "test"), "split must be train|test"
        self.subtask = subtask
        self.size = size
        self.split = split
        self.pool = _get_word_pool()

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        op = OPS[self.subtask]

        # Draw a word and build the (fields, answer) pair; retry on op failure.
        result = None
        for _ in range(16):
            word = rng.choice(self.pool)
            result = op(word, rng, self.pool)
            if result is not None:
                break
        assert result is not None, f"could not build a {self.subtask} example"
        fields, answer = result

        # Phrase the question: pick a template, wrap values, jitter case/punct.
        template = rng.choice(TEMPLATES[self.subtask])
        if rng.random() < 0.3:
            template = template.lower()
        wrapped = {k: _wrap(v, rng) for k, v in fields.items()}
        user_msg = template.format(**wrapped)
        if rng.random() < 0.5:
            user_msg += "?"

        # Terse assistant turn: the quoted answer in CUTE surface form.
        assistant_msg = f'"{answer}"'

        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "answer": answer,
            "subtask": self.subtask,
        }

    def evaluate(self, conversation, assistant_response):
        gold = conversation["answer"]
        pred = extract_cute_answer(assistant_response, prefilled=False)
        return int(pred == gold)


if __name__ == "__main__":
    # Preview a few examples per subtask. Inject a tiny offline stand-in pool
    # so this runs without the words_alpha / leukas/cute downloads.
    _word_pool = ["strawberry", "tongue", "puzzle", "rhythm", "balloon",
                  "mississippi", "kitten", "orange", "letter", "banana",
                  "wonderful", "syzygy", "cellar", "address", "mammal"]
    for subtask in CUTE_CHAR_LEVEL:
        print("=" * 80)
        print(f"=== {subtask} ===")
        task = CuteChat(subtask=subtask, size=4)
        for i in range(4):
            ex = task.get_example(i)
            u = ex["messages"][0]["content"]
            a = ex["messages"][1]["content"]
            print(f"  U: {u!r}")
            print(f"  A: {a!r}")
