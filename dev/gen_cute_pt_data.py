"""
Generate synthetic CUTE-format completion training data.

Produces a parquet file with a 'text' column whose rows are full
CUTE-style documents -- 4-shot demo prefix + Question + Answer in quotes --
matching the published `leukas/cute` benchmark format byte-for-byte so the
model trains on the same surface form it sees at eval time (with the same
`ANSWER_PREFILL` convention).

7 char-level subtasks are emitted (spell, spell_inverse, contains_char,
ins_char, del_char, sub_char, swap_char) plus orth (Levenshtein selection,
generated from synthetic candidate pairs). Word-level subtasks are out of
scope: the byte-vs-BPE structural advantage is on character manipulation,
and word-level tasks add an unrelated semantic-similarity load.

Test contamination is avoided by loading all 14 CUTE subtasks and
excluding every quoted string that appears in any `Question:` line.

The dataloader splits a parquet directory by file (train = all but last,
val = last), so this writer always emits two shards: shard_00000.parquet
(train) and shard_00001.parquet (val). Use --val-frac to size the val
slice. Default 1%.

Default --num-words is intentionally tiny (1000) so the cheapest possible
finetune runs end-to-end first; bump it up once the pipeline is verified.

Usage:
    .venv/bin/python dev/gen_cute_pt_data.py \\
        --out-dir $NANOCHAT_BASE_DIR/cute_pt_data \\
        --num-words 1000 \\
        --seed 0
"""

import argparse
import os
import random
import re
import string
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from nanochat.common import download_file_with_lock

WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"

# Verbatim 4-shot prefixes pulled from leukas/cute row[0] of each subtask
# (`prompt` field, sliced up to "Question:"). DO NOT edit the whitespace --
# the dataset has 12-space indentation, the trailing demo block ends with
# "\n            \n            " before "Question:". The model is going to
# see this surface form at eval time, so training has to match.
PREFIXES = {
    "spell": (
        'Spell out the word, putting spaces between each letter, based on the following examples:\n'
        '            \n'
        '            1. Spell out the word " alphabet ". Answer: " a l p h a b e t "\n'
        '            2. Spell out the word " hello ". Answer: " h e l l o "\n'
        '            3. Spell out the word " zebra ". Answer: " z e b r a "\n'
        '            4. Spell out the word " tongue ". Answer: " t o n g u e "\n'
        '            \n'
        '            '
    ),
    "spell_inverse": (
        'Write the word that is spelled out, without any spaces, based on the following examples:\n'
        '            \n'
        '            1. Write the word " a l p h a b e t ".  Answer: " alphabet "\n'
        '            2. Write the word " h e l l o ". Answer: " hello "\n'
        '            3. Write the word " z e b r a ". Answer: " zebra "\n'
        '            4. Write the word " t o n g u e ". Answer: " tongue "\n'
        '            \n'
        '            '
    ),
    "contains_char": (
        'Answer whether the specified letter is in the given word, based on the following examples:\n'
        '            \n'
        '            1. Is there a " a " in " alphabet "? Answer: "Yes"\n'
        '            2. Is there a " z " in " alphabet "? Answer: "No"\n'
        '            3. Is there a " u " in " hello "? Answer: "No"\n'
        '            4. Is there a " o " in " hello "? Answer: "Yes"\n'
        '            \n'
        '            '
    ),
    "orth": (
        'Select the word that is closer in Levenshtein distance to the given word based on the following examples:\n'
        '            \n'
        '            1. Closer in Levenshtein distance to "bold": "cold", "brave". Answer: "cold"\n'
        '            2. Closer in Levenshtein distance to "computer": "completed", "laptop". Answer: "completed"\n'
        '            3. Closer in Levenshtein distance to "happy": "glad, "apply". Answer: "apply"\n'
        '            4. Closer in Levenshtein distance to "camp": "ramp", "tent". Answer: "ramp"\n'
        '            \n'
        '            '
    ),
    "ins_char": (
        'Add the specified letter after every instance of the second specified letter in a given word, based on the following examples:\n'
        '            \n'
        '            1. Add an " e " after every " a " in " alphabet ". Answer: " aelphaebet "\n'
        '            2. Add an " l " after every " l " in " hello ". Answer: " hellllo "\n'
        '            3. Add an " t " after every " z " in " zebra ". Answer: " ztebra "\n'
        '            4. Add an " f " after every " u " in " tongue ". Answer: " tongufe "\n'
        '            \n'
        '            '
    ),
    "del_char": (
        'Delete every instance of a specified letter in a given word, based on the following examples:\n'
        '            \n'
        '            1. Delete every instance of " a " in " alphabet ". Answer: " lphbet "\n'
        '            2. Delete every instance of " l " in " hello ". Answer: " heo "\n'
        '            3. Delete every instance of " z " in " zebra ". Answer: " ebra "\n'
        '            4. Delete every instance of " u " in " tongue ". Answer: " tonge "\n'
        '            \n'
        '            '
    ),
    "sub_char": (
        'Substitute the first specified letter with the second specified letter in a given word, based on the following examples:\n'
        '            \n'
        '            1. Substitute " a " with " b " in " alphabet ". Answer: " blphbbet "\n'
        '            2. Substitute " h " with " e " in " hello ". Answer: " eello "\n'
        '            3. Substitute " z " with " a " in " zebra ". Answer: " aebra "\n'
        '            4. Substitute " u " with " e " in " tongue ". Answer: " tongee "\n'
        '            \n'
        '            '
    ),
    "swap_char": (
        'Swap the positions of two specified letters in a given word, based on the following examples:\n'
        '            \n'
        '            1. Swap " l " and " b " in " alphabet ". Answer: " abphalet "\n'
        '            2. Swap " h " and " e " in " hello ". Answer: " ehllo "\n'
        '            3. Swap " z " and " a " in " zebra ". Answer: " aebrz "\n'
        '            4. Swap " u " and " e " in " tongue ". Answer: " tongeu "\n'
        '            \n'
        '            '
    ),
}

# Demo words used in the 4-shot prefixes; never used as targets.
DEMO_WORDS = {"alphabet", "hello", "zebra", "tongue", "bold", "cold", "brave",
              "computer", "completed", "laptop", "happy", "glad", "apply",
              "camp", "ramp", "tent"}


# -----------------------------------------------------------------------------
# Per-subtask example builders. Each returns (question_str, answer_str).
# answer_str does NOT include surrounding quotes; the writer wraps it.

def _swap_chars(s, x, y):
    out = []
    for c in s:
        if c == x: out.append(y)
        elif c == y: out.append(x)
        else: out.append(c)
    return "".join(out)


def make_spell(word, rng):
    return f'Question: Spell out the word " {word} ".', " ".join(word)


def make_spell_inverse(word, rng):
    return f'Question: Write the word " {" ".join(word)} ".', word


def make_contains_char(word, rng):
    # 50/50: a letter that's in the word vs one that isn't, matching the
    # 2-yes-2-no demo split.
    if rng.random() < 0.5 and len(set(word)) < 26:
        letter = rng.choice([c for c in string.ascii_lowercase if c not in word])
        ans = "No"
    else:
        letter = rng.choice(list(set(word))) if word else rng.choice(string.ascii_lowercase)
        ans = "Yes" if letter in word else "No"
    return f'Question: Is there a " {letter} " in " {word} "?', ans


def make_ins_char(word, rng):
    # Add letter X after every Y in word; pick Y from the word's letters
    # so the answer is non-trivial.
    y = rng.choice(list(set(word)))
    x = rng.choice(string.ascii_lowercase)
    answer = "".join(c + x if c == y else c for c in word)
    return f'Question: Add an " {x} " after every " {y} " in " {word} ".', answer


def make_del_char(word, rng):
    letter = rng.choice(list(set(word)))
    answer = word.replace(letter, "")
    return f'Question: Delete every instance of " {letter} " in " {word} ".', answer


def make_sub_char(word, rng):
    x = rng.choice(list(set(word)))
    y = rng.choice([c for c in string.ascii_lowercase if c != x])
    answer = word.replace(x, y)
    return f'Question: Substitute " {x} " with " {y} " in " {word} ".', answer


def make_swap_char(word, rng):
    letters = list(set(word))
    if len(letters) < 2:
        # rare for a useful swap_char; fall back to swapping with a random
        # letter not in the word (effectively a substitute, but the format is
        # still "swap")
        x = letters[0]
        y = rng.choice([c for c in string.ascii_lowercase if c != x])
    else:
        x, y = rng.sample(letters, 2)
    answer = _swap_chars(word, x, y)
    return f'Question: Swap " {x} " and " {y} " in " {word} ".', answer


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


def make_orth(word, rng, *, near_pool, far_pool):
    # Build a one-edit-away candidate (by single-char substitution) and a
    # totally-different candidate, then verify the near one is actually closer.
    # Try a few times; if we can't get a valid pair, return None and skip.
    for _ in range(8):
        # near: change one random position to a different random letter
        i = rng.randrange(len(word))
        new_c = rng.choice([c for c in string.ascii_lowercase if c != word[i]])
        near = word[:i] + new_c + word[i + 1:]
        far = rng.choice(far_pool)
        if near == far or near == word or far == word:
            continue
        d_near = _levenshtein(word, near)
        d_far = _levenshtein(word, far)
        if d_near < d_far:
            # randomize order to avoid the answer always being first
            if rng.random() < 0.5:
                question = f'Question: Closer in Levenshtein distance to " {word} ": " {near} ", " {far} ".'
            else:
                question = f'Question: Closer in Levenshtein distance to " {word} ": " {far} ", " {near} ".'
            return question, near
    return None  # caller skips


SUBTASKS = {
    "spell": make_spell,
    "spell_inverse": make_spell_inverse,
    "contains_char": make_contains_char,
    "ins_char": make_ins_char,
    "del_char": make_del_char,
    "sub_char": make_sub_char,
    "swap_char": make_swap_char,
    # orth handled separately because it needs a far-pool argument
}

# Per-subtask answer formatting. Tasks where the demo prints the answer with
# a leading/trailing space inside the quotes need to match exactly; tasks
# whose demos use bare-quoted answers (Yes/No, single-word selection) don't.
ANSWER_HAS_SPACES = {
    "spell": True,
    "spell_inverse": True,
    "contains_char": False,
    "orth": False,
    "ins_char": True,
    "del_char": True,
    "sub_char": True,
    "swap_char": True,
}


def format_document(subtask, question, answer):
    """Build the full document text: prefix + question + Answer line."""
    quoted = f'" {answer} "' if ANSWER_HAS_SPACES[subtask] else f'"{answer}"'
    return f'{PREFIXES[subtask]}{question}\n\nAnswer: {quoted}'


# -----------------------------------------------------------------------------
def collect_cute_target_words():
    """Load all 14 CUTE subtasks and collect every quoted string from each
    Question line. These are off-limits as training words to avoid test-set
    contamination."""
    excluded = set(DEMO_WORDS)
    subtasks_all = ['spell', 'spell_inverse', 'contains_char', 'contains_word',
                    'orth', 'sem', 'ins_char', 'ins_word', 'del_char',
                    'del_word', 'sub_char', 'sub_word', 'swap_char', 'swap_word']
    for s in subtasks_all:
        ds = load_dataset("leukas/cute", split=s)
        for row in ds:
            q = row["prompt"].split("Question:")[-1]
            for q_str in re.findall(r'"\s*([^"]*?)\s*"', q):
                excluded.add(q_str)
    return excluded


def load_word_list():
    path = download_file_with_lock(WORD_LIST_URL, WORD_LIST_URL.split("/")[-1])
    with open(path, encoding="utf-8") as f:
        words = [w.strip() for w in f]
    # Filter to lowercase 3-12 letter alpha words (typical CUTE word length).
    out = [w for w in words if 3 <= len(w) <= 12 and w.isalpha() and w.islower()]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", help="directory for the two output shards")
    parser.add_argument("--num-words", type=int, default=1000,
                        help="number of source words to use; total examples = num_words * 8 subtasks (minus orth fails)")
    parser.add_argument("--val-frac", type=float, default=0.01, help="fraction of generated docs put in shard_00001.parquet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=int, default=0,
                        help="if >0, print this many rendered examples per subtask and exit (no parquet written)")
    args = parser.parse_args()
    if not args.preview and not args.out_dir:
        parser.error("--out-dir is required unless --preview is used")

    rng = random.Random(args.seed)

    print("loading words_alpha.txt and CUTE exclusion set...")
    words_all = load_word_list()
    excluded = collect_cute_target_words()
    print(f"  word source: {len(words_all)} candidates")
    print(f"  CUTE excluded: {len(excluded)} target strings (from all 14 subtasks)")
    candidates = [w for w in words_all if w not in excluded]
    print(f"  after exclusion: {len(candidates)} candidates")

    rng.shuffle(candidates)
    if args.num_words > len(candidates):
        raise SystemExit(f"requested {args.num_words} words but only {len(candidates)} are available after exclusion")
    pool = candidates[:args.num_words]

    # far-pool for orth candidates: a wider random sample so orth has variety
    # of "obviously different" candidates to choose from
    far_pool = candidates[args.num_words:args.num_words + 5000] or candidates[:5000]

    docs = []
    counts = Counter()

    if args.preview:
        for subtask in list(SUBTASKS.keys()) + ["orth"]:
            print("=" * 80)
            print(f"=== {subtask} ===")
            for w in pool[:args.preview]:
                if subtask == "orth":
                    res = make_orth(w, rng, near_pool=pool, far_pool=far_pool)
                    if res is None:
                        continue
                    q, a = res
                else:
                    q, a = SUBTASKS[subtask](w, rng)
                print(format_document(subtask, q, a))
                print("---")
        return

    for subtask, fn in SUBTASKS.items():
        for w in pool:
            q, a = fn(w, rng)
            docs.append(format_document(subtask, q, a))
            counts[subtask] += 1

    for w in pool:
        res = make_orth(w, rng, near_pool=pool, far_pool=far_pool)
        if res is None:
            continue
        q, a = res
        docs.append(format_document("orth", q, a))
        counts["orth"] += 1

    rng.shuffle(docs)

    print(f"\ngenerated {len(docs)} examples:")
    for s, n in counts.most_common():
        print(f"  {s:<16s} {n}")

    n_val = max(1, int(len(docs) * args.val_frac))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "shard_00000.parquet")
    val_path = os.path.join(args.out_dir, "shard_00001.parquet")
    pq.write_table(pa.table({"text": train_docs}), train_path)
    pq.write_table(pa.table({"text": val_docs}), val_path)
    print(f"wrote {train_path}  ({len(train_docs):,} docs, {os.path.getsize(train_path)/1e6:.1f} MB)  [train]")
    print(f"wrote {val_path}  ({len(val_docs):,} docs, {os.path.getsize(val_path)/1e6:.1f} MB)  [val]")


if __name__ == "__main__":
    main()
