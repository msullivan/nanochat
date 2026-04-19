"""
Tasks intended to make nanochat better at basic arithmetic.

The current d24 model gets simple addition wrong (e.g. claims 2 + 4 = 5).
Mixing lots of arithmetic problems with shown work should help.

There are two tasks in this file:
1. Addition: addition problems. Mix of 2-term (most common) and
   3/4/5-term, with multi-term answers reducing one pair at a time.
2. Multiplication: multiplication problems. Small ones direct; larger
   ones broken into partial products by place value.

The answer format mirrors GSM8K, so the existing #### answer extractor
works without changes.

To preview example conversations, run:
    python -m tasks.arithmetic
"""

import re
import random
from tasks.common import Task


# Identical to gsm8k's answer extraction
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """Extract the numerical answer after the #### marker."""
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


# Fallback: last integer in the response. Used when the model hasn't learned
# to emit `#### <answer>` yet — for arithmetic prompts the last integer is
# usually the intended answer anyway.
LAST_INT_RE = re.compile(r"\d[\d,]*")
def extract_answer_loose(completion):
    """Prefer the #### marker; otherwise return the last integer in the text."""
    strict = extract_answer(completion)
    if strict is not None:
        return strict
    matches = LAST_INT_RE.findall(completion)
    if matches:
        return matches[-1].replace(",", "")
    return None


# A number bigger than any plausible dataset size to separate train/test seeds
TEST_RANDOM_SEED_OFFSET = 10_000_000


# Templates for 2-term addition. {op} is rendered as "+" or "plus".
ADD2_TEMPLATES = [
    "What is {a} {op} {b}?",
    "What is {a} {op} {b}",
    "{a} + {b} = ?",
    "{a} + {b}",
    "{a} {op} {b}",
    "Compute {a} + {b}",
    "Calculate {a} {op} {b}",
    "What's {a} {op} {b}",
    "Add {a} and {b}",
    "What is the sum of {a} and {b}?",
    "Sum: {a} + {b}",
    "Find {a} + {b}",
    "Evaluate {a} + {b}",
    "What does {a} {op} {b} equal?",
    "Solve {a} + {b}",
    "{a} {op} {b} equals what?",
    "Tell me {a} + {b}",
    "Give me the sum of {a} and {b}",
    "Quick: {a} + {b}",
    "I need {a} + {b}",
]

# Templates for n-term addition rendered as a full expression like "12 + 34 + 56".
ADDN_EXPR_TEMPLATES = [
    "What is {expr}?",
    "{expr} = ?",
    "{expr}",
    "Compute {expr}",
    "Calculate {expr}",
    "What's {expr}",
    "Sum: {expr}",
    "Find {expr}",
    "Evaluate {expr}",
    "Solve {expr}",
    "Tell me {expr}",
]

# Templates for n-term addition rendered as a comma-separated list.
ADDN_LIST_TEMPLATES = [
    "What is the sum of {nums}?",
    "Add {nums}",
    "Sum these numbers: {nums}",
    "What's the total of {nums}?",
    "Total of {nums}",
    "Find the sum of {nums}",
]

# Templates for multiplication. {op} is "*", "×", or "times".
MUL_TEMPLATES = [
    "What is {a} {op} {b}?",
    "What is {a} {op} {b}",
    "{a} * {b} = ?",
    "{a} × {b} = ?",
    "{a} {op} {b}",
    "Compute {a} * {b}",
    "Calculate {a} × {b}",
    "What's {a} times {b}?",
    "Multiply {a} and {b}",
    "What is the product of {a} and {b}?",
    "Find {a} × {b}",
    "Evaluate {a} * {b}",
    "Tell me {a} times {b}",
    "Solve {a} × {b}",
]


def _rand_number(rng, n_digits):
    """Return a random non-negative integer with exactly n_digits (n_digits=1 allows 0)."""
    if n_digits == 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (n_digits - 1), 10 ** n_digits - 1)


def _format_nums_list(nums, rng):
    """Render [12, 34, 56] as one of a few comma-and-'and' styles."""
    s = [str(n) for n in nums]
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        return f"{s[0]} and {s[1]}"
    style = rng.choice(["oxford", "no_oxford", "plain"])
    if style == "oxford":
        return ", ".join(s[:-1]) + ", and " + s[-1]
    if style == "no_oxford":
        return ", ".join(s[:-1]) + " and " + s[-1]
    return ", ".join(s)


def _addition_work(nums):
    """
    Shown-work string for n-term addition. For n==2 we do a single line:
        a + b = c

        #### c
    For n>=3 we reduce leftmost pair one step at a time:
        a + b + c + d
        = (a+b) + c + d
        = ((a+b)+c) + d
        = ...
    """
    total = sum(nums)
    if len(nums) == 2:
        return f"{nums[0]} + {nums[1]} = {total}\n\n#### {total}"
    lines = [" + ".join(str(n) for n in nums)]
    current = list(nums)
    while len(current) > 1:
        head = current[0] + current[1]
        current = [head] + current[2:]
        lines.append("= " + " + ".join(str(n) for n in current))
    return "\n".join(lines) + f"\n\n#### {total}"


def _multiplication_work(a, b):
    """
    Shown-work string for a * b. Small enough → direct one-liner.
    Otherwise break the larger operand by place value and sum the partial products.
    """
    product = a * b
    if min(a, b) < 10 or max(a, b) < 30:
        return f"{a} × {b} = {product}\n\n#### {product}"
    # Break the larger operand by place value; multiply by the smaller.
    if a >= b:
        to_break, mul, break_on_left = a, b, True
    else:
        to_break, mul, break_on_left = b, a, False
    digits = str(to_break)
    n = len(digits)
    parts, partial_values = [], []
    for i, d in enumerate(digits):
        dv = int(d)
        if dv == 0:
            continue
        coeff = dv * (10 ** (n - 1 - i))
        partial_values.append(mul * coeff)
        if break_on_left:
            parts.append(f"{coeff} × {mul}")
        else:
            parts.append(f"{mul} × {coeff}")
    if len(parts) <= 1:
        return f"{a} × {b} = {product}\n\n#### {product}"
    lines = [
        f"{a} × {b}",
        "= " + " + ".join(parts),
        "= " + " + ".join(str(p) for p in partial_values),
    ]
    current = list(partial_values)
    while len(current) > 1:
        head = current[0] + current[1]
        current = [head] + current[2:]
        lines.append("= " + " + ".join(str(n) for n in current))
    return "\n".join(lines) + f"\n\n#### {product}"


def _jitter_user_msg(user_msg, rng):
    """Small augmentations: lowercase sometimes, sometimes add/drop '?'."""
    if rng.random() < 0.3:
        user_msg = user_msg.lower()
    if user_msg.endswith("?"):
        if rng.random() < 0.4:
            user_msg = user_msg[:-1]
    else:
        if rng.random() < 0.3:
            user_msg += "?"
    return user_msg


class Addition(Task):
    """
    Addition practice. Mix of 2-term (most common) with occasional 3/4/5-term
    problems that reduce leftmost pair-by-pair. Each assistant answer shows
    work and ends in `#### answer` so the GSM8K-style evaluator works.
    """

    # default weights for (2, 3, 4, 5)-term problems
    DEFAULT_N_TERMS_WEIGHTS = (0.70, 0.15, 0.10, 0.05)

    def __init__(self, size=1000, split="train", n_terms_weights=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "Addition split must be train|test"
        self.size = size
        self.split = split
        self.n_terms_weights = n_terms_weights or self.DEFAULT_N_TERMS_WEIGHTS

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        n_terms = rng.choices([2, 3, 4, 5], weights=self.n_terms_weights, k=1)[0]

        # Per-problem digit distribution: wider for 2-term, smaller for many-term.
        if n_terms == 2:
            digit_choices, digit_weights = [1, 2, 3, 4], [0.10, 0.40, 0.35, 0.15]
        elif n_terms == 3:
            digit_choices, digit_weights = [1, 2, 3], [0.20, 0.55, 0.25]
        elif n_terms == 4:
            digit_choices, digit_weights = [1, 2, 3], [0.30, 0.55, 0.15]
        else:  # 5
            digit_choices, digit_weights = [1, 2], [0.50, 0.50]

        # Each operand independently samples its digit count.
        nums = []
        for _ in range(n_terms):
            d = rng.choices(digit_choices, weights=digit_weights, k=1)[0]
            nums.append(_rand_number(rng, d))

        # Build the user message.
        if n_terms == 2:
            template = rng.choice(ADD2_TEMPLATES)
            op = rng.choice(["+", "plus"])
            user_msg = template.format(a=nums[0], b=nums[1], op=op)
        else:
            # 60/40 between expression-style and list-style templates.
            if rng.random() < 0.6:
                template = rng.choice(ADDN_EXPR_TEMPLATES)
                expr = " + ".join(str(n) for n in nums)
                user_msg = template.format(expr=expr)
            else:
                template = rng.choice(ADDN_LIST_TEMPLATES)
                user_msg = template.format(nums=_format_nums_list(nums, rng))

        user_msg = _jitter_user_msg(user_msg, rng)

        assistant_text = _addition_work(nums)
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_text},
        ]
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        """0 = wrong, 1 = correct. Uses loose extraction on the prediction
        (prefers `####`, falls back to last integer) so this still gives a
        meaningful signal on models that haven't learned the `####` format."""
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        ref_content = assistant_message['content']
        ref_text = ref_content[-1]['text'] if isinstance(ref_content, list) else ref_content
        ref_num = extract_answer(ref_text)
        pred_num = extract_answer_loose(assistant_response)
        return int(pred_num == ref_num)

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))


class Multiplication(Task):
    """
    Multiplication practice. Small numbers answered directly; larger ones
    broken into partial products by place value before the final sum.
    """

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "Multiplication split must be train|test"
        self.size = size
        self.split = split

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        d_a = rng.choices([1, 2, 3], weights=[0.30, 0.50, 0.20], k=1)[0]
        d_b = rng.choices([1, 2, 3], weights=[0.40, 0.50, 0.10], k=1)[0]
        a = _rand_number(rng, d_a)
        b = _rand_number(rng, d_b)
        # 0 is uninteresting; bump to 1-9 so we get real practice.
        if a == 0:
            a = rng.randint(1, 9)
        if b == 0:
            b = rng.randint(1, 9)

        template = rng.choice(MUL_TEMPLATES)
        op = rng.choice(["*", "×", "times"])
        user_msg = template.format(a=a, b=b, op=op)
        user_msg = _jitter_user_msg(user_msg, rng)

        assistant_text = _multiplication_work(a, b)
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_text},
        ]
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant"
        ref_content = assistant_message['content']
        ref_text = ref_content[-1]['text'] if isinstance(ref_content, list) else ref_content
        ref_num = extract_answer(ref_text)
        pred_num = extract_answer_loose(assistant_response)
        return int(pred_num == ref_num)

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))


if __name__ == "__main__":
    print("=" * 100)
    print("Addition examples:")
    print("=" * 100)
    task = Addition()
    for i in range(20):
        ex = task.get_example(i)
        print("-" * 100)
        print("USER:     ", ex['messages'][0]['content'])
        print("ASSISTANT:")
        print(ex['messages'][1]['content'])
    print()
    print("=" * 100)
    print("Multiplication examples:")
    print("=" * 100)
    task = Multiplication()
    for i in range(15):
        ex = task.get_example(i)
        print("-" * 100)
        print("USER:     ", ex['messages'][0]['content'])
        print("ASSISTANT:")
        print(ex['messages'][1]['content'])
