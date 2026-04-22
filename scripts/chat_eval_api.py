"""
Run nanochat chat evals against an OpenAI-compatible /v1/chat/completions endpoint.

Example:
    OPENAI_API_KEY=... python -m scripts.chat_eval_api -a GSM8K --model gpt-4o-mini
    python -m scripts.chat_eval_api -a Addition \
        --base-url http://localhost:8000/v1 --model my-model --api-key-env NONE

For categorical tasks (MMLU, ARC-*) we can't access logits, so we just prompt
the model and scan the response for the first matching letter.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Task imports are lazy (via build_task) so tasks that pull in torch (e.g. SpellingBee
# via nanochat.common) don't block the script from running on a torch-less machine.
TASK_NAMES = ['HumanEval', 'MMLU', 'ARC-Easy', 'ARC-Challenge', 'GSM8K', 'SpellingBee', 'Addition', 'Multiplication']


def build_task(name):
    if name == 'HumanEval':
        from tasks.humaneval import HumanEval
        return HumanEval()
    if name == 'MMLU':
        from tasks.mmlu import MMLU
        return MMLU(subset="all", split="test")
    if name == 'ARC-Easy':
        from tasks.arc import ARC
        return ARC(subset="ARC-Easy", split="test")
    if name == 'ARC-Challenge':
        from tasks.arc import ARC
        return ARC(subset="ARC-Challenge", split="test")
    if name == 'GSM8K':
        from tasks.gsm8k import GSM8K
        return GSM8K(subset="main", split="test")
    if name == 'SpellingBee':
        from tasks.spellingbee import SpellingBee
        return SpellingBee(size=256, split="test")
    if name == 'Addition':
        from tasks.arithmetic import Addition
        return Addition(size=256, split="test")
    if name == 'Multiplication':
        from tasks.arithmetic import Multiplication
        return Multiplication(size=256, split="test")
    raise ValueError(f"unknown task {name!r}. known: {TASK_NAMES}")


def strip_gold(conversation):
    messages = conversation['messages']
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    return messages


def parse_letter(response, letters):
    letterset = set(letters)
    for ch in response:
        if ch in letterset:
            return ch
    return None


class OpenAIClient:
    def __init__(self, base_url, api_key, model, timeout=180, max_retries=5):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, messages, *, max_tokens, temperature, n=1):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if n != 1:
            payload["n"] = n
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code == 429 or 500 <= r.status_code < 600:
                last_err = requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                time.sleep(min(2 ** attempt, 30))
                continue
            if not r.ok:
                raise requests.HTTPError(f"{r.status_code}: {r.text[:500]}")
            body = r.json()
            return [c['message']['content'] or "" for c in body['choices']]
        raise RuntimeError(f"exhausted retries: {last_err}")


def run_eval(task_object, client, *, num_samples, max_tokens, temperature,
             concurrency, max_problems, show_failures, verbose):
    n_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    is_categorical = task_object.eval_type == 'categorical'

    def process(i):
        conversation = task_object[i]
        messages = strip_gold(conversation)
        completions = client.complete(
            messages, max_tokens=max_tokens, temperature=temperature, n=num_samples,
        )
        if is_categorical:
            letters = conversation['letters']
            outcomes = []
            for c in completions:
                pred = parse_letter(c, letters)
                outcomes.append(task_object.evaluate(conversation, pred) if pred is not None else 0)
        else:
            outcomes = [task_object.evaluate(conversation, c) for c in completions]
        return i, any(outcomes), completions, conversation

    num_passed, total, errors = 0, 0, 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(process, i) for i in range(n_problems)]
        for fut in as_completed(futures):
            try:
                i, passed, completions, conversation = fut.result()
            except Exception as e:
                errors += 1
                print(f"\n[error] {type(e).__name__}: {e}", flush=True)
                continue
            total += 1
            num_passed += int(passed)
            if verbose or (show_failures and not passed):
                user_msg = conversation['messages'][0]['content']
                gold_msg = conversation['messages'][-1]['content'] if conversation['messages'][-1]['role'] == 'assistant' else ''
                tag = 'PASS' if passed else 'FAIL'
                print(f"\n--- {tag} #{i} ---\nUSER: {user_msg}\nGOLD: {gold_msg}\nMODEL: {completions[0]}\n", flush=True)
            print(f"\r\033[K{num_passed}/{total} ({100*num_passed/max(total,1):.2f}%)", end='', flush=True)
    print()
    if errors:
        print(f"warning: {errors} problem(s) errored out and were skipped", file=sys.stderr)
    return num_passed / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--task-name', type=str, default=None,
                        help=f"Task name, or pipe-separated list. Known: {'|'.join(TASK_NAMES)}. Default = all.")
    parser.add_argument('--base-url', type=str, default='https://api.openai.com/v1',
                        help='OpenAI-compatible endpoint base URL (default: OpenAI)')
    parser.add_argument('--model', type=str, required=True, help='Model id to send in the request')
    parser.add_argument('--api-key-env', type=str, default='OPENAI_API_KEY',
                        help='Env var holding the API key. Use NONE to send unauthenticated.')
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-c', '--concurrency', type=int, default=8)
    parser.add_argument('-x', '--max-problems', type=int, default=None)
    parser.add_argument('--show-failures', action='store_true', help='Print each failed (user, gold, model) triple')
    parser.add_argument('--verbose', action='store_true', help='Print every (user, gold, model) triple, pass or fail')
    args = parser.parse_args()

    api_key = None if args.api_key_env.upper() == 'NONE' else os.environ.get(args.api_key_env)
    if args.api_key_env.upper() != 'NONE' and not api_key:
        print(f"warning: ${args.api_key_env} not set; sending unauthenticated", file=sys.stderr)

    if args.num_samples > 1 and args.temperature == 0.0:
        print("warning: --num-samples > 1 with temperature=0 produces identical completions", file=sys.stderr)

    client = OpenAIClient(args.base_url, api_key, args.model)
    task_names = TASK_NAMES if args.task_name is None else args.task_name.split('|')

    results = {}
    for task_name in task_names:
        print(f"=== {task_name} ({args.model} @ {args.base_url}) ===")
        task_object = build_task(task_name)
        acc = run_eval(
            task_object, client,
            num_samples=args.num_samples,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            max_problems=args.max_problems,
            show_failures=args.show_failures,
            verbose=args.verbose,
        )
        results[task_name] = acc
        print(f"{task_name}: {100*acc:.2f}%")

    if len(results) > 1:
        print("\n=== Summary ===")
        for k, v in results.items():
            print(f"  {k}: {100*v:.2f}%")


if __name__ == "__main__":
    main()
