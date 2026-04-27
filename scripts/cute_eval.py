"""
CUTE benchmark eval driver. Supports three checkpoint/mode combinations:

  --source base --mode completion   # ICL on base model (paper-style methodology)
  --source sft  --mode completion   # ICL on SFT'd model (still works! we own the API)
  --source sft  --mode chat         # SFT model with the prompt as a user turn

(--source base --mode chat is rejected: base models have no chat tokens.)

Greedy decoding (temperature=0). Prompt is the HF leukas/cute prompt with
`\\n\\nAnswer: "` appended as prefill; we parse up to the next `"`.

Examples:
    python -m scripts.cute_eval -i base --mode completion -g d24-byte
    torchrun --nproc_per_node=8 -m scripts.cute_eval -i sft --mode completion
    python -m scripts.cute_eval -i sft --mode chat --subtasks swap_char,ins_char
"""

import argparse
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.cute import CUTE, CUTE_SUBTASKS, CUTE_CHAR_LEVEL


def run_cute_subtask(task_object, tokenizer, engine, max_new_tokens, max_problems=None, debug_n=0):
    from tasks.cute import extract_cute_answer
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = engine.model.get_device() if hasattr(engine, "model") else None

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        if task_object.mode == "chat":
            encoded_prompt = tokenizer.render_for_completion(conversation)
        else:
            encoded_prompt = tokenizer.encode(conversation["prompt_text"], prepend="<|bos|>")

        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_k=50,
        )
        prefix_length = len(encoded_prompt)
        completion = tokenizer.decode(results[0][prefix_length:])

        outcome = task_object.evaluate(conversation, completion)
        total += 1
        num_passed += int(outcome)

        if debug_n > 0 and i < debug_n and ddp_rank == 0:
            pred = extract_cute_answer(completion, prefilled=task_object.prefill)
            print(f"\n[debug {task_object.subtask} #{i}] gold={conversation['answer']!r} pred={pred!r} ok={outcome}")
            print(f"  raw completion (first 120 chars): {completion[:120]!r}")

        print(f"\r\033[KRank {ddp_rank} | {task_object.subtask} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end="", flush=True)

    print()

    if ddp:
        device = device or torch.device(f"cuda:{ddp_local_rank}")
        num_passed_t = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_t = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        num_passed = num_passed_t.item()
        total = total_t.item()

    return num_passed, total


def main():
    parser = argparse.ArgumentParser(description="CUTE benchmark eval")
    parser.add_argument("-i", "--source", type=str, required=True, choices=["base", "sft", "rl"], help="Checkpoint source")
    parser.add_argument("--mode", type=str, default="completion", choices=["completion", "chat"], help="Prompt format")
    parser.add_argument("--subtasks", type=str, default=None, help="Comma-separated subset of subtasks (default: all 14). Use 'char' for the char-level subset.")
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    parser.add_argument("-m", "--max-new-tokens", type=int, default=64, help="Max new tokens per generation (CUTE answers are short)")
    parser.add_argument("-x", "--max-problems", type=int, default=None, help="Max problems per subtask")
    parser.add_argument("--debug-n", type=int, default=0, help="Print raw completion + parse for the first N examples per subtask")
    parser.add_argument("--no-prefill", action="store_true", help='Do not append \\n\\nAnswer: " to the prompt; let the model emit the answer turn itself')
    parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"])
    args = parser.parse_args()

    if args.source == "base" and args.mode == "chat":
        parser.error("--source base --mode chat is invalid: base models have no chat tokens")

    if args.subtasks is None:
        subtasks = CUTE_SUBTASKS
    elif args.subtasks == "char":
        subtasks = CUTE_CHAR_LEVEL
    else:
        subtasks = [s.strip() for s in args.subtasks.split(",")]
        for s in subtasks:
            if s not in CUTE_SUBTASKS:
                parser.error(f"Unknown subtask: {s}. Valid: {CUTE_SUBTASKS}")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    print0(f"CUTE eval | source={args.source} | mode={args.mode} | model={args.model_tag or 'default'} step={meta.get('step', '?')}")
    print0(f"Subtasks: {subtasks}")

    results = {}
    for subtask in subtasks:
        task = CUTE(subtask=subtask, mode=args.mode, prefill=not args.no_prefill)
        num_passed, total = run_cute_subtask(task, tokenizer, engine, args.max_new_tokens, max_problems=args.max_problems, debug_n=args.debug_n)
        acc = num_passed / total if total > 0 else 0.0
        results[subtask] = acc
        print0(f"{subtask}: {num_passed}/{total} ({100*acc:.2f}%)")

    print0("=" * 60)
    print0(f"{'subtask':<20} {'accuracy':>10}")
    print0("-" * 60)
    for subtask in subtasks:
        print0(f"{subtask:<20} {100*results[subtask]:>9.2f}%")
    avg = sum(results.values()) / len(results)
    print0("-" * 60)
    print0(f"{'mean':<20} {100*avg:>9.2f}%")

    from nanochat.report import get_report
    get_report().log(section=f"CUTE eval {args.source} {args.mode}", data=[
        vars(args),
        {f"cute/{k}": v for k, v in results.items()},
        {"cute/mean": avg},
    ])

    compute_cleanup()


if __name__ == "__main__":
    main()
