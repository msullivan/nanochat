"""Breakdown report comparing two GSM8K eval runs by per-problem correctness.

Reads --save-records JSONL files (one record per problem: index, question, gold,
completion, pred, correct), aligns by index (records share the seed-42 shuffle),
partitions problems into A-only / B-only / both-correct / both-wrong, and writes a
markdown report with counts + N examples per group.

Usage:
    python analysis/gsm8k_breakdown.py \
        --a byte:analysis/gsm8k_records/byte_rl_2k.jsonl \
        --b bpe:analysis/gsm8k_records/bpe_rl.jsonl \
        --n 5 --out analysis/gsm8k_breakdown.md
"""
import argparse, json, os


def load(path):
    return {r["index"]: r for r in (json.loads(l) for l in open(path))}


def fmt_example(i, a_name, b_name, a, b):
    out = []
    out.append(f"### Problem {i}  ·  gold = `{a[i].get('gold')}`")
    out.append("")
    out.append(f"**Q:** {a[i].get('question','').strip()}")
    out.append("")
    for name, rec in [(a_name, a[i]), (b_name, b[i])]:
        mark = "✓" if rec["correct"] else "✗"
        out.append(f"**{name} [{mark}] pred=`{rec.get('pred')}`:**")
        out.append("```")
        out.append(rec["completion"].strip())
        out.append("```")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="name:path.jsonl")
    ap.add_argument("--b", required=True, help="name:path.jsonl")
    ap.add_argument("--n", type=int, default=5, help="examples per group")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a_name, a_path = args.a.split(":", 1)
    b_name, b_path = args.b.split(":", 1)
    a, b = load(a_path), load(b_path)
    idx = sorted(set(a) & set(b))
    N = len(idx)

    groups = {
        f"{a_name}-only (correct only here)": [i for i in idx if a[i]["correct"] and not b[i]["correct"]],
        f"{b_name}-only (correct only here)": [i for i in idx if b[i]["correct"] and not a[i]["correct"]],
        "both correct": [i for i in idx if a[i]["correct"] and b[i]["correct"]],
        "both wrong": [i for i in idx if not a[i]["correct"] and not b[i]["correct"]],
    }

    a_acc = sum(r["correct"] for r in a.values()) / N
    b_acc = sum(r["correct"] for r in b.values()) / N

    L = []
    L.append(f"# GSM8K breakdown: {a_name} vs {b_name}")
    L.append("")
    L.append(f"Aligned by problem index over N={N} problems (shared seed-42 shuffle).")
    L.append("")
    L.append(f"- **{a_name}** accuracy: {a_acc:.4f}")
    L.append(f"- **{b_name}** accuracy: {b_acc:.4f}")
    L.append("")
    L.append("| group | count | % of N |")
    L.append("|---|---:|---:|")
    for name, members in groups.items():
        L.append(f"| {name} | {len(members)} | {100*len(members)/N:.1f}% |")
    L.append("")
    # "no answer produced" tallies (failure-mode asymmetry)
    a_noans = sum(1 for i in idx if a[i].get("pred") is None)
    b_noans = sum(1 for i in idx if b[i].get("pred") is None)
    L.append(f"No-answer (no `####` extracted): {a_name}={a_noans}, {b_name}={b_noans}")
    L.append("")

    for name, members in groups.items():
        L.append(f"## {name} — {len(members)} problems")
        L.append("")
        if not members:
            L.append("_(none)_\n")
            continue
        for i in members[: args.n]:
            L.append(fmt_example(i, a_name, b_name, a, b))
        L.append("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out}: N={N}, " + ", ".join(f"{k.split(' ')[0]}={len(v)}" for k, v in groups.items()))


if __name__ == "__main__":
    main()
