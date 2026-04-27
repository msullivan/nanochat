# CUTE benchmark — notes for nanochat-byte

Written 2026-04-26. Source: Edman, Schmid, Fraser, *CUTE: Measuring LLMs' Understanding of Their Tokens*, EMNLP 2024 ([arXiv:2409.15452](https://arxiv.org/abs/2409.15452)).

- **GitHub:** https://github.com/Leukas/CUTE — TSV data + generation scripts
- **HuggingFace:** https://huggingface.co/datasets/leukas/cute — 14 splits × 1k rows, pre-rendered `{prompt, answer}`
- Local cached source: `~/.cache/nanochat/knowledge/2409.15452/acl_latex.tex`

## TL;DR

CUTE has **14 subtasks** (7 conceptual categories × char/word level, minus similarity which has no word version). We currently have one of them (`spell` ≈ `SimpleSpelling`) and one related-but-not-CUTE task (letter counting in `SpellingBee`, closest CUTE relative is `contains_char`). The most additive subtasks to add — both for training and as an eval headline — are the **char-level manipulation** ones: `ins_char`, `swap_char`, `sub_char`, plus `spell_inverse` and `contains_char`. These are the subtasks where BPE LLMs catastrophically fail and a byte model is structurally well-positioned.

## The full task list

Composition (4):
- **`spell`**: `Spell out the word "alphabet"` → `a l p h a b e t`. *(have it: `SimpleSpelling`)*
- **`spell_inverse`**: `Write the word that is spelled out: a l p h a b e t` → `alphabet`
- **`contains_char`**: `Is there a 'c' in 'alphabet'?` → `No`
- **`contains_word`**: `Is there a 'the' in 'the sky is blue'?` → `Yes`

Similarity (2, no word-level pair):
- **`orth`**: `Which is closer in Levenshtein distance to "happy"? glad or apply` → `apply`
- **`sem`**: `Which is more semantically related to "happy"? glad or apply` → `glad`

Manipulation (8 = 4 ops × 2 levels):
- **`ins_char` / `ins_word`**: `Add 'b' after every 'a' in 'alphabet'` → `ablphabbet`
- **`del_char` / `del_word`**: `Delete every 'a' in 'alphabet'` → `lphbet`
- **`sub_char` / `sub_word`**: `Replace every 'a' with 'b' in 'alphabet'` → `blphbbet`
- **`swap_char` / `swap_word`**: `Swap 'a' and 'b' in 'back'` → `abck`

What's *not* in CUTE: letter counting (our `SpellingBee` is unique), reversal, unscrambling.

## Where BPE LLMs fail (most discriminating subtasks)

Numbers from the paper's main table; "best" = best across the 13 BPE LLMs evaluated (mostly Cmd-R+ 104B, sometimes DBRX 132B):

| Task | Best BPE | Llama-2 7B | Discrimination |
|---|---|---|---|
| `swap_char` | **10.0%** | 3.5% | catastrophic |
| `ins_char` | 8.9% (best Mistral-47B 15.9%) | 8.1% | catastrophic |
| `sub_char` | 55.5% | 11.1% | huge headroom |
| `del_char` | 72.0% | 13.0% | only one solved at scale |
| `contains_char` | 79.7% | 67.0% | ~30 pts above random |
| `orth` | 86.5% | 32.1% (below random) | bimodal — most BPE below random |
| `spell` | 82–100% | — | already saturating |
| `spell_inverse` | 73–100% | — | mostly saturating |

**Char-vs-word manipulation gap (Cmd-R+ 104B):** insertion 8.9% char vs 81.7% word (72.8 pt gap); swap 10% vs 81% (71 pt); substitution 55% vs 95% (40 pt); deletion ~equal. The pattern is unambiguous — BPE hides char-level structure, and operations that require addressing characters individually break.

**Random-string ablation (Appendix C):** Replacing real words with random consonant strings (BPE ~1.6 chars/token) makes manipulation accuracy go *up*. Strongest argument that byte-level should dominate; a clean experiment to actually run on our byte fork.

## Dataset structure

- **1,000 examples per task × 14 tasks = 14k total.** Single split (test only — it's an eval benchmark, not a training set). Total 967 kB.
- **Char-level tasks:** drawn from the 1,000 most frequent ≥3-char English words (Google Web Trillion Word Corpus via Kaggle `rtatman/english-word-frequency`).
- **Word-level tasks:** TinyStories sentences of length 3–10 words.
- **Similarity tasks:** filtered with fastText embeddings (cosine ≥0.5/≤0.2) and normalized Levenshtein (≥0.7/≤0.3).
- **Format:** TSV in `/data/`, columns are op-specific "ingredients" (e.g. `ins_char.tsv` is `input1, input2, input3, label`). HuggingFace version has prompts pre-rendered as `{prompt, answer}` strings with the 4-shot template baked in.
- **Generation scripts** in `/data_gen/` are re-runnable — we can produce arbitrarily many training examples by pulling a longer word list and rerunning.

### Prompt template (Appendix B / Figure 4)

4 in-context examples + question + `Answer: "` opener; greedy decoding, parsed up to closing quote. Generations like `H-E-L-L-O` (wrong delimiter) count as wrong.

```
[INST] Spell out the word, putting spaces between each letter, based on the following examples:

1. Spell out the word "alphabet". Answer: "a l p h a b e t"
2. Spell out the word "hello". Answer: "h e l l o"
3. Spell out the word "zebra". Answer: "z e b r a"
4. Spell out the word "tongue". Answer: "t o n g u e"

Question: Spell out the word "cow". [/INST]
Answer: "
```

## Coverage gap vs what we already have

| CUTE task | Status in our repo |
|---|---|
| `spell` | Have it (`SimpleSpelling`) |
| `spell_inverse` | **Missing** — trivial to add (reverse `SimpleSpelling`'s template) |
| `contains_char` | **Missing** — closely related to `SpellingBee` (count → ≥1) |
| `contains_word` | Missing (low priority — BPE solves it) |
| `ins_char` | **Missing** — top priority, BPE ceiling 8–16% |
| `del_char` | **Missing** — moderate priority, BPE solves at scale |
| `sub_char` | **Missing** — high priority, BPE ceiling 56% |
| `swap_char` | **Missing** — top priority, BPE ceiling 10% |
| `*_word` (4) | Missing (low priority — BPE controls) |
| `orth` | Missing — needs fastText for negative generation |
| `sem` | Missing (doesn't probe char capability — skip) |

We also have **letter counting** (`SpellingBee`) which is *not* in CUTE — it's a strictly harder version of `contains_char`. Worth keeping; report it as a nanochat-specific extension.

## Suggested rollout

**Phase 1 — Eval-only.** Vendor the 14 TSVs (or load from HF) into the nanochat eval harness; produce a CUTE table on the d24 byte baseline vs. the d24 BPE baseline vs. the published BPE numbers. This alone may be the cleanest "byte beats BPE" headline available at this scale.

**Phase 2 — SFT training mix.** Add CUTE-flavored synthetic generators to the SFT mix in priority order: `swap_char`, `ins_char`, `sub_char`, `spell_inverse`, `contains_char`. Each is ~10 lines of Python. Generate 50–100k examples per task. Hold out the released 1k for eval (don't contaminate).

**Phase 3 — Random-string ablation.** Run the paper's random-consonant variant. If our byte model holds up where BPE collapses (which the paper predicts), that's the cleanest mechanistic confirmation that the win is from byte-level addressing, not memorization of frequent words.

## Published results table (full)

From the paper's commented-out LaTeX table (`tab:results` in `acl_latex.tex`). The published version of the paper uses a figure (`all.pdf`) instead of a printed table.

| Task | Llama-2 7B | Llama-2 13B | Llama-2 70B | Mistral 7B | Mistral 47B | Gemma 7B | Cmd-R 35B | Cmd-R+ 104B | DBRX 132B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spelling | 93.4 | 99.6 | 99.9 | 87.0 | 98.4 | 82.2 | 98.9 | **100** | 99.9 |
| Inverse Spelling | 84.5 | 92.9 | 92.7 | 91.1 | 95.7 | 73.0 | 97.8 | **100** | 99.9 |
| Contains Character | 67.0 | 69.8 | 70.1 | 61.5 | 68.1 | 64.0 | 68.9 | **79.7** | 47.8 |
| Contains Word | 78.0 | 84.8 | 94.6 | 94.2 | 90.8 | 84.8 | 87.4 | **99.7** | 97.2 |
| Orthographic Sim. | 32.1 | 46.3 | 42.6 | 55.3 | 50.8 | 54.5 | 43.4 | **86.5** | 47.6 |
| Semantic Sim. | 76.8 | 76.3 | 82.6 | 84.7 | 81.8 | 82.0 | 81.7 | **92.6** | 90.3 |
| Insert Character | 8.1 | 11.5 | 7.3 | 4.8 | 15.9 | 10.7 | 7.3 | 8.9 | **9.9** |
| Delete Character | 13.0 | 30.5 | 26.4 | 34.2 | 47.8 | 30.4 | 37.7 | **72.0** | 64.0 |
| Substitute Character | 11.1 | 14.2 | 19.6 | 21.0 | 34.6 | 15.1 | 28.7 | **55.5** | 47.7 |
| Swap Character | 3.5 | 2.5 | 5.6 | 2.8 | 6.4 | 2.5 | 5.3 | **10.0** | 7.5 |
| Insert Word | 20.1 | 36.7 | 31.6 | 11.3 | 40.8 | 36.5 | 42.2 | **81.7** | 53.2 |
| Delete Word | 34.8 | 56.7 | 46.8 | 53.8 | 56.5 | 56.7 | 55.8 | 65.7 | **86.3** |
| Substitute Word | 72.9 | 70.7 | 86.6 | 70.0 | 90.7 | 70.7 | 75.8 | **95.1** | 93.6 |
| Swap Word | 10.9 | 17.2 | 36.9 | 15.9 | 37.8 | 17.2 | 28.3 | **81.4** | 60.3 |

Llama-3 8B/70B and Aya-23 8B/35B are listed in `tab:models` but their numbers don't appear in the commented LaTeX — likely shown only in the figure.

**Implications for nanochat scale:**
- No model under 7B is in the table; the smallest is Llama-2-7B chat (~10× our d24, with RLHF). Our 0% on `spell` is bad but not necessarily shocking given the size + training-mix gap.
- Manipulation tasks are catastrophic even at 7B-100B+ instruction-tuned: `swap_char` tops out at 10%, `ins_char` at 16%. These are the cleanest places a byte model could in principle win — the bar is on the floor.
- The paper explicitly notes there is **no character-level instruction-tuned model evaluated** (line 506: "we do not evaluate any character-level models since there are no instruction-tuned versions"). nanochat-byte would be the first published byte number on CUTE.

## EXECUTE (quick note)

EXECUTE is essentially CUTE generalized to multiple languages — same task taxonomy plus non-English phenomena (diacritics, non-Latin scripts). For an English-first byte nanochat, **CUTE is the right benchmark**; EXECUTE is what to add later if going multilingual.
