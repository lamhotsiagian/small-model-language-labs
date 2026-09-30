"""
Lab 10, step 4: RULER-style synthetic long-context probes (Hsieh et al., 2024).

Four task families, each generated at any target length with a known answer:

  niah_single     one "needle" (a key -> 7-digit value) hidden in filler text
  niah_multikey   the needle plus 3 distractor keys with different values
  niah_multivalue one key with 4 values; the model must return all of them
  vt              variable tracking: X1 = 12345, X2 = X1, ... ; which vars equal 12345?

Needle depth is swept (0-100%), so "lost in the middle" failures show up
as a depth-dependent accuracy dip (N. F. Liu et al., 2024).

Usage:
    python ruler_lite.py --model checkpoints/smollm2_1p7b_32k --lengths 4096 8192 16384 32768 --n 40
    python ruler_lite.py --demo      # print one example of each task (no model)
"""
from __future__ import annotations

import argparse
import json
import random
import re

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
          "There and back again. ")


def _haystack(tok, n_tokens: int, rng) -> list[str]:
    per = len(tok(FILLER).input_ids) if tok else 20
    return [FILLER] * max(1, n_tokens // per)


def make_task(kind: str, n_tokens: int, depth: float, rng, tok=None) -> tuple[str, list[str]]:
    hay = _haystack(tok, n_tokens, rng)
    key = f"{rng.choice(['amber', 'cobalt', 'violet', 'jade'])}-{rng.randint(100, 999)}"
    val = str(rng.randint(1_000_000, 9_999_999))
    if kind == "niah_single":
        inserts, q, ans = [f"The special magic number for {key} is {val}. "], key, [val]
    elif kind == "niah_multikey":
        inserts = [f"The special magic number for {key} is {val}. "]
        for _ in range(3):
            k2 = f"{key.split('-')[0]}-{rng.randint(100, 999)}"
            inserts.append(f"The special magic number for {k2} is {rng.randint(1_000_000, 9_999_999)}. ")
        q, ans = key, [val]
    elif kind == "niah_multivalue":
        vals = [str(rng.randint(1_000_000, 9_999_999)) for _ in range(4)]
        inserts, q, ans = [f"One of the special magic numbers for {key} is {v}. " for v in vals], key, vals
    else:  # variable tracking
        v0 = str(rng.randint(10_000, 99_999))
        names = [f"X{i}" for i in range(1, 6)]
        inserts = [f"VAR {names[0]} = {v0}. "] + [f"VAR {names[i]} = {names[i - 1]}. " for i in range(1, 5)]
        q, ans = v0, names
    positions = sorted(int(depth * len(hay)) if len(inserts) == 1 else rng.randint(0, len(hay))
                       for _ in inserts)
    for p, ins in sorted(zip(positions, inserts), reverse=True):
        hay.insert(p, ins)
    ctx = "".join(hay)
    if kind == "vt":
        question = f"Find all variables that are assigned the value {q}. Answer with the variable names."
    elif kind == "niah_multivalue":
        question = f"What are all the special magic numbers for {q}? List them."
    else:
        question = f"What is the special magic number for {q}? Answer with the number only."
    return f"{ctx}\n\n{question}", ans


def score(output: str, answers: list[str]) -> float:
    found = set(re.findall(r"[A-Za-z]?\d+", output))
    return sum(a in found for a in answers) / len(answers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    ap.add_argument("--tasks", nargs="+", default=["niah_single", "niah_multikey", "niah_multivalue", "vt"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    rng = random.Random(0)
    if a.demo:
        for t in a.tasks:
            prompt, ans = make_task(t, 300, 0.5, rng)
            print(f"--- {t} (answer {ans}) ---\n...{prompt[-260:]}\n")
        return
    from vllm import LLM, SamplingParams  # type: ignore
    llm = LLM(a.model, max_model_len=max(a.lengths) + 256)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    results = {}
    for L in a.lengths:
        for t in a.tasks:
            items = [make_task(t, L - 200, (i % 11) / 10, rng, tok) for i in range(a.n)]
            outs = llm.generate([p for p, _ in items], sp)
            acc = sum(score(o.outputs[0].text, ans) for o, (_, ans) in zip(outs, items)) / a.n
            results[f"{t}@{L}"] = round(acc, 3)
            print(f"{t:<16} {L:>6}  acc={acc:.3f}")
    import os
    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/ruler_lite.json", "w"), indent=2)


if __name__ == "__main__":
    main()
