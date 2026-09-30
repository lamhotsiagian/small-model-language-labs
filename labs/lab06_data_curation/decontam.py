"""
Lab 6, step 6: benchmark contamination detection by n-gram overlap.

Builds a set of 13-gram shingles from every evaluation benchmark you will
report (GSM8K, MMLU, HellaSwag, ARC, your domain eval) and flags any training
document that shares at least `min_hits` of them. 13-grams follow the GPT-3
contamination study; shorter n-grams over-flag common phrases.

Flagged documents are REMOVED from training (not just reported), and the
report lists which benchmark each hit came from so you can disclose it.

Usage:
    python decontam.py --train data/edu.jsonl --bench data/benchmarks.jsonl --out data/decontam.jsonl
    python decontam.py --demo
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter


def ngrams(text: str, n: int = 13):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_index(bench_rows, n: int = 13) -> dict:
    index = {}
    for r in bench_rows:
        for g in ngrams(r["text"], n):
            index[g] = r["benchmark"]
    return index


def scan(docs, index, n: int = 13, min_hits: int = 1):
    clean, report = [], Counter()
    for d in docs:
        hits = [index[g] for g in ngrams(d["text"], n) if g in index]
        if len(hits) >= min_hits:
            report.update(set(hits))
        else:
            clean.append(d)
    return clean, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--bench"); ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        q = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
             "clips in May. How many clips did Natalia sell altogether in April and May?")
        bench = [{"benchmark": "gsm8k", "text": q}]
        docs = [{"id": 1, "text": "Here is a practice problem. " + q + " Answer: 72."},
                {"id": 2, "text": "Clips are small tools used to hold paper together in offices."}]
    else:
        bench = [json.loads(l) for l in open(a.bench)]
        docs = [json.loads(l) for l in open(a.train)]
    clean, report = scan(docs, build_index(bench, a.n), a.n)
    print(f"[lab06] removed {len(docs) - len(clean)} of {len(docs)} documents; hits by benchmark: {dict(report)}")
    if a.out:
        with open(a.out, "w") as f:
            for d in clean:
                f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
