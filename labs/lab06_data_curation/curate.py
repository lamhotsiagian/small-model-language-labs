"""
Lab 6, steps 1-3: heuristic filtering, PII scrubbing, and deduplication.

Pipeline stages (Chapter 6, sections 6.2, 6.3, 6.7):

  1. heuristic_filter   Gopher / C4-style document rules: length, symbol and
                        bullet ratios, repeated lines, stop-word presence,
                        boilerplate phrases. Cheap, transparent, and removes
                        the worst junk before any model runs.
  2. scrub_pii          Regex redaction of emails, phone numbers, IPs, and
                        card-like digit runs. Replace, do not drop: dropping
                        biases the corpus against documents that mention people.
  3. exact_dedup        SHA-1 of normalised text.
  4. minhash_dedup      MinHash over word 5-gram shingles + LSH banding to find
                        near-duplicates at Jaccard >= threshold, keep one per cluster.

Usage:
    python curate.py --input data/raw.jsonl --output data/clean.jsonl
    python curate.py --demo            # runs on a small synthetic sample
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import numpy as np

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with", "yang", "dan", "di", "ini"}
BOILERPLATE = ("lorem ipsum", "click here", "all rights reserved", "cookie policy",
               "javascript is disabled", "terms of use")


# ---------------------------------------------------------------------------
# 1. Heuristic filters
# ---------------------------------------------------------------------------
def heuristic_filter(text: str) -> tuple[bool, str]:
    """Return (keep, reason). Thresholds follow the spirit of Gopher (Rae et al., 2021)."""
    words = text.split()
    n = len(words)
    if n < 50 or n > 100_000:
        return False, "length"
    mean_len = sum(len(w) for w in words) / n
    if not 3 <= mean_len <= 10:
        return False, "mean_word_length"
    if text.count("#") / n > 0.1 or text.count("...") / n > 0.1:
        return False, "symbol_ratio"
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and sum(l.lstrip().startswith(("-", "*", "•")) for l in lines) / len(lines) > 0.9:
        return False, "bullet_ratio"
    if lines and len(set(lines)) / len(lines) < 0.7:
        return False, "repeated_lines"
    if sum(w.lower() in STOP_WORDS for w in words) < 2:
        return False, "no_stop_words"
    low = text.lower()
    if any(b in low for b in BOILERPLATE):
        return False, "boilerplate"
    alpha = sum(any(c.isalpha() for c in w) for w in words) / n
    if alpha < 0.8:
        return False, "non_alpha_words"
    return True, "ok"


# ---------------------------------------------------------------------------
# 2. PII scrubbing
# ---------------------------------------------------------------------------
def _luhn_ok(digits: str) -> bool:
    """Luhn checksum: real card numbers pass, most phone numbers and IDs do not."""
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (re.compile(r"(?:\+?\d{1,3}[ -]?)?(?:\(?\d{2,4}\)?[ -]?)\d{3,4}[ -]?\d{3,4}\b"), "<PHONE>"),
]


def scrub_pii(text: str) -> tuple[str, int]:
    """Regex PII redaction. Order matters, and every pattern has false positives:
    cards are validated with Luhn BEFORE the looser phone pattern runs."""
    hits = 0

    def card_repl(m):
        nonlocal hits
        digits = re.sub(r"\D", "", m.group(0))
        if _luhn_ok(digits):
            hits += 1
            return "<CARD>"
        return m.group(0)

    text = CARD.sub(card_repl, text)
    for pat, repl in PII_PATTERNS:
        text, k = pat.subn(repl, text)
        hits += k
    return text, hits


# ---------------------------------------------------------------------------
# 3. Exact dedup
# ---------------------------------------------------------------------------
def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def exact_dedup(docs: List[dict]) -> List[dict]:
    seen, out = set(), []
    for d in docs:
        h = hashlib.sha1(normalise(d["text"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# 4. MinHash + LSH near-dedup (self-contained, numpy only)
# ---------------------------------------------------------------------------
_P = (1 << 61) - 1


def shingles(text: str, k: int = 5) -> set:
    w = normalise(text).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _P, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _P, num_perm, dtype=np.uint64)

    def signature(self, sh: set) -> np.ndarray:
        hv = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
                       % _P for s in sh], dtype=np.uint64)
        # (a*h + b) mod p for every permutation, then min over shingles.
        # Python ints avoid uint64 overflow in the product.
        sig = [min(((int(a) * int(h) + int(b)) % _P) for h in hv) for a, b in zip(self.a, self.b)]
        return np.array(sig, dtype=np.uint64)


def minhash_dedup(docs: List[dict], threshold: float = 0.8, num_perm: int = 128,
                  bands: int = 16) -> tuple[List[dict], int]:
    """LSH: split each signature into `bands` bands of r rows. Two docs collide
    in a band with probability s^r; they become candidates if ANY band collides:
    P = 1 - (1 - s^r)^bands, an S-curve centred near (1/bands)^(1/r)."""
    mh, r = MinHasher(num_perm), num_perm // bands
    sigs = [mh.signature(shingles(d["text"])) for d in docs]
    buckets = defaultdict(list)
    for i, s in enumerate(sigs):
        for b in range(bands):
            buckets[(b, s[b * r:(b + 1) * r].tobytes())].append(i)
    parent = list(range(len(docs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ids in buckets.values():
        for j in ids[1:]:
            # verify the candidate with the estimated Jaccard before merging
            if (sigs[ids[0]] == sigs[j]).mean() >= threshold:
                parent[find(j)] = find(ids[0])
    keep = {find(i) for i in range(len(docs))}
    return [d for i, d in enumerate(docs) if i in keep], len(docs) - len(keep)


def run(docs: Iterable[dict]) -> tuple[List[dict], dict]:
    stats = defaultdict(int)
    kept = []
    for d in docs:
        ok, reason = heuristic_filter(d["text"])
        stats[f"filter_{reason}"] += 1
        if not ok:
            continue
        d["text"], hits = scrub_pii(d["text"])
        stats["pii_redactions"] += hits
        kept.append(d)
    before = len(kept)
    kept = exact_dedup(kept)
    stats["exact_dups"] = before - len(kept)
    kept, near = minhash_dedup(kept)
    stats["near_dups"] = near
    stats["kept"] = len(kept)
    return kept, dict(stats)


def demo_docs() -> List[dict]:
    base = ("The small model was trained on curated educational text about physics and "
            "chemistry. Each lesson explains the concept, gives an example, and ends with an "
            "exercise so that the reader can check the idea before moving on to the next topic. ")
    docs = [{"id": "a", "text": base * 3},
            {"id": "b", "text": base * 3},                                    # exact dup
            {"id": "c", "text": (base * 3).replace("physics", "biology")},    # near dup
            {"id": "d", "text": "click here to accept the cookie policy " * 20},
            {"id": "e", "text": ("Contact budi@example.com or +62 812 3456 7890 for the syllabus. Card on file "
                         "4111 1111 1111 1111 is not needed. The "
                         "course covers linear algebra, probability, and optimisation, with weekly "
                         "problem sets and a final project in which students train and evaluate "
                         "a small classifier on a dataset of their choice and present the results "
                         "to the class with a short written report and a live demonstration.")},
            {"id": "f", "text": "- item\n" * 80}]
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output", default="data/clean.jsonl")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    docs = demo_docs() if a.demo else [json.loads(l) for l in open(a.input)]
    kept, stats = run(docs)
    print(json.dumps(stats, indent=2))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        for d in kept:
            f.write(json.dumps(d) + "\n")
    if a.demo:
        for d in kept:
            print(d["id"], "|", d["text"][:90])


if __name__ == "__main__":
    main()
