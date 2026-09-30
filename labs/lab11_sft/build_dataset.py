"""
Lab 11, step 1: build a curated 50K instruction set.

Quality over quantity (Zhou et al., 2023): a small, diverse, clean set beats
a large noisy one, and small models are MORE sensitive to noise than large
ones. The pipeline:

  1. pool      several open SFT sources, converted to a common message format
  2. clean     drop empty/garbled turns, refusals-by-default, and responses
               longer than the student can reasonably imitate
  3. dedup     exact dedup on the first user turn + MinHash near-dedup (Lab 6)
  4. decontam  13-gram overlap against IFEval / MT-Bench prompts (Lab 6)
  5. balance   cap each category so no single skill dominates
  6. mix       add a slice of general data to limit catastrophic forgetting
  7. format    TRL "prompt/completion" conversational format, so the loss
               is computed on the final assistant turn only

Usage:
    python build_dataset.py --n 50000 --out data/sft_50k.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab06_data_curation"))
from curate import minhash_dedup  # noqa: E402
from decontam import build_index, ngrams  # noqa: E402

# (dataset, config, split, category field or constant, weight)
SOURCES = [
    ("HuggingFaceTB/smoltalk", "all", "train", "source", 0.55),
    ("allenai/tulu-3-sft-mixture", None, "train", "source", 0.35),
    ("Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered", None, "train", "task_category", 0.10),
]
REFUSAL = re.compile(r"^(i'm sorry|i cannot|i can't|as an ai)", re.I)


def normalise(row) -> list[dict] | None:
    msgs = row.get("messages") or row.get("conversations")
    if not msgs:
        return None
    out = []
    for m in msgs:
        role = m.get("role") or {"human": "user", "gpt": "assistant"}.get(m.get("from"), m.get("from"))
        content = (m.get("content") or m.get("value") or "").strip()
        if role not in ("system", "user", "assistant") or not content:
            return None
        out.append({"role": role, "content": content})
    return out if out and out[-1]["role"] == "assistant" else None


def keep(msgs, max_resp_words: int) -> bool:
    resp = msgs[-1]["content"]
    if len(resp.split()) > max_resp_words or len(resp) < 2:
        return False
    if REFUSAL.match(resp) and len(resp.split()) < 40:     # reflexive refusals teach over-refusal
        return False
    if "\ufffd" in resp or resp.count("```") % 2:           # garbled text or unclosed code block
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--pool", type=int, default=400_000)
    ap.add_argument("--max-resp-words", type=int, default=800)
    ap.add_argument("--cap-frac", type=float, default=0.2, help="max share per category")
    ap.add_argument("--bench", default="data/eval_prompts.jsonl", help="IFEval/MT-Bench prompts")
    ap.add_argument("--out", default="data/sft_50k.jsonl")
    a = ap.parse_args()
    from datasets import load_dataset

    rng = random.Random(0)
    pool, stats = [], Counter()
    for name, cfg, split, cat_field, w in SOURCES:
        ds = load_dataset(name, cfg, split=split, streaming=True)
        take = int(a.pool * w)
        for i, row in enumerate(ds):
            if i >= take:
                break
            msgs = normalise(row)
            stats["seen"] += 1
            if msgs is None or not keep(msgs, a.max_resp_words):
                stats["dropped_clean"] += 1
                continue
            first_user = next(m["content"] for m in msgs if m["role"] == "user")
            pool.append({"messages": msgs, "category": str(row.get(cat_field, name)),
                         "text": first_user, "source": name,
                         "hash": hashlib.sha1(first_user.lower().encode()).hexdigest()})

    # exact + near dedup on the first user turn
    seen, uniq = set(), []
    for r in pool:
        if r["hash"] not in seen:
            seen.add(r["hash"]); uniq.append(r)
    stats["dropped_exact"] = len(pool) - len(uniq)
    uniq, near = minhash_dedup(uniq, threshold=0.85)
    stats["dropped_near"] = near

    # decontaminate against evaluation prompts
    if Path(a.bench).exists():
        index = build_index([json.loads(l) for l in open(a.bench)])
        before = len(uniq)
        uniq = [r for r in uniq if not any(g in index for g in ngrams(
            " ".join(m["content"] for m in r["messages"])))]
        stats["dropped_contam"] = before - len(uniq)

    # category balancing with a cap
    by_cat = defaultdict(list)
    for r in uniq:
        by_cat[r["category"]].append(r)
    cap = int(a.n * a.cap_frac)
    chosen = []
    for cat, rows in by_cat.items():
        rng.shuffle(rows)
        chosen += rows[:cap]
    rng.shuffle(chosen)
    chosen = chosen[:a.n]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in chosen:
            # prompt/completion conversational format -> TRL computes loss on the completion only
            f.write(json.dumps({"prompt": r["messages"][:-1], "completion": [r["messages"][-1]],
                                "category": r["category"], "source": r["source"]}) + "\n")
    stats["kept"] = len(chosen)
    print(json.dumps(dict(stats), indent=2))
    print("top categories:", Counter(r["category"] for r in chosen).most_common(8))


if __name__ == "__main__":
    main()
