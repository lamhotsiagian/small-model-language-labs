"""
Lab 12, step 1: build ~10K on-policy preference pairs with an LLM judge.

  1. sample K=4 responses per prompt FROM THE POLICY (the Lab 11 SFT model):
     on-policy pairs teach the model about its own mistakes
  2. a judge scores each response 1-10 with a fixed rubric (absolute grading
     is cheaper than all-pairs comparison: K calls instead of K(K-1)/2)
  3. chosen = best, rejected = worst; drop pairs with margin < 2 (noise) and
     ties; this is the "best-vs-worst" recipe used by UltraFeedback-style sets
  4. report length statistics: if chosen responses are systematically longer,
     the judge has a length bias that DPO will amplify (Chapter 12, 12.5)

Usage:
    python make_pairs.py --policy ../lab11_sft/checkpoints/sft --judge Qwen/Qwen2.5-7B-Instruct \
        --prompts HuggingFaceH4/ultrafeedback_binarized --n 12000 --out data/pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import statistics

RUBRIC = """Rate the assistant's response to the user on a 1-10 scale for helpfulness,
correctness, instruction following, and clarity. A concise correct answer should score
at least as high as a longer answer with the same content; do not reward length itself.
Reply with a one-sentence justification, then "Score: <1-10>".

[User]
{prompt}

[Assistant]
{response}"""


def parse(text: str):
    m = re.search(r"Score:\s*(10|[1-9])", text)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompts", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--n", type=int, default=12_000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-margin", type=int, default=2)
    ap.add_argument("--out", default="data/pairs.jsonl")
    a = ap.parse_args()
    from datasets import load_dataset
    from vllm import LLM, SamplingParams  # type: ignore

    ds = load_dataset(a.prompts, split="train_prefs").shuffle(seed=0).select(range(a.n))
    prompts = [r["prompt"] for r in ds]
    policy = LLM(a.policy, max_model_len=4096, gpu_memory_utilization=0.35)
    outs = policy.chat([[{"role": "user", "content": p}] for p in prompts],
                       SamplingParams(n=a.k, temperature=0.8, top_p=0.95, max_tokens=768))
    cands = [[c.text for c in o.outputs] for o in outs]
    del policy

    judge = LLM(a.judge, max_model_len=6144, gpu_memory_utilization=0.55)
    flat = [(i, j) for i in range(len(prompts)) for j in range(a.k)]
    reqs = [[{"role": "user", "content": RUBRIC.format(prompt=prompts[i], response=cands[i][j])}]
            for i, j in flat]
    scores = [parse(o.outputs[0].text) for o in judge.chat(reqs, SamplingParams(temperature=0.0, max_tokens=96))]
    by_prompt = {}
    for (i, j), s in zip(flat, scores):
        if s is not None:
            by_prompt.setdefault(i, []).append((s, j))

    kept, len_w, len_l = 0, [], []
    with open(a.out, "w") as f:
        for i, sc in by_prompt.items():
            if len(sc) < 2:
                continue
            (s_hi, j_hi), (s_lo, j_lo) = max(sc), min(sc)
            if s_hi - s_lo < a.min_margin:
                continue
            w, l = cands[i][j_hi], cands[i][j_lo]
            f.write(json.dumps({"prompt": [{"role": "user", "content": prompts[i]}],
                                "chosen": [{"role": "assistant", "content": w}],
                                "rejected": [{"role": "assistant", "content": l}],
                                "score_chosen": s_hi, "score_rejected": s_lo}) + "\n")
            kept += 1
            len_w.append(len(w.split())); len_l.append(len(l.split()))
    print(f"[lab12] {kept} pairs (margin >= {a.min_margin}) from {len(prompts)} prompts")
    print(f"[lab12] median words chosen {statistics.median(len_w)} vs rejected {statistics.median(len_l)}; "
          f"chosen longer in {sum(x > y for x, y in zip(len_w, len_l)) / max(1, kept):.0%} of pairs")


if __name__ == "__main__":
    main()
