"""
Lab 11, step 4b: MT-Bench-style pairwise comparison with an LLM judge.

For each prompt, both models answer; a judge sees the two answers and picks
A, B, or tie. Every pair is judged TWICE with the order swapped, and only
consistent verdicts count as wins (position-bias control; Zheng et al., 2023).

Usage:
    python judge_pairwise.py --a checkpoints/sft --b Qwen/Qwen2.5-0.5B-Instruct \
        --judge Qwen/Qwen2.5-7B-Instruct --prompts data/mtbench_style.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

JUDGE = """Please act as an impartial judge and evaluate the quality of the responses provided by
two AI assistants to the user question below. Consider helpfulness, relevance, accuracy, depth,
and instruction following. Do not let response length or position influence your decision.
Output your final verdict strictly as "[[A]]", "[[B]]", or "[[C]]" for a tie.

[User Question]
{q}

[The Start of Assistant A's Answer]
{a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{b}
[The End of Assistant B's Answer]"""


def verdict(text: str) -> str:
    m = re.search(r"\[\[([ABC])\]\]", text)
    return m.group(1) if m else "C"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompts", default="data/mtbench_style.jsonl")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams  # type: ignore
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts)]
    sp = SamplingParams(temperature=0.0, max_tokens=768)
    answers = {}
    for name in (a.a, a.b):
        llm = LLM(name, max_model_len=4096, gpu_memory_utilization=0.4)
        answers[name] = [o.outputs[0].text for o in llm.chat([[{"role": "user", "content": p}]
                                                              for p in prompts], sp)]
        del llm
    judge = LLM(a.judge, max_model_len=8192)
    jp = SamplingParams(temperature=0.0, max_tokens=512)
    ab = [JUDGE.format(q=q, a=x, b=y) for q, x, y in zip(prompts, answers[a.a], answers[a.b])]
    ba = [JUDGE.format(q=q, a=y, b=x) for q, x, y in zip(prompts, answers[a.a], answers[a.b])]
    v1 = [verdict(o.outputs[0].text) for o in judge.chat([[{"role": "user", "content": p}] for p in ab], jp)]
    v2 = [verdict(o.outputs[0].text) for o in judge.chat([[{"role": "user", "content": p}] for p in ba], jp)]
    win = sum(x == "A" and y == "B" for x, y in zip(v1, v2))
    loss = sum(x == "B" and y == "A" for x, y in zip(v1, v2))
    inconsistent = sum((x, y) not in {("A", "B"), ("B", "A"), ("C", "C")} for x, y in zip(v1, v2))
    n = len(prompts)
    print(f"{a.a} vs {a.b}: win {win / n:.1%}  loss {loss / n:.1%}  "
          f"tie/inconsistent {(n - win - loss) / n:.1%}  (position-inconsistent {inconsistent / n:.1%})")
    mean_len = {k: sum(len(x.split()) for x in v) / n for k, v in answers.items()}
    print("mean response words:", {k: round(v) for k, v in mean_len.items()})


if __name__ == "__main__":
    main()
