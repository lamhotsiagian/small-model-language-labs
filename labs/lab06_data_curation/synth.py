"""
Lab 6, step 5: synthetic "textbook quality" data (Phi / Cosmopedia approach).

The trick is DIVERSITY, not volume. Asking a teacher model "write a textbook"
50,000 times yields 50,000 near-duplicates. Cosmopedia-style generation
crosses three independent axes so every prompt is different:

    topic   (seeded from a curated topic list or from web documents)
  x format  (textbook section, worked example, story for children, Q&A, ...)
  x audience(primary school, high school, university, professional)

Each generated sample is tagged with its seed tuple so the mix can be audited
and re-balanced, and then goes through the same dedup + decontamination as web
data (synthetic data can memorise benchmarks too).

Usage:
    python synth.py --topics topics.txt --n 200000 --teacher Qwen/Qwen2.5-7B-Instruct \
        --out data/synthetic.jsonl
    python synth.py --dry-run --n 5          # print prompts only
"""
from __future__ import annotations

import argparse
import itertools
import json
import random

FORMATS = {
    "textbook": "Write a clear, rigorous textbook section on \"{topic}\" for {audience}. "
                "Define key terms, explain the core idea step by step, give one worked "
                "example, and end with two exercises with answers.",
    "worked_example": "Write a detailed worked example that teaches \"{topic}\" to {audience}. "
                      "State the problem, solve it step by step with reasoning, and point out "
                      "one common mistake.",
    "story": "Write a short, engaging story for {audience} in which the characters "
             "discover and explain \"{topic}\" accurately.",
    "qa": "Write five question-and-answer pairs that teach \"{topic}\" to {audience}. "
          "Questions should build on each other from basic to advanced.",
}
AUDIENCES = ["primary school students", "high school students",
             "first-year university students", "working professionals"]
DEFAULT_TOPICS = ["photosynthesis", "binary search", "compound interest", "Newton's second law",
                  "the water cycle", "hash tables", "probability of independent events",
                  "supply and demand", "gradient descent", "the structure of DNA"]


def prompts(topics, n: int, seed: int = 0):
    """Yield n unique (topic, format, audience) prompts, round-robin over the grid."""
    grid = list(itertools.product(topics, FORMATS, AUDIENCES))
    random.Random(seed).shuffle(grid)
    for topic, fmt, aud in itertools.islice(itertools.cycle(grid), n):
        yield {"topic": topic, "format": fmt, "audience": aud,
               "prompt": FORMATS[fmt].format(topic=topic, audience=aud)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="data/synthetic.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    topics = [t.strip() for t in open(a.topics)] if a.topics else DEFAULT_TOPICS
    ps = list(prompts(topics, a.n))
    if a.dry_run:
        for p in ps:
            print(f"[{p['format']} | {p['audience']}] {p['prompt']}")
        return

    # vLLM is the production path (batched, ~100x faster than HF generate).
    try:
        from vllm import LLM, SamplingParams  # type: ignore
        llm = LLM(a.teacher)
        sp = SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=a.max_new_tokens)
        msgs = [[{"role": "user", "content": p["prompt"]}] for p in ps]
        outs = [o.outputs[0].text for o in llm.chat(msgs, sp)]
    except ImportError:
        import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        from slmlab.hfutils import chat_generate, load_model
        tok, model = load_model(a.teacher)
        outs = [chat_generate(tok, model, [{"role": "user", "content": p["prompt"]}],
                              max_new_tokens=a.max_new_tokens, temperature=a.temperature).text
                for p in ps]
    with open(a.out, "w") as f:
        for p, text in zip(ps, outs):
            f.write(json.dumps({**p, "text": text, "source": f"synthetic:{a.teacher}"}) + "\n")
    print(f"[lab06] wrote {len(outs)} synthetic documents to {a.out}")


if __name__ == "__main__":
    main()
