"""
Lab 13, step 5: test-time compute on GSM8K (Chapter 13, section 13.5).

  greedy            1 sample
  self-consistency  N samples, majority vote over extracted answers (X. Wang et al., 2023)
  best-of-N oracle  N samples, correct if ANY is correct (upper bound = pass@N)

Plotting accuracy vs N shows how much of a small model's capability is
locked behind sampling, and what an ideal verifier could unlock.

Usage:
    python test_time.py --model checkpoints/grpo --n 1 4 8 16 --limit 500
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab08_distillation"))
from gsm8k import extract_answer, load, messages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--limit", type=int, default=500)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams  # type: ignore
    rows = load("test", a.limit)
    llm = LLM(a.model, max_model_len=2048)
    N = max(a.n)
    outs = llm.chat([messages(r["question"]) for r in rows],
                    SamplingParams(n=N, temperature=0.8, top_p=0.95, max_tokens=512))
    greedy = llm.chat([messages(r["question"]) for r in rows], SamplingParams(temperature=0.0, max_tokens=512))
    g_acc = sum(extract_answer(o.outputs[0].text) == r["answer"] for o, r in zip(greedy, rows)) / len(rows)
    print(f"greedy: {g_acc:.3f}")
    print(f"{'N':>4}{'self-consistency':>18}{'pass@N (oracle)':>17}{'tokens/question':>17}")
    for n in a.n:
        sc = orc = 0
        toks = 0
        for o, r in zip(outs, rows):
            answers = [extract_answer(c.text) for c in o.outputs[:n]]
            toks += sum(len(c.token_ids) for c in o.outputs[:n])
            vote = Counter(x for x in answers if x is not None).most_common(1)
            sc += bool(vote) and vote[0][0] == r["answer"]
            orc += r["answer"] in answers
        print(f"{n:>4}{sc / len(rows):>18.3f}{orc / len(rows):>17.3f}{toks / len(rows):>17.0f}")


if __name__ == "__main__":
    main()
