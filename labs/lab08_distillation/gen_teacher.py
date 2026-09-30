"""
Lab 8, step 1: sequence-level distillation data from the teacher.

For every GSM8K training question, sample k solutions from the teacher and
keep only those whose final answer matches the reference (rejection
sampling). Correct-only traces are the single biggest quality lever for
SFT-style distillation of reasoning: the student imitates reasoning that
worked, not reasoning that merely sounds right.

Optionally stores the teacher's top-k log-probabilities for every response
token (``--topk``) so logit KD can run later WITHOUT the teacher in memory
(offline KD, Chapter 8 section 8.3).

Usage:
    python gen_teacher.py --teacher Qwen/Qwen2.5-7B-Instruct --k 4 --out data/teacher.jsonl
    python gen_teacher.py --teacher Qwen/Qwen2.5-7B-Instruct --k 1 --topk 20 --out data/teacher_topk.jsonl
"""
from __future__ import annotations

import argparse
import json

from gsm8k import extract_answer, load, messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--n", type=int)
    ap.add_argument("--out", default="data/teacher.jsonl")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams  # type: ignore
    rows = load("train", a.n)
    llm = LLM(a.teacher, max_model_len=2048)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95, max_tokens=512,
                        logprobs=a.topk or None)
    outs = llm.chat([messages(r["question"]) for r in rows], sp)
    kept = total = 0
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            for c in o.outputs:
                total += 1
                if extract_answer(c.text) != r["answer"]:
                    continue                                # rejection sampling
                rec = {"question": r["question"], "answer": r["answer"], "response": c.text}
                if a.topk:
                    # per position: list of (token_id, logprob) for the teacher's top-k
                    rec["topk"] = [[(tid, lp.logprob) for tid, lp in pos.items()]
                                   for pos in c.logprobs]
                    rec["token_ids"] = list(c.token_ids)
                f.write(json.dumps(rec) + "\n")
                kept += 1
                break                                       # one correct trace per question
    print(f"[lab08] kept {kept} correct traces from {total} samples "
          f"({kept / max(1, len(rows)):.1%} of questions covered)")


if __name__ == "__main__":
    main()
