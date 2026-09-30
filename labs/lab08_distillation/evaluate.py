"""
Lab 8, step 3: GSM8K accuracy for the baseline student, the three distilled
students, and the teacher, plus a cost table.

Usage:
    python evaluate.py --models Qwen/Qwen2.5-0.5B-Instruct checkpoints/student_sft \
        checkpoints/student_logit checkpoints/student_gkd Qwen/Qwen2.5-7B-Instruct --n 1319
"""
from __future__ import annotations

import argparse
import json

from gsm8k import accuracy, load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1319)          # full GSM8K test set
    a = ap.parse_args()
    from vllm import LLM, SamplingParams  # type: ignore
    rows = load("test", a.n)
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    report = {}
    for m in a.models:
        llm = LLM(m, max_model_len=2048, gpu_memory_utilization=0.85)
        gen = lambda batch: [o.outputs[0].text for o in llm.chat(batch, sp)]
        report[m] = round(accuracy(gen, rows, batch=len(rows)), 4)
        print(f"{m:<45} GSM8K = {report[m]:.3f}")
        del llm
    import os
    os.makedirs("results", exist_ok=True)
    json.dump(report, open("results/gsm8k.json", "w"), indent=2)


if __name__ == "__main__":
    main()
