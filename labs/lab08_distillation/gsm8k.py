"""
GSM8K helpers shared by Labs 8 and 13: prompt format, answer extraction,
and a batched accuracy evaluator.
"""
from __future__ import annotations

import re

SYSTEM = ("Solve the math word problem. Think step by step, then give the final "
          "answer on its own line as '#### <number>'.")


def messages(question: str):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def extract_answer(text: str) -> str | None:
    """Prefer the '#### n' convention; fall back to the last number in the text."""
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    else:
        s = m.group(1)
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return None


def gold(answer_field: str) -> str:
    """GSM8K reference solutions end with '#### <answer>'."""
    return extract_answer("#### " + answer_field.split("####")[-1])


def load(split: str = "test", n: int | None = None):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = [{"question": r["question"], "answer": gold(r["answer"])} for r in ds]
    return rows[:n] if n else rows


def accuracy(generate_fn, rows, batch: int = 32) -> float:
    """generate_fn(list[messages]) -> list[str]."""
    correct = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        outs = generate_fn([messages(r["question"]) for r in chunk])
        correct += sum(extract_answer(o) == r["answer"] for o, r in zip(outs, chunk))
    return correct / max(1, len(rows))
