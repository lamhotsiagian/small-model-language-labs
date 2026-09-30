"""
Lab 6, step 4: model-based quality filtering (FineWeb-Edu style).

Two-stage design, because running an LLM over billions of documents is
unaffordable but running it over 50K is cheap:

  1. annotate   an LLM judge scores a sample of documents 0-5 for
                educational value with a fixed rubric prompt
  2. distil     a tiny classifier (TF-IDF + logistic regression here; a small
                embedding model + linear head in production) learns to
                reproduce the judge, then scores the whole corpus at CPU speed
  3. threshold  keep documents with predicted score >= 3 (the FineWeb-Edu cut)

Usage:
    python quality.py annotate --input data/clean.jsonl --n 50000 --judge Qwen/Qwen2.5-7B-Instruct
    python quality.py train    --labels data/annotations.jsonl
    python quality.py filter   --input data/clean.jsonl --output data/edu.jsonl --threshold 3
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from pathlib import Path

RUBRIC = """Below is an extract from a web page. Evaluate whether it has high educational
value for teaching at primary to university level, using this additive 5-point scale:
+1 if it provides some basic information relevant to an educational topic, even if mixed
   with ads or irrelevant material.
+1 if it addresses elements pertinent to education but does not align closely with
   educational standards, or is superficial.
+1 if it is appropriate for educational use and introduces key concepts relevant to
   school curricula, is coherent, and is not overly complex.
+1 if it is highly relevant and beneficial for educational purposes at a grade level,
   with a clear and consistent writing style, like a textbook chapter or tutorial.
+1 if it is outstanding in educational value, with no non-educational content.

Extract:
{text}

After examining the extract, briefly justify your score in at most 60 words, then end
with the line: "Educational score: <points>"."""


def parse_score(reply: str) -> int | None:
    m = re.search(r"Educational score:\s*([0-5])", reply)
    return int(m.group(1)) if m else None


def annotate(inp: str, n: int, judge: str, out: str) -> None:
    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from slmlab.hfutils import chat_generate, load_model
    docs = [json.loads(l) for l in open(inp)]
    random.Random(0).shuffle(docs)
    tok, model = load_model(judge)
    with open(out, "w") as f:
        for d in docs[:n]:
            r = chat_generate(tok, model, [{"role": "user",
                                            "content": RUBRIC.format(text=d["text"][:3000])}],
                              max_new_tokens=120)
            s = parse_score(r.text)
            if s is not None:                       # drop unparsable judgements
                f.write(json.dumps({"text": d["text"], "score": s}) + "\n")


def train(labels: str, model_out: str) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    rows = [json.loads(l) for l in open(labels)]
    X = [r["text"] for r in rows]
    y = [r["score"] for r in rows]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=0)
    vec = TfidfVectorizer(max_features=200_000, ngram_range=(1, 2), sublinear_tf=True)
    # Regression, not classification: the score is ordinal, and the threshold is
    # chosen afterwards on the continuous prediction.
    reg = Ridge(alpha=1.0).fit(vec.fit_transform(Xtr), ytr)
    pred = reg.predict(vec.transform(Xte))
    f1 = f1_score([v >= 3 for v in yte], [p >= 3 for p in pred])
    print(f"[lab06] held-out binary F1 at threshold 3: {f1:.3f}")
    pickle.dump((vec, reg), open(model_out, "wb"))


def filt(inp: str, out: str, model_path: str, threshold: float) -> None:
    vec, reg = pickle.load(open(model_path, "rb"))
    kept = total = 0
    with open(out, "w") as f:
        for line in open(inp):
            d = json.loads(line)
            total += 1
            s = float(reg.predict(vec.transform([d["text"]]))[0])
            if s >= threshold:
                d["edu_score"] = round(s, 2)
                f.write(json.dumps(d) + "\n")
                kept += 1
    print(f"[lab06] kept {kept}/{total} ({kept / max(total, 1):.1%}) at threshold {threshold}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("annotate")
    a1.add_argument("--input", required=True); a1.add_argument("--n", type=int, default=50_000)
    a1.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    a1.add_argument("--out", default="data/annotations.jsonl")
    a2 = sub.add_parser("train")
    a2.add_argument("--labels", default="data/annotations.jsonl")
    a2.add_argument("--model-out", default="data/quality_clf.pkl")
    a3 = sub.add_parser("filter")
    a3.add_argument("--input", required=True); a3.add_argument("--output", required=True)
    a3.add_argument("--model", default="data/quality_clf.pkl")
    a3.add_argument("--threshold", type=float, default=3.0)
    a = ap.parse_args()
    Path("data").mkdir(exist_ok=True)
    if a.cmd == "annotate":
        annotate(a.input, a.n, a.judge, a.out)
    elif a.cmd == "train":
        train(a.labels, a.model_out)
    else:
        filt(a.input, a.output, a.model, a.threshold)


if __name__ == "__main__":
    main()
