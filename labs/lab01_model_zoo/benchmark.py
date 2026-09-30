"""
Lab 1: SLM Model Zoo Benchmark
==============================

Run the same 50 prompts (5 task types x 10) through several small models and
record, per model and task type:
  * quality     mean scorer value in [0, 1]
  * latency     TTFT (prefill) and TPOT (decode) in milliseconds
  * memory      weights footprint and peak accelerator memory
  * throughput  output tokens per second

Results land in results/zoo_results.json; decision_matrix.py turns them into a
per-task recommendation.

Usage:
    python benchmark.py --preset smoke           # 2 tiny models, 10 prompts
    python benchmark.py                           # full zoo, 50 prompts
    python benchmark.py --models Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.hfutils import chat_generate, free, load_model, peak_memory
from tasks import build_suite

# Pin exact revisions before quoting numbers (see README, step 1).
DEFAULT_ZOO = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",     # gated: accept the license on the Hub first
    "google/gemma-3-1b-it",                 # gated: accept the license on the Hub first
    "microsoft/Phi-3.5-mini-instruct",
]
SMOKE_ZOO = ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]


def run_model(model_id: str, suite, max_new_tokens: int, dtype: str) -> dict:
    tok, model = load_model(model_id, dtype=dtype)
    weight_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20
    per_type = defaultdict(lambda: {"scores": [], "ttft": [], "tpot": [], "tps": []})

    with peak_memory() as mem:
        for task in suite:
            # Some chat templates (Gemma) reject a system role: fold it into the user turn.
            msgs = [{"role": "system", "content": task.system},
                    {"role": "user", "content": task.user}]
            try:
                r = chat_generate(tok, model, msgs, max_new_tokens=max_new_tokens)
            except Exception:
                msgs = [{"role": "user", "content": f"{task.system}\n\n{task.user}"}]
                r = chat_generate(tok, model, msgs, max_new_tokens=max_new_tokens)
            d = per_type[task.task_type]
            d["scores"].append(task.scorer(r.text, task.reference))
            d["ttft"].append(r.ttft_s * 1000)
            d["tpot"].append(r.tpot_s * 1000)
            d["tps"].append(r.new_tokens / max(r.total_s, 1e-6))

    summary = {}
    for t, d in per_type.items():
        summary[t] = {"quality": round(statistics.mean(d["scores"]), 3),
                      "ttft_ms_p50": round(statistics.median(d["ttft"]), 1),
                      "tpot_ms_p50": round(statistics.median(d["tpot"]), 1),
                      "tok_per_s": round(statistics.mean(d["tps"]), 1)}
    n_params = sum(p.numel() for p in model.parameters())
    device = str(model.device)
    free(model)
    return {"model": model_id, "params": n_params, "weights_mb": round(weight_mb, 1),
            "peak_mb": mem["peak_mb"], "device": device,
            "per_task": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--preset", choices=["smoke", "full"], default="full")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", default="results/zoo_results.json")
    args = ap.parse_args()

    suite = build_suite()
    models = args.models or (SMOKE_ZOO if args.preset == "smoke" else DEFAULT_ZOO)
    if args.preset == "smoke":
        suite = suite[::5]                     # 10 prompts, 2 per task type

    results = []
    for m in models:
        print(f"[lab01] {m}")
        try:
            results.append(run_model(m, suite, args.max_new_tokens, args.dtype))
        except Exception as exc:               # gated model, OOM, missing template...
            print(f"   skipped: {type(exc).__name__}: {exc}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[lab01] wrote {args.out}")


if __name__ == "__main__":
    main()
