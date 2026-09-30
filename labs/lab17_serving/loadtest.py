"""
Lab 17, step 3: streaming load test against any OpenAI-compatible server
(vLLM, SGLang, llama.cpp server, Ollama, TensorRT-LLM via its OpenAI frontend).

For each concurrency level, `concurrency` virtual users send requests
back-to-back for `duration` seconds. Per request we record:
  TTFT   time to first streamed token          (prefill + queueing)
  TPOT   mean time between subsequent tokens   (decode speed seen by one user)
  E2E    total request latency
Aggregate: output tokens/s across all users, requests/s, and p50/p95/p99.

Prompt lengths are drawn from a fixed-seed distribution so every engine sees
the same workload. Tokens are counted from streamed chunks (one chunk ~ one
token for these servers; use the usage field when available).

Usage:
    python loadtest.py --url http://localhost:8000/v1 --model Qwen/Qwen2.5-1.5B-Instruct \
        --concurrency 1 8 32 128 --duration 60 --max-tokens 256 --out results/vllm.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time

WORDS = ("system design small model latency throughput memory bandwidth cache token batch "
         "quantization prefill decode request queue schedule kernel attention").split()


def make_prompt(rng: random.Random, n_words: int) -> str:
    return "Summarize the following notes in three bullet points:\n" + " ".join(
        rng.choice(WORDS) for _ in range(n_words))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


async def one_request(client, url, model, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft, stamps = None, []
    async with client.stream("POST", f"{url}/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True}) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
            if delta:
                now = time.perf_counter()
                ttft = ttft or now - t0
                stamps.append(now)
    e2e = time.perf_counter() - t0
    tpot = (stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else float("nan")
    return {"ttft": ttft or e2e, "tpot": tpot, "e2e": e2e, "tokens": len(stamps)}


async def run_level(url, model, conc, duration, max_tokens, seed):
    import httpx
    rng = random.Random(seed)
    results, stop = [], time.perf_counter() + duration
    limits = httpx.Limits(max_connections=conc * 2)
    async with httpx.AsyncClient(timeout=300, limits=limits) as client:
        async def user(uid):
            r = random.Random(seed * 1000 + uid)
            while time.perf_counter() < stop:
                prompt = make_prompt(r, r.choice([50, 200, 800]))
                results.append(await one_request(client, url, model, prompt, max_tokens))
        t0 = time.perf_counter()
        await asyncio.gather(*(user(i) for i in range(conc)))
        wall = time.perf_counter() - t0
    toks = sum(r["tokens"] for r in results)
    return {"concurrency": conc, "requests": len(results), "req_per_s": round(len(results) / wall, 2),
            "out_tok_per_s": round(toks / wall, 1),
            "ttft_p50_ms": round(pct([r["ttft"] for r in results], 50) * 1000),
            "ttft_p95_ms": round(pct([r["ttft"] for r in results], 95) * 1000),
            "tpot_p50_ms": round(statistics.median([r["tpot"] for r in results if r["tpot"] == r["tpot"]]) * 1000, 1),
            "tpot_p95_ms": round(pct([r["tpot"] for r in results if r["tpot"] == r["tpot"]], 95) * 1000, 1),
            "e2e_p99_s": round(pct([r["e2e"] for r in results], 99), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32, 128])
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="results/loadtest.csv")
    a = ap.parse_args()
    rows = []
    for c in a.concurrency:
        row = asyncio.run(run_level(a.url, a.model, c, a.duration, a.max_tokens, seed=c))
        print(json.dumps(row))
        rows.append(row)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
