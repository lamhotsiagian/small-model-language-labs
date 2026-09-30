"""
Lab 14, step 5b: hit the multi-LoRA server with a mixed workload.

Sends the same log lines to each adapter concurrently through the
OpenAI-compatible API, and reports per-adapter accuracy and latency. The
point: three "models" cost one base model of GPU memory.

Usage:
    python client.py --url http://localhost:8000/v1 --n 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

ADAPTERS = ["soc_lora_r8", "soc_dora_r8", "soc_qlora_r64"]


async def main() -> None:
    import httpx
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="data/test.jsonl")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data)][: a.n]
    stats = {m: {"ok": 0, "lat": []} for m in ADAPTERS}
    async with httpx.AsyncClient(timeout=60) as client:
        async def one(model, row):
            t0 = time.perf_counter()
            r = await client.post(f"{a.url}/chat/completions", json={
                "model": model, "messages": row["prompt"], "max_tokens": 8, "temperature": 0})
            text = r.json()["choices"][0]["message"]["content"].strip().split()
            stats[model]["lat"].append(time.perf_counter() - t0)
            stats[model]["ok"] += bool(text) and text[0].strip(".,") == row["label"]
        await asyncio.gather(*(one(m, row) for row in rows for m in ADAPTERS))
    for m, s in stats.items():
        lat = sorted(s["lat"])
        print(f"{m:<15} acc={s['ok'] / len(rows):.3f}  p50={lat[len(lat) // 2] * 1000:.0f}ms  "
              f"p95={lat[int(len(lat) * 0.95)] * 1000:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
