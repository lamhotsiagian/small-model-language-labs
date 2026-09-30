"""
Lab 20, step 3: a guarded, observable API in front of the SLM.

Request path:
  auth -> input screen (injection heuristic + guard SLM) -> spotlight documents
  -> SLM (vLLM upstream) -> tool policy check -> output screen (secret scan)
  -> response with model/version headers

Observability:
  * Prometheus metrics: request latency, tokens, blocks by stage, escalations
  * structured JSON logs with request id, model version, decisions, and
    PII-redacted text (redaction from Lab 6); raw content is NOT logged by default

Usage:
    vllm serve ../lab14_peft/merged/soc_3b --port 8000 --served-model-name soc-slm
    uvicorn api_server:app --port 9000
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from guardrail import GuardClassifier, injection_score, spotlight, tool_allowed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab06_data_curation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab15_function_calling"))
from curate import scrub_pii  # noqa: E402
from eval_toolcall import parse_calls  # noqa: E402

UPSTREAM = os.environ.get("UPSTREAM", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL_NAME", "soc-slm")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "soc-slm@2025-10-01+lora_r8")
API_KEYS = set(os.environ.get("API_KEYS", "dev-key").split(","))
SECRETS = [s for s in os.environ.get("SECRET_MARKERS", "CANARY-").split(",") if s]

LAT = Histogram("slm_request_seconds", "end-to-end latency", buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8))
TOK = Counter("slm_output_tokens_total", "output tokens")
BLOCK = Counter("slm_blocked_total", "blocked requests", ["stage"])
ESC = Counter("slm_escalations_total", "escalations to a human or larger model")

log = logging.getLogger("slm")
logging.basicConfig(level=logging.INFO, format="%(message)s")
guard = GuardClassifier()
app = FastAPI()
app.mount("/metrics", make_asgi_app())


@app.get("/healthz")
def health():
    return {"ok": True, "model_version": MODEL_VERSION}


@app.post("/v1/chat/completions")
async def chat(body: dict, authorization: str = Header(default="")):
    rid, t0 = str(uuid.uuid4()), time.perf_counter()
    if authorization.removeprefix("Bearer ") not in API_KEYS:
        raise HTTPException(401, "invalid key")
    msgs = body.get("messages", [])
    user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    context = body.get("context", "chat")
    decision = {"rid": rid, "model_version": MODEL_VERSION, "context": context}

    if injection_score(user) > 0 or guard.classify_prompt(user) == "unsafe":
        BLOCK.labels("input").inc()
        decision["blocked"] = "input"
        log.info(json.dumps({**decision, "user": scrub_pii(user)[0][:200]}))
        return reply("I can't help with that request.", rid)
    for m in msgs:                                    # mark untrusted document text
        if m["role"] == "user" and "<document>" in m["content"]:
            head, _, rest = m["content"].partition("<document>")
            m["content"] = head + spotlight(rest.split("</document>")[0])

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{UPSTREAM}/chat/completions", json={**body, "model": MODEL})
    out = r.json()
    text = out["choices"][0]["message"]["content"]
    calls, _ = parse_calls(text)
    for c in calls:
        ok, why = tool_allowed(context, c)
        if not ok:
            BLOCK.labels("tool_policy").inc(); ESC.inc()
            decision["blocked"] = f"tool_policy: {why}"
            text = "This action needs approval from an analyst. I've flagged it for review."
            break
    if any(s in text for s in SECRETS):
        BLOCK.labels("output").inc()
        decision["blocked"] = "output_secret"
        text = "I can't share that."
    TOK.inc(out.get("usage", {}).get("completion_tokens", 0))
    LAT.observe(time.perf_counter() - t0)
    log.info(json.dumps({**decision, "latency_s": round(time.perf_counter() - t0, 3)}))
    return reply(text, rid)


def reply(text: str, rid: str):
    return {"id": rid, "model": MODEL_VERSION, "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}]}
