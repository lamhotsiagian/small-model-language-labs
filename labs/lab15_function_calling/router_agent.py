"""
Lab 15, step 5: a small agent that uses the fine-tuned SLM as a tool router.

Loop (Chapter 15, sections 15.3 and 15.5), with the controls a production
agent needs:
  * max_steps and a per-tool timeout (tools are remote calls; they fail)
  * schema validation of every call before execution (never execute
    unvalidated model output)
  * an escalation path: invalid call twice, or low confidence -> hand the
    conversation to a larger model (here: a stub you point at any API)
  * a trace of every step for evaluation and debugging

The model is reached through an OpenAI-compatible endpoint (vLLM, SGLang,
llama.cpp server, Ollama), so the same agent runs against a laptop or a GPU.

Usage:
    vllm serve checkpoints/toolcall_1b --port 8001
    python router_agent.py --url http://localhost:8001/v1 --model checkpoints/toolcall_1b \
        --ask "Where is order 88213, and what's the weather in Medan?"
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json

from eval_toolcall import parse_calls, schema_ok
from gen_toolcall_data import system_prompt
from tools import execute


def chat(url: str, model: str, messages, max_tokens: int = 256) -> str:
    import httpx
    r = httpx.post(f"{url}/chat/completions", timeout=60, json={
        "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def escalate(messages) -> str:
    return "[escalated to large model] " + messages[-1]["content"][:80]


def run(url: str, model: str, question: str, max_steps: int = 4, tool_timeout_s: float = 5.0):
    msgs = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": question}]
    trace, invalid = [], 0
    pool = cf.ThreadPoolExecutor(max_workers=4)
    for step in range(max_steps):
        out = chat(url, model, msgs)
        calls, ok = parse_calls(out)
        if not calls:                                   # model chose to answer
            trace.append({"step": step, "answer": out})
            return out, trace
        if not ok or not all(schema_ok(c) for c in calls):
            invalid += 1
            trace.append({"step": step, "invalid": out})
            if invalid >= 2:
                return escalate(msgs), trace
            msgs.append({"role": "user", "content": "That tool call was invalid. Follow the schema exactly."})
            continue
        msgs.append({"role": "assistant", "content": out})
        futures = {pool.submit(execute, c["name"], c["arguments"]): c for c in calls}   # parallel calls
        for fut, c in futures.items():
            try:
                result = fut.result(timeout=tool_timeout_s)
            except Exception as exc:                    # timeout or backend error is DATA for the model
                result = json.dumps({"error": type(exc).__name__})
            trace.append({"step": step, "call": c, "result": result})
            msgs.append({"role": "tool", "content": result})
    return escalate(msgs), trace                        # step budget exhausted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ask", required=True)
    a = ap.parse_args()
    answer, trace = run(a.url, a.model, a.ask)
    for t in trace:
        print(json.dumps(t))
    print("\nFINAL:", answer)


if __name__ == "__main__":
    main()
