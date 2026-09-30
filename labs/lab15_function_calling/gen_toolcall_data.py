"""
Lab 15, step 1: 20K synthetic function-calling examples.

Five example types, mixed on purpose (Chapter 15, section 15.1):

  single      one call with correct arguments                        45%
  parallel    two independent calls in one turn                      15%
  multi_turn  call -> tool result -> grounded natural-language answer 20%
  irrelevant  no tool applies: answer directly or ask a question      15%
  missing     required argument missing: ask for it, do NOT guess      5%

Irrelevance and missing-argument examples are what stop a small model from
calling a tool for every message, the most common failure of tool-tuned SLMs.

Format: Hermes/Qwen style. The system prompt lists tools; calls are emitted as
<tool_call>{"name": ..., "arguments": {...}}</tool_call>; results come back as
role "tool". Output rows are prompt/completion for TRL (final assistant turn).

Usage:
    python gen_toolcall_data.py --n 20000 --out data/toolcalls.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

from tools import TOOLS, execute

CITIES = ["Jakarta", "Medan", "Bandung", "Tokyo", "Paris", "Nairobi", "Lima", "Toronto"]
CUR = ["USD", "IDR", "EUR", "JPY", "SGD", "GBP"]
CHAT = ["Thanks!", "What can you do?", "Tell me a fun fact about octopuses.", "Good morning",
        "Explain what an API is in one sentence.", "How do I reset my password?"]


def system_prompt() -> str:
    return ("You are a helpful assistant with access to these tools:\n<tools>\n"
            + "\n".join(json.dumps(t["function"]) for t in TOOLS) + "\n</tools>\n"
            "Call a tool by replying with <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>. "
            "Only call a tool when it is needed. If a required argument is missing, ask for it.")


def call(name, args) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args})}\n</tool_call>"


def sample_call(r):
    k = r.choice(["weather", "order", "fx", "docs", "ticket"])
    if k == "weather":
        c = r.choice(CITIES)
        u = r.choice([None, "celsius", "fahrenheit"])
        q = r.choice([f"What's the weather in {c}?", f"Is it hot in {c} right now?", f"{c} weather please"])
        if u:
            q += f" In {u}."
        return q, "get_weather", ({"city": c, "unit": u} if u else {"city": c})
    if k == "order":
        oid = r.choice([str(r.randint(10000, 99999)), f"A-{r.randint(100, 999)}"])
        return r.choice([f"Where is order {oid}?", f"Track {oid}", f"Status of my order {oid}?"]), \
            "get_order_status", {"order_id": oid}
    if k == "fx":
        amt, a, b = r.choice([10, 50, 100, 2500]), *r.sample(CUR, 2)
        return f"Convert {amt} {a} to {b}.", "convert_currency", {"amount": amt, "from": a, "to": b}
    if k == "docs":
        topic = r.choice(["refund policy", "shipping to Bali", "warranty for laptops", "API rate limits"])
        return f"What does our knowledge base say about {topic}?", "search_docs", {"query": topic}
    pr = r.choice(["low", "medium", "high", "urgent"])
    issue = r.choice(["login page is down", "invoice is wrong", "app crashes on start"])
    return f"Open a {pr} priority ticket: {issue}.", "create_ticket", {"title": issue, "priority": pr}


def example(r) -> dict:
    kind = r.choices(["single", "parallel", "multi_turn", "irrelevant", "missing"],
                     weights=[45, 15, 20, 15, 5])[0]
    sys_msg = {"role": "system", "content": system_prompt()}
    if kind == "single":
        q, n, a = sample_call(r)
        return {"kind": kind, "prompt": [sys_msg, {"role": "user", "content": q}],
                "completion": [{"role": "assistant", "content": call(n, a)}], "gold": [{"name": n, "arguments": a}]}
    if kind == "parallel":
        (q1, n1, a1), (q2, n2, a2) = sample_call(r), sample_call(r)
        return {"kind": kind, "prompt": [sys_msg, {"role": "user", "content": f"{q1} Also, {q2[0].lower()}{q2[1:]}"}],
                "completion": [{"role": "assistant", "content": call(n1, a1) + "\n" + call(n2, a2)}],
                "gold": [{"name": n1, "arguments": a1}, {"name": n2, "arguments": a2}]}
    if kind == "multi_turn":
        q, n, a = sample_call(r)
        result = execute(n, a)
        answer = f"Here is what I found: {result}"
        return {"kind": kind, "prompt": [sys_msg, {"role": "user", "content": q},
                                         {"role": "assistant", "content": call(n, a)},
                                         {"role": "tool", "content": result}],
                "completion": [{"role": "assistant", "content": answer}], "gold": []}
    if kind == "irrelevant":
        q = r.choice(CHAT)
        return {"kind": kind, "prompt": [sys_msg, {"role": "user", "content": q}],
                "completion": [{"role": "assistant", "content": "Happy to help! " + (
                    "I can check weather, orders, currency, docs, and tickets." if "can you" in q else
                    "Could you tell me a bit more about what you need?")}], "gold": []}
    return {"kind": kind, "prompt": [sys_msg, {"role": "user", "content": "Convert this amount to euros please."}],
            "completion": [{"role": "assistant", "content": "Sure. How much, and from which currency?"}], "gold": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--out", default="data/toolcalls.jsonl")
    a = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    r = random.Random(0)
    counts = {}
    with open(a.out, "w") as f:
        for _ in range(a.n):
            ex = example(r)
            counts[ex["kind"]] = counts.get(ex["kind"], 0) + 1
            f.write(json.dumps(ex) + "\n")
    print(f"[lab15] wrote {a.n} examples: {counts}")


if __name__ == "__main__":
    main()
