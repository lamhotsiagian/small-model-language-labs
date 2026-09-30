"""
Lab 15, step 4: BFCL-style evaluation of tool calls by AST matching.

A call is correct when (Patil et al., 2025):
  * it parses (valid JSON inside <tool_call> ... </tool_call>)
  * the function name exists and matches the gold call
  * every required argument is present, no unknown arguments appear,
    types match the schema, enum values are valid, values match gold
Parallel calls are matched as a multiset (order-insensitive).
Irrelevance: when gold is empty, emitting ANY call is an error.

Usage:
    python eval_toolcall.py --pred results/preds.jsonl
    python eval_toolcall.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re

from tools import BY_NAME

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
PY_TYPES = {"string": str, "number": (int, float), "integer": int, "object": dict, "boolean": bool}


def parse_calls(text: str):
    """Return (calls, parse_ok)."""
    calls, ok = [], True
    for blob in CALL_RE.findall(text):
        try:
            calls.append(json.loads(blob))
        except json.JSONDecodeError:
            ok = False
    if "<tool_call>" in text and not calls:
        ok = False
    return calls, ok


def schema_ok(call) -> bool:
    fn = BY_NAME.get(call.get("name"))
    if fn is None or not isinstance(call.get("arguments"), dict):
        return False
    props, args = fn["parameters"]["properties"], call["arguments"]
    if any(k not in args for k in fn["parameters"].get("required", [])):
        return False
    for k, v in args.items():
        if k not in props or not isinstance(v, PY_TYPES[props[k]["type"]]):
            return False
        if "enum" in props[k] and v not in props[k]["enum"]:
            return False
    return True


def norm(call):
    return json.dumps({"name": call.get("name"), "arguments": call.get("arguments")}, sort_keys=True)


def score(text: str, gold: list) -> dict:
    calls, ok = parse_calls(text)
    if not gold:                                   # irrelevance / clarification
        return {"parse": ok, "schema": True, "exact": not calls, "irrelevance_case": True}
    return {"parse": ok and bool(calls), "schema": bool(calls) and all(schema_ok(c) for c in calls),
            "exact": sorted(map(norm, calls)) == sorted(map(norm, gold)), "irrelevance_case": False}


def summarise(rows):
    n = len(rows)
    rel = [r for r in rows if not r["irrelevance_case"]]
    irr = [r for r in rows if r["irrelevance_case"]]
    return {"n": n, "parse_rate": sum(r["parse"] for r in rel) / max(1, len(rel)),
            "schema_valid": sum(r["schema"] for r in rel) / max(1, len(rel)),
            "call_accuracy": sum(r["exact"] for r in rel) / max(1, len(rel)),
            "irrelevance_accuracy": sum(r["exact"] for r in irr) / max(1, len(irr))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        g = [{"name": "get_weather", "arguments": {"city": "Medan"}}]
        cases = [
            ('<tool_call>{"name": "get_weather", "arguments": {"city": "Medan"}}</tool_call>', g),
            ('<tool_call>{"name": "get_weather", "arguments": {"city": "Medan", "unit": "kelvin"}}</tool_call>', g),
            ('<tool_call>{"name": "get_weather", "arguments": {"city": "Medan"</tool_call>', g),
            ('<tool_call>{"name": "get_forecast", "arguments": {"city": "Medan"}}</tool_call>', g),
            ("It is probably sunny in Medan.", g),
            ('<tool_call>{"name": "search_docs", "arguments": {"query": "hi"}}</tool_call>', []),
            ("You're welcome!", []),
        ]
        for text, gold in cases:
            print(f"{json.dumps(score(text, gold))}  <- {text[:60]}")
        print(summarise([score(t, g) for t, g in cases]))
        return
    rows = [json.loads(l) for l in open(a.pred)]
    print(json.dumps(summarise([score(r["output"], r["gold"]) for r in rows]), indent=2))


if __name__ == "__main__":
    main()
