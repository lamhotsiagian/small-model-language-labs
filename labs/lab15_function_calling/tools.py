"""
Lab 15: tool registry (JSON Schema) and mock executors.

Tool schemas are API contracts: the model is trained on them, the grammar
constrains decoding to them, and the evaluator scores against them. Keeping
all three in one registry is what makes the system testable.
"""
from __future__ import annotations

import json
import random

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_order_status", "description": "Look up an order by its ID.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "convert_currency", "description": "Convert an amount between ISO currencies.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
            "required": ["amount", "from", "to"]}}},
    {"type": "function", "function": {
        "name": "search_docs", "description": "Search the internal knowledge base.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_ticket", "description": "Open a support ticket.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}},
            "required": ["title", "priority"]}}},
]
BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}


def execute(name: str, args: dict) -> str:
    """Mock backends with deterministic-ish outputs (replace with real APIs or MCP servers)."""
    rng = random.Random(json.dumps([name, args], sort_keys=True))
    if name == "get_weather":
        return json.dumps({"city": args["city"], "temp_c": rng.randint(18, 34), "condition": "cloudy"})
    if name == "get_order_status":
        return json.dumps({"order_id": args["order_id"], "status": rng.choice(["shipped", "processing"])})
    if name == "convert_currency":
        return json.dumps({"result": round(float(args["amount"]) * rng.uniform(0.5, 2.0), 2)})
    if name == "search_docs":
        return json.dumps({"results": [f"doc-{rng.randint(1, 99)}: refund policy is 30 days"]})
    if name == "create_ticket":
        return json.dumps({"ticket_id": f"T-{rng.randint(1000, 9999)}", "priority": args["priority"]})
    raise KeyError(name)
