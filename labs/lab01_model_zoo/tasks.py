"""
Lab 1 task suite: 50 prompts, 10 per task type, each with an automatic scorer.

The five task types are the ones Chapter 1 argues SLMs win most often:
classification, extraction, routing, summarization, and tool calling. Every
scorer returns a value in [0, 1] so task scores are comparable and can be
weighted into a decision matrix.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class Task:
    task_type: str
    system: str
    user: str
    reference: object
    scorer: Callable[[str, object], float]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
def score_label(output: str, ref: str) -> float:
    """1.0 if the first label-like word matches the reference label."""
    m = re.search(r"[A-Za-z_]+", output.strip().lower())
    return float(bool(m) and m.group(0) == ref.lower())


def _first_json(text: str):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def score_extraction(output: str, ref: dict) -> float:
    """Field-level accuracy over the reference keys (0 if JSON is invalid)."""
    obj = _first_json(output)
    if not isinstance(obj, dict):
        return 0.0
    hits = sum(str(obj.get(k, "")).strip().lower() == str(v).lower() for k, v in ref.items())
    return hits / len(ref)


def score_summary(output: str, ref: List[str]) -> float:
    """Keyword recall with a length penalty: summaries must be short AND cover facts."""
    out = output.lower()
    recall = sum(k.lower() in out for k in ref) / len(ref)
    words = len(output.split())
    penalty = 1.0 if words <= 40 else max(0.0, 1 - (words - 40) / 60)
    return recall * penalty


def score_tool_call(output: str, ref: dict) -> float:
    """0.4 valid JSON, +0.3 correct tool name, +0.3 correct arguments."""
    obj = _first_json(output)
    if not isinstance(obj, dict):
        return 0.0
    s = 0.4
    if obj.get("name") == ref["name"]:
        s += 0.3
        args = obj.get("arguments", {})
        if isinstance(args, dict) and all(str(args.get(k)).lower() == str(v).lower()
                                          for k, v in ref["arguments"].items()):
            s += 0.3
    return s


# ---------------------------------------------------------------------------
# Prompts (10 per type). Short, unambiguous, and cheap to score.
# ---------------------------------------------------------------------------
CLS_SYS = ("Classify the customer message into exactly one label: billing, "
           "technical, account, shipping, other. Reply with the label only.")
CLS = [("I was charged twice for my subscription this month.", "billing"),
       ("The app crashes every time I open the camera screen.", "technical"),
       ("How do I change the email address on my profile?", "account"),
       ("My package says delivered but it is not here.", "shipping"),
       ("Do you have an office in Jakarta?", "other"),
       ("Refund has not arrived after 10 days.", "billing"),
       ("Login page returns error 502.", "technical"),
       ("Please delete my account and all data.", "account"),
       ("Can I change the delivery address for order 5531?", "shipping"),
       ("What are your opening hours on holidays?", "other")]

EXT_SYS = ("Extract the fields as JSON with keys name, date, amount. "
           "Use ISO dates (YYYY-MM-DD) and numbers without currency symbols. Output JSON only.")
EXT = [("Invoice for Maria Chen dated March 3, 2025, total $1,250.", {"name": "Maria Chen", "date": "2025-03-03", "amount": "1250"}),
       ("Budi Santoso paid 300 on 2024-12-01.", {"name": "Budi Santoso", "date": "2024-12-01", "amount": "300"}),
       ("Receipt: Ana Lopez, 15 Jan 2025, amount 89.", {"name": "Ana Lopez", "date": "2025-01-15", "amount": "89"}),
       ("On 2025-07-04 John Park transferred 4200.", {"name": "John Park", "date": "2025-07-04", "amount": "4200"}),
       ("Amount 75 received from Siti Rahma on May 9, 2025.", {"name": "Siti Rahma", "date": "2025-05-09", "amount": "75"}),
       ("Lee Wong, 2023-11-30, 560 due.", {"name": "Lee Wong", "date": "2023-11-30", "amount": "560"}),
       ("Payment of 999 by Omar Ali on 1 February 2025.", {"name": "Omar Ali", "date": "2025-02-01", "amount": "999"}),
       ("Emma Stone was billed 42 on 2025-08-18.", {"name": "Emma Stone", "date": "2025-08-18", "amount": "42"}),
       ("2024-06-21: Raj Patel, 1800.", {"name": "Raj Patel", "date": "2024-06-21", "amount": "1800"}),
       ("Dewi Lestari settled 610 on October 2, 2024.", {"name": "Dewi Lestari", "date": "2024-10-02", "amount": "610"})]

ROUTE_SYS = ("You are a router. Decide which backend should answer: 'local' for simple "
             "FAQ, greetings, and short lookups; 'large' for multi-step reasoning, code, "
             "legal, or medical questions. Reply with local or large only.")
ROUTE = [("Hi there!", "local"), ("What is your return policy?", "local"),
         ("Write a Python function that merges k sorted linked lists and analyse its complexity.", "large"),
         ("Is this clause in my employment contract enforceable in California?", "large"),
         ("What time do you close today?", "local"),
         ("Plan a 3-step migration from MySQL to Postgres with zero downtime.", "large"),
         ("Thanks, that helped!", "local"),
         ("Given my symptoms and medications, what dosage adjustment is safe?", "large"),
         ("Where can I download the invoice?", "local"),
         ("Prove that the sum of the first n odd numbers is n squared.", "large")]

SUM_SYS = "Summarize the text in one sentence of at most 30 words."
SUM = [("The quarterly report shows revenue grew 12% to $4.1M, driven by the new mobile app, while costs rose 5% due to cloud spend.", ["12%", "mobile", "cloud"]),
       ("A storm closed the airport for six hours on Friday; 140 flights were cancelled and operations resumed Saturday morning.", ["airport", "140", "saturday"]),
       ("The team migrated the search service to a 1B model on CPU, cutting latency from 900 ms to 120 ms and cost by 70%.", ["1b", "120", "70%"]),
       ("Researchers found that deduplicating training data reduced memorization and improved downstream accuracy by two points.", ["dedup", "memorization", "two"]),
       ("The city council approved a new bike lane network covering 40 km, to be built over three years starting in 2026.", ["bike", "40", "2026"]),
       ("After the firmware update, battery life on the sensor improved from 9 to 14 months according to field tests.", ["firmware", "14", "battery"]),
       ("Support tickets dropped 30% after the chatbot started answering password reset questions automatically.", ["30%", "password", "chatbot"]),
       ("The bakery will open a second branch in Bandung in June, hiring 15 staff and adding a gluten-free line.", ["bandung", "june", "15"]),
       ("The model was quantized to 4-bit, shrinking from 3 GB to 0.9 GB, with a one-point drop on the internal benchmark.", ["4-bit", "0.9", "one-point"]),
       ("A new policy requires two-factor authentication for all admin accounts by the end of the quarter.", ["two-factor", "admin", "quarter"])]

TOOL_SYS = ("You can call one tool. Tools: get_weather(city), get_order_status(order_id), "
            "convert_currency(amount, from, to). Respond ONLY with JSON: "
            '{"name": <tool>, "arguments": {...}}')
TOOL = [("What's the weather in Jakarta?", {"name": "get_weather", "arguments": {"city": "Jakarta"}}),
        ("Where is my order 88213?", {"name": "get_order_status", "arguments": {"order_id": "88213"}}),
        ("Convert 100 USD to IDR.", {"name": "convert_currency", "arguments": {"amount": "100", "from": "USD", "to": "IDR"}}),
        ("Is it raining in Paris right now?", {"name": "get_weather", "arguments": {"city": "Paris"}}),
        ("Status of order A-771 please.", {"name": "get_order_status", "arguments": {"order_id": "A-771"}}),
        ("How much is 50 EUR in USD?", {"name": "convert_currency", "arguments": {"amount": "50", "from": "EUR", "to": "USD"}}),
        ("Weather for Tokyo today.", {"name": "get_weather", "arguments": {"city": "Tokyo"}}),
        ("Track order 10045.", {"name": "get_order_status", "arguments": {"order_id": "10045"}}),
        ("Change 2500 JPY to SGD.", {"name": "convert_currency", "arguments": {"amount": "2500", "from": "JPY", "to": "SGD"}}),
        ("Temperature in Medan?", {"name": "get_weather", "arguments": {"city": "Medan"}})]


def build_suite() -> List[Task]:
    suite: List[Task] = []
    suite += [Task("classification", CLS_SYS, u, r, score_label) for u, r in CLS]
    suite += [Task("extraction", EXT_SYS, u, r, score_extraction) for u, r in EXT]
    suite += [Task("routing", ROUTE_SYS, u, r, score_label) for u, r in ROUTE]
    suite += [Task("summarization", SUM_SYS, u, r, score_summary) for u, r in SUM]
    suite += [Task("tool_calling", TOOL_SYS, u, r, score_tool_call) for u, r in TOOL]
    return suite


if __name__ == "__main__":
    s = build_suite()
    print(len(s), "tasks:", sorted({t.task_type for t in s}))
