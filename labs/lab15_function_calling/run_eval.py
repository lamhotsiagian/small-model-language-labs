"""
Lab 15, step 4: accuracy with and without constrained decoding.

Two-stage decoding, because a JSON-only grammar cannot say "no tool needed":
  1. decide:  constrained CHOICE between "call" and "respond"
  2. call:    JSON-schema constrained decoding of one tool call (tool_call_schema)
     respond: free text

The unconstrained baseline generates freely and is parsed as-is.

Usage:
    python run_eval.py --model checkpoints/toolcall_1b --data data/test.jsonl --mode both
Official BFCL: pip install bfcl-eval, then follow its README to score a served model.
"""
from __future__ import annotations

import argparse
import json

from constrained import guided_params
from eval_toolcall import score, summarise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--mode", choices=["free", "constrained", "both"], default="both")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams  # type: ignore
    rows = [json.loads(l) for l in open(a.data) if json.loads(l)["kind"] in ("single", "irrelevant", "missing")]
    llm = LLM(a.model, max_model_len=4096)
    msgs = [r["prompt"] for r in rows]
    report = {}
    if a.mode in ("free", "both"):
        outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=256))
        report["free"] = summarise([score(o.outputs[0].text, r["gold"]) for o, r in zip(outs, rows)])
    if a.mode in ("constrained", "both"):
        try:
            from vllm.sampling_params import GuidedDecodingParams
            choose = SamplingParams(temperature=0.0, max_tokens=4,
                                    guided_decoding=GuidedDecodingParams(choice=["call", "respond"]))
        except ImportError:
            from vllm.sampling_params import StructuredOutputsParams
            choose = SamplingParams(temperature=0.0, max_tokens=4,
                                    structured_outputs=StructuredOutputsParams(choice=["call", "respond"]))
        decide = [m + [{"role": "user", "content": "Reply 'call' if a tool is required, else 'respond'."}]
                  for m in msgs]
        d = [o.outputs[0].text.strip() for o in llm.chat(decide, choose)]
        call_idx = [i for i, x in enumerate(d) if x == "call"]
        calls = llm.chat([msgs[i] for i in call_idx], guided_params())
        texts = ["(direct answer)"] * len(rows)
        for i, o in zip(call_idx, calls):
            texts[i] = f"<tool_call>{o.outputs[0].text}</tool_call>"
        report["constrained"] = summarise([score(t, r["gold"]) for t, r in zip(texts, rows)])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
