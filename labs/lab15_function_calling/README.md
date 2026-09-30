# Lab 15: Function-Calling SLM

**Chapter:** 15, Tool Use, Structured Output, and Agentic SLMs
**Goal:** Fine-tune a 1B model on 20K synthetic tool-call examples, add
grammar-constrained decoding, and evaluate with BFCL-style AST matching. Build
a small agent that uses the model as a tool router. Deliver accuracy with and
without constraints, plus a demo agent.

## Files

| File | Purpose |
|---|---|
| `tools.py` | Five tools as JSON Schemas + mock executors (swap for real APIs or MCP servers) |
| `gen_toolcall_data.py` | 20K examples: single, parallel, multi-turn with tool results, irrelevance, missing-argument |
| `constrained.py` | Constrained-decoding mechanism demo + production `tool_call_schema()` and vLLM `guided_params()` |
| `eval_toolcall.py` | AST matching: parse rate, schema validity, exact call accuracy, irrelevance accuracy; `--selftest` |
| `run_eval.py` | Free vs two-stage constrained decoding (choice -> JSON schema) |
| `router_agent.py` | Agent loop with step budget, timeouts, schema validation, parallel tool calls, escalation |

## Flow

1. **Data.** `python gen_toolcall_data.py --n 20000 --out data/toolcalls.jsonl`, then split
   (`head -n 18000` train, the rest test).
2. **Mechanism.** `python constrained.py` and `python eval_toolcall.py --selftest`.
3. **Fine-tune.** Reuse Lab 11's trainer:
   `python ../lab11_sft/train_sft.py --base meta-llama/Llama-3.2-1B-Instruct --data data/train.jsonl --lr 2e-5 --epochs 2 --out checkpoints/toolcall_1b`
4. **Evaluate.** `python run_eval.py --model checkpoints/toolcall_1b --data data/test.jsonl --mode both`
   for the base instruct model and the fine-tune: four numbers per model.
   Optionally run the official BFCL harness (`pip install bfcl-eval`) on the served model.
5. **Agent.** `vllm serve checkpoints/toolcall_1b --port 8001` then
   `python router_agent.py --model checkpoints/toolcall_1b --ask "Where is order 88213, and what's the weather in Medan?"`.

## Deliverable

Table: {base, fine-tuned} x {free, constrained} x {parse, schema-valid, call
accuracy, irrelevance accuracy}; an agent trace for three scenarios
(single call, parallel calls, no call needed).

## Reference output (`python constrained.py`, computed)

```
unconstrained  valid:  45.1%   plausible-but-wrong:  48.0%   malformed:   6.9%
constrained    valid: 100.0%   plausible-but-wrong:   0.0%   malformed:   0.0%
```
