"""
Lab 11, step 2 (optional): Magpie-style self-synthesis of instructions.

Magpie (Z. Xu et al., 2025) exploits the fact that an aligned chat model,
given ONLY the template prefix for a user turn, autoregressively writes a
plausible user instruction. A second call answers it. No seed prompts, no
human data: diversity comes from sampling the model's own prior over user
requests. Filter the results (length, dedup, quality judge) before use.

Usage:
    python magpie_synth.py --model Qwen/Qwen2.5-7B-Instruct --n 20000 --out data/magpie.jsonl
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--out", default="data/magpie.jsonl")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams  # type: ignore
    llm = LLM(a.model, max_model_len=4096)
    tok = llm.get_tokenizer()
    # The pre-query template: everything up to (and including) the user-turn header.
    pre_query = tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False)
    pre_query = pre_query[: pre_query.index("X")]
    q_params = SamplingParams(temperature=1.0, top_p=0.99, max_tokens=256,
                              stop=["<|im_end|>", "<|eot_id|>"])
    instructions = [o.outputs[0].text.strip() for o in llm.generate([pre_query] * a.n, q_params)]
    instructions = [q for q in dict.fromkeys(instructions) if 8 <= len(q.split()) <= 200]
    r_params = SamplingParams(temperature=0.0, max_tokens=1024)
    answers = llm.chat([[{"role": "user", "content": q}] for q in instructions], r_params)
    with open(a.out, "w") as f:
        for q, r in zip(instructions, answers):
            f.write(json.dumps({"messages": [{"role": "user", "content": q},
                                             {"role": "assistant", "content": r.outputs[0].text}],
                                "source": f"magpie:{a.model}"}) + "\n")
    print(f"[lab11] {len(instructions)} unique instructions written to {a.out}")


if __name__ == "__main__":
    main()
