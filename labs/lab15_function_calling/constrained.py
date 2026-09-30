"""
Lab 15, step 3: constrained decoding, from the mechanism to production.

Mechanism (runnable on CPU): at every step, mask every token that would make
the output an invalid prefix of the grammar, then sample from what remains.
Here the "grammar" is a finite set of strings (tool names), the vocabulary is
characters, and the "model" is a noisy scorer that likes plausible-but-wrong
names; that is exactly how small models fail at tool names.

Production: the same idea compiled from a JSON Schema into a token-level
automaton (XGrammar, Outlines, llama.cpp GBNF). `tool_call_schema()` builds
the schema; `guided_params()` returns vLLM sampling params that enforce it.

Usage:
    python constrained.py            # mechanism demo
"""
from __future__ import annotations

import math
import random

from tools import TOOLS


def tool_call_schema() -> dict:
    """JSON Schema accepting exactly one valid call to any registered tool."""
    return {"anyOf": [{
        "type": "object",
        "properties": {"name": {"const": t["function"]["name"]},
                       "arguments": t["function"]["parameters"]},
        "required": ["name", "arguments"], "additionalProperties": False}
        for t in TOOLS]}


def guided_params(max_tokens: int = 256):
    """vLLM structured-output params (API name differs across vLLM versions)."""
    from vllm import SamplingParams  # type: ignore
    schema = tool_call_schema()
    try:
        from vllm.sampling_params import GuidedDecodingParams  # vLLM 0.6-0.9
        return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                              guided_decoding=GuidedDecodingParams(json=schema))
    except ImportError:
        from vllm.sampling_params import StructuredOutputsParams  # newer vLLM
        return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                              structured_outputs=StructuredOutputsParams(json=schema))


class PrefixConstrainedSampler:
    """Character-level constrained sampling against a finite language."""

    def __init__(self, allowed: list[str], end: str = "$") -> None:
        self.allowed = [a + end for a in allowed]
        self.end = end

    def valid_next(self, prefix: str) -> set[str]:
        return {a[len(prefix)] for a in self.allowed if a.startswith(prefix) and len(a) > len(prefix)}

    def sample(self, score, rng, constrained: bool, alphabet: str) -> str:
        out = ""
        while not out.endswith(self.end) and len(out) < 40:
            cand = self.valid_next(out) if constrained else set(alphabet + self.end)
            logits = {c: score(out, c) for c in cand}
            z = max(logits.values())
            probs = {c: math.exp(v - z) for c, v in logits.items()}
            tot = sum(probs.values())
            x, acc = rng.random() * tot, 0.0
            for c, p in probs.items():
                acc += p
                if acc >= x:
                    out += c
                    break
        return out.rstrip(self.end)


PLAUSIBLE_WRONG = ["get_forecast", "order_status", "currency_convert", "search_doc", "new_ticket"]


def noisy_model(target_names: list[str]):
    """Scores characters like a small model that half-remembers tool names:
    it prefers continuations of real names but also of plausible variants."""
    variants = target_names + PLAUSIBLE_WRONG

    def score(prefix: str, c: str) -> float:
        s = -7.0
        for v in variants:
            v2 = v + "$"
            if v2.startswith(prefix + c):
                s = max(s, 1.5 if v in target_names else 1.2)
        return s
    return score


def main() -> None:
    names = [t["function"]["name"] for t in TOOLS]
    sampler = PrefixConstrainedSampler(names)
    alphabet = "abcdefghijklmnopqrstuvwxyz_"
    score = noisy_model(names)
    for constrained in (False, True):
        rng = random.Random(0)
        outs = [sampler.sample(score, rng, constrained, alphabet) for _ in range(1000)]
        valid = sum(o in names for o in outs) / len(outs)
        plausible = [o for o in outs if o in PLAUSIBLE_WRONG]
        print(f"{'constrained' if constrained else 'unconstrained':<14} valid: {valid:6.1%}   "
              f"plausible-but-wrong: {len(plausible) / len(outs):6.1%}   "
              f"malformed: {1 - valid - len(plausible) / len(outs):6.1%}")
    n_branches = len(tool_call_schema()["anyOf"])
    print(f"\nproduction schema: anyOf over {n_branches} tools; pass guided_params() to vLLM")


if __name__ == "__main__":
    main()
