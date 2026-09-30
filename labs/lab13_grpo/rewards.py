"""
Lab 13: verifiable rewards and the GRPO group advantage (Chapter 13).

Rewards are PROGRAMS, not models: nothing to hack except the checker itself,
so write checkers carefully (Chapter 13, section 13.4).

  correctness_reward   1.0 if the extracted final answer equals the reference
  format_reward        0.2 if the response has <think>...</think> then an answer line
  length_penalty       optional soft penalty beyond a token budget (DAPO-style overlong shaping)

GRPO (Shao et al., 2024) replaces PPO's value network with a GROUP baseline:
sample G responses per prompt, and use each response's reward standardised
within its group as the advantage of every token in that response.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab08_distillation"))
from gsm8k import extract_answer  # noqa: E402

THINK = re.compile(r"^\s*<think>.+?</think>\s*.*####\s*-?[\d,.]+", re.S)


def _text(c) -> str:
    """TRL passes conversational completions as [{"role": "assistant", "content": ...}]."""
    return c[0]["content"] if isinstance(c, list) else c


def correctness_reward(completions, answer, **_):
    return [1.0 if extract_answer(_text(c)) == a else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions, **_):
    return [0.2 if THINK.match(_text(c)) else 0.0 for c in completions]


def length_penalty(completions, budget: int = 400, window: int = 100, **_):
    """0 below budget, linear to -0.5 over the next `window` words (soft overlong penalty)."""
    out = []
    for c in completions:
        n = len(_text(c).split())
        out.append(0.0 if n <= budget else -0.5 * min(1.0, (n - budget) / window))
    return out


def group_advantages(rewards: torch.Tensor, group: int, scale: bool = True) -> torch.Tensor:
    """rewards: [B*G] flat, grouped consecutively. Returns per-response advantages.

    scale=False gives Dr. GRPO's unscaled variant (Z. Liu et al., 2025b), which avoids
    up-weighting groups whose rewards are nearly identical (very easy or very hard prompts).
    """
    r = rewards.view(-1, group)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale:
        adv = adv / (r.std(dim=1, keepdim=True) + 1e-4)
    return adv.view(-1)


if __name__ == "__main__":
    comps = ["<think>48/2 = 24 in May; 48 + 24 = 72</think>\nThe answer is\n#### 72",
             "She sold 72 clips.\n#### 72",
             "<think>48 + 48 = 96</think>\n#### 96",
             "I am not sure."]
    ans = ["72"] * 4
    c, f, l = correctness_reward(comps, ans), format_reward(comps), length_penalty(comps)
    total = torch.tensor([a + b + d for a, b, d in zip(c, f, l)])
    print("rewards (correct, format, total):")
    for i, t in enumerate(total.tolist()):
        print(f"  response {i}: {c[i]:.1f} + {f[i]:.1f} = {t:.2f}")
    print("GRPO advantages (group of 4):   ", [round(x, 3) for x in group_advantages(total, 4).tolist()])
    print("Dr. GRPO advantages (unscaled): ", [round(x, 3) for x in group_advantages(total, 4, False).tolist()])
    same = torch.tensor([1.0, 1.0, 1.0, 1.0])
    print("all-correct group -> advantages:", group_advantages(same, 4).tolist(), "(no learning signal)")
