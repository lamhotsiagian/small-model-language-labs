"""
Lab 18, step 5: hybrid on-device / cloud routing policy.

Decide per request where to run, from signals the device already has:
  privacy class   'local_only' requests never leave the device
  connectivity    offline -> local, whatever the quality
  length          prompts beyond the local context budget -> cloud (if allowed)
  confidence      mean token log-prob of the local draft answer; low -> cloud
  battery/thermal low battery or hot device -> prefer cloud for long generations

The policy is a pure function so it can be unit tested and shipped as
configuration, and every decision is logged for later threshold tuning.

Usage:
    python hybrid_router.py      # prints decisions for a table of scenarios
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signals:
    privacy: str            # "local_only" | "any"
    online: bool
    prompt_tokens: int
    local_confidence: float  # mean log-prob of the on-device draft, e.g. -0.3 good, -1.5 poor
    battery_pct: int
    temp_c: float


@dataclass
class Policy:
    local_ctx: int = 4096
    min_conf: float = -0.9
    low_battery: int = 15
    hot_c: float = 42.0


def route(s: Signals, p: Policy = Policy()) -> tuple[str, str]:
    if s.privacy == "local_only":
        return "local", "privacy: data may not leave the device"
    if not s.online:
        return "local", "offline"
    if s.prompt_tokens > p.local_ctx:
        return "cloud", "prompt exceeds local context budget"
    if s.battery_pct < p.low_battery or s.temp_c > p.hot_c:
        return "cloud", "device constrained (battery/thermal)"
    if s.local_confidence < p.min_conf:
        return "cloud", f"low local confidence ({s.local_confidence:.2f} < {p.min_conf})"
    return "local", "confident local answer"


if __name__ == "__main__":
    cases = [
        ("health note summary", Signals("local_only", True, 900, -1.4, 80, 35)),
        ("airplane mode", Signals("any", False, 500, -1.2, 60, 33)),
        ("long contract", Signals("any", True, 12000, -0.4, 70, 34)),
        ("hot phone", Signals("any", True, 800, -0.3, 70, 44)),
        ("hard question", Signals("any", True, 300, -1.3, 90, 31)),
        ("easy reply", Signals("any", True, 120, -0.2, 90, 31)),
    ]
    for name, s in cases:
        where, why = route(s)
        print(f"{name:<22} -> {where:<6} ({why})")
