"""
Lab 18, step 1: will it fit, and how fast can it possibly go? (computed)

For each (device, model, weight format) the planner adds weights + KV cache
at the target context + activations + runtime overhead, compares against the
memory an app can realistically use (NOT the device's total RAM), and prints
the bandwidth roofline for batch-1 decode.

App memory budgets are deliberately conservative: mobile OSes kill apps well
before physical RAM is exhausted, and the model shares memory with the OS,
the UI, and other apps.

Usage:
    python memory_plan.py --context 4096
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> import slmlab
from slmlab.budget import BYTES, ZOO, decode_tokens_per_sec_bound, memory_plan, param_breakdown  # noqa: E402

# name: (usable app memory GB, memory bandwidth GB/s) -- approximate, check your device
DEVICES = {
    "phone 8GB (flagship)": (3.0, 60),
    "phone 12GB (flagship)": (5.0, 75),
    "Raspberry Pi 5 8GB": (6.0, 17),
    "Jetson Orin Nano 8GB": (6.0, 68),
    "laptop 16GB unified": (8.0, 100),
    "laptop 36GB unified (Max-class)": (24.0, 400),
}
MODELS = ["Qwen2.5-0.5B", "Llama-3.2-1B", "Qwen2.5-1.5B", "Llama-3.2-3B"]
FORMATS = ["q4_k_m", "q8_0", "bf16"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--kv", default="fp16")
    a = ap.parse_args()
    print(f"context {a.context}, KV {a.kv}; 'ok' = fits app budget; t/s = batch-1 decode ceiling\n")
    header = f"{'model / format':<24}" + "".join(f"{d[:18]:>20}" for d in DEVICES)
    print(header)
    for m in MODELS:
        spec = ZOO[m]
        for fmt in FORMATS:
            plan = memory_plan(spec, weight_dtype=fmt, kv_dtype=a.kv, context=a.context)
            need_gb = plan["total_mb"] / 1024
            wbytes = param_breakdown(spec)["total"] * BYTES[fmt]
            cells = []
            for budget_gb, bw in DEVICES.values():
                tps = decode_tokens_per_sec_bound(wbytes, bw)
                cells.append(f"{'ok' if need_gb <= budget_gb else 'NO':>3} {need_gb:4.1f}G {tps:5.0f}t/s")
            print(f"{m + ' ' + fmt:<24}" + "".join(f"{c:>20}" for c in cells))


if __name__ == "__main__":
    main()
