"""
Lab 14, step 6: merge adapters (linear, TIES, DARE) or fold one into the base.

  --merge-only        W' = W + s * B A for a single adapter, saved as a dense model
                      (zero inference overhead; required by some runtimes)
  --combine a b ...   combine several adapters with PEFT's add_weighted_adapter:
                      linear  weighted average of task vectors
                      ties    trim small deltas, elect sign, merge agreeing (Yadav et al., 2023)
                      dare    randomly drop deltas and rescale before merging (Yu et al., 2024)

Usage:
    python merge.py --adapter checkpoints/lora_r8 --merge-only --out merged/soc_3b
    python merge.py --combine checkpoints/lora_r8 checkpoints/other_task --method ties --density 0.5
"""
from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--combine", nargs="*")
    ap.add_argument("--method", default="ties", choices=["linear", "ties", "dare_ties", "dare_linear"])
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--out", default="merged/out")
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.base)
    if a.merge_only:
        model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    else:
        names = [f"t{i}" for i in range(len(a.combine))]
        model = PeftModel.from_pretrained(base, a.combine[0], adapter_name=names[0])
        for n, path in zip(names[1:], a.combine[1:]):
            model.load_adapter(path, adapter_name=n)
        model.add_weighted_adapter(names, [1.0] * len(names), adapter_name="merged",
                                   combination_type=a.method, density=a.density)
        model.set_adapter("merged")
        model = model.merge_and_unload()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"[lab14] wrote {a.out}")


if __name__ == "__main__":
    main()
