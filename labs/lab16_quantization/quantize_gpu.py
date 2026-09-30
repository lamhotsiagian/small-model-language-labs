"""
Lab 16, step 2: GPU quantization formats for server deployment.

  int8   bitsandbytes LLM.int8 (mixed-precision outlier decomposition; Dettmers et al., 2022)
  awq    4-bit weight-only, activation-aware scaling (Lin et al., 2024) via llm-compressor
  gptq   4-bit weight-only, second-order error compensation (Frantar et al., 2023) via llm-compressor

AWQ and GPTQ need calibration data. Use text from YOUR domain: the lab's
default is a FineWeb-Edu sample, but calibrating on support tickets for a
support bot measurably changes which weights get protected.

Usage:
    python quantize_gpu.py --method awq  --model Qwen/Qwen2.5-1.5B-Instruct --out quant/awq
    python quantize_gpu.py --method gptq --model Qwen/Qwen2.5-1.5B-Instruct --out quant/gptq
"""
from __future__ import annotations

import argparse


def calibration(tok, n: int = 256, seq: int = 512):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    rows = []
    for r in ds:
        if len(r["text"]) > 2000:
            rows.append({"text": r["text"][:4000]})
        if len(rows) >= n:
            break
    from datasets import Dataset
    return Dataset.from_list(rows).map(
        lambda r: tok(r["text"], truncation=True, max_length=seq), remove_columns=["text"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["int8", "awq", "gptq"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=128)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    if a.method == "int8":
        from transformers import BitsAndBytesConfig
        m = AutoModelForCausalLM.from_pretrained(
            a.model, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
        m.save_pretrained(a.out); tok.save_pretrained(a.out)
        return

    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import GPTQModifier
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
    # W4A16: 4-bit weights, 16-bit activations; keep lm_head in 16-bit (it is tied and sensitive)
    recipe = (AWQModifier(targets=["Linear"], scheme="W4A16", ignore=["lm_head"]) if a.method == "awq"
              else GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]))
    oneshot(model=model, dataset=calibration(tok), recipe=recipe, max_seq_length=512,
            num_calibration_samples=256)
    model.save_pretrained(a.out, save_compressed=True); tok.save_pretrained(a.out)
    print(f"[lab16] wrote {a.out} (servable by vLLM)")


if __name__ == "__main__":
    main()
