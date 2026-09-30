# small-model-language-labs

Companion code for **Small Language Model System Design** (AI Engineering Insider).

Twenty labs and one capstone that take you from "what is a small language model"
to a quantized, evaluated, red-teamed SLM running on a laptop, a phone, and a
server. Each lab maps one-to-one to a chapter of the ebook and has its own
`README.md` that describes the flow step by step. The ebook's "Lab" section in
each chapter follows that README exactly and quotes the scripts here.

- Book preview: https://drive.google.com/file/d/1qW2fWd4zwwhhbYK4hqcgnixJHMq4SUCo/view
- Book link: https://shop.beacons.ai/aiengineeringinsider/35896e19-1a63-4df7-8b80-4902f87c87ca

<img width="1241" height="1754" alt="preview-small-model-language-1-14_page-0001" src="https://github.com/user-attachments/assets/28b765c2-1df5-4d38-810f-d27befd171b4" />


## Layout

```
small-model-language-labs/
├── slmlab/                 shared library used by every lab
│   ├── model.py            pure-PyTorch Llama-style SLM (GQA, RoPE, SwiGLU, RMSNorm, tied embeddings)
│   ├── budget.py           parameter, FLOPs, KV-cache, and device-memory calculators + model zoo configs
│   ├── data.py             corpora, byte tokenizer, sequence packing
│   ├── train.py            AdamW + WSD/cosine loop with spike rollback and MFU accounting
│   └── hfutils.py          Hugging Face loading, chat generation with TTFT/TPOT timing, log-probs
├── labs/
│   ├── lab01_model_zoo/            Ch 1   benchmark five SLMs, build a decision matrix
│   ├── lab02_transformer_from_scratch/ Ch 2  10M decoder + FLOPs/memory calculator
│   ├── lab03_tokenizer_tradeoffs/  Ch 3   BPE at 16K/32K/64K, fertility, embedding share
│   ├── lab04_architecture_ablations/ Ch 4 deep-thin vs wide-shallow, MHA vs GQA, tied vs untied
│   ├── lab05_scaling_laws/         Ch 5   fit L(N, D) on tiny models, predict a bigger one
│   ├── lab06_data_curation/        Ch 6   dedup, quality classifier, synthetic textbooks
│   ├── lab07_pretrain_150m/        Ch 7   pretrain a Llama-style 150M model
│   ├── lab08_distillation/         Ch 8   SFT-on-teacher vs logit KD vs on-policy GKD
│   ├── lab09_pruning/              Ch 9   width + depth pruning, then distillation recovery
│   ├── lab10_long_context/         Ch 10  YaRN extension 8K -> 32K, RULER-style probes
│   ├── lab11_sft/                  Ch 11  instruction-tune a base model with TRL
│   ├── lab12_preference/           Ch 12  DPO vs ORPO vs SimPO on judge-labelled pairs
│   ├── lab13_grpo/                 Ch 13  GRPO with verifiable math rewards
│   ├── lab14_peft/                 Ch 14  full FT vs LoRA vs QLoRA vs DoRA, multi-LoRA serving
│   ├── lab15_function_calling/     Ch 15  tool-call fine-tune + grammar-constrained decoding + router agent
│   ├── lab16_quantization/         Ch 16  FP16 -> INT8 -> AWQ/GPTQ -> GGUF k-quant ladder
│   ├── lab17_serving/              Ch 17  vLLM vs SGLang vs llama.cpp load test, speculative decoding
│   ├── lab18_on_device/            Ch 18  GGUF + MLC/ExecuTorch on a laptop or phone
│   ├── lab19_evaluation/           Ch 19  300-item domain eval, judge rubric, leaderboard
│   ├── lab20_red_team_mlops/       Ch 20  attack suite, guardrail, monitored API
│   └── bonus_interview/            Ch 21  estimates for the 20 system design cases
└── capstone/                       end-to-end 1B domain SLM under 1 GB
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional, hardware specific: bitsandbytes, vllm, llama-cpp-python, mlx-lm ...
export PYTHONPATH=$PWD      # so `import slmlab` works from any lab folder
```

Quick sanity check (CPU, under a minute):

```bash
python -m slmlab.budget                                  # zoo parameter + KV table
python labs/lab02_transformer_from_scratch/train_tiny.py --steps 200 --preset smoke
```

## Compute tiers

| Tier | Hardware | Labs that fit |
|---|---|---|
| Laptop | CPU / Apple Silicon, 16 GB | 1 (small models), 2, 3, 5 (reduced), 16 (GGUF), 18 |
| Single GPU | 24 GB (RTX 4090 / L4 / A10) | 1, 4, 6, 8 (0.5B student), 11, 12, 13 (1.5B), 14 (QLoRA), 15, 16, 17, 19, 20 |
| Rented node | 1-8 x 80 GB (A100 / H100) | 7, 9, 10, 13 (full), 14 (full FT 3B), 17 (8B target) |

Every script exposes `--preset smoke` (runs in minutes on CPU to prove the
pipeline) and a full preset that reproduces the chapter's experiment.

## Conventions

- Each lab writes results to `labs/<lab>/results/` as JSON plus a Markdown report.
- Model identifiers are pinned in each lab's `config.py` or CLI defaults; pin a
  `--revision` commit SHA before you quote any number.
- Numbers in the ebook marked *computed* come from closed-form calculators in
  this repo; numbers marked *illustrative* show the expected shape of a result
  and must be reproduced on your hardware before you rely on them.
# small-model-language-labs
