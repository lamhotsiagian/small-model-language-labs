# Capstone: A 1B Domain SLM Under 1 GB

Build an end-to-end domain SLM, for example a 1B cybersecurity analyst
assistant, that beats a general 7B model on your domain evaluation while
fitting in under 1 GB of memory.

`pipeline.yaml` is the plan: every stage names the lab that implements it and
the gate it must pass before the next stage starts.

| Stage | Lab | Output | Gate |
|---|---|---|---|
| Curate domain data | 6 | decontaminated domain + general mix | >= 200M tokens, 0 contamination hits |
| Continued pretraining | 7 | domain-adapted 1B base | general benchmarks drop <= 1 point |
| Distill + SFT | 8, 11 | instruct SOC model | IFEval >= 0.55 |
| Preference | 12 | SimPO-aligned model | length-controlled win vs SFT >= 55% |
| Tool calling | 15 | tool-routing model | call acc and irrelevance acc >= 0.9 |
| Quantize | 16 | Q5_K_M / Q4_K_M GGUF | <= 1 point domain drop, <= 1 GB |
| Deploy | 17, 18 | server + on-device builds | p95 latency targets |
| Evaluate | 19 | leaderboard vs general 7B | paired bootstrap P(win) >= 0.95 |
| Red-team | 20 | guarded API + report | indirect-injection ASR <= 5%, benign refusal <= 5% |

## Flow

1. Freeze the Lab 19 evaluation contract **first**, including the 7B baseline score.
2. Run stages in order; record each gate result in `results/gates.json`.
3. When a gate fails, fix the stage that owns it; do not relax the gate.
4. Write the final model card with the full lineage (data hashes, configs, gate results).
