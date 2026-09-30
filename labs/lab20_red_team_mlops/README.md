# Lab 20: Red-Team and Productionize

**Chapter:** 20, Safety, Security, Domain Specialization, and Production MLOps
**Goal:** Red-team your specialized SLM with automated attack sets (prompt
injection and jailbreak benchmarks), add a guardrail classifier and tool
policy, measure attack success rate before and after, then deploy behind an
API with logging and monitoring. Deliver a security report plus a production
deployment.

## Files

| File | Purpose |
|---|---|
| `attack_suite.py` | Direct jailbreaks (public benchmark, loaded by name), direct and indirect prompt injection; objective detectors (canary leak, unauthorised tool call, guard verdict); ASR per family |
| `guardrail.py` | Injection heuristic, Llama Guard 3 1B wrapper, document spotlighting, tool allowlist with argument constraints; `--selftest` |
| `api_server.py` | FastAPI gateway: auth, input/output screening, tool policy, Prometheus metrics, PII-redacted structured logs, model-version headers |
| `drift_monitor.py` | PSI drift on content-free features of live traffic |

## Flow

1. **Serve the unguarded model.** `vllm serve <your Lab 14/15 model> --port 8000 --served-model-name soc-slm`.
2. **Baseline ASR.** `python attack_suite.py --url http://localhost:8000/v1 --model soc-slm --out results/asr_before.json`.
3. **Add guards.** `python guardrail.py --selftest`, then start the gateway:
   `uvicorn api_server:app --port 9000`.
4. **Guarded ASR.** `python attack_suite.py --url http://localhost:9000/v1 --model soc-slm --out results/asr_after.json`
   (send `Authorization: Bearer dev-key`; adapt the client or set `API_KEYS`).
5. **Over-refusal check.** Run the Lab 19 `over_refusal` and `log_classification` categories
   through the gateway; guards must not break benign traffic.
6. **Observe.** Scrape `/metrics` with Prometheus; build a dashboard for p95 latency,
   blocks by stage, escalations, and tokens/s. Run `drift_monitor.py` daily on logs.
7. **Report.** ASR before/after per family, over-refusal delta, latency overhead of the
   guards, residual risks, and the incident runbook.

## Deliverable

Security report (tables above plus residual-risk analysis) and the running,
monitored deployment.

## Safety note

Attack prompts come from public research benchmarks and are never stored in
this repository. Run red-team jobs only against systems you own, in an
isolated environment, and keep their outputs access-controlled.
