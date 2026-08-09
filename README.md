<div align="center">

# Janus
### Local / cloud LLM gateway experiment

`self-hosted` · `provider routing` · `local fallback` · `explicit claim boundaries`

</div>

Janus is a small Python gateway for routing chat-style requests between configured cloud providers and local/self-hosted model endpoints.

It is a **supporting engineering project**, not one of the account's flagship research results.

Machine-readable status: [`PROJECT_STATUS.json`](PROJECT_STATUS.json)

## Scope

The repository explores:

- a single chat-completion-style entry point;
- provider selection and fallback;
- local-model integration, including Ollama-style endpoints;
- shared state for small companion applications;
- operation on modest self-hosted hardware.

Provider compatibility, latency, uptime and fallback behavior depend on the deployed configuration and external APIs.

## Claim boundary

```text
PROJECT_CLASS = ENGINEERING_PROTOTYPE
PRODUCTION_UPTIME_GUARANTEE = NOT_CLAIMED
SECURITY_CERTIFICATION = NOT_ESTABLISHED
BENCHMARKED_PROVIDER_SUPERIORITY = NOT_CLAIMED
MEDICAL_DIAGNOSIS_AUTHORITY = FALSE
FINANCIAL_OR_GAMBLING_OUTCOME_GUARANTEE = FALSE
HISTORICAL_PRIORITY_OVER_OTHER_GATEWAYS = NOT_CLAIMED
```

Applications connected to this gateway may include games, creative interfaces, health-information prototypes or other experiments. Their presence does not make the gateway a medical device, financial service, safety authority or independently audited production platform.

## Security

Keep API keys, Wi-Fi credentials, private endpoints and local deployment data outside committed source files. Any public deployment should receive its own authentication, authorization, rate-limit, logging, dependency and threat-model review.

## Portfolio navigation

- **Research:** [Janus-Fundamentum](https://github.com/Hawkar-usls/Janus-Fundamentum), [AIFC](https://github.com/Hawkar-usls/AIFC), [janus-io-public](https://github.com/Hawkar-usls/janus-io-public)
- **Embedded engineering:** [janus-distributed-ai-swarm](https://github.com/Hawkar-usls/janus-distributed-ai-swarm)
- **Creative technology:** [Janus_Genesis](https://github.com/Hawkar-usls/Janus_Genesis)
- **Machine-readable account index:** [`portfolio-index.json`](portfolio-index.json)
- **Profile README source:** [`PROFILE_README_SOURCE.md`](PROFILE_README_SOURCE.md)

## Status

This repository is retained as a compact gateway/integration project. Scientific claims should be evaluated in the dedicated research repositories above, not inferred from this codebase.

---

Hawkar / Oleksandr Ahapov · Ukraine
