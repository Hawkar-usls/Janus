<div align="center">

# Janus
### Supporting LLM gateway and public metadata hub

![Status](https://img.shields.io/badge/status-active%20prototype-2f81f7)
![Class](https://img.shields.io/badge/class-supporting%20engineering-6e7681)

</div>

## Status

**Active Prototype / Supporting Infrastructure.** Janus is a small local/cloud LLM gateway and the account's public metadata hub. It is not a flagship scientific result or an independently audited production service.

## Abstract

The gateway routes chat-style requests between configured cloud providers and local/self-hosted model endpoints. The repository also hosts the account-wide portfolio index, maturity/visibility policy, metadata schemas, and canonical profile README source.

## Implemented scope

- a chat-completion-style entry point;
- provider selection and fallback experiments;
- local-model integration, including Ollama-style endpoints;
- shared state for small companion applications;
- public portfolio/status metadata.

Provider compatibility, latency, uptime, and fallback behavior depend on the deployed configuration and external APIs.

## Boundary

```text
MATURITY = ACTIVE_PROTOTYPE
PROJECT_CLASS = SUPPORTING_ENGINEERING
FLAGSHIP_RESEARCH = FALSE
PRODUCTION_UPTIME_GUARANTEE = NOT_CLAIMED
SECURITY_CERTIFICATION = NOT_ESTABLISHED
BENCHMARKED_PROVIDER_SUPERIORITY = NOT_CLAIMED
MEDICAL_DIAGNOSIS_AUTHORITY = FALSE
FINANCIAL_OR_GAMBLING_OUTCOME_GUARANTEE = FALSE
```

Applications connected through this gateway keep their own claim boundaries. Their presence does not turn the gateway into a medical, financial, safety, or scientific authority.

## Public metadata

- [`portfolio-index.json`](portfolio-index.json) — project classification and claim boundaries.
- [`portfolio-visibility.json`](portfolio-visibility.json) — Featured / WIP / Legacy / Archive / Upstream recommendations.
- [`public-metadata-coverage.json`](public-metadata-coverage.json) — metadata coverage audit.
- [`schemas/registry.json`](schemas/registry.json) — `schema_id` → JSON Schema / vocabulary mapping.
- [`docs/PUBLIC_REPOSITORY_PRESENTATION_STANDARD.md`](docs/PUBLIC_REPOSITORY_PRESENTATION_STANDARD.md) — public README/claim standard.
- [`PROFILE_README_SOURCE.md`](PROFILE_README_SOURCE.md) — canonical profile front page source.

## Primary portfolio

- [Janus-Fundamentum](https://github.com/Hawkar-usls/Janus-Fundamentum)
- [AIFC](https://github.com/Hawkar-usls/AIFC)
- [janus-io-public](https://github.com/Hawkar-usls/janus-io-public)
- [janus-distributed-ai-swarm](https://github.com/Hawkar-usls/janus-distributed-ai-swarm)
- [Janus_Genesis](https://github.com/Hawkar-usls/Janus_Genesis) — creative technology

## Security

Keep API keys, Wi-Fi credentials, private endpoints, and deployment secrets outside committed source. Any public deployment requires its own authentication, authorization, rate-limit, dependency, logging, and threat-model review.

Metadata/schema validity describes structure and declared boundaries; it does not establish scientific truth, novelty, replication, or peer review.

Presentation follows the account's academic/minimalist standard. No affiliation with MIT is implied.

---

Hawkar / Oleksandr Ahapov · Ukraine
