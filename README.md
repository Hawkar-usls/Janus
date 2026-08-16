<div align="center">

# Janus
### First Portal · supporting LLM gateway · public metadata hub

![Status](https://img.shields.io/badge/status-active%20prototype-2f81f7)
![Class](https://img.shields.io/badge/class-supporting%20engineering-6e7681)
![Portal](https://img.shields.io/badge/first%20portal-reference%20door-8957e5)

</div>

## Status

**Active Prototype / Supporting Infrastructure.** Janus is the account's **First Portal**: a small typed door for discovering bounded JANUS destinations, plus the repository's historical local/cloud LLM gateway and public metadata hub. It is not a flagship scientific result or an independently audited production service.

## First Portal

The reference Portal intentionally stays smaller than the historical `janus_core.py` monolith:

```text
OPEN DOOR
-> CHOOSE DESTINATION
-> RECEIVE ROUTE RECEIPT
-> DESTINATION APPLIES ITS OWN GATES
```

```text
PORTAL_ROUTE != WORLD_EFFECT
PORTAL_DISCOVERY != AUTHORITY
DESTINATION_VISIBILITY != DESTINATION_PERMISSION
ANYWHERE != ARBITRARY_ENDPOINT
```

The Portal is implemented in [`portal/`](portal/) as a deterministic typed destination manifest and route resolver. It performs no network call, stores no provider credentials and grants no destination effect authority.

General destinations include `DEMIHEAD`, `GENESIS`, `HRAIN`, `INAIHR`, `META_REGISTRY` and a symbolic `LOCAL_MODEL` route.

The Portal also exposes four short typed doors to exact-head-tested DemiHead reference contracts:

```text
DEMIHEAD_CORRECTIONS  -> explicit correction lineage
DEMIHEAD_LANGUAGE     -> protected uk/ru/en semantics
DEMIHEAD_REVIEW       -> reviewer readiness / DISAGREEMENT
DEMIHEAD_APPEAL       -> append-only Human Appeal
```

These are **doors, not effects**:

```text
CORRECTION_ROUTE != CORRECTION_APPLICATION
LANGUAGE_ROUTE != TRANSLATION_TRUTH
REVIEW_ROUTE != REVIEW_CONSENSUS
APPEAL_ROUTE != APPEAL_SUBMISSION
```

The catalog is extensible: “anywhere” means adding a reviewed typed destination, not proxying an arbitrary URL.

```bash
python portal/portal.py --list
python portal/portal.py --resolve DEMIHEAD --language uk
python portal/portal.py --resolve DEMIHEAD_CORRECTIONS
python portal/portal.py --resolve DEMIHEAD_LANGUAGE --language ru
python portal/portal.py --resolve DEMIHEAD_REVIEW
python portal/portal.py --resolve DEMIHEAD_APPEAL
python portal/portal.py --self-test
```

See [`docs/FIRST_PORTAL.md`](docs/FIRST_PORTAL.md).

## Historical gateway scope

The historical gateway routes chat-style requests between configured cloud providers and local/self-hosted model endpoints and also contains several legacy application integrations. It remains useful lineage, but it is **not** treated as the new constitutional Portal boundary.

Implemented historical scope includes:

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
FIRST_PORTAL = REFERENCE_ROUTE_DISCOVERY
FIRST_PORTAL_NETWORK_EFFECT = NONE
FIRST_PORTAL_ARBITRARY_URL_PROXY = FALSE
FIRST_PORTAL_AUTHORITY_DELTA = 0
PRODUCTION_UPTIME_GUARANTEE = NOT_CLAIMED
SECURITY_CERTIFICATION = NOT_ESTABLISHED
BENCHMARKED_PROVIDER_SUPERIORITY = NOT_CLAIMED
MEDICAL_DIAGNOSIS_AUTHORITY = FALSE
FINANCIAL_OR_GAMBLING_OUTCOME_GUARANTEE = FALSE
```

Applications connected through this gateway or discovered through the Portal keep their own claim boundaries. Their presence does not turn the Portal into a medical, financial, safety, scientific, civic, review, appeal, or effect authority.

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

Keep API keys, Wi-Fi credentials, private endpoints, and deployment secrets outside committed source. Any public deployment requires its own authentication, authorization, rate-limit, dependency, logging, CORS and threat-model review.

The First Portal reference implementation deliberately has no listener and no arbitrary URL forwarding. The historical `janus_core.py` aiohttp server must not be treated as a production Portal wrapper without a separate security review.

Metadata/schema validity describes structure and declared boundaries; it does not establish scientific truth, novelty, replication, or peer review.

Presentation follows the account's academic/minimalist standard. No affiliation with MIT is implied.

---

Hawkar / Oleksandr Ahapov · Ukraine
