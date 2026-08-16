# JANUS First Portal

The First Portal is the smallest public-facing routing contract in the JANUS repository.

It intentionally does less than the historical `janus_core.py` monolith.

```text
OPEN DOOR
-> CHOOSE DESTINATION
-> RECEIVE ROUTE RECEIPT
-> DESTINATION APPLIES ITS OWN GATES
```

The reference Portal is not a proxy server, not a truth engine and not an authority broker.

```text
PORTAL_ROUTE != WORLD_EFFECT
PORTAL_DISCOVERY != AUTHORITY
DESTINATION_VISIBILITY != DESTINATION_PERMISSION
ROUTE_RECEIPT != PROVIDER_REALIZATION
ANYWHERE != ARBITRARY_ENDPOINT
```

## Why it is a separate layer

Historically, `janus_core.py` already behaves like a gateway: it routes model calls and exposes multiple `/api/...` application endpoints. That file also mixes several unrelated legacy trust domains in one process, including memory, RPG/narrative state, health-information experiments, slot/casino/payment handlers and provider routing.

That historical code remains useful lineage, but it is deliberately **not** the constitutional Portal boundary.

The new Portal stores only typed destination metadata and produces deterministic route receipts. It performs no network call and authorizes no external effect.

## Destination manifest

[`../portal/manifest.json`](../portal/manifest.json) exposes bounded symbolic destinations.

General doors:

- `DEMIHEAD` — Guardian Mesh evidence/routing head;
- `GENESIS` — Genesis constitutional/creative runtime;
- `HRAIN` — human-visible graph interface;
- `INAIHR` — semantic decomposition interface;
- `META_REGISTRY` — provenance and policy memory;
- `LOCAL_MODEL` — symbolic deployment-resolved local model route.

After the corresponding DemiHead contracts passed exact-head software CI, the Portal catalog gained four more explicit subdoors:

- `DEMIHEAD_CORRECTIONS` — correction-propagation route;
- `DEMIHEAD_LANGUAGE` — protected `uk/ru/en` semantic-invariance route;
- `DEMIHEAD_REVIEW` — reviewer readiness and disagreement-preservation route;
- `DEMIHEAD_APPEAL` — append-only Human Appeal route.

These are route-discovery entries only:

```text
CORRECTION_ROUTE != CORRECTION_APPLICATION
LANGUAGE_ROUTE != TRANSLATION_TRUTH
REVIEW_ROUTE != REVIEW_CONSENSUS
APPEAL_ROUTE != APPEAL_SUBMISSION
```

The catalog is extensible. “Anywhere” means adding a reviewed typed destination to the manifest, not forwarding arbitrary URLs.

Allowed `route_ref` forms are symbolic only:

```text
repo:owner/name
service:SYMBOLIC_SERVICE
catalog:SYMBOLIC_CATALOG
```

A raw `http://` or `https://` route is rejected by the reference validator.

## Usage

List doors:

```bash
python portal/portal.py --list
```

Inspect a destination:

```bash
python portal/portal.py --inspect DEMIHEAD
```

Resolve the default door:

```bash
python portal/portal.py
```

Resolve a protected subdoor:

```bash
python portal/portal.py --resolve DEMIHEAD_CORRECTIONS
python portal/portal.py --resolve DEMIHEAD_LANGUAGE --language uk
python portal/portal.py --resolve DEMIHEAD_REVIEW
python portal/portal.py --resolve DEMIHEAD_APPEAL
```

Resolving `DEMIHEAD_APPEAL` does not file an appeal. Resolving `DEMIHEAD_REVIEW` does not create consensus.

Decline routing:

```bash
python portal/portal.py --decline
```

Self-test:

```bash
python portal/portal.py --self-test
python -m unittest discover -s tests -v
```

## Language boundary

The Portal may carry a requested presentation language (`uk`, `ru`, `en`) as route metadata. It does not translate evidence and it does not alter evidence status.

The stronger invariant belongs downstream in DemiHead:

```text
LANGUAGE_CHANGE != EVIDENCE_STATUS_CHANGE
LANGUAGE_CHANGE != UNCERTAINTY_CHANGE
LANGUAGE_CHANGE != USER_RIGHTS_CHANGE
```

## Correction boundary

The Portal may route a user toward a correction-aware destination. It does not rewrite source lineage itself.

Known correction propagation belongs to DemiHead or another destination that explicitly owns the corresponding evidence contract.

```text
PORTAL_CARRIES_ROUTE
DEMIHEAD_APPLIES_CORRECTION_GATES
```

## Review and appeal boundaries

The Portal can expose the road toward review or appeal, but does not manufacture either operation:

```text
PORTAL_ROUTE != REVIEW_CONSENSUS
PORTAL_ROUTE != APPEAL_SUBMISSION
```

The review gate owns package binding and `DISAGREEMENT` semantics. The appeal gate owns exact decision binding and append-only appeal history.

## Security boundary

The reference Portal has no listener, no provider credentials, no arbitrary URL forwarding and no external side effects.

A future HTTP wrapper would require a separate review covering authentication, authorization, rate limits, CORS, request-size limits, logging/retention, SSRF, destination allowlisting and secret handling.

The legacy `janus_core.py` must not silently become that wrapper merely because it already starts an aiohttp server.

## Claim ceiling

The current implementation establishes deterministic typed route discovery only. It does not establish production security, public internet readiness, destination uptime, provider availability, real review/appeal operations, or permission to execute effects at any destination.
