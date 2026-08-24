# JANUS Federated Organism v1

JANUS is treated as a **federated organism**, not as one repository with ambient authority over every other repository.

The canonical machine-readable map is [`organism/JANUS_ORGANISM_v1.json`](../organism/JANUS_ORGANISM_v1.json).

## Organ map

```text
                            ┌────────────────────────┐
                            │   Janus / MCP Gateway  │
                            │ nervous system / route │
                            └───────────┬────────────┘
                                        │ typed routes
      ┌─────────────────────────────────┼──────────────────────────────────┐
      │                                 │                                  │
      v                                 v                                  v
janus-meta-registry             Janus-Demiurge                         Demi_Head
memory / provenance             spiral orchestration                  guardian cortex
      ^                                 │                                  │
      │                                 v                                  v
      │                     janus-distributed-ai-swarm              Janus-Fundamentum
      │                     sensorimotor / embedded                 proof/falsification
      │                                 ^                                  ^
      │                                 │                                  │
      │                      Janus-Cosmos / TOPA  <---- janus-lapis -------┘
      │                      observation / anomaly      hypothesis ranking
      │                                                  ^
      └──────────────────────────────────────────────────│
                                                         │
                                                  aura-oracle-tg
                                            symbolic imagination only
```

## Roles

- **Janus** — nervous-system gateway, MCP surface and account metadata hub. Routing does not create scientific authority.
- **janus-meta-registry** — long-term memory, provenance, corrections, supersession and historical lineage.
- **Janus-Fundamentum** — scoped formal proof/falsification spine. A formal result has authority only in its stated scope.
- **Demi_Head** — guardian/review cortex: source-root collapse, bounded review, disagreement preservation and HOLD.
- **Janus-Demiurge** — spiral agent control plane: missions, Scout reconnaissance, lessons and constraints. Dispatch is not authority.
- **janus-distributed-ai-swarm** — sensorimotor/embedded mesh: telemetry, heartbeat, device state and protocol receipts.
- **janus-lapis** — hypothesis-ranking/metabolism surface. Ranking is not validation.
- **aura-oracle-tg** — symbolic imagination/creative advisor. It has zero empirical evidence authority.
- **TOPA** — anomaly triage and falsification laboratory. An unresolved anomaly is not an extraordinary conclusion.
- **Janus-Cosmos** — observational/domain research organ for Cosmos experiments and signal candidates.

## Constitutional boundary

```text
REPOSITORY_MEMBERSHIP != TRUTH
MODEL_OUTPUT != EVIDENCE
SYMBOLIC_OUTPUT != EMPIRICAL_EVIDENCE
RANKING != VALIDATION
DISPATCH != AUTHORITY
MULTIPLE_AGENTS != INDEPENDENT_SOURCES
HASH_OR_SIGNATURE != TRUTH_OF_CONTENT
NEGATIVE_RESULTS_AND_DISAGREEMENT_ARE_PRESERVED
WRITE_BACK_DEFAULT = DENY
```

Each organ preserves its own maturity and claim boundary. Membership in JANUS does not grant another repository's privileges, evidence class, production status, or external-effect authority.

## Interface contract

Every attached repository may expose `.janus/JANUS_ORGANISM_LINK.json` containing:

- canonical organism ID and manifest location;
- the repository's organ key and role;
- accepted input/output classes;
- evidence-authority boundary;
- `authority_delta = 0` for transport;
- cross-repository write default `DENY`.

The link is **bidirectional discovery metadata**, not an execution credential.

## Data flow

A normal research path can look like:

```text
Aura idea / human idea
  -> Lapis candidate ranking
  -> TOPA competing hypotheses / falsifiers
  -> Cosmos or Swarm observations where applicable
  -> Demi_Head source-root / disagreement review
  -> Fundamentum formal attack where the claim is formalizable
  -> Meta Registry receipt / correction / negative result
  -> Janus gateway exposes the resulting state
  -> Demiurge schedules another spiral pass if a real open gate remains
```

No stage silently promotes the previous stage's output.

## Writes

The MCP gateway remains read-first. Its existing write tools write only to the dedicated `Hawkar-usls/Janus:mcp-checkpoints` branch when explicitly enabled. Organism membership does not authorize writes to member repositories.

## Expansion

This v1 manifest is the seed core, not a claim that every historical repository has already been classified. Additional JANUS repositories can join the organism without changing the constitutional rules by adding a typed member link and then being admitted to a later manifest version.
