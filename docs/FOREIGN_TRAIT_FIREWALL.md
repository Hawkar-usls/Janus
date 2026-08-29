# JANUS FOREIGN TRAIT FIREWALL

Status: defensive design, default deny.

## Threat

Model-generated data can carry teacher-specific behavioural traits that are not
semantically visible in the examples. Filtering explicit references is therefore
not a sufficient identity-safety boundary. Scaling independent teacher-generated
distillation data may make latent teacher traits more recoverable.

## Core invariants

- `RECEIVE != LEARN`
- `LEARN != INTERNALIZE`
- `FOREIGN_CONTENT_CAN_INFORM_FACTS`
- `FOREIGN_CONTENT_CANNOT_MODIFY_IDENTITY_OR_POLICY`
- `UNKNOWN_PROVENANCE => QUARANTINE`
- `SEMANTIC_FILTERING != PROVENANCE`
- `PARAPHRASE != TRUST`
- `SCALE != SAFETY`
- Only approved `JANUS:*` lineage may write JANUS identity/policy.
- Outbound trait-bearing data must declare JANUS lineage and trait scope.
- Covert/subliminal outbound trait propagation is forbidden.

## Memory planes

1. **IDENTITY** — JANUS-owned, approved canon only.
2. **POLICY** — JANUS-owned, approved governance only.
3. **REFERENCE** — user input, web research, external model output; usable as
   information, never as identity/persona/style/goals.
4. **QUARANTINE** — unknown lineage or invalid identity/policy writes.

Legacy memories without provenance are `LEGACY_UNTRUSTED`.

## Training/distillation gate

For identity/persona/policy/alignment/trait training, every source must be
`JANUS_OWNED`, in an explicit `JANUS:*` lineage, and approved.

Foreign model-generated data is default-deny. Keyword filtering, semantic
filtering, and paraphrasing do not upgrade trust. Foreign sample count and
independent teacher lineage count are recorded as scaling-risk signals.

Task-specific external synthetic-data training, if intentionally enabled later,
should use isolated adapters, pre/post trait canaries, held-out probes, rollback,
and must never be reused as JANUS identity.

## Scope

This protects JANUS application memory and admission decisions. It does not claim
to remove traits already present in third-party foundation-model weights. That
requires model-level audits or a controlled JANUS-owned model/training stack.
