# Public metadata convention

This directory defines the small machine-readable layer used to classify public `Hawkar-usls` repositories.

## Entry points

- [`registry.json`](registry.json) — resolves stable IDs to schemas/vocabularies.
- [`hawkar.project-status.v1.schema.json`](hawkar.project-status.v1.schema.json) — shared schema for per-repository `PROJECT_STATUS.json` files.
- [`hawkar.public-portfolio-index.v1.schema.json`](hawkar.public-portfolio-index.v1.schema.json) — schema for the account-wide [`../portfolio-index.json`](../portfolio-index.json).
- [`portfolio-classes.json`](portfolio-classes.json) — definitions of `portfolio_class` values.

## Interpretation rule

A valid JSON document means only that its **metadata structure** is valid.

```text
SCHEMA_VALID = STRUCTURE_VALID
SCHEMA_VALID != SCIENTIFIC_CLAIM_TRUE
CI_PASS != PEER_REVIEW
INTERNAL_VERIFICATION != WORLD_NOVELTY
```

Evidence must still be evaluated from the linked code, reports, artifacts, experiments, proofs and external review status.

## Stability

`schema_id` is the stable discriminator. Repository-specific status files may add fields beyond the common core when those fields make a claim boundary easier to audit.
