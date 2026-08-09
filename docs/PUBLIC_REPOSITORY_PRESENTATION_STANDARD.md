# Public Repository Presentation Standard

> **MIT-inspired academic minimalism — no affiliation with MIT is implied.**

This document defines the public presentation standard for repositories in the `Hawkar-usls` account. It governs presentation and claim boundaries only; it does not alter code ownership, scientific status, licenses, or upstream attribution.

## Objective

A new reader should be able to answer five questions in under one minute:

1. What is this repository?
2. What is its current maturity/status?
3. What is actually implemented or established?
4. What is explicitly not claimed or not yet established?
5. Where should a reviewer inspect evidence, code, or machine-readable metadata?

## Required front-door structure

Preferred README order:

```text
Title
One-line technical subtitle
Status / maturity
Abstract
Implemented scope
Current boundary / non-claims
Reviewer or run path
Machine-readable status
License / attribution
```

Long historical narratives, speculative terminology, screenshots, roadmaps, and archived experiments should follow the front-door material or live in dedicated documents.

## Maturity vocabulary

Use one primary maturity label:

- `FLAGSHIP` — externally reviewable primary work.
- `ACTIVE_PROTOTYPE` — implemented prototype under continued development.
- `WORK_IN_PROGRESS` — incomplete; interfaces, behavior, or validation may change.
- `LEGACY` — preserved for lineage; not a current primary implementation.
- `ARCHIVE` — historical record, not a current claim surface.
- `UPSTREAM_DERIVED` — base project authorship belongs to upstream authors; local work must be evaluated as a diff.

Do not replace an uncertain status with promotional language.

## Evidence vocabulary

Preferred machine-readable terms:

```text
IMPLEMENTED
TESTED_IN_STATED_SCOPE
ESTABLISHED_IN_STATED_SCOPE
NOT_ESTABLISHED
NOT_CLAIMED
NOT_PERFORMED
INCOMPLETE
SUPERSEDED
ARCHIVAL
```

Avoid unsupported superlatives such as `revolutionary`, `breakthrough`, `world first`, `production ready`, `clinically ready`, or `100%` unless a narrowly defined object and evidence source justify the exact statement.

## Claim discipline

Permanent rules:

```text
CI_PASS != PEER_REVIEW
SCHEMA_VALID != SCIENTIFICALLY_TRUE
INTERNAL_REPLAY != INDEPENDENT_REPLICATION
FINITE_TEST != ASYMPTOTIC_THEOREM
PROJECT_NAME != PHYSICAL_CLAIM
METAPHOR != EVIDENCE
ROADMAP != IMPLEMENTATION
UPSTREAM_COPY != LOCAL_AUTHORSHIP
```

Negative, null, obstruction, and fail-closed results should remain visible.

## Work-in-progress repositories

A WIP repository should say so on the first screen. Recommended wording:

> **Status: Work in Progress.** This repository is incomplete. APIs, behavior, documentation, and validation may change. It should not be treated as a finished product or established result.

A WIP repository may still be valuable. The status is a maturity statement, not a dismissal of the work.

## Legacy and archive repositories

Legacy code should direct readers to its current successor when one exists. Archive repositories should preserve lineage without presenting historical vocabulary as current evidence.

## Upstream-derived repositories

The first screen must identify the upstream project and state that base-code/research authorship is not claimed locally. Local contribution should be evaluated by an explicit diff against a corresponding upstream revision.

## Machine-readable status

Where practical, repositories expose `PROJECT_STATUS.json` using the shared `hawkar.project-status.v1` metadata convention. Account-wide schemas and vocabularies are registered at:

- `https://github.com/Hawkar-usls/Janus/blob/main/schemas/registry.json`
- `https://github.com/Hawkar-usls/Janus/blob/main/schemas/portfolio-classes.json`

Schema validity establishes metadata structure only.

## Licensing

Presentation style does not change licensing.

- Existing Apache/MIT/GPL/other licenses remain in force.
- Upstream-derived repositories retain upstream and third-party licensing obligations.
- Do not add an MIT license merely to make a repository look "MIT-style".

## Visual style

Prefer:

- short headings;
- restrained badges;
- one technical sentence per paragraph where possible;
- neutral language;
- links to primary evidence rather than decorative claims;
- no institutional logos or wording that could imply affiliation or endorsement.

The desired impression is a readable laboratory notebook: precise, inspectable, modest about uncertainty, and easy to audit.
