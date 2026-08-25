# JANUS NAS ↔ GitHub Link

This directory defines the **GitHub side** of the future JANUS NAS link.

It does not deploy to a NAS, restart containers, expose a webhook, or grant runtime authority.

## Contract

Canonical contract:

- `.janus/NAS_GIT_LINK_CONTRACT.json`
- canonical organism topology: `organism/JANUS_ORGANISM_v1.json`
- repository ecology: `organism/JANUS_REPOSITORY_ECOLOGY_v1.3.json`

Core law:

```text
SYNC != EXECUTION AUTHORITY
LOCAL_CHANGE != AUTOMATIC_MAIN_MERGE
NAS_RUNTIME_STATE != GITHUB_PUBLIC_STATE
```

## Intended flow

```text
GitHub canonical repositories
        │
        │ fetch / ff-only when clean
        ▼
future NAS working trees
        │
        ├─ local tests/runtime awareness
        └─ sanitized checkpoint branch
                 │
                 ▼
        nas/<host>/checkpoint
                 │
                 ▼
          validation + human review
```

The NAS side should preserve dirty or divergent local trees and must not force-reset them.

## Branches

Machine-originated work is expected under:

- `nas/<host>/checkpoint` — sanitized state receipts only
- `nas/<host>/integration` — proposed integration changes
- `nas/<host>/runtime` — code/runtime proposals that still require review

Direct NAS writes to `main` are outside this contract.

## Sanitized checkpoints

Schema: `checkpoint.schema.json`

Validator:

```bash
python nas_sync/validate_checkpoint.py nas_sync/examples/checkpoint.json
```

Unit tests:

```bash
python -m unittest discover -s nas_sync/tests -v
```

A public checkpoint may contain repository names, branches, commit IDs, aggregate sync states and test states. It must not contain secrets, tokens, credentials, private IPs, absolute NAS paths, Telegram metadata, environment values, databases, WAL files or raw runtime logs.

## Authentication boundary

The future NAS should authenticate outbound with either a dedicated SSH key or a fine-grained GitHub token with the smallest repository scope that satisfies the chosen workflow. No credential belongs in this repository.

The GitHub-side contract intentionally does not require inbound NAS ports or a public webhook.

## Deployment boundary

A successful Git sync is only evidence that source state is synchronized. It is not permission to:

- restart containers;
- execute changed code;
- promote generated modules;
- write into `modules_live`;
- merge to `main`;
- touch unrelated or Telegram services.

Deployment, if introduced later, is a separate explicitly authorized gate.
