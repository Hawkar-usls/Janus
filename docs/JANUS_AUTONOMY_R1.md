# JANUS R1: observed map and bounded growth proposals

This stage gives JANUS a repeatable public repository observation cycle and a
reviewable growth protocol. It does not activate an autonomous source writer,
fork executor, model trainer, organ runtime or production promoter.

## What “the map” means

The Meta Registry chat index is a navigation aid. The historical working-map
v1.2 preserves anatomy, habitat, Nexus and epistemic/proof graphs. Current
membership comes from `organism/JANUS_ORGANISM_v1.json`, ecology v1.3 and the
reviewed ecology v1.4/Simptomat admission and route-refinement deltas. These
historical records are preserved; the observer builds a new derived projection.
The existing gateway's older allowlist is recorded as drift, without granting
additional runtime access.

`autonomy/inventory.py` reads public metadata from the fixed Hawkar-usls owner.
It records each repository's real default branch, immutable head, member-link
Git blob/digest, the latest Actions run and all available head check-runs. It
does not substitute an old successful run when the latest run is pending or
failed. Missing, stale, inaccessible or paginated-away evidence stays UNKNOWN.
Discovered repositories without a reviewed classification stay CATALOG_ONLY.
Private nodes are opaque slots; their repository names and content are excluded.

The collector accepts only five endpoint shapes on api.github.com, only GET,
with no redirects. Limits are 100 public repositories, 300 requests by default
(hard ceiling 500), 2 MB per response and a 240 second collection deadline.
It reads JSON metadata, never runs discovered source code. Authentication is
explicit and is used only on that host; a read-only workflow token can still
be unable to observe checks in another repository. Partial results are saved.

## Independent observation gates

`autonomy/health.py` does not import the collector. It independently reconstructs
membership, ecology classifications, flows, private slots and refined route
gates from a separately checked-out base map. The expected source commit is
supplied outside the report. It recomputes hashes and rejects incomplete,
future-dated, stale, reclassified or authority-escalating metadata. For canonical
public organs it requires a safe matching member-link, current-head latest
Actions success and fresh successful check-runs (default maximum age 24 hours).

Its gates remain distinct:

| Gate | Meaning |
| --- | --- |
| Integrity | Report matches the pinned map and its declared observation data |
| Observation | Required fresh GitHub metadata is available and successful |
| Operational | NOT_ASSESSED: needs service-specific, independent probes |
| Independent admission | HOLD: external review and a separately trusted executor are absent |

A hash detects changes relative to an anchor; it does not authenticate an
unanchored report or prove its content true. This candidate verifier itself
requires review. A separate CI job provides process separation from the
collector, not independent authorship, scientific replication or immunity to a
malicious repository administrator. Protected required checks and an externally
anchored verifier are operator setup work, not effects performed by this PR.

## Dependencies: catalog → quarantine → review → fork proposal

`autonomy/admission.py` is a pure, bounded data state machine. A candidate binds
its canonical GitHub source, default branch assertion, 40-character commit,
tree, SPDX license, license Git blob/SHA256 and a bounded byte-bundle manifest.
Quarantine hashes supplied bytes, checks license text hints, secret patterns,
paths and install/workflow/binary hazards without importing or executing them.
The static scan covers supplied bytes only, not complete history or the entire
upstream tree. License text hints are not legal approval.

Review must refer to that exact candidate hash and a different asserted reviewer.
Because strings cannot authenticate a human, the state is explicitly
REVIEW_ASSERTED_UNVERIFIED. The final state is FORK_PROPOSED, execution_allowed
false. There is no network writer or real fork API. A future separately scoped
operator must authenticate review, re-fetch the exact source and default branch,
inspect all tree modes/submodules/history, independently scan secrets, verify
license applicability, disable foreign Actions/hooks and verify the resulting
fork head. A changed upstream or evidence hash restarts review.

The default policy permits 16 catalog candidates, four lifetime proposals, one
fork proposal, depth one and zero new repositories. Bundles are limited to
128 files/1 MiB; the ledger to 128 events. Hard ceilings prevent a policy file
from arbitrarily expanding these budgets. Rollback/revocation never refunds
consumed budgets. The emergency kill switch blocks forward work and allows
revocation. CAS expected-head, idempotency keys and full replay from empty
genesis detect stale updates, skips and state tampering. Durable atomic storage
and an external pinned ledger head remain the caller's responsibility.

## Birth of an organ

Prefer a module in an existing repository. A proposal requires a distinct
function (checked against an explicitly supplied existing-function catalog),
typed inputs/outputs, bounded CPU/memory/disk, source-bound test and health
receipts, separate review, a rollback parent commit and TRAIN_ONLY_NO_EXAM_ACCESS.
The default repository budget is zero. A structurally valid module still ends
PROPOSED_NOT_ACTIVE; receipts are unverified assertions until authenticated by
the outer operator. No supplied test, scanner or model can approve itself into
production. No current weights, research exams or learning workflows are used.

## Running and retaining evidence

Run from this repository, supplying the actual checked-out source commit:

```text
python -m unittest discover -s tests -p 'test_autonomy_*.py' -v
python -m autonomy.inventory --live --source-commit <commit> --out <new-inventory.json>
python -m autonomy.inventory --live --authenticated --source-commit <commit> --previous <previous.json> --out <new-inventory.json>
python -m autonomy.health <new-inventory.json> --trusted-root <separate-base-checkout> --expected-source-commit <commit> --out <health.json>
```

Authenticated collection requires GITHUB_TOKEN from the operator's existing
credential store; never put it in a receipt. Every inventory output is create-new.
`--previous` binds drift to a previous snapshot digest. A missing/incomplete owner
listing cannot be interpreted as mass deletion. Add
`--require-observation-pass` when an outer system needs a failing exit on health
HOLD; normal engineering CI preserves the HOLD result without equating it to a
code regression or granting admission.

The new workflow runs offline controls, collects metadata, then verifies the
artifact in a separate credential-free job against a base-map checkout. All
four infrastructure Actions are pinned to full commits, credential persistence
is disabled, permissions are read-only, and jobs have time/concurrency limits.
No workflow_run privileged consumer, pull_request_target, auto-merge, dispatch,
learning, real fork or repository creation is introduced. Artifacts expire after
14 days; durable reviewed receipts belong in Meta Registry.

The daily schedule runs only after the workflow reaches the default branch.
While this PR is a draft, push/PR runs demonstrate the implementation but do
not enable a permanent service. Scheduled observations update the derived map
artifact automatically; they do not rewrite the reviewed constitutional map.
Optional drift comparison uses a prior pinned snapshot, never an unvalidated
artifact downloaded from an arbitrary run.

## Limits and recovery

GitHub schedules can be delayed or dropped, public schedules can be disabled
after 60 days of inactivity, and default-branch, token, Actions minutes, rate,
artifact retention and service availability limits apply. See GitHub's
[workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows),
[Actions limits](https://docs.github.com/en/actions/reference/limits) and
[trigger troubleshooting](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows).
GitHub alone cannot guarantee continuous autonomous operation or an independent
watchdog for its own outage. The next deployment stage needs a separately
operated watcher, externally anchored receipts, durable storage, authenticated
reviews, least-privilege credentials and service-specific probes with recovery
drills. None is claimed present by a green metadata/contract run.

To roll back this draft, close the PR. After any reviewed merge, revert the PR
commit and disable the new observer workflow. Preserve evidence and append a
superseding/revocation receipt; do not erase failures. This PR changes no active
ESSENCE kernel, provider configuration, model weights or organ permissions.
