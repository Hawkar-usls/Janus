# JANUS autonomy R1: source audit and limits

Observed 2026-11-23. This audit identifies existing engineering mechanisms and
the missing controls needed for a bounded next stage. Repository text, a hash,
a successful CI job, and agreement between model agents do not establish
independent operational or scientific validation.

## Source snapshots

| Repository | Audited commit |
|---|---|
| `Hawkar-usls/Janus` | `929cabcab2c1886432ae6f075448c823a6f40a50` |
| `Hawkar-usls/janus-meta-registry` | `8abdad46f16694bb76a6127c2d0b779d1cd8052e` |
| `Hawkar-usls/Janus-Demiurge` | `5a90ddb2ffc893559d66cc0c442bc1f112333851` |

These are audit inputs, not floating references or production adoption. Later
results must identify their own exact source commit and execution evidence.
Private repository contents and locators were excluded from this public audit.

## What the owner calls the map

There is no single complete, continuously verified map. Several projections
describe different questions and must retain their own provenance.

| Projection | Repository path | Observed Git blob |
|---|---|---|
| Chat routing index | Registry `data/JANUS-CHAT-CONTEXT-INDEX.json` | `c3d10c5ea67a36fc7a515b144e7338350b99bfc1` |
| Historical working anatomy | Registry `data/JANUS-WORKING-MAP-REPOSITORY-ANATOMY-AND-INTEGRATION-ROADMAP-2026-08-19-v1.2.json` | `a897ac25e945fbb9550d8234387aed836dcf0970` |
| Canonical organ topology | Janus `organism/JANUS_ORGANISM_v1.json` | `24c4876e93913f231cd544797defb1c05f89df04` |
| Repository ecology | Janus `organism/JANUS_REPOSITORY_ECOLOGY_v1.3.json` | `c050447b56b0bf86392a76506ac04bf7e9da9e31` |
| Simptomat ecology delta | Janus `organism/JANUS_REPOSITORY_ECOLOGY_v1.4.json` | `970f27656855f564d50c4af09c5ddca0a95f003f` |
| Public discovery contour | Registry `agent-observatory/JANUS_CONTOUR_MANIFEST.json` | `bf29cd30b2c9d61066e0f51da565b0b6f2efe0fa` |
| Existing inventory verifier | Registry `tools/janus_contour_inventory_audit.py` | `068b755dce9ba52c94bec67776341e8847d51e7e` |

The historical working map explicitly separates `HABITAT_GRAPH` (existence and
provenance), `NEXUS_GRAPH` (admitted typed transfers), and
`EPISTEMIC_PROOF_GRAPH` (bounded evidence authority). A live repository census
updates the first projection. It must not silently promote a discovered link
into an admitted transport or a proof edge.

The chat index contains project pointers rather than complete repository
coverage. The contour was last marked reviewed on 2026-10-28. Its classifications
are declarations, not live execution witnesses. The newer Simptomat delta
coexists with older topology and runtime projections: the audited
`mcp_gateway/organism_tools.py` public allowlist omits Simptomat, and
`mcp_gateway/server.py::health()` advertises execution `1.2.0` and ecology `1.3.0`.
This is observable projection drift; resolving it requires explicit reviewed
integration work, not automatic authority inheritance.

## Existing mechanisms and their actual scope

| Mechanism | Actual behavior | Remaining boundary |
|---|---|---|
| Registry `tools/janus_contour_inventory_audit.py` and `.github/workflows/janus-contour-inventory-health.yml` | Daily read-only public owner reconciliation; rejects missing, new, duplicate or misclassified entries | Does not propose map updates or verify real runtime integration |
| Registry `tools/essence_v14_fleet_audit.py` | Reads public manifests at immutable commits; runs only its pinned local ESSENCE kernel, never downloaded target code | Frozen 48-name cohort and `main` assumptions; CI observations lack freshness and exact-head admission |
| Registry `tools/essence_v52_security_fleet.py` | Static workflow security census at commit pins | Signals for review, not a comprehensive secret scan or proof of exploitability |
| Janus `.janus/essence/kernel.py` | Bounded deterministic offline observations and child proposals | No trained model, runtime activation, network, shell or repository mutation |
| Janus `.janus/essence/heritage_v08.py` | Thirteen advisory historical roles over six existing microcores; source-root checks | Advisory lenses are not three independent trained or signed evaluators |
| Janus `.github/workflows/essence-readonly-heartbeat.yml` | Weekly synthetic heartbeat, root checks and artifact receipts | Synthetic health does not establish deployed service health or delivery |
| Demiurge `.janus/essence/module_governor_v13.py` | Pure-function reconstruction and data-only research blueprints | Historical synthetic cases and post-test errata are not pristine independent exams |
| Demiurge `.github/workflows/janus-rex-organogenesis.yml` | Bounded ModuleSpec candidate creation, audit and separate admission paths | Candidate creation is not general repository admission or external fork safety |
| Demiurge `tools/janus_all_star_intake_catalog.py` | Static AST inspection of a fixed curated local source list | Does not implement a general external dependency lifecycle |

One relevant weakness in the older fleet auditor is
`latest_workflow()`: it can select an older completed success when the newest
run is pending. A new health gate must require the intended commit, expected
workflow, unambiguous latest attempt and bounded evidence age. Missing or
pending evidence must remain `UNKNOWN` or `HOLD`.

The original ESSENCE kernel blob is
`de5199978fbafcbda836b77da370943f7ea2514e`. Its audited manifest requires
`enabled=false`, `EXPLICIT_READONLY_TEST_ONLY`, and `PROPOSAL_ONLY`; network,
shell, repository writes, workflow dispatch, secrets and hardware are denied.
The next stage must preserve these defaults. A new observer is not permission
to change ESSENCE roots, working model weights or established gateways.

No general `catalog -> quarantine -> review -> fork` implementation was found
in these audited mechanisms. Experiment-specific external source locks already
record upstream commits, blobs and licenses, but do not grant dependency or
execution admission. A fork is a provenance-preserving copy for review and is
never automatically an organ or executable dependency.

## Dimensions the map still needs

1. **Resource budgets.** Record per-tick API calls, response bytes, repositories,
   candidate count, active work, runtime, retained artifacts and retry limits.
   Admission must fail closed at exhaustion. Budget changes need separate
   review; a model proposal cannot increase its own quota.
2. **Incident and recovery state.** Record a kill switch, last known reviewed
   revision, candidate disable/revoke state, rollback target, recovery owner,
   idempotency key and audit-chain linkage. Rollback verification is distinct
   from a written rollback plan. Preserve failed and superseded receipts.
3. **Source provenance and quarantine.** Record canonical upstream identity,
   public visibility, default branch, immutable commit, license file and bytes,
   content hashes, scan scope and limitations. Unknown license, changed source,
   unresolved branch, secrets signals or incomplete scan must block admission.
   Never execute remote installation hooks or downloaded workflows to inspect
   a candidate.
4. **Independent admission gates.** Keep collection, deterministic verification,
   trusted approvals and execution capabilities separate. A self-reported
   health result or a user-editable `approved=true` field is not an independent
   witness. Require exact candidate digests, authenticated external approval
   where effects are requested, expiry and anti-replay checks. Engineering CI
   does not establish scientific gain.
5. **Capability freshness.** Each capability and typed edge needs the observed
   source SHA, test/run identity, observation time, expiry, state and revocation
   reason. Distinguish declared, code-present, fixture-tested, delivered and
   independently observed states. Discovery must not auto-enroll new organs.
6. **Birth and retirement contracts.** A candidate module or repository needs a
   distinct purpose, typed interfaces, parent, dependency closure, denied
   capabilities, resource limits, owner, review path and rollback. Enforce
   depth, fan-out and global candidate caps. Avoid duplication of an existing
   organ. A rejected candidate cannot reproduce or grant admission to children.
7. **Evaluation custody.** Keep training and development sources separate from
   frozen examinations. Record first-look status and independent custody.
   Disclosed exams may serve only their explicitly permitted archival role;
   do not debug, retune or claim new independent success on them.

## Honest autonomy state and GitHub limitations

At the audited Registry commit, `data/autonomy_loop/LATEST.json` observed
`2026-11-23T16:12:36Z` reports `AUTONOMY_PARTIAL`, phase `P0.3`, and
`full_issue_162_acceptance=false`. `SELF_ASSEMBLY_TASK.json` retains BEETLE
freshness/delivery blockers and independent witness blockers. This audit does
not resolve or overwrite them.

GitHub Actions provides bounded jobs and event triggers, not an always-on,
guaranteed autonomous organism. Schedules depend on the default-branch workflow,
platform scheduling and repository state; queueing, rate limits, permissions,
quota and retention can interrupt the loop. A workflow successful on one head
does not witness a later head. Artifacts expire. Jobs triggered using the
repository token have event-propagation limitations. Cross-repository access
and protected environments require separately configured scopes and review.

A GitHub-only control plane also cannot independently guarantee the integrity
of its own administrators, branch protections, credentials, external evidence,
physical systems or disaster recovery. External custody and an independent
watchdog remain necessary before stronger endurance or autonomy claims.

R1 therefore provides reviewable map observations, deterministic health holds,
and bounded admission proposals. It must retain `authority_delta=0`, prohibit
self-promotion, arbitrary external code execution and uncontrolled reproduction,
and keep actual external effects unavailable until separately authenticated,
reviewed capability configuration exists.
