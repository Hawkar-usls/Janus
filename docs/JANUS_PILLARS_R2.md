# JANUS Pillars R2 — 13 external architecture inspirations

Status: DRAFT / RESEARCH ONLY / NO AUTONOMY ADMISSION. This is original
JANUS code layered on existing autonomy.inventory, autonomy.health and
autonomy.admission. No foreign source was copied, installed or executed.

## Pinning and legality

Source GitHub branch HEAD, repository URL, GitHub license metadata and
architectural pattern are frozen in autonomy/pillars/CATALOG.json. License
metadata is NOT a byte-level legal clearance or full dependency review.

| Pillar | JANUS-native pattern |
| --- | --- |
| aeonfun/aeon | Scheduled repair triage and health self-monitoring |
| github/gh-aw | Hardened GitHub Actions, reviewed output |
| malakazlan/NEXUS | Capability need discovery, typed tools |
| Narco995/nexusforge-ai-organism | Organ registry, nervous/immune loops |
| paperclipai/paperclip | Roles, delegated budgets, accountability |
| open-gitagent/gitagent | Portable versioned identity/skills |
| hanfour/multi-repo-agent | Repository dependency/impact graph |
| jennyzzt/dgm | Parent-child lineage and regression selection |
| SakanaAI/ShinkaEvolve | Variation archive and bounded search |
| ANative-Lab/EvoAgentX | Pairwise workflow fitness checks |
| aiming-lab/Agent0 | Teacher-curated tasks distinct from truth |
| microsoft/Agensh | Bounded multi-agent coordination and review |
| EvoMap/evolver | Gene/capsule/event evidence ledger pattern |

NEXUS and NexusForge have NO license listed by GitHub metadata, EvoAgentX
shows NOASSERTION, EvoMap/evolver GPL-3.0 (outside the JANUS admission
allowlist). Those four are PATTERN-ONLY with explicit CODE-IMPORT HOLD.
The other nine are MIT or Apache-2.0 according to GitHub metadata but
remain CODE-IMPORT HOLD until full license bytes, provenance, history,
all dependencies and an external reviewer are checked. No automatic forks.

## R2 working research loop

The sixteen capability needs in autonomy/pillars/NEEDS.json are observed
RESEARCH_ONLY or NOT_DEMONSTRATED from the frozen 49-repo R1 Meta Registry
map. They are evidence-limited claims, not assertions of complete absence.
The existing health verifier was HOLD. Plan generation never overrides it.

- engine.py validates 13 exact refs, 16 gaps and JANUS admission policy.
  Uses fixed benefit/risk/cost ranking. Emits at most four typed,
  source-anchored module-only blueprints and a SHA-linked Gene-Capsule
  proposal chain. Does not install or activate those candidate modules.
- health_triage reads and digest-checks the historical R1 health receipt,
  counts fixed error categories and proposes no privileged remediation.
- upstream_watch.py issues a maximum of 26 GET requests to public
  api.github.com repository metadata and default-branch HEADs. Drift,
  missing evidence, license changes and lack of access remain HOLD.
- verifier.py is an independently written structural checker in a
  separate no-credential CI job, not an independent outside custodian.
- tests/test_autonomy_pillars.py contains adversarial unit controls for
  authority inflation, license spoofing, fork creation, audit tampering,
  source unavailability and exact source pinning.

New GitHub Actions workflow runs contract tests, plans candidates using
the original pinned R1 health HOLD, independently checks the proposal
chain, observes current public upstream HEAD/metadata and uploads
bounded JSON artifacts. Its scheduled daily event only becomes active
on GitHub's DEFAULT BRANCH after reviewed merge. No 24/7 service or
production self-repair is claimed while this draft PR is unmerged.

## Real organ birth gate

1. Prove the new function is needed by current independent service health
   and cannot already be served by a working JANUS organ.
2. Bind external source commit, complete license, full repository tree
   and history. Quarantine. No execution of foreign setup/workflow code.
3. Require typed interfaces, CPU/memory/time quota, synthetic sandbox,
   exact rollback commit and anti-contamination training policy.
4. Independently test on fresh tasks alongside strong parent. Neither
   same-owner scoring nor disclosed tests become a graduation exam.
5. Review protected evidence with an authenticated authorizer that the
   candidate cannot modify. Only then a limited, reversible PR is possible.
6. Do not enable actual fork creation or production model replacement
   without a separate, explicit and independently verified gate.

Current JANUS P0.3, external custody, private ACK operational credentials,
independent model-learning benefit and outage recovery remain HOLD.
The Terminal fork expansion workflow remains disabled.

To rollback: close this draft PR. Preserve all failure receipts and pins;
revert reviewed future merges by exact commit, never rewrite old evidence.
