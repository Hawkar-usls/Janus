# R8.4 — bounded durable replay gate

Research only. This first substage is **archive replay, not live admission or training**.
The archived R8 signed exchange is verified against pinned commits/blobs and its original
verification window, not admitted as a fresh packet. Two separate GitHub runners commit
bounded state transitions to the one research branch. A separate read-only workflow observes
the final committed state. The two wakes must both skip training on the same data hash.
Stop, budget exhaustion, altered archive, unapproved queue and training budget are tested.

## Before enabling a schedule
- Provide independently managed Ed25519 public-key fingerprints and secrets via approved GitHub
  environments; private keys must never enter a repository, artifact or log. Historical
  fingerprints authenticate only the archived exchange, not future packets.
- Implement fresh signed packet admission, durable PENDING/CLAIMED/VERIFIED/REJECTED transitions,
  idempotent recovery after interruption, signed revocation and a new closed regression exam.
- Set finite per-cycle and total budgets and an operator-controlled stop, concurrency lock,
  non-force branch publication and no production promotion.
- GitHub cron runs on the default branch. Draft PR configuration does not demonstrate a live cron.
  No schedule or deployment is enabled by this experiment; after gates pass, review an explicit
  default-branch configuration (e.g. every 30 minutes, at most a fixed number of cycles).
- External participants require explicit consent and separate onboarding.

No new learning claim, no live autonomous recurrence, no production change.
