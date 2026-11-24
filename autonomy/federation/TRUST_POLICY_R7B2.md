# JFP R7B.2 trust contract (research)

This layer reuses the R7B.1 signed envelope; it does not create trust from discovery. Only the existing fixed public registry fixture is supported.

- Each operator maintains a local policy with schema `janus.jfp.trust.v1`, local node ID, integer epoch, and admitted node rows.
- A row contains an Ed25519 public PEM, SHA-256 of its DER public key, explicit ACTIVE/REVOKED state, validity bounds in Unix seconds, and exactly the supported capability. Each node and key must be unique.
- A trusted deployment supplies the expected canonical policy SHA-256 independently of the packet and policy file. Never obtain the pin from an untrusted peer.
- The private key must be provisioned outside Git and must match the local policy. This research workflow uses generated disposable keys only; it does not provision GitHub secrets.
- SQLite stores the policy head and replay entries. The same protected database must be preserved across calls. Loss or rollback of the database defeats the cross-run protection; missing durable state in production must fail closed. The tests establish local file persistence, not persistence across hosted runners.
- Revocation requires an operator-approved higher policy epoch and trusted pin. Older epochs and conflicting content at the same epoch are rejected. No automatic trust updates.
- A result is tied to the verified signed request, its digest, addressed peer, fixed oracle answer and expiry; one accepted result per request.
- The CLI creates output files exclusively. A crash between admission and output can require a new request; transactional delivery/recovery is a separate milestone.

## Local verification

`python3 autonomy/federation/jfp_gateway.py test`

`python3 autonomy/federation/jfp_trust.py test`

## Next gate

Review the trust policy and durable-state recovery before enabling any bounded cross-repository transport. Use existing owned repositories, explicit local consent and restricted GitHub credentials, no foreign workflow execution, no secrets in artifacts, no external discovery-to-execution, no production promotion. Test failure, duplicate delivery, revocation, crash recovery and expiry on separate runners.

