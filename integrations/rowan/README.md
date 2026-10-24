# JANUS × Rowan × TOPA × Meta Registry — bounded research bridge v0.1

Status: research prototype in Draft PR, not production or scientific validation.
Currently implemented: Rowan microscopic pKa only. Docking, co-folding, FEP, molecular
dynamics, screening and TOPA/Meta Registry automatic writes are not implemented.

## Scientific chain and authority

1. GOLEM or AURA proposes a hypothesis, alternative, and independent falsifier.
   An AURA output is not evidence; sealed AURA records remain in the AURA system.
2. TOPA supplies or reviews source provenance and falsification gates. The bridge
   accepts a TOPA origin reference but cannot certify TOPA review.
3. Rowan predicts microscopic pKa. A workflow UUID is not experimental evidence.
4. JANUS Meta Registry may archive a human-reviewed receipt. Local receipts are
   only candidates. This adapter never pushes or approves registry material.

Related repos:
- GOLEM/JANUS gateway: https://github.com/Hawkar-usls/Janus
- AURA: https://github.com/Hawkar-usls/aura-oracle-tg
- TOPA: https://github.com/Hawkar-usls/TOPA
- Archival registry: https://github.com/Hawkar-usls/janus-meta-registry

Provider documentation (Python API v3):
https://docs.rowansci.com/api/python/v3/
https://docs.rowansci.com/api/python/v3/workflows/pka/
https://docs.rowansci.com/tutorials/account/api-keys

## Usage

Run from repository root. This default mode is entirely OFFLINE:

    python integrations/rowan/janus_rowan_bridge.py \
      --case integrations/rowan/examples/phenol_case.json \
      --output /tmp/rowan-plan.json

Create a remote Rowan draft (requires a key, does not execute calculations):

    pip install rowan-python
    export ROWAN_API_KEY="read-from-secure-secret-store"
    python integrations/rowan/janus_rowan_bridge.py \
      --case integrations/rowan/examples/phenol_case.json \
      --output /tmp/rowan-draft.json --create-draft

Run ONE real workflow only with explicit credit-spend authorization:

    python integrations/rowan/janus_rowan_bridge.py \
      --case integrations/rowan/examples/phenol_case.json \
      --output /tmp/rowan-submit.json --submit --authorize-credit-use

Collect an existing result by Rowan UUID (read-only):

    python integrations/rowan/janus_rowan_bridge.py \
      --case integrations/rowan/examples/phenol_case.json \
      --output /tmp/rowan-result.json --collect \
      --workflow-id UUID-FROM-SUBMISSION-RECEIPT

The adapter passes a per-workflow ceiling of 1-10 credits to the Rowan v3 API.
This is not an account-wide balance guarantee. Rowan has free credits subject to
current provider conditions; the bridge NEVER purchases additional credits.

## Receipt semantics

- input_sha256: SHA-256 of canonicalized original input JSON. Integrity fingerprint,
  not source authentication or scientific evidence.
- provider.smiles_sha256: hashed structure identifier, not plaintext SMILES.
- event: actual bridge state (PLAN, DRAFT, SUBMITTED, PENDING, COLLECTED).
- workflow_uuid: populated following a real successful Rowan API response.
- research.scientific_status: always UNVERIFIED.
- research.topa_review: always REQUIRED.
- meta_registry.auto_published: always false.
- result: if present, Rowan model prediction, NOT an experimental value.

Avoid putting raw private molecules, personal health records, provider keys,
patient data or confidential research into public GitHub. A public receipt hash
alone is insufficient for reproducing chemistry: securely retain private inputs
and account-scoped data. Independent validation requires measured reference data,
protocol/method and conditions, error analysis and provenance review.

## Safety, limitations, and verification

- Offline by default; external draft requires credentials but no calculation.
- Executing a calculation requires BOTH submit and authorize-credit-use flags.
- No uncontrolled batch jobs, webhook posting, autonomous retry or scheduling.
- No automatic archival/publication, diagnosis, treatment or clinical authority.
- Never construe prediction as evidence of causation, efficacy or novelty.
- Outputs are write-once. If remote submission succeeds but the process fails
  before receipt creation, audit the Rowan account before retrying to avoid
  duplicate submissions or unexpected credit use.
- Collect enforces a case-name/workflow-type correlation, not a cryptographic
  proof that a workflow used the exact input SMILES.
- Test suite uses a fake SDK, not provider credentials or external computation.

Offline tests:

    python -m unittest discover -s integrations/rowan/tests -v

## Next gate

A project-scoped, short-lived Rowan API key and explicit free-credit budget are
needed to test a single benign phenol pKa prediction. Compare against an independent
experimental measurement and uncertainties, request TOPA review, then separately
approve archival into Meta Registry. Each future provider workflow requires
dedicated costs, evidence rules, and permissions.
