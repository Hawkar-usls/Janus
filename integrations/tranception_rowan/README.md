# JANUS Tranception × Rowan — provenance-gated experiment v0.1

**Scientific status:** PROTOCOL + OFFLINE TESTS ONLY. NO live Rowan work, no claim about fitness, medication, neurodevelopment, P vs NP, Osiris, or the JANUS reasoning analogy.

## Provenance inventory

The frozen `discovery_manifest.json` enumerates **22 JSON paths** whose *names* contain TRANCEPTION in the untruncated 6,936-entry tree of `Hawkar-usls/janus-meta-registry` on 2026-10-25. Each carries its Git blob SHA and evidence-scope label. File-name matching is NOT a complete search of file contents; other documents can mention the name.

Two direct scientific orientation documents are:
- `data/JANUS-TRANCEPTION-FORWARD-REVERSE-FITNESS-LIFE-GATE-v1.0.json`
- `data/JANUS-TRANCEPTION-CHRONIC-THREAT-NEURODEVELOPMENT-RESEARCH-CHEATSHEET-v1.0.json`

Their existence does not provide a chemically valid SMILES input, a raw amino acid sequence, measured fitness results, or authorization for clinical inference. Most other files use *Transception* as a JANUS analogy applied to reverse reasoning, history, mathematics, or literature. Do not send those texts as protein molecules.

Source base: `https://github.com/Hawkar-usls/tranception` is explicitly an **upstream-derived working copy** of `https://github.com/OATML-Markslab/Tranception`. Tranception's protein-fitness scores are not synonymous with Rowan/Boltz fold confidence.

## Independently sourced, harmless control

The official ProteinGym reference table `OATML-Markslab/ProteinGym/reference_files/DMS_substitutions.csv`, pinned by blob SHA in the manifest, provides a **64-residue Saccharomyces cerevisiae BBC1** protein domain for `BBC1_YEAST_Tsuboyama_2023_1TG0`. Two test arms are:
- wild-type as published in ProteinGym;
- exactly one specified `K5A` substitution derived from this control.

The existence of an experimentally measured **K5A** activity label was **NOT established**. This pair is intended for checking sequence ingestion and changes in **predicted structural confidence** only. A correlation test against experimental mutational fitness requires separately retrieved, independently pinned DMS labels, and more than two sequences.

## Rowan API support

The official Rowan Python v3 `submit_protein_cofolding_workflow` accepts a sequence list, model, credit ceiling, and a draft flag:
https://docs.rowansci.com/api/python/v3/workflows/protein-cofolding/

Use `model="boltz-2"`, one diffusion sample and one sequence per run. Store the provider UUID with SHA-256 fingerprints. Do not publish raw confidential sequences or credentials. `pTM` and `avg_lddt` estimate confidence in **structure prediction**, never function, mutation fitness, pathogenicity, or medical outcomes.

**Default/offline**, from repository root:

    python integrations/tranception_rowan/probe.py --arm wildtype --mode plan --output /tmp/bbc1-wt-plan.json
    python integrations/tranception_rowan/probe.py --arm K5A --mode plan --output /tmp/bbc1-k5a-plan.json

**Check account**, only if a trusted runtime has a `ROWAN_API_KEY` (no computation). The account check establishes successful authentication, NOT an independently validated free-credit balance:

    python integrations/tranception_rowan/probe.py --arm wildtype --mode auth --output /tmp/bbc1-auth.json

**Create a Rowan draft** (writes draft to your Rowan account, no compute):

    python integrations/tranception_rowan/probe.py --arm wildtype --mode draft --output /tmp/bbc1-wt-draft.json

**Run a single provider calculation**, only with an independently checked free-credit balance and explicit authorization (10-credit ceiling; may terminate without a result):

    python integrations/tranception_rowan/probe.py --arm wildtype --mode submit --max-credits 10 --authorize-credit-use --output /tmp/bbc1-wt-submit.json

Collect:

    python integrations/tranception_rowan/probe.py --arm wildtype --mode collect --workflow-id YOUR_UUID --output /tmp/bbc1-wt-collect.json

Repeat the same operations for K5A with separate file paths. No automatic batching, retries, publication, charging, or promotion of scientific findings. If a provider submission succeeds and local receipt writing fails, verify the Rowan account first, **never retry blindly**.

## Gates

1. Frozen upstream metadata and exact sequences — PASS for control.
2. Registry crosswalk — 22 name-matched JSON refs indexed with SHA.
3. Offline input and fail-closed computational budget tests — CI required.
4. Published Rowan SDK structural API signature check — CI required.
5. Rowan authentication and balance — **NOT VERIFIED**.
6. Rowan cloud draft submission — **NOT ATTEMPTED**.
7. Real Boltz-2 predictions for both arms — **NOT ATTEMPTED**.
8. Tranception model outputs and matching experimental DMS labels — **NOT RUN**.
9. TOPA independent biological evidence review — **NOT DONE**.
10. Meta Registry scientific promotion — **FORBIDDEN UNTIL GATES 5–9 PASS**.

The test script does NOT treat a compiler/lint/unit-test PASS as evidence that a Rowan computation was actually performed. Existing `ROWAN_API_KEY` shared in a chat has not been copied into GitHub code or logs.

## Confirmed cost and budget boundary (2026-10-25)

Rowan's public FAQ reports that a protein co-folding job **typically uses around 10 credits**; actual usage depends on size and options:
https://rowansci.co/blog/credits-faq

Therefore, the previous one-credit cap might cut off Boltz-2 before completion. The new suggested single-job ceiling is **10 credits**, with the CLI enforcing 1–15 credits (and the GitHub on-demand workflow defaulting to 10). **The credit cap is a limit, not a price estimate or a promise of completion**.

Rowan credits are spent in this order: weekly user → weekly organization → purchased user → purchased organization. This integration makes **no credit purchases** but a successful Rowan computation could still consume already purchased credits if free credits are insufficient. Rowan's `whoami()` confirms authentication, but we do not currently have a verified, authoritative SDK balance endpoint and we will not invent an available balance:
https://docs.rowansci.com/tutorials/account/credits-and-billing

Before the first submit, verify the **free** balance in Rowan Account → Billing. Keep the existing project key in a secured runtime, not in code, YAML, console logs, or GitHub commit history. The read-only auth workflow now uploads a one-day private result artifact containing no secret, username, or email.

A completed provider job has not been performed until a real Rowan workflow UUID and verified cloud result exist. The compiled/mock CI run only checks software mechanics. The Meta Registry evidence status remains OPEN.
