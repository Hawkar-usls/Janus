# JANUS Rowan BBC1 autonomous stage 1

**Scope**: two preregistered Boltz-2 protein-cofolding jobs, and no others.
The two cases are BBC1 wildtype and BBC1 K5A, as frozen in discovery_manifest.json.

The protected GitHub Environment `rowan-research` stores ROWAN_API_KEY; it runs only from main.
The workflow is scheduled daily at 05:37 UTC, serializes executions, and submits at most one
missing case per run. Case identity is checked against provider-side listings for the exact
approved Rowan folder. It never auto-retries any failed case.

An individual job has provider max_credits=10. Thus the maximum authorized requested budget
for the pair is 20 credits. Weekly organization credits must be >=30 before each submission.
Even though an additional >=450 organization nonweekly credits were read, this stage
does not intentionally draw from that reserve. Rowan billing docs specify weekly credits
are debited first. No billing subscription, top-up, project promotion or purchases are initiated.

Result collection is read-only and emits only safe status and bounded pTM / avg lDDT values
if present, never raw structure coordinates, account names, keys, personal balances or tokens.
Temporary evidence artifacts and provider-side workflow identity form the audit trail.

Interpretation firewall: pTM / avg lDDT are structural confidence, not experimental fitness,
causal validation or mutation free energy. TOPA independent review remains OPEN. Previous
ProteinGym assay values and historical Tranception correlations are separate evidence types.

No autonomous discovery outside this fixed two-arm agenda is authorized by the workflow.
The remaining 500-credit nonweekly allowance will not be depleted blindly. Expand only
after outcomes, scientific criteria, pricing, and provider balance classifications are reviewed.

GitHub scheduled runs are best-effort and may be late or skipped. They do not provide
a permanently running agent. Existing NAS or portal services are unaffected.
