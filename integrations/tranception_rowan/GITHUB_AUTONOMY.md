# Autonomous GitHub Rowan connection: security and limits

**Current scope:** zero-credit, read-only scheduled connectivity, not automatic paid science.

The canonical public repository `Hawkar-usls/Janus` uses the protected
`rowan-research` GitHub **Environment**, which is explicitly restricted to
the default `main` branch. Do not put Rowan secrets into public or private
repository files, commit history, artifacts, issues, Pull Requests or logs.

## First-time setup (only one secret)

GitHub UI: `Hawkar-usls/Janus` → **Settings** → **Environments** →
**rowan-research** → **Environment secrets** → **Add secret**:

- Name: `ROWAN_API_KEY`
- Value: a dedicated **regular project Rowan API key**, preferably scoped
  read-only and to the appropriate Rowan project; **never a master key**.

The value is encrypted by GitHub when uploaded and injected only into an
eligible job. Environment branch policy allows only `main`; the source
workflow has `permissions: contents: read` and does not check out
unreviewed Pull Request code. GitHub masks secrets in logs but masking is
not an absolute guarantee against malicious scripts; review changes to
workflows before merging and restrict who can modify `main`.

After the secret is installed, the public workflow
`.github/workflows/rowan-autonomous-readonly.yml` runs every six hours
and can be triggered from the Actions tab. It only calls:

- `rowan.whoami()` (read-only);
- `rowan.retrieve_folder("70480946-24c3-4078-91d1-ad99009262bf")`;
- up to ten direct folder entries via read-only listing (no entry details
  are printed or uploaded).

The job returns sanitized PASS / HTTP code / BLOCKED statuses. It does
not print usernames, email addresses, balances, folder contents, API
responses, project keys or session cookies. It does not use `submit_*`,
spend credits, change folder permissions, or configure subscriptions.

## Geographic access

On Yaksa the provider returned **HTTP 403** for both read-only calls.
Because that machine did not have a VPN running at the time, IP-level
restrictions are a plausible explanation but NOT confirmed. GitHub-hosted
runners use a different network. Testing the same safe read-only endpoints
from a scheduled runner helps determine if GitHub egress is permitted,
but a different status code alone will not establish the underlying policy.

If GitHub-hosted Rowan traffic is also restricted, use only an authorized
provider-supported egress arrangement. For a NAS-based worker, attach a
dedicated Rowan container's namespace to a dedicated VPN container with
a kill switch: do not alter the NAS default gateway, other containers,
inbound NAS ports, or JANUS Genesis networking. Do not route around
contractual or legal geographic access restrictions.

## Scientific gates

Even after read-only access PASS, no Rowan calculations run automatically:
project credit budgets, free/purchased credit waterfall, identity and
folder permissions, official assay identity, model validation, TOPA
independent review and non-duplication receipts remain separate gates.
The published ProteinGym assay numbers and Tranception benchmark scores
are not proof of an executed cloud Rowan computation.

## Operational limitations

GitHub Actions is **scheduled execution**, not an always-running JANUS.
Scheduled runs may be delayed or missed, and public-repository schedules
can be disabled after 60 days without activity. No automatic posting to
Meta Registry is enabled by this watcher because cross-repository writes
would require a separately scoped GitHub App or token, reviewable PRs
and a provenance-filtered receipt format.

Source: GitHub Actions security and scheduling documentation. Protect
the Rowan key separately; a private repository alone is not a secrets vault.
