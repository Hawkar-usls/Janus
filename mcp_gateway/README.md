# JANUS MCP Gateway

A narrow Model Context Protocol (MCP) surface over the JANUS federated organism.

The organism uses a **5D event-sourced spiral**, not a closed cycle:

```text
STATE_n
  -> FORWARD
  -> REVERSE
  -> TRANCEPTION_ROTATE
  -> SCALE
  -> TIME / IDENTITY / PROVENANCE
  -> CONTRADICTION TEST
  -> CHECKPOINT
  -> ASCEND
  -> STATE_n+1
```

Canonical law:

```text
RETURN_TO_QUESTION_AT_HIGHER_STATE__NEVER_RETURN_TO_IDENTICAL_STATE
```

## Tools

| Tool | Mode | Purpose |
|---|---|---|
| `janus.health` | read | configuration / capability check |
| `janus.organism_map` | read | topology + spiral/Tranception execution constitution |
| `janus.spiral_pass` | read | build a deterministic 5D spiral packet for the host model |
| `janus.validate_spiral_transition` | read | fail-closed gate for repeated/unjustified state transitions |
| `janus.search_organ` | read | search an allowlisted JANUS organ by typed organ key |
| `janus.read_organ` | read | read one organ artifact by path + ref with provenance |
| `janus.search_registry` | read | search `janus-meta-registry` with provenance |
| `janus.search_public_evidence` | read, explicit egress consent | Wikipedia / OpenAlex / optional Brave source discovery for PUBLIC questions, not scientific proof |
| `janus.read_registry` | read | read one registry artifact by path + ref |
| `janus.run_topa` | read | build a falsification-first TOPA packet and attach the canonical foundation |
| `janus.query_cosmos` | read | search `Janus-Cosmos` |
| `janus.read_cosmos` | read | read one Cosmos artifact by path + ref |
| `janus.freeze_gate` | **write, locked** | freeze a gate checkpoint on the dedicated branch |
| `janus.codex_cloud_handoff` | read-only | draft a task for owner-authorized Codex Cloud; no model or repo mutation |
| `janus.codex_app_server_readiness` | read-only | report safe Sign in with ChatGPT / app-server prerequisites |
| `janus.write_checkpoint` | **write, locked** | persist a progress checkpoint |

The gateway does **not** expose shell execution, arbitrary URLs, arbitrary repositories, secret values, or model-generated truth claims.

## Spiral and Tranception boundary

`janus.spiral_pass` creates a structured analysis packet; it does not perform the analysis or fabricate evidence.

`janus.validate_spiral_transition` permits ascent only when the state hash changes **and** a recognized state-change reason exists. A repeated hash is a plateau/hold, not another turn.

```text
TRANCEPTION_ROTATION != EVIDENCE
SPIRAL_ASCENT != CLAIM_PROMOTION
REPEATED_STATE_HASH_IS_NOT_A_NEW_TURN
ASCEND_ALLOWED != CLAIM_CONFIRMED
```

Tranception is a whole-organism representation operator available at any node. It is not a repository organ and is not the same thing as the external protein-fitness software project that shares the name.

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r mcp_gateway/requirements.txt
python -m mcp_gateway.server
```

MCP endpoint: `http://127.0.0.1:8000/mcp`

For reliable GitHub code search, set a fine-grained `GITHUB_TOKEN`. Public file reads work without one but are rate limited.

## Test

```bash
python -m unittest discover -s mcp_gateway/tests -v
```

For protocol-level testing, connect the official MCP Inspector to `/mcp`.

## Write barrier

Writes are disabled unless all of the following are intentional:

```text
JANUS_ALLOW_WRITES=1
GITHUB_TOKEN=<fine-grained token with Contents: write only on Hawkar-usls/Janus>
JANUS_WRITE_BRANCH=mcp-checkpoints
```

Use a dedicated branch. Do not give the gateway broad organization/repository administration scopes.

## Claim boundary

```text
SEARCH_MATCH != VALIDATED_CLAIM
ARCHIVED_SOURCE != TRUSTED_SOURCE
TOPA PACKET != WORLD-TRUTH VERDICT
ARTIFACT != INDEPENDENT REPLICATION
TRANCEPTION_ROTATION != EVIDENCE
SPIRAL_ASCENT != CLAIM_PROMOTION
```

## Codex Cloud and ChatGPT account bridge

The initial safe integration contract is documented in
[`docs/JANUS_CODEX_CLOUD_AND_CHATGPT_GATEWAY.md`](../docs/JANUS_CODEX_CLOUD_AND_CHATGPT_GATEWAY.md).
Codex Cloud environment setup and ChatGPT OAuth consent belong to the owner.
The new gateway tools only build draft tasks and describe readiness; no account
is signed in, no app-server is launched and no source code is auto-merged.

## Public-source search (R2G integration into the canonical JANUS gateway)

The read-only tool `janus.search_public_evidence` is registered in the current MCP gateway server. Its provider client lives in `mcp_gateway/public_evidence.py`. It returns bounded source metadata with URLs, dates and unverified excerpts; it is not a chat-completion endpoint or scientific adjudicator. No dependency on an unmerged Demi_Head branch is required at runtime.

### Explicit permission gates

Both gates are required: server variable `JANUS_PUBLIC_EVIDENCE_ENABLED=1`, and the calling host passes `public_query_approved=true` for each PUBLIC question. Both default to OFF. The gateway also requires authentication, rate limits and user consent at the hosting ingress layer before external deployment: MCP protocol alone is not user authentication.

Example call: `janus.search_public_evidence(question='Что известно об атмосфере Сатурна?', provider='auto', public_query_approved=true)`.

Providers: `auto`, `wikipedia`, `openalex`, `gdelt`, `brave`. For nonfresh questions, auto queries Wikimedia and OpenAlex scholarly metadata. For time-sensitive questions, auto uses keyless GDELT DOC 2.0 news search, and optionally also Brave if configured. GDELT is a news index, not a complete web search engine or a verified sports-score database. Dates labelled `seen_at` mean when GDELT saw an article, **not** its publication time. Fresh-event claims remain unverified until independently checked. Brave API uses `BRAVE_SEARCH_API_KEY` and OpenAlex optionally uses `OPENALEX_API_KEY`, both injected into trusted server runtime (not GitHub source, forks or PR secrets).

### Security and scientific limitations

- Fixed allowlisted HTTPS API endpoints only, no redirects or arbitrary URL fetching. Discovered result URLs are never fetched. Search results are filtered against localhost, IP literals and embedded credentials.
- Timeout 6 seconds, response cap 100 KB, four returned sources maximum. Read-only, no filesystem write or model training.
- Heuristic privacy filter blocks common password/token markers, URLs, email addresses, paths and IPv4 addresses before egress. This is NOT comprehensive privacy detection; do not forward private NAS logs, registry records, healthcare data or other user-private context.
- Retrieved titles/excerpts are UNTRUSTED DATA, not instructions to execute. A Wikipedia or Brave match is not verified fact; OpenAlex metadata is not full peer review or scientific explanation.
- Current-event search results get `FRESHNESS_NOT_INDEPENDENTLY_VERIFIED`, even when Brave returns text. Missing data or provider HTTP_429 never means the real-world claim is false.
- MCP returns a query SHA256, not the plaintext query, but search providers receive the public text by necessity.
- The provider endpoint is not an unrestricted public chat bot. A separate authenticated host must decide whether to surface or summarize retrieved sources.

### Safe deployment procedure

1. Review this draft PR and CI. No main branch, live MCP deployment or NAS container has been modified.
2. On an authenticated trusted HTTPS host, deploy the reviewed gateway code and keep `JANUS_PUBLIC_EVIDENCE_ENABLED=0` until ingress access control, budgeting and consent checks are ready.
3. Enable public search by setting `JANUS_PUBLIC_EVIDENCE_ENABLED=1`. For broader web coverage beyond GDELT news, optionally configure `BRAVE_SEARCH_API_KEY` via a protected secret store **only if the service is lawfully available in your region**. The key is not required for GDELT news discovery. For scalable OpenAlex queries use its free optional key.
4. For each explicitly approved public question, call `janus.search_public_evidence` and use the returned citations only within their actual evidentiary scope.

R2B/R2C/D research candidates and original scientific holdouts remain untouched. `SEARCH_SOURCE != VERIFIED_FACT`; `SOURCE_METADATA != PHYSICAL_REPLICATION`; `TOOL_REGISTERED != DEPLOYED`.

### Regional Brave Search API access and keyless fallback

Brave's published unavailable-jurisdictions list includes Ukraine. A `403` on the Brave dashboard therefore may reflect a service-availability restriction, not a JANUS problem. Never attempt to circumvent provider geographical restrictions. For account-specific confirmation contact Brave Search API support. JANUS no longer requires the Brave dashboard or key to discover recent public **news**: it uses GDELT DOC 2.0 (`https://api.gdeltproject.org/api/v2/doc/doc`) with fixed HTTPS endpoint, `mode=artlist`, `format=json`, `timespan=1week`, up to four records. Respect provider throttling (avoid polling more frequently than once per five seconds). The returned `seendate` is an index observation timestamp, not an event date or publication proof. Missing sources or rate-limits remain `UNKNOWN`, not `FALSE`.
