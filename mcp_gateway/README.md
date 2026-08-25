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
| `janus.read_registry` | read | read one registry artifact by path + ref |
| `janus.run_topa` | read | build a falsification-first TOPA packet and attach the canonical foundation |
| `janus.query_cosmos` | read | search `Janus-Cosmos` |
| `janus.read_cosmos` | read | read one Cosmos artifact by path + ref |
| `janus.freeze_gate` | **write, locked** | freeze a gate checkpoint on the dedicated branch |
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
