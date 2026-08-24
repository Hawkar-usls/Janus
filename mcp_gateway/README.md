# JANUS MCP Gateway

A narrow Model Context Protocol (MCP) surface over JANUS research repositories.

## Tools

| Tool | Mode | Purpose |
|---|---|---|
| `janus.health` | read | configuration / capability check |
| `janus.search_registry` | read | search `janus-meta-registry` with provenance |
| `janus.read_registry` | read | read one registry artifact by path + ref |
| `janus.run_topa` | read | build a falsification-first TOPA packet and attach the canonical foundation |
| `janus.query_cosmos` | read | search `Janus-Cosmos` |
| `janus.read_cosmos` | read | read one Cosmos artifact by path + ref |
| `janus.freeze_gate` | **write, locked** | freeze a gate checkpoint on the dedicated branch |
| `janus.write_checkpoint` | **write, locked** | persist a progress checkpoint |

The gateway does **not** expose shell execution, arbitrary URLs, arbitrary repositories, secret values, or model-generated truth claims.

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
```
