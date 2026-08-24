# JANUS MCP Gateway v0.1

## Purpose

Turn the root JANUS gateway into a standards-based tool surface that MCP-capable hosts can call directly instead of manually traversing GitHub repositories.

```text
HOST / LLM
   ↓ MCP (Streamable HTTP)
JANUS MCP Gateway
   ├── janus-meta-registry   → provenance/search/read
   ├── TOPA                  → falsification packet
   ├── Janus-Cosmos          → search/read
   └── Janus checkpoints     → gated writes only
```

## Security model

1. **No arbitrary shell.** The MCP server never executes repository code.
2. **Repository allowlist.** Only explicitly configured JANUS repositories are reachable.
3. **Secrets stay server-side.** `GITHUB_TOKEN` is never returned by a tool.
4. **Read first.** All research/search tools are read-only.
5. **Writes default OFF.** `janus.freeze_gate` and `janus.write_checkpoint` require `JANUS_ALLOW_WRITES=1`.
6. **Dedicated write branch.** Default target is `mcp-checkpoints`, not `main`.
7. **Proof-carrying checkpoints.** Every generated checkpoint includes a canonical SHA-256 digest.
8. **No epistemic promotion by transport.** A tool result, repository hit, CI pass or checkpoint does not establish scientific truth.

## ChatGPT availability note — 2026-08-25

As of 2026-08-25, OpenAI's current documentation says direct custom/full MCP developer-mode connections are available on workspace plans (Business / Enterprise / Edu), while Pro has a more limited developer-mode path. A personal Plus account should therefore treat this server as **built and deployment-ready, but not directly attachable as a private custom MCP app yet**.

This is a product-plan limitation, not a protocol limitation. The server remains usable with MCP Inspector and other MCP-capable hosts, and is ready for ChatGPT once the account/workspace exposes the required custom-app capability or if the app is distributed through a supported publication route.

## Initial API contract

- `janus.search_registry(query, max_results=10, path_prefix=None)` — registry search with provenance.
- `janus.read_registry(path, ref="main")` — exact artifact read by path/ref.
- `janus.run_topa(claim, mode="empirical", include_foundation=True)` — falsification-first packet; no manufactured verdict.
- `janus.query_cosmos(query, max_results=10, path_prefix=None)` — Cosmos search.
- `janus.read_cosmos(path, ref="main")` — exact Cosmos artifact read.
- `janus.freeze_gate(...)` — locked write action for immutable gate checkpoints.
- `janus.write_checkpoint(...)` — locked write action for progress checkpoints.

## Deployment

The server uses MCP Streamable HTTP at `/mcp`. Deploy the provided Dockerfile to an HTTPS container host and inject environment variables as secrets.

Recommended first deployment is read-only:

```text
GITHUB_TOKEN=<fine-grained read-only token>
JANUS_ALLOW_WRITES=0
```

Only after read-only inspection passes should a separate fine-grained Contents-write token and dedicated branch be considered.
