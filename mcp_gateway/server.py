from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from mcp.server import MCPServer

from .core import checkpoint_document, safe_slug, topa_packet
from .github_api import GitHubAPI

GITHUB_OWNER = os.getenv("JANUS_GITHUB_OWNER", "Hawkar-usls")
REGISTRY_REPO = os.getenv("JANUS_REGISTRY_REPO", f"{GITHUB_OWNER}/janus-meta-registry")
TOPA_REPO = os.getenv("JANUS_TOPA_REPO", f"{GITHUB_OWNER}/TOPA")
COSMOS_REPO = os.getenv("JANUS_COSMOS_REPO", f"{GITHUB_OWNER}/Janus-Cosmos")
CHECKPOINT_REPO = os.getenv("JANUS_CHECKPOINT_REPO", f"{GITHUB_OWNER}/Janus")
WRITE_BRANCH = os.getenv("JANUS_WRITE_BRANCH", "mcp-checkpoints")
ALLOW_WRITES = os.getenv("JANUS_ALLOW_WRITES", "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_TEXT_BYTES = max(1024, int(os.getenv("JANUS_MAX_TEXT_BYTES", "120000")))

ALLOWED_REPOS = {REGISTRY_REPO, TOPA_REPO, COSMOS_REPO, CHECKPOINT_REPO}

mcp = MCPServer(
    "JANUS MCP Gateway",
    instructions=(
        "JANUS research gateway. Preserve provenance and claim boundaries. "
        "Repository presence is not evidence of truth. Prefer read tools. "
        "Write tools are explicitly gated and must never be used to fabricate evidence."
    ),
)
api = GitHubAPI()


def _clip(text: str) -> tuple[str, bool]:
    data = text.encode("utf-8")
    if len(data) <= MAX_TEXT_BYTES:
        return text, False
    clipped = data[:MAX_TEXT_BYTES].decode("utf-8", errors="ignore")
    return clipped, True


def _write_guard() -> None:
    if not ALLOW_WRITES:
        raise PermissionError(
            "JANUS write tools are locked. Set JANUS_ALLOW_WRITES=1 and provide a fine-grained GITHUB_TOKEN "
            "with contents:write only for the designated checkpoint repository."
        )
    if CHECKPOINT_REPO not in ALLOWED_REPOS:
        raise PermissionError("checkpoint repository is outside the JANUS allowlist")


@mcp.tool(name="janus.health")
def health() -> dict[str, Any]:
    """Return gateway configuration without exposing secrets."""
    return {
        "service": "JANUS MCP Gateway",
        "version": "0.1.0",
        "transport": "streamable-http",
        "write_enabled": ALLOW_WRITES,
        "write_branch": WRITE_BRANCH,
        "repos": {"registry": REGISTRY_REPO, "topa": TOPA_REPO, "cosmos": COSMOS_REPO, "checkpoint": CHECKPOINT_REPO},
        "github_token_present": bool(os.getenv("GITHUB_TOKEN")),
    }


@mcp.tool(name="janus.search_registry")
def search_registry(query: str, max_results: int = 10, path_prefix: str | None = None) -> dict[str, Any]:
    """Search the JANUS meta-registry by code/text index. Read-only; returns provenance paths, not truth claims."""
    return {"query": query, "repository": REGISTRY_REPO, "results": api.search_code(REGISTRY_REPO, query, max_results, path_prefix), "boundary": "SEARCH_MATCH != VALIDATED_CLAIM"}


@mcp.tool(name="janus.read_registry")
def read_registry(path: str, ref: str = "main") -> dict[str, Any]:
    """Read one UTF-8 text/JSON artifact from the JANUS meta-registry with SHA provenance."""
    item = api.get_file(REGISTRY_REPO, path, ref)
    item["content"], item["truncated"] = _clip(item["content"])
    item["boundary"] = "ARCHIVED_SOURCE != TRUSTED_SOURCE"
    return item


@mcp.tool(name="janus.run_topa")
def run_topa(claim: str, mode: str = "empirical", include_foundation: bool = True) -> dict[str, Any]:
    """Build a falsification-first TOPA analysis packet for the host model. It does not declare the claim true or false."""
    packet = topa_packet(claim, mode)
    packet["repository"] = TOPA_REPO
    packet["boundary"] = "TOPA PACKET != WORLD-TRUTH VERDICT"
    if include_foundation:
        foundation = api.get_file(TOPA_REPO, "protocols/TOPA_FOUNDATION.json", "main")
        foundation_text, truncated = _clip(foundation["content"])
        try:
            parsed: Any = json.loads(foundation_text)
        except json.JSONDecodeError:
            parsed = foundation_text
        packet["foundation"] = {"path": foundation["path"], "sha": foundation["sha"], "html_url": foundation["html_url"], "truncated": truncated, "content": parsed}
    return packet


@mcp.tool(name="janus.query_cosmos")
def query_cosmos(query: str, max_results: int = 10, path_prefix: str | None = None) -> dict[str, Any]:
    """Search Janus-Cosmos for evidence, configs, receipts or experiment artifacts. Read-only."""
    return {"query": query, "repository": COSMOS_REPO, "results": api.search_code(COSMOS_REPO, query, max_results, path_prefix), "boundary": "SEARCH_MATCH != TECHNOSIGNATURE OR ANOMALY"}


@mcp.tool(name="janus.read_cosmos")
def read_cosmos(path: str, ref: str = "main") -> dict[str, Any]:
    """Read one Janus-Cosmos artifact from an explicit branch/ref with SHA provenance."""
    item = api.get_file(COSMOS_REPO, path, ref)
    item["content"], item["truncated"] = _clip(item["content"])
    item["boundary"] = "ARTIFACT != INDEPENDENT REPLICATION"
    return item


@mcp.tool(name="janus.freeze_gate")
def freeze_gate(gate_id: str, statement: str, evidence_paths: list[str] | None = None, status: str = "FROZEN", falsifier: str | None = None) -> dict[str, Any]:
    """WRITE: freeze a named JANUS gate as an immutable checkpoint JSON on the dedicated write branch."""
    _write_guard()
    doc = checkpoint_document(kind="gate", project="JANUS", summary=statement, state=status, evidence=evidence_paths, next_action=falsifier, metadata={"gate_id": gate_id, "falsifier": falsifier})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"mcp/checkpoints/gates/{stamp}-{safe_slug(gate_id, 'gate')}.json"
    result = api.put_file(CHECKPOINT_REPO, path, json.dumps(doc, ensure_ascii=False, indent=2) + "\n", f"MCP freeze gate: {gate_id}", WRITE_BRANCH)
    return {"checkpoint": doc, "write": result}


@mcp.tool(name="janus.write_checkpoint")
def write_checkpoint(project: str, summary: str, state: str, evidence: list[str] | None = None, next_action: str | None = None) -> dict[str, Any]:
    """WRITE: persist a proof-carrying progress checkpoint on the dedicated JANUS write branch."""
    _write_guard()
    doc = checkpoint_document(kind="progress", project=project, summary=summary, state=state, evidence=evidence, next_action=next_action)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"mcp/checkpoints/progress/{stamp}-{safe_slug(project, 'project')}.json"
    result = api.put_file(CHECKPOINT_REPO, path, json.dumps(doc, ensure_ascii=False, indent=2) + "\n", f"MCP checkpoint: {project}", WRITE_BRANCH)
    return {"checkpoint": doc, "write": result}


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("MCP_PORT", "8000")))
    mcp.run(transport="streamable-http", host=host, port=port, streamable_http_path="/mcp")
