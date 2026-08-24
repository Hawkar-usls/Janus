from __future__ import annotations

import json
from typing import Any, Callable

ORGAN_REPOS: dict[str, str] = {
    "gateway": "Hawkar-usls/Janus",
    "memory": "Hawkar-usls/janus-meta-registry",
    "proof_spine": "Hawkar-usls/Janus-Fundamentum",
    "guardian_cortex": "Hawkar-usls/Demi_Head",
    "orchestrator": "Hawkar-usls/Janus-Demiurge",
    "sensorimotor_mesh": "Hawkar-usls/janus-distributed-ai-swarm",
    "hypothesis_metabolism": "Hawkar-usls/janus-lapis",
    "symbolic_imagination": "Hawkar-usls/aura-oracle-tg",
    "anomaly_lab": "Hawkar-usls/TOPA",
    "observatory": "Hawkar-usls/Janus-Cosmos",
}

ORGAN_BOUNDARIES: dict[str, str] = {
    "gateway": "ROUTING != TRUTH",
    "memory": "REGISTRY_PRESENCE != TRUTH",
    "proof_spine": "SCOPED_RESULT != UNIVERSAL_RESULT",
    "guardian_cortex": "REVIEW != WORLD_TRUTH",
    "orchestrator": "DISPATCH != AUTHORITY",
    "sensorimotor_mesh": "TELEMETRY != PREDICTION",
    "hypothesis_metabolism": "RANKING != VALIDATION",
    "symbolic_imagination": "SYMBOLIC_OUTPUT != EMPIRICAL_EVIDENCE",
    "anomaly_lab": "UNRESOLVED != EXTRAORDINARY",
    "observatory": "SIGNAL_CANDIDATE != TECHNOSIGNATURE",
}


def _resolve_organ(organ: str) -> tuple[str, str]:
    key = (organ or "").strip().lower()
    repo = ORGAN_REPOS.get(key)
    if repo is None:
        raise ValueError(f"unknown JANUS organ: {organ!r}; allowed={sorted(ORGAN_REPOS)}")
    return key, repo


def install_organism_tools(mcp: Any, api: Any, clip: Callable[[str], tuple[str, bool]], root_repo: str) -> None:
    """Register read-only tools for the typed JANUS federated organism."""

    @mcp.tool(name="janus.organism_map")
    def organism_map() -> dict[str, Any]:
        """Return the canonical JANUS organism manifest with provenance."""
        item = api.get_file(root_repo, "organism/JANUS_ORGANISM_v1.json", "main")
        text, truncated = clip(item["content"])
        try:
            content: Any = json.loads(text)
        except json.JSONDecodeError:
            content = text
        return {
            "repo": item["repo"],
            "path": item["path"],
            "ref": item["ref"],
            "sha": item["sha"],
            "truncated": truncated,
            "content": content,
            "boundary": "ORGANISM_MEMBERSHIP != AUTHORITY_INHERITANCE",
        }

    @mcp.tool(name="janus.search_organ")
    def search_organ(organ: str, query: str, max_results: int = 10, path_prefix: str | None = None) -> dict[str, Any]:
        """Search one allowlisted JANUS organ by typed organ key; arbitrary repositories are not accepted."""
        key, repo = _resolve_organ(organ)
        return {
            "organ": key,
            "repository": repo,
            "query": query,
            "results": api.search_code(repo, query, max_results, path_prefix),
            "boundary": ORGAN_BOUNDARIES[key],
        }

    @mcp.tool(name="janus.read_organ")
    def read_organ(organ: str, path: str, ref: str = "main") -> dict[str, Any]:
        """Read one text/JSON artifact from an allowlisted JANUS organ with exact repository/ref/SHA provenance."""
        key, repo = _resolve_organ(organ)
        item = api.get_file(repo, path, ref)
        item["content"], item["truncated"] = clip(item["content"])
        item["organ"] = key
        item["boundary"] = ORGAN_BOUNDARIES[key]
        return item
