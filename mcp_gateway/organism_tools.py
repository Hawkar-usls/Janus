from __future__ import annotations

import json
import os
from typing import Any, Callable

# Public/default MCP-visible organs. Repository names here are an explicit allowlist.
ORGAN_REPOS: dict[str, str] = {
    "gateway": "Hawkar-usls/Janus",
    "memory": "Hawkar-usls/janus-meta-registry",
    "proof_spine": "Hawkar-usls/Janus-Fundamentum",
    "causal_witness": "Hawkar-usls/AIFC",
    "guardian_cortex": "Hawkar-usls/Demi_Head",
    "orchestrator": "Hawkar-usls/Janus-Demiurge",
    "operator_hands": "Hawkar-usls/-Terminal-for-Janus",
    "left_context": "Hawkar-usls/Hrain",
    "right_association": "Hawkar-usls/iNaiHR",
    "measurement_bench": "Hawkar-usls/janus-io-public",
    "sensorimotor_mesh": "Hawkar-usls/janus-distributed-ai-swarm",
    "hypothesis_metabolism": "Hawkar-usls/janus-lapis",
    "symbolic_imagination": "Hawkar-usls/aura-oracle-tg",
    "anomaly_lab": "Hawkar-usls/TOPA",
    "observatory": "Hawkar-usls/Janus-Cosmos",
    "simulation_habitat": "Hawkar-usls/Janus_Genesis",
    "voice": "Hawkar-usls/The-Voice-of-Janus",
    "physical_voice": "Hawkar-usls/Echo-Pyramid",
    "materials_lab": "Hawkar-usls/SCOBY-D0",
    "feline_timing_lab": "Hawkar-usls/Fast-CAT-SHAiTan",
    "social_membrane": "Hawkar-usls/janus-first-followers-club-",
}

# Private organ locators are never hard-coded into the public gateway source.
# A deployment must bind them explicitly and provide credentials that can read them.
PRIVATE_ORGAN_ENVS: dict[str, str] = {
    "somatosensory_skin": "JANUS_PRIVATE_SOMATOSENSORY_REPO",
}

ORGAN_BOUNDARIES: dict[str, str] = {
    "gateway": "ROUTING != TRUTH",
    "memory": "REGISTRY_PRESENCE != TRUTH",
    "proof_spine": "SCOPED_RESULT != UNIVERSAL_RESULT",
    "causal_witness": "CAUSAL_ORDER_PASS != PHYSICAL_MECHANISM",
    "guardian_cortex": "REVIEW != WORLD_TRUTH",
    "orchestrator": "DISPATCH != AUTHORITY",
    "operator_hands": "COMMAND != EVIDENCE",
    "left_context": "STRUCTURE != COMMAND",
    "right_association": "ASSOCIATION != EVIDENCE",
    "measurement_bench": "MEASUREMENT != MECHANISM",
    "sensorimotor_mesh": "TELEMETRY != PREDICTION",
    "hypothesis_metabolism": "RANKING != VALIDATION",
    "symbolic_imagination": "SYMBOLIC_OUTPUT != EMPIRICAL_EVIDENCE",
    "anomaly_lab": "UNRESOLVED != EXTRAORDINARY",
    "observatory": "SIGNAL_CANDIDATE != TECHNOSIGNATURE",
    "simulation_habitat": "SIMULATION != WORLD_EVIDENCE",
    "voice": "ACOUSTIC_MODEL != MEASURED_RESONANCE",
    "physical_voice": "DEVICE_OUTPUT != SCIENTIFIC_CLAIM",
    "somatosensory_skin": "SENSOR_OUTPUT != CALIBRATED_MEASUREMENT",
    "materials_lab": "MATERIAL_CANDIDATE != FIELD_VALIDATION",
    "feline_timing_lab": "CANDIDATE_GEOMETRY != BIOLOGICAL_EVENT_TRUTH",
    "social_membrane": "PUBLIC_HANDSHAKE != CONSENT_OR_AUTHORITY",
}


def _resolve_organ(organ: str) -> tuple[str, str]:
    key = (organ or "").strip().lower()
    repo = ORGAN_REPOS.get(key)
    if repo is not None:
        return key, repo

    env_name = PRIVATE_ORGAN_ENVS.get(key)
    if env_name is not None:
        private_repo = (os.getenv(env_name) or "").strip()
        if not private_repo:
            raise ValueError(
                f"private JANUS organ {key!r} is not bound; set {env_name} explicitly on the MCP deployment"
            )
        if "/" not in private_repo or private_repo.startswith("/") or private_repo.endswith("/"):
            raise ValueError(f"invalid private JANUS repository locator in {env_name}")
        return key, private_repo

    allowed = sorted(set(ORGAN_REPOS) | set(PRIVATE_ORGAN_ENVS))
    raise ValueError(f"unknown JANUS organ: {organ!r}; allowed={allowed}")


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
            "private_binding": key in PRIVATE_ORGAN_ENVS,
        }

    @mcp.tool(name="janus.read_organ")
    def read_organ(organ: str, path: str, ref: str = "main") -> dict[str, Any]:
        """Read one text/JSON artifact from an allowlisted JANUS organ with exact repository/ref/SHA provenance."""
        key, repo = _resolve_organ(organ)
        item = api.get_file(repo, path, ref)
        item["content"], item["truncated"] = clip(item["content"])
        item["organ"] = key
        item["boundary"] = ORGAN_BOUNDARIES[key]
        item["private_binding"] = key in PRIVATE_ORGAN_ENVS
        return item
