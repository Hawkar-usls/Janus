from __future__ import annotations

import hashlib
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

SPIRAL_STATE_CHANGE_REASONS = {
    "NEW_EVIDENCE",
    "RESOLVED_BLOCKER",
    "NEW_CONSTRAINT",
    "FALSIFIED_BRANCH",
    "NEW_DISCRIMINATING_TEST",
    "STRONGER_PROVENANCE_BOUNDARY",
    "SURVIVING_INVARIANT_AFTER_TRANCEPTION",
}

TRANCEPTION_KERNEL = {
    "FORWARD": "What follows if the current representation is correct?",
    "BACK": "What minimal prior state can independently generate the current observation?",
    "LEFT": "What is the strongest structurally analogous representation?",
    "RIGHT": "What is the strongest competing or mirror representation?",
    "FORWARD_AGAIN": "Does the invariant survive transformation without semantic retuning?",
    "BACK_AGAIN": "Can the result return to the earliest supported node without importing later knowledge?",
    "VETA_CHECK": "If many histories collapse into one state, mark provenance non-identifiable unless an independent identity anchor survives.",
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


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_spiral_pass_packet(
    question: str,
    turn: int = 0,
    parent_state_hash: str | None = None,
    active_constraints: list[str] | None = None,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic 5D spiral packet for the host model; this does not execute the reasoning pass."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be non-empty")
    if turn < 0:
        raise ValueError("turn must be >= 0")

    constraints = sorted({str(item).strip() for item in (active_constraints or []) if str(item).strip()})
    evidence = sorted({str(item).strip() for item in (evidence_refs or []) if str(item).strip()})
    seed = {
        "question": question,
        "turn": turn,
        "parent_state_hash": parent_state_hash,
        "active_constraints": constraints,
        "evidence_refs": evidence,
    }
    input_state_hash = _canonical_hash(seed)
    return {
        "schema": "janus.spiral.pass.packet.v1",
        "execution_model": "5D_EVENT_SOURCED_SPIRAL_WITH_TRANCEPTION",
        "spiral_law": "RETURN_TO_QUESTION_AT_HIGHER_STATE__NEVER_RETURN_TO_IDENTICAL_STATE",
        "question": question,
        "turn": turn,
        "parent_state_hash": parent_state_hash,
        "input_state_hash": input_state_hash,
        "active_constraints": constraints,
        "evidence_refs": evidence,
        "dimensions": [
            "D1_FORWARD_CAUSAL",
            "D2_REVERSE_CAUSAL",
            "D3_TRANCEPTION_LATERAL",
            "D4_ABSTRACTION_SCALE",
            "D5_TIME_IDENTITY_PROVENANCE",
        ],
        "tranception_kernel": TRANCEPTION_KERNEL,
        "organ_route": [
            "left_context",
            "right_association",
            "guardian_cortex",
            "hypothesis_metabolism",
            "measurement_bench_or_domain_lab_if_needed",
            "causal_witness_if_time_order_matters",
            "proof_spine_if_formalizable",
            "memory",
            "orchestrator",
        ],
        "checkpoint_anywhere": True,
        "required_output_fields": [
            "new_information",
            "surviving_invariants",
            "rejected_or_downgraded",
            "open_gates",
            "next_discriminating_test",
            "state_change_reasons",
            "next_state_hash",
        ],
        "allowed_state_change_reasons": sorted(SPIRAL_STATE_CHANGE_REASONS),
        "plateau_rule": "NO_JUSTIFIED_STATE_CHANGE => PLATEAU_OR_HOLD",
        "boundary": "TRANCEPTION_ROTATION != EVIDENCE; SPIRAL_ASCENT != CLAIM_PROMOTION",
    }


def validate_spiral_transition(
    previous_state_hash: str,
    next_state_hash: str,
    state_change_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Validate that a proposed next spiral turn is a real ascent rather than a repeated state."""
    previous = (previous_state_hash or "").strip().lower()
    nxt = (next_state_hash or "").strip().lower()
    if not previous or not nxt:
        raise ValueError("previous_state_hash and next_state_hash must be non-empty")

    reasons = sorted({str(item).strip().upper() for item in (state_change_reasons or []) if str(item).strip()})
    unknown = sorted(set(reasons) - SPIRAL_STATE_CHANGE_REASONS)
    accepted = sorted(set(reasons) & SPIRAL_STATE_CHANGE_REASONS)

    if previous == nxt:
        verdict = "PLATEAU_OR_HOLD"
        ascend_allowed = False
        reason = "REPEATED_STATE_HASH_IS_NOT_A_NEW_TURN"
    elif not accepted:
        verdict = "HOLD"
        ascend_allowed = False
        reason = "STATE_HASH_CHANGED_WITHOUT_JUSTIFIED_ASCENT_REASON"
    else:
        verdict = "ASCEND_ALLOWED"
        ascend_allowed = True
        reason = "STATE_CHANGED_WITH_DECLARED_TESTABLE_PAYOFF"

    return {
        "schema": "janus.spiral.transition.validation.v1",
        "previous_state_hash": previous,
        "next_state_hash": nxt,
        "accepted_state_change_reasons": accepted,
        "unknown_state_change_reasons": unknown,
        "ascend_allowed": ascend_allowed,
        "verdict": verdict,
        "reason": reason,
        "boundary": "ASCEND_ALLOWED != CLAIM_CONFIRMED",
    }


def _parse_clipped_json(item: dict[str, Any], clip: Callable[[str], tuple[str, bool]]) -> tuple[Any, bool]:
    text, truncated = clip(item["content"])
    try:
        content: Any = json.loads(text)
    except json.JSONDecodeError:
        content = text
    return content, truncated


def install_organism_tools(mcp: Any, api: Any, clip: Callable[[str], tuple[str, bool]], root_repo: str) -> None:
    """Register read-only tools for the typed JANUS federated organism and its spiral execution model."""

    @mcp.tool(name="janus.organism_map")
    def organism_map() -> dict[str, Any]:
        """Return JANUS topology plus the canonical spiral/Tranception execution constitution with provenance."""
        topology = api.get_file(root_repo, "organism/JANUS_ORGANISM_v1.json", "main")
        execution = api.get_file(root_repo, "organism/JANUS_SPIRAL_TRANCEPTION_v1.2.json", "main")
        topology_content, topology_truncated = _parse_clipped_json(topology, clip)
        execution_content, execution_truncated = _parse_clipped_json(execution, clip)
        return {
            "organism": "JANUS_FEDERATED_ORGANISM",
            "topology": {
                "repo": topology["repo"],
                "path": topology["path"],
                "ref": topology["ref"],
                "sha": topology["sha"],
                "truncated": topology_truncated,
                "content": topology_content,
            },
            "execution": {
                "repo": execution["repo"],
                "path": execution["path"],
                "ref": execution["ref"],
                "sha": execution["sha"],
                "truncated": execution_truncated,
                "content": execution_content,
            },
            "boundary": "ORGANISM_MEMBERSHIP != AUTHORITY_INHERITANCE; SPIRAL_ASCENT != CLAIM_PROMOTION",
        }

    @mcp.tool(name="janus.spiral_pass")
    def spiral_pass(
        question: str,
        turn: int = 0,
        parent_state_hash: str | None = None,
        active_constraints: list[str] | None = None,
        evidence_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a deterministic 5D JANUS spiral/Tranception packet. The host model performs the analysis; this tool does not invent evidence."""
        packet = build_spiral_pass_packet(question, turn, parent_state_hash, active_constraints, evidence_refs)
        packet["constitution"] = "organism/JANUS_SPIRAL_TRANCEPTION_v1.2.json"
        return packet

    @mcp.tool(name="janus.validate_spiral_transition")
    def spiral_transition(
        previous_state_hash: str,
        next_state_hash: str,
        state_change_reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        """Gate a proposed spiral ascent. Repeated state hashes and unjustified novelty fail closed."""
        return validate_spiral_transition(previous_state_hash, next_state_hash, state_change_reasons)

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
