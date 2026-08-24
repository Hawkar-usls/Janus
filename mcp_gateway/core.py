from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_repo_path(path: str) -> str:
    """Return a safe GitHub repository-relative path or raise ValueError."""
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    path = path.strip().replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")
    path = path.lstrip("/")
    if not path or path in {".", ".."}:
        raise ValueError("path must not be empty")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path traversal is not allowed")
    return "/".join(parts)


def safe_slug(value: str, fallback: str = "item", max_len: int = 80) -> str:
    value = _SAFE_NAME.sub("-", (value or "").strip()).strip("-._")
    value = value or fallback
    return value[:max_len]


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def checkpoint_document(
    *,
    kind: str,
    project: str,
    summary: str,
    state: str,
    evidence: Iterable[str] | None = None,
    next_action: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": "janus.mcp.checkpoint.v1",
        "kind": kind,
        "project": project,
        "summary": summary,
        "state": state,
        "evidence": list(evidence or []),
        "next_action": next_action,
        "metadata": metadata or {},
        "created_at": utc_now_iso(),
    }
    doc["content_sha256"] = sha256_json(doc)
    return doc


def topa_packet(claim: str, mode: str = "empirical") -> dict[str, Any]:
    mode = (mode or "empirical").strip().lower()
    if mode not in {"empirical", "mathematical"}:
        raise ValueError("mode must be 'empirical' or 'mathematical'")
    common = {
        "claim": claim.strip(),
        "mode": mode,
        "epistemic_laws": [
            "ANOMALY IS A QUESTION, NOT A CONCLUSION",
            "UNKNOWN != SUPERNATURAL",
            "NOT_REFUTED != TRUE",
            "MULTI_CHANNEL != MULTI_SOURCE",
            "MISSING_DATA_STAYS_MISSING",
            "I_DO_NOT_KNOW = VALID OUTPUT",
        ],
    }
    if mode == "empirical":
        common["host_workflow"] = [
            "preserve_raw_provenance",
            "freeze_time_location_claim",
            "split_observation_from_interpretation",
            "build_competing_hypotheses",
            "test_mundane_models_early",
            "name_falsifiers",
            "check_source_and_sensor_independence",
            "attack_both_sides",
            "update_confidence",
            "resolve_falsify_or_keep_unresolved",
            "spiral_and_reattack",
        ]
        common["required_output"] = {
            "observation": None,
            "interpretation": None,
            "provenance": [],
            "competing_hypotheses": [],
            "falsifiers": [],
            "independence_audit": None,
            "evidence_level": "O0",
            "status": "UNRESOLVED",
        }
    else:
        common["host_workflow"] = [
            "freeze_claim",
            "normalize_definitions_and_parameters",
            "verify_encoding_and_object_identity",
            "search_proof_and_counterexample_routes",
            "charge_full_algorithmic_and_certificate_cost",
            "classify_proved_refuted_or_open",
            "ascend_and_reattack",
        ]
        common["required_output"] = {
            "definitions": [],
            "parameters": [],
            "proof_routes": [],
            "counterexample_routes": [],
            "cost_audit": None,
            "status": "OPEN",
        }
    return common
