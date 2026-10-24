#!/usr/bin/env python3
"""Offline/opt-in JANUS Rowan → TOPA → Meta Registry candidate pipeline.

No automatic external publication or scientific-status promotion.
Use a NEW output directory; never modify an existing receipt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from janus_rowan_bridge import canonical_bytes, operate, sha256


def render_envelopes(case: dict, receipt: dict) -> dict[str, dict]:
    """Construct a hash-linked review package. Every status remains unverified."""
    receipt_hash = sha256(receipt)
    topa = {
        "schema_id": "janus.topa.rowan_review_request.v1",
        "case_id": receipt["case_id"],
        "source": receipt["source"],
        "canonical_method": "https://github.com/Hawkar-usls/TOPA/blob/main/protocols/TOPA_FOUNDATION.json",
        "status": "AWAITING_INDEPENDENT_REVIEW",
        "rowan_receipt_sha256": receipt_hash,
        "hypothesis": receipt["research"]["hypothesis"],
        "alternative": receipt["research"]["alternative"],
        "falsifier": receipt["research"]["falsifier"],
        "required_gates": [
            "Verify source identity, consent, rights, and original conditions",
            "Reconstruct exact molecular input from a private trustworthy record",
            "Compare with an independent experimental reference",
            "Examine model validity range, uncertainty, and alternative explanations",
            "Preserve negative and unresolved results",
        ],
        "science_claim": "HYPOTHESIS_ONLY",
        "reviewer_identity": None,
        "review_outcome": None,
    }
    topa_hash = sha256(topa)
    registry = {
        "schema_id": "janus.meta_registry.rowan_candidate.v1",
        "artifact_id": receipt["case_id"] + "-ROWAN-CANDIDATE",
        "source_system": "JANUS_GOLEM_AURA_TOPA",
        "status": "REVIEW_REQUIRED_NOT_IMPORTED",
        "rowan_workflow_uuid": receipt["workflow_uuid"],
        "rowan_event": receipt["event"],
        "input_sha256": receipt["input_sha256"],
        "rowan_receipt_sha256": receipt_hash,
        "topa_request_sha256": topa_hash,
        "source_reference": receipt["source"]["reference"],
        "evidence_class": "HYPOTHESIS_ONLY",
        "automatic_evidence_promotion": False,
        "automatic_publication": False,
        "registry_ref": "https://github.com/Hawkar-usls/janus-meta-registry",
        "notes": (
            "This is an import candidate, NOT a validated experiment or a "
            "published archive entry. No scientific promotion without TOPA review."
        ),
    }
    return {
        "rowan.receipt.json": receipt,
        "topa.review-request.json": topa,
        "meta-registry.candidate.json": registry,
    }


def write_package(output_dir: Path, envelopes: dict[str, dict]) -> None:
    """Best effort transactional local staging; refuses overwrites."""
    out = output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    parent = out.parent
    if not parent.is_dir():
        raise FileNotFoundError("Output parent directory does not exist")
    staging = Path(tempfile.mkdtemp(prefix=".janus-rowan-stage-", dir=parent))
    try:
        os.chmod(staging, 0o700)
        for name, payload in envelopes.items():
            file = staging / name
            fd = os.open(str(file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical_bytes(payload) + b"\n")
        if out.exists():
            raise FileExistsError("Destination appeared during staging")
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["plan", "auth", "draft", "submit", "collect"],
                   default="plan")
    p.add_argument("--authorize-credit-use", action="store_true")
    p.add_argument("--workflow-id")
    args = p.parse_args()
    if args.authorize_credit_use and args.mode != "submit":
        p.error("Only a submit can use --authorize-credit-use")
    if args.mode == "collect" and not args.workflow_id:
        p.error("Collection requires an existing workflow UUID")
    try:
        with args.case.open(encoding="utf-8") as f:
            case = json.load(f)
        receipt = operate(
            case, args.mode, authorize_spend=args.authorize_credit_use,
            workflow_id=args.workflow_id,
        )
        write_package(args.output_dir, render_envelopes(case, receipt))
        print("JANUS_ROWAN_PIPELINE=" + receipt["event"])
        print("TOPA_REVIEW=REQUIRED")
        print("META_REGISTRY=NOT_IMPORTED")
        return 0
    except (OSError, PermissionError, ValueError) as exc:
        # Never print raw provider HTTP exceptions; keep tokens out of logs.
        print("JANUS_ROWAN_PIPELINE_BLOCKED=" + type(exc).__name__)
        return 2
    except Exception:
        print("JANUS_ROWAN_PIPELINE_REMOTE_ERROR")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
