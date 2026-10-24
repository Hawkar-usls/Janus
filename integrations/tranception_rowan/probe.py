#!/usr/bin/env python3
"""JANUS Tranception/ProteinGym -> Rowan Boltz-2 structure-probe gateway.

Only structurally predictive. Never interprets Boltz pTM/pLDDT as fitness.
Offline by default; real jobs require environment credentials and explicit
per-job credit authorization. One arm per invocation, no auto-retry.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST = Path(__file__).with_name("discovery_manifest.json")
AMINO = set("ACDEFGHIKLMNPQRSTVWY")
SCHEMA = "janus.tranception.rowan.probe_receipt.v1"
NAME_PREFIX = "JANUS-TRANCEPTION-ROWAN-BBC1-"


def digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(
        data, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def prepare(manifest: dict, arm: str) -> dict:
    archive = manifest["archive"]
    control = manifest["independent_control"]
    files = archive["files"]
    if (archive["match_count"] != 22 or len(files) != 22 or
            archive["tree_truncated"] or len({x["path"] for x in files}) != 22):
        raise ValueError("Archive corpus not the frozen 22-file index")
    if any(not re.fullmatch(r"[0-9a-f]{40}", x["git_blob_sha"]) for x in files):
        raise ValueError("Invalid immutable source blob SHA")
    if manifest["origin"]["class"] != "UPSTREAM_DERIVED_WORKING_COPY":
        raise ValueError("Cannot assert original authorship")
    seq = control["wildtype"]
    if (not isinstance(seq, str) or len(seq) != control["length"]
            or not set(seq).issubset(AMINO)):
        raise ValueError("Invalid anchored protein sequence")
    if control["mutation"] != "K5A" or seq[4] != "K":
        raise ValueError("Expected exact preregistered K5A position")
    mutant = seq[:4] + "A" + seq[5:]
    if control["mutant"] != mutant:
        raise ValueError("Mutant mismatch")
    if arm not in {"wildtype", "K5A"}:
        raise ValueError("Unknown protein arm")
    if (control["mutation_label_proven_experimentally"] or
            control["experimental_fitness_labels_available_in_manifest"]):
        raise ValueError("Control evidence provenance changed")
    return {
        "id": NAME_PREFIX + arm,
        "sequence": seq if arm == "wildtype" else mutant,
        "arm": arm,
        "reference": control["assay_id"],
        "reference_blob_sha": control["reference_blob_sha"],
    }


def execute(manifest: dict, arm: str, mode: str, *,
            max_credits: int = 1, authorize_credit_use: bool = False,
            workflow_id: str | None = None, sdk: Any = None) -> dict:
    case = prepare(manifest, arm)
    if type(max_credits) is not int or not 1 <= max_credits <= 10:
        raise ValueError("max_credits must be an integer in range 1..10")
    receipt = {
        "schema_id": SCHEMA,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": digest(manifest),
        "provider": "Rowan",
        "workflow_type": "protein_cofolding",
        "model": "boltz-2",
        "arm": arm,
        "case_name": case["id"],
        "sequence_sha256": hashlib.sha256(case["sequence"].encode()).hexdigest(),
        "sequence_length": len(case["sequence"]),
        "source_assay": case["reference"],
        "source_blob_sha": case["reference_blob_sha"],
        "max_credits": max_credits,
        "workflow_uuid": None,
        "event": "OFFLINE_PLAN_ONLY",
        "structural_confidence": None,
        "tranception_model_run": False,
        "protein_fitness_measured": False,
        "biological_fitness_inferred": False,
        "topa_review": "NOT_DONE",
        "registry_published": False,
    }
    if mode == "plan":
        return receipt
    if mode not in {"auth", "draft", "submit", "collect"}:
        raise ValueError("Unknown mode")
    if mode == "submit" and not authorize_credit_use:
        raise PermissionError("Explicit credit authorization required")
    if mode != "submit" and authorize_credit_use:
        raise ValueError("Credit authorization is only valid with submit")
    if mode == "collect" and not workflow_id:
        raise ValueError("Collect must identify an existing workflow ID")
    if mode != "collect" and workflow_id:
        raise ValueError("Unexpected workflow ID")
    if not os.environ.get("ROWAN_API_KEY"):
        raise PermissionError("ROWAN_API_KEY missing from secured runtime")
    api = sdk if sdk is not None else importlib.import_module("rowan")
    if mode == "auth":
        user = api.whoami()
        receipt["event"] = "ROWAN_AUTH_CHECKED_NO_CHEMISTRY_JOB"
        receipt["pka_available"] = "pka" in (
            getattr(user, "enabled_workflows", []) or [])
        return receipt
    if mode in {"draft", "submit"}:
        wf = api.submit_protein_cofolding_workflow(
            initial_protein_sequences=[case["sequence"]],
            model="boltz-2",
            num_samples=1,
            use_msa_server=True,
            max_credits=max_credits,
            name=case["id"],
            is_draft=(mode == "draft"),
        )
        receipt["event"] = (
            "ROWAN_DRAFT_CREATED_NO_COMPUTATION" if mode == "draft"
            else "ROWAN_SUBMITTED_RESULT_PENDING")
    else:
        wf = api.retrieve_workflow(workflow_id, result_type=api.ProteinCofoldingResult)
        if wf.name != case["id"] or wf.workflow_type != "protein_cofolding":
            raise ValueError("Wrong case name or wrong workflow type")
        wf = wf.fetch_latest()
        receipt["event"] = "ROWAN_RESULT_PENDING"
        if wf.done():
            result = wf.result(wait=False)
            scores = getattr(result, "scores", None)
            receipt["structural_confidence"] = {
                "ptm": getattr(scores, "ptm", None),
                "avg_lddt": getattr(scores, "avg_lddt", None),
                "kind": "STRUCTURE_CONFIDENCE_NOT_MUTATION_FITNESS",
            }
            receipt["event"] = "ROWAN_STRUCTURE_PREDICTION_COLLECTED_UNVERIFIED"
    receipt["workflow_uuid"] = str(wf.uuid)
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--arm", choices=["wildtype", "K5A"], required=True)
    p.add_argument("--mode", choices=["plan", "auth", "draft", "submit", "collect"],
                   default="plan")
    p.add_argument("--max-credits", type=int, default=1)
    p.add_argument("--authorize-credit-use", action="store_true")
    p.add_argument("--workflow-id")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Receipt exists: refusing to overwrite or duplicate submission")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        receipt = execute(
            manifest, args.arm, args.mode, max_credits=args.max_credits,
            authorize_credit_use=args.authorize_credit_use,
            workflow_id=args.workflow_id,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        print("JANUS_ROWAN_TRANCEPTION=" + receipt["event"])
        return 0
    except (OSError, PermissionError, ValueError) as err:
        print("JANUS_ROWAN_BLOCKED=" + type(err).__name__, file=sys.stderr)
        return 2
    except Exception:
        print("JANUS_ROWAN_REMOTE_ERROR_NO_CREDENTIALS_LOGGED", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
