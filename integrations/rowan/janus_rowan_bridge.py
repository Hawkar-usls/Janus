#!/usr/bin/env python3
"""JANUS ↔ Rowan pKa bridge. Safe-by-default research adapter (v0.1).

PLAN is entirely local and requires no Rowan package, account, or network.
REMOTE_DRAFT creates a Rowan draft but never executes it.
SUBMIT requires an explicit credit-spend flag and a per-workflow credit ceiling.
COLLECT reads an existing workflow without starting a new computation.
No operation publishes to TOPA or Meta Registry automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METHODS = {
    "starling", "chemprop_nevolianis2025",
    "aimnet2_wagen2024", "gxtb_wagen2026",
}
SCHEMA = "janus.rowan.research_receipt.v1"
MAX_CREDIT_CAP = 10  # Upper policy cap; explicit lower per-case ceiling required.


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_str(obj: dict, key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid non-empty field: {key}")
    return value.strip()


def validate(case: dict) -> dict:
    if not isinstance(case, dict):
        raise ValueError("Case must be a JSON object")
    case_id = require_str(case, "case_id")
    hypothesis = require_str(case, "hypothesis")
    alternative = require_str(case, "alternative")
    falsifier = require_str(case, "falsifier")
    source = case.get("source")
    molecule = case.get("molecule")
    workflow = case.get("workflow")
    if not isinstance(source, dict) or not isinstance(molecule, dict):
        raise ValueError("source and molecule must be objects")
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be an object")
    system = require_str(source, "system")
    if system not in {"TOPA", "AURA", "JANUS"}:
        raise ValueError("Unknown source system")
    reference = require_str(source, "reference")
    smiles = require_str(molecule, "smiles")
    if len(smiles) > 2048 or any(ord(c) < 32 for c in smiles):
        raise ValueError("Invalid SMILES text length/controls")
    # Rowan, not this validator, decides chemical validity.
    if workflow.get("type") != "pka":
        raise ValueError("v0.1 only supports pka; never silently substitute workflows")
    method = workflow.get("method", "starling")
    if method not in METHODS:
        raise ValueError("Unrecognized pKa method")
    rng = workflow.get("pka_range", [2, 12])
    if (not isinstance(rng, list) or len(rng) != 2
            or any(type(v) is not int for v in rng)
            or not -30 <= rng[0] < rng[1] <= 50):
        raise ValueError("Invalid pka_range")
    credits = workflow.get("max_credits")
    if type(credits) is not int or not 1 <= credits <= MAX_CREDIT_CAP:
        raise ValueError("Explicit max_credits required (integer from 1 to 10)")
    return {
        "case_id": case_id, "hypothesis": hypothesis,
        "alternative": alternative, "falsifier": falsifier,
        "source": {"system": system, "reference": reference},
        "smiles": smiles, "method": method,
        "pka_range": rng, "max_credits": credits,
    }


def base_receipt(case: dict, params: dict) -> dict:
    return {
        "schema_id": SCHEMA,
        "event_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": params["case_id"],
        "input_sha256": sha256(case),
        "source": params["source"],
        "research": {
            "hypothesis": params["hypothesis"],
            "alternative": params["alternative"],
            "falsifier": params["falsifier"],
            "scientific_status": "UNVERIFIED",
            "topa_review": "REQUIRED",
        },
        "provider": {
            "name": "Rowan",
            "api_docs": "https://docs.rowansci.com/api/python/v3/",
            "workflow_type": "pka",
            "method": params["method"],
            "smiles": params["smiles"],
            "pka_range": params["pka_range"],
            "max_credits": params["max_credits"],
        },
        "meta_registry": {
            "authority": "CANDIDATE_FOR_HUMAN_REVIEW",
            "auto_published": False,
            "note": "A workflow receipt is not evidence of molecular or clinical validity.",
        },
        "event": "PLANNED_NO_EXTERNAL_EFFECT",
        "workflow_uuid": None,
        "result": None,
    }


def operate(case: dict, mode: str = "plan", *, authorize_spend: bool = False,
            workflow_id: str | None = None, sdk: Any = None) -> dict:
    params = validate(case)
    receipt = base_receipt(case, params)
    if mode == "plan":
        return receipt
    if mode not in {"draft", "submit", "collect"}:
        raise ValueError("Unknown mode")
    if mode == "submit" and not authorize_spend:
        raise PermissionError("Submission blocked: --authorize-credit-use required")
    if mode == "collect" and not workflow_id:
        raise ValueError("Collect requires --workflow-id")
    if not os.environ.get("ROWAN_API_KEY"):
        raise PermissionError("ROWAN_API_KEY is required for remote operations")

    api = sdk if sdk is not None else importlib.import_module("rowan")
    if mode in {"draft", "submit"}:
        wf = api.submit_pka_workflow(
            initial_molecule=params["smiles"],
            method=params["method"],
            pka_range=tuple(params["pka_range"]),
            max_credits=params["max_credits"],
            name="JANUS " + params["case_id"][:90],
            is_draft=(mode == "draft"),
        )
        receipt["event"] = (
            "ROWAN_DRAFT_CREATED_NO_EXECUTION"
            if mode == "draft" else "ROWAN_SUBMITTED_RESULT_UNVERIFIED"
        )
    else:
        wf = api.retrieve_workflow(workflow_id)
        # Reject unrelated jobs; keep the claim boundary explicit.
        if getattr(wf, "workflow_type", None) != "pka":
            raise ValueError("Retrieved Rowan workflow is not pka")
        wf = wf.fetch_latest()
        receipt["event"] = "ROWAN_RESULT_PENDING"
        if wf.done():
            result = wf.result(wait=False)
            receipt["result"] = {
                "strongest_acid_pka": getattr(result, "strongest_acid", None),
                "strongest_base_pka": getattr(result, "strongest_base", None),
                "kind": "MODEL_PREDICTION_NOT_EXPERIMENT",
            }
            receipt["event"] = "ROWAN_RESULT_COLLECTED_UNVERIFIED"
    receipt["workflow_uuid"] = str(wf.uuid)
    receipt["provider"]["rowan_status"] = str(getattr(wf, "status", "UNKNOWN"))
    receipt["provider"]["credits_charged_reported"] = getattr(
        wf, "credits_charged", None
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True, help="Private case JSON input")
    p.add_argument("--output", type=Path, required=True,
                   help="Write-once JSON receipt; never auto-publishes")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--create-draft", action="store_true")
    group.add_argument("--submit", action="store_true")
    group.add_argument("--collect", action="store_true")
    p.add_argument("--authorize-credit-use", action="store_true")
    p.add_argument("--workflow-id")
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Output exists: refusing to overwrite receipt or resubmit")
    if args.authorize_credit_use and not args.submit:
        p.error("--authorize-credit-use applies only to --submit")
    mode = ("draft" if args.create_draft else "submit" if args.submit
            else "collect" if args.collect else "plan")
    try:
        with args.case.open(encoding="utf-8") as f:
            case = json.load(f)
        receipt = operate(
            case, mode, authorize_spend=args.authorize_credit_use,
            workflow_id=args.workflow_id,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
    except (ValueError, PermissionError, OSError) as exc:
        # Don't print remote server responses or tokens to logs.
        print(f"JANUS_ROWAN_BLOCKED: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("JANUS_ROWAN_REMOTE_ERROR: review credentials/SDK/network privately",
              file=sys.stderr)
        return 3
    print(f"JANUS_ROWAN_EVENT={receipt['event']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
