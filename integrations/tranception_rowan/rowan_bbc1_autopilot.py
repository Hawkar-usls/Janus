#!/usr/bin/env python3
"""JANUS BBC1 two-arm Rowan Boltz-2 autopilot, tightly budgeted.

Only two preregistered distinct protein cases; one submit max per run, one
submit max per case EVER without manual scientific reauthorization.
Only spend from sufficiently large *weekly* credit pool (priority 1/2
in Rowan docs); never rely on unidentified nonweekly purchased/promo pool.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import os
from pathlib import Path

from probe import prepare, TARGET_ROWAN_FOLDER_UUID
from budget_gate import finite_nonnegative

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/"discovery_manifest.json"
CREDIT_CAP=10
WEEKLY_FLOOR=30
CASES=("wildtype","K5A")
PAGE_SIZE=100
MAX_PAGES=15

class BudgetHold(RuntimeError):
    pass

def require_budget(user):
    org=getattr(user,"organization",None)
    if org is None:
        raise BudgetHold("NO_ORGANIZATION_WEEKLY_POOL")
    weekly=finite_nonnegative(getattr(org,"weekly_credits",None))
    if weekly is None or weekly < WEEKLY_FLOOR:
        raise BudgetHold("WEEKLY_POOL_BELOW_SAFETY_FLOOR")
    if getattr(user,"individual_subscription",None) is not None:
        raise BudgetHold("SUBSCRIPTION_STATUS_REQUIRES_REVIEW")
    return "ORG_WEEKLY_POOL_ABOVE_FLOOR"

def existing_by_name(api, target_folder_uuid, names):
    found={name:[] for name in names}
    for page in range(MAX_PAGES):
        batch=api.list_workflows(parent_uuid=target_folder_uuid,page=page,size=PAGE_SIZE)
        if len(batch)>PAGE_SIZE:
            raise BudgetHold("WORKFLOW_LIST_OVERSIZE")
        for w in batch:
            name=str(getattr(w,"name",""))
            if name in found:
                parent=str(getattr(w,"parent_uuid",""))
                wf_type=str(getattr(w,"workflow_type",""))
                if parent != target_folder_uuid or wf_type!="protein_cofolding":
                    raise BudgetHold("EXISTING_JOB_IDENTITY_DRIFT")
                found[name].append(w)
        if len(batch)<PAGE_SIZE:
            break
    else:
        raise BudgetHold("WORKFLOW_PAGINATION_NOT_EXHAUSTED")
    if any(len(v)>1 for v in found.values()):
        raise BudgetHold("DUPLICATE_CASE_NEEDS_MANUAL_REVIEW")
    return found

def single_step(api, manifest, authorized):
    if not authorized:
        raise BudgetHold("EXPLICIT_APPROVAL_SWITCH_REQUIRED")
    user=api.whoami()
    budget_type=require_budget(user)
    folder=api.retrieve_folder(TARGET_ROWAN_FOLDER_UUID)
    if str(getattr(folder,"uuid",None)) != TARGET_ROWAN_FOLDER_UUID:
        raise BudgetHold("ROWAN_FOLDER_ID_DRIFT")
    cases=[prepare(manifest,arm) for arm in CASES]
    matches=existing_by_name(api,TARGET_ROWAN_FOLDER_UUID,[c["id"] for c in cases])
    for case in cases:
        if matches[case["id"]]:
            continue
        # Enforce max one provider submission per run; full pair never
        # attempted simultaneously and no auto retry on provider failure.
        wf=api.submit_protein_cofolding_workflow(
            initial_protein_sequences=[case["sequence"]],
            model=api.CofoldingModel.BOLTZ_2,
            num_samples=1,
            use_msa_server=True,
            max_credits=CREDIT_CAP,
            folder_uuid=TARGET_ROWAN_FOLDER_UUID,
            name=case["id"],
            is_draft=False,
        )
        uid=str(getattr(wf,"uuid",""))
        if not uid:
            raise BudgetHold("ROWAN_SUBMISSION_NO_UUID_VERIFY_IN_FOLDER")
        return {
            "state":"SUBMITTED_ONE_AND_STOPPED",
            "arm":case["arm"],
            "max_credits":CREDIT_CAP,
            "credit_source_gate":budget_type,
            "workflow_uuid_digest_prefix":hashlib.sha256(uid.encode()).hexdigest()[:16],
            "provider_results":"PENDING",
            "tranception_model_inference":False,
            "protein_fitness_validated":False,
        }
    return {"state":"ALL_PREREGISTERED_CASES_ALREADY_EXIST_NO_SUBMISSION",
            "credit_source_gate":budget_type,"max_credits":0,
            "tranception_model_inference":False,"protein_fitness_validated":False}

def main():
    if os.environ.get("JANUS_ROWAN_APPROVED_WEEKLY_RUN") != "TRUE":
        print("JANUS_ROWAN_AUTOPILOT=BLOCKED_EXPLICIT_SWITCH")
        return 2
    if not os.environ.get("ROWAN_API_KEY"):
        print("JANUS_ROWAN_AUTOPILOT=BLOCKED_MISSING_PROVIDER_KEY")
        return 2
    try:
        manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
        api=importlib.import_module("rowan")
        receipt=single_step(api,manifest,authorized=True)
    except BudgetHold as exc:
        # Only our fixed codes, never raw provider response text or key.
        print("JANUS_ROWAN_AUTOPILOT_HOLD="+str(exc))
        return 0
    except Exception:
        # Failure after submission may be ambiguous: no retry until
        # list_workflows verifies case identity in a future invocation.
        print("JANUS_ROWAN_AUTOPILOT=PROVIDER_FAILURE_MANUAL_REVIEW")
        return 3
    print("JANUS_ROWAN_AUTOPILOT="+receipt["state"])
    if "arm" in receipt:
        print("JANUS_ROWAN_ARM="+receipt["arm"])
        print("JANUS_ROWAN_MAX_CREDITS="+str(receipt["max_credits"]))
        print("JANUS_ROWAN_WORKFLOW_HASH="+receipt["workflow_uuid_digest_prefix"])
    print("JANUS_ROWAN_CREDIT_GATE="+receipt["credit_source_gate"])
    print("JANUS_ROWAN_PROTEIN_FITNESS_VALIDATED=FALSE")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
