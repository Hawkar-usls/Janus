#!/usr/bin/env python3
"""Non-sensitive, read-only collection of the two fixed BBC1 Rowan cases.

Never publishes raw provider workflow, atomic coordinates, private user
metadata or credentials. Structural confidence is NOT mutational fitness.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from probe import prepare,TARGET_ROWAN_FOLDER_UUID
from rowan_bbc1_autopilot import CASES,existing_by_name,BudgetHold,CREDIT_CAP
from budget_gate import finite_nonnegative

ROOT=Path(__file__).resolve().parent

def confidence(v):
    if type(v) in (int,float) and math.isfinite(v) and 0<=v<=1:
        return float(v)
    return None

def collect(api,manifest):
    api.whoami()
    folder=api.retrieve_folder(TARGET_ROWAN_FOLDER_UUID)
    if str(getattr(folder,"uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
        raise BudgetHold("FOLDER_ID_DRIFT")
    cases=[prepare(manifest,arm) for arm in CASES]
    matches=existing_by_name(api,TARGET_ROWAN_FOLDER_UUID,[c["id"] for c in cases])
    rows=[]
    for case in cases:
        records=matches[case["id"]]
        if not records:
            rows.append({"arm":case["arm"],"status":"NOT_SUBMITTED"})
            continue
        w=records[0].fetch_latest()
        if str(getattr(w,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
            raise BudgetHold("WORKFLOW_MOVED_FROM_APPROVED_FOLDER")
        uid=str(getattr(w,"uuid",""))
        identity_hash=hashlib.sha256(uid.encode()).hexdigest()[:16] if uid else "MISSING"
        charge=finite_nonnegative(getattr(w,"credits_charged",None))
        item={"arm":case["arm"],"status":"PENDING_OR_FAILED", "provider_uuid_hash":identity_hash,
              "provider_credits_charged":charge, "credit_cap":CREDIT_CAP,
              "charge_over_cap_review":charge is not None and charge>CREDIT_CAP}
        try:
            if w.done():
                result=w.result(wait=False)
                s=getattr(result,"scores",None)
                ptm=confidence(getattr(s,"ptm",None))
                lddt=confidence(getattr(s,"avg_lddt",None))
                item["status"]="ROWAN_RESULT_RETURNED_NO_FITNESS_CLAIM"
                item["ptm"]=ptm
                item["avg_lddt"]=lddt
        except Exception:
            item["status"]="RESULT_UNREADABLE_REQUIRES_MANUAL_REVIEW"
        rows.append(item)
    paired={}
    if len(rows)==2 and all(r.get("status")=="ROWAN_RESULT_RETURNED_NO_FITNESS_CLAIM" for r in rows):
        for field in ("ptm","avg_lddt"):
            before,after=rows[0].get(field),rows[1].get(field)
            if before is not None and after is not None:
                paired["K5A_minus_wildtype_"+field]=round(after-before,6)
    return {
        "schema":"janus.rowan.bbc1.provider_structural_status.v2",
        "paired_confidence_deltas_not_fitness":paired,
        "source_kind":"REAL_PROVIDER_READ_ONLY_STATUS",
        "case_order":list(CASES),
        "entries":rows,
        "tranception_model_run":False,
        "protein_fitness_validated":False,
        "experimental_assay_independent_review":"OPEN",
        "provider_spend_by_collector":False,
    }

def main()->int:
    if not os.getenv("ROWAN_API_KEY"):
        print("JANUS_ROWAN_COLLECT=NO_KEY");return 2
    try:
        manifest=json.loads((ROOT/"discovery_manifest.json").read_text(encoding="utf-8"))
        api=importlib.import_module("rowan")
        r=collect(api,manifest)
    except Exception:
        print("JANUS_ROWAN_COLLECT=READ_OR_IDENTITY_ERROR");return 3
    for i in r["entries"]:
        print("JANUS_ROWAN_COLLECT_ARM="+i["arm"]+"_STATUS="+i["status"])
        if i.get("provider_credits_charged") is not None:
            print("JANUS_ROWAN_COLLECT_"+i["arm"]+"_PROVIDER_CREDITS="+repr(i["provider_credits_charged"]))
        if i.get("charge_over_cap_review"):
            print("JANUS_ROWAN_COLLECT_"+i["arm"]+"_CREDIT_CAP_ALERT=TRUE")
        if i.get("ptm") is not None:
            print("JANUS_ROWAN_COLLECT_"+i["arm"]+"_PTM="+repr(i["ptm"]))
        if i.get("avg_lddt") is not None:
            print("JANUS_ROWAN_COLLECT_"+i["arm"]+"_LDDT="+repr(i["avg_lddt"]))
    target=os.environ.get("JANUS_ROWAN_RECEIPT_PATH")
    if target:
        p=Path(target)
        if p.exists(): print("JANUS_ROWAN_COLLECT=IMMUTABLE_RECEIPT_EXISTS");return 3
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("x",encoding="utf-8") as f:
            json.dump(r,f,indent=2,allow_nan=False);f.write("\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
