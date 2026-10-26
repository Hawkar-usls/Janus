#!/usr/bin/env python3
"""Exactly one preregistered 1EY0 T8A Rowan BOLTZ_2 mutant; no other mutation.

Source: JANUS Meta Registry ROWAN-1EY0-T8A-SINGLE-MUTANT-STRUCTURAL-DELTA-
PREREG-2026-10-26-v1.json frozen before either mutant prediction.
Source archived FireProt PDB bytes HOLD; RCSB modeled sequence verified 136/136.
One 10-credit max provider job using organization weekly >=30 only, never
nonweekly unknown credits; repeated invocations NEVER resubmit; collect read-only.
"""
from __future__ import annotations
import hashlib,importlib,json,os
from pathlib import Path
from budget_gate import finite_nonnegative
from probe import TARGET_ROWAN_FOLDER_UUID

from rowan_1ey0_weekly_api import WT,WT_SHA256

MUT="T8A"
NAME="JANUS-ROWAN-1EY0-RCSB-136AA-T8A-20261026-R0"
MUT_SHA256="1c9ea97ff41ea9f659d1712e422c2ea1f4aa3373463b1e17d3d682c156cddacb"
CAP=10
WEEKLY_FLOOR=30
PAGE_SIZE=100
MAX_PAGES=15
class Hold(ValueError):pass
def sha(b):return hashlib.sha256(b).hexdigest()
def mutant():
    if len(WT)!=136 or sha(WT.encode())!=WT_SHA256:
        raise Hold("FROZEN_WT_SEQUENCE_DRIFT")
    if WT[7]!="T":raise Hold("MUTATION_SOURCE_T8_IDENTITY_DRIFT")
    sequence=WT[:7]+"A"+WT[8:]
    if sha(sequence.encode())!=MUT_SHA256:raise Hold("MUTANT_136AA_FROZEN_SHA_DRIFT")
    return sequence
def check_budget(api):
    u=api.whoami()
    if getattr(u,"individual_subscription",None) is not None:
        raise Hold("INDIVIDUAL_SUBSCRIPTION_NEEDS_REVIEW")
    org=getattr(u,"organization",None)
    if org is None:raise Hold("NO_VERIFIED_ORGANIZATION_POOL")
    n=finite_nonnegative(getattr(org,"weekly_credits",None))
    if n is None or n<WEEKLY_FLOOR:
        raise Hold("ORG_WEEKLY_BELOW_30_OR_UNKNOWN")
    return "ORG_WEEKLY_MIN_30_VERIFIED"
def existing(api):
    fd=api.retrieve_folder(TARGET_ROWAN_FOLDER_UUID)
    if str(getattr(fd,"uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
        raise Hold("FOLDER_IDENTITY_DRIFT")
    matches=[]
    for p in range(MAX_PAGES):
        batch=api.list_workflows(parent_uuid=TARGET_ROWAN_FOLDER_UUID,page=p,size=PAGE_SIZE)
        if len(batch)>PAGE_SIZE:raise Hold("PAGE_OVERFLOW")
        for w in batch:
            if str(getattr(w,"name",None))==NAME:
                if (str(getattr(w,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID
                    or str(getattr(w,"workflow_type",None))!="protein_cofolding"):
                    raise Hold("DUPLICATE_NAME_IN_WRONG_TYPE_OR_FOLDER")
                matches.append(w)
        if len(batch)<PAGE_SIZE:break
    else:raise Hold("UNBOUNDED_PAGINATION")
    if len(matches)>1:raise Hold("DUPLICATE_MATCHING_JOBS")
    return matches[0] if matches else None
def collect_one(w):
    try:
        w=w.fetch_latest()
        if str(getattr(w,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
            raise Hold("COLLECT_WRONG_FOLDER")
        if str(getattr(w,"name",None))!=NAME:
            raise Hold("COLLECT_WRONG_MUTATION_CASE")
        if not w.done():
            return {"provider_result_state":"PENDING_OR_IN_PROGRESS","new_jobs":0}
        rs=w.result(wait=False)
        scores=getattr(rs,"scores",None)
        ptm=finite_nonnegative(getattr(scores,"ptm",None))
        pl=finite_nonnegative(getattr(scores,"avg_lddt",None))
        credits=finite_nonnegative(getattr(w,"credits_charged",None))
        if ptm is None or pl is None or ptm>1 or pl>1:
            return {"provider_result_state":"PROVIDER_SCORE_UNVERIFIED",
                    "credits_charged":credits,"new_jobs":0}
        return {"provider_result_state":"PROVIDER_RESULT_RETURNED",
                "ptm":ptm,"avg_lddt":pl,"credits_charged":credits,"new_jobs":0}
    except Hold:raise
    except Exception:
        return {"provider_result_state":"PROVIDER_RESULT_UNREADABLE","new_jobs":0}
def execute(api,action,authorized):
    seq=mutant()
    if action not in ("collect","submit"):raise Hold("UNKNOWN_ACTION")
    prior=existing(api)
    receipt={
      "schema_id":"janus.rowan.1ey0_T8A.single_weekly_provider_job.v1",
      "case_name":NAME,
      "source_mutant_sha256":MUT_SHA256,
      "wt_sha256":WT_SHA256,"length":136,"variant":MUT,
      "max_credits_per_new_job":CAP,
      "new_jobs":0,"actual_provider_status":"UNSEEN",
      "model":"BOLTZ_2","num_samples":1,"use_msa_server":True,
      "FireProt_210_inference_authorized":False,
      "mutation_level_ddg_claim":False,
      "published_experimental_labels_accessed":False,
      "shadow_model_weights_modified":False,
      "original_zip_PDB_bytes_attested":False
    }
    if prior is not None:
        uid=str(getattr(prior,"uuid",""))
        if not uid:raise Hold("PRIOR_PROVIDER_JOB_NO_UUID")
        receipt.update({"status":"EXISTING_SINGLE_MUTANT_NO_SPEND",
                        "workflow_uuid":uid,"actual_provider_status":str(getattr(prior,"status",None))})
        if action=="collect":receipt.update(collect_one(prior))
        return receipt
    if action=="collect":
        receipt["status"]="NO_MUTANT_JOB_TO_COLLECT"
        return receipt
    if not authorized:raise Hold("NO_NEW_JOB_EXPLICIT_APPROVAL")
    source=check_budget(api)
    w=api.submit_protein_cofolding_workflow(
        initial_protein_sequences=[seq],
        model=api.CofoldingModel.BOLTZ_2,
        num_samples=1,use_msa_server=True,max_credits=CAP,
        folder_uuid=TARGET_ROWAN_FOLDER_UUID,name=NAME,is_draft=False)
    uid=str(getattr(w,"uuid",""))
    if not uid:raise Hold("SUBMISSION_UUID_ABSENT_NEVER_AUTO_RETRY")
    receipt.update({"status":"ONE_PREREGISTERED_T8A_PROVIDER_JOB_SUBMITTED",
      "actual_provider_status":str(getattr(w,"status",None)),
      "credit_source":source,"new_jobs":1,"workflow_uuid":uid})
    return receipt
def main():
    action=os.environ.get("JANUS_ROWAN_1EY0_T8A_ACTION","collect")
    approved=os.environ.get("JANUS_ROWAN_1EY0_T8A_CREDIT_APPROVED","FALSE")=="TRUE"
    if not os.environ.get("ROWAN_API_KEY"):
        print("JANUS_1EY0_T8A=HOLD_MISSING_PROVIDER_SECRET");return 2
    if action=="submit" and not approved:
        print("JANUS_1EY0_T8A=HOLD_NEW_JOB_NOT_AUTHORIZED");return 2
    if action=="collect" and approved:
        print("JANUS_1EY0_T8A=HOLD_APPROVAL_IN_COLLECT_MODE");return 2
    try:
        rowan=importlib.import_module("rowan")
        r=execute(rowan,action,approved)
        print("JANUS_1EY0_T8A="+r["status"])
        print("JANUS_1EY0_T8A_PROVIDER_STATE="+str(r.get("provider_result_state",r["actual_provider_status"])))
        print("JANUS_1EY0_T8A_NEW_JOBS="+str(r["new_jobs"]))
        out=os.environ.get("JANUS_ROWAN_1EY0_T8A_RECEIPT")
        if out:
            f=Path(out)
            if f.exists():raise Hold("RECEIPT_IMMUTABLE_OUTPUT_EXISTS")
            f.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        return 0
    except Hold as e:
        print("JANUS_1EY0_T8A=HOLD_"+str(e));return 2
    except Exception:
        print("JANUS_1EY0_T8A=HOLD_PROVIDER_ERROR_NO_AUTO_RETRY");return 3
if __name__=="__main__":raise SystemExit(main())
