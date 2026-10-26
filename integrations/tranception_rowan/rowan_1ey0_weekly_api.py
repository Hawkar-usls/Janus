#!/usr/bin/env python3
"""ONE Rowan Boltz-2 1EY0 WT workflow, using ONLY verifiable weekly credits.

2026-10-26, JANUS. User directly authorized a credit-based Rowan API run.
Existing archived FireProt PDB MEMBER hash remains HOLD. This run uses the
independently RCSB-confirmed 136-aa modeled WT sequence for a technical
provider comparison. No mutant, no ΔΔG/stability claim, no auto-retries.

Safety: explicit switch; preflight weekly pool >= 30, per-job max_credits 10;
folder identity, all-page duplicate detection before submission; fail closed
on unknown provider fields; no use of non-weekly credit balance for gating.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from budget_gate import finite_nonnegative
from probe import TARGET_ROWAN_FOLDER_UUID

WT="KLHKEPATLIKAIDGDTVKLMYKGQPMTFRLLLVDTPETKHPKKGVEKYGPEASAFTKKMVENAKKIEVEFDKGQRTDKYGRGLAYIYADGKMVNEALVRQGLAKVAYVYKPNNTHEQHLRKSEAQAKKEKLNIWS"
# The exact frozen 136-residue PDB-modeled 1EY0 sequence; assert against
# independent SHA rather than trusting the text literal.
WT_SHA256="94a6f4324114f9e643076e2bfcb727145d400345f4c5feff0607e5c1fb787be1"
NAME="JANUS-ROWAN-1EY0-RCSB-136AA-WT-20261026-R0"
CAP=10
ORG_WEEKLY_MIN=30
PAGE_SIZE=100
MAX_PAGES=15
ALLOWED=("submit","collect")

class Hold(RuntimeError):pass

def digest(s):return hashlib.sha256(s.encode("ascii")).hexdigest()

def check_static():
    if len(WT)!=136 or digest(WT)!=WT_SHA256:
        raise Hold("FROZEN_WT_SOURCE_SEQUENCE_SHA_DRIFT")
    if not NAME.startswith("JANUS-ROWAN-1EY0-"):
        raise Hold("CASE_IDENTITY_DRIFT")

def budget(user):
    org=getattr(user,"organization",None)
    if org is None:raise Hold("MISSING_ORGANIZATION_WEEKLY_POOL")
    if getattr(user,"individual_subscription",None) is not None:
        raise Hold("SUBSCRIPTION_STATUS_NEEDS_REVIEW")
    available=finite_nonnegative(getattr(org,"weekly_credits",None))
    if available is None or available<ORG_WEEKLY_MIN:
        raise Hold("ORG_WEEKLY_POOL_BELOW_30_OR_UNKNOWN")
    return "ORG_WEEKLY_POOL_GTE_30"

def existing(api):
    found=[]
    for page in range(MAX_PAGES):
        batch=api.list_workflows(parent_uuid=TARGET_ROWAN_FOLDER_UUID,page=page,size=PAGE_SIZE)
        if len(batch)>PAGE_SIZE:raise Hold("ROWAN_PAGE_TOO_LARGE")
        for w in batch:
            if str(getattr(w,"name",""))!=NAME:continue
            if (str(getattr(w,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID
                or str(getattr(w,"workflow_type",""))!="protein_cofolding"):
                raise Hold("EXISTING_JOB_DIFFERENT_FOLDER_OR_WORKFLOW_TYPE")
            found.append(w)
        if len(batch)<PAGE_SIZE:break
    else:raise Hold("WORKFLOW_PAGINATION_NOT_EXHAUSTED")
    if len(found)>1:raise Hold("EXISTING_DUPLICATE_SUBMISSIONS_MANUAL_REVIEW")
    return found[0] if found else None

def check_folder(api):
    folder=api.retrieve_folder(TARGET_ROWAN_FOLDER_UUID)
    if str(getattr(folder,"uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
        raise Hold("APPROVED_FOLDER_IDENTITY_DRIFT")

def process(api,action,approve):
    check_static()
    if action not in ALLOWED:raise Hold("UNSUPPORTED_ACTION")
    check_folder(api)
    previous=existing(api)
    if previous is not None:
        uid=str(getattr(previous,"uuid",""))
        if not uid:raise Hold("EXISTING_WORKFLOW_UUID_MISSING")
        result={
            "status":"ALREADY_EXISTS_NO_NEW_SUBMISSION",
            "existing_uuid":uid,
            "case_name":NAME,
            "workflow_state":str(getattr(previous,"status",None)),
            "new_jobs":0,"credit_spend_cap":0
        }
    elif action=="collect":
        return {"status":"NO_ROWAN_JOB_TO_COLLECT","case_name":NAME,
                "new_jobs":0,"credit_spend_cap":0}
    else:
        if approve is not True:raise Hold("EXPLICIT_ROWAN_WEEKLY_CREDIT_AUTHORIZATION_REQUIRED")
        source=budget(api.whoami())
        job=api.submit_protein_cofolding_workflow(
            initial_protein_sequences=[WT],
            model=api.CofoldingModel.BOLTZ_2,
            num_samples=1,
            use_msa_server=True,
            max_credits=CAP,
            folder_uuid=TARGET_ROWAN_FOLDER_UUID,
            name=NAME,
            is_draft=False,
        )
        uid=str(getattr(job,"uuid",""))
        if not uid:raise Hold("PROVIDER_SUBMISSION_UUID_UNKNOWN_NO_RETRY")
        result={"status":"ONE_1EY0_ROWAN_BOLTZ2_WT_SUBMITTED",
                "existing_uuid":uid,"case_name":NAME,
                "workflow_state":str(getattr(job,"status",None)),
                "credit_source_gate":source,"new_jobs":1,"credit_spend_cap":CAP}
    result.update({
       "schema_id":"janus.rowan.external_1ey0_direct_api_weekly_source_job.v1",
       "input_sequence_sha256":WT_SHA256,"input_length":136,
       "boltz_model":"BOLTZ_2","num_samples":1,"use_msa_server":True,
       "rowan_folder_uuid":TARGET_ROWAN_FOLDER_UUID,
       "experimental_label_access":False,"fireprot_210_mutations_submitted":0,
       "shadow_model_retrained":False,"fitness_validated":False,
       "archived_fireprot_pdb_member_verified":False,
       "cross_provider_feature_equivalence_certified":False,
       "scientific_use":"single technical private Rowans-vs-local WT model score comparison"
    })
    return result

def main():
    action=os.environ.get("JANUS_ROWAN_1EY0_ACTION","collect")
    approved=os.environ.get("JANUS_ROWAN_1EY0_WEEKLY_AUTH")=="TRUE"
    if not os.environ.get("ROWAN_API_KEY"):
        print("ROWAN_1EY0_API=HOLD_MISSING_API_KEY");return 2
    if action=="submit" and not approved:
        print("ROWAN_1EY0_API=HOLD_MISSING_EXPLICIT_WEEKLY_APPROVAL");return 2
    try:
        api=importlib.import_module("rowan")
        report=process(api,action,approved)
        # stdout doesn't include account identity, key or raw results.
        print("ROWAN_1EY0_API="+report["status"])
        print("ROWAN_1EY0_CASE_SHA256="+digest(NAME))
        print("ROWAN_1EY0_JOB_COUNT="+str(report["new_jobs"]))
        print("ROWAN_1EY0_CREDIT_CAP="+str(report["credit_spend_cap"]))
        if report.get("existing_uuid"):
            print("ROWAN_1EY0_PROVIDER_JOB_URL=https://labs.rowansci.com/protein-cofolding/"+report["existing_uuid"])
        path=os.environ.get("JANUS_ROWAN_1EY0_RECEIPT")
        if path:
            Path(path).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        return 0
    except Hold as e:
        print("ROWAN_1EY0_API=HOLD_"+str(e));return 2
    except Exception:
        # Provider failures after submission are ambiguous: NEVER auto-retry.
        print("ROWAN_1EY0_API=HOLD_PROVIDER_ERROR_CHECK_EXISTING_JOB_BEFORE_RETRY")
        return 3

if __name__=="__main__":raise SystemExit(main())
