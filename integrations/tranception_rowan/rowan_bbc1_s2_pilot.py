#!/usr/bin/env python3
"""Preregistered BBC1 mutant diversity pilot; never a general fitness predictor.

Eight variants are chosen deterministically across published *processed* assay
label quantiles, with distinct mutated positions. Selection cannot adapt to
Rowan outputs. Strict SHA256 source verification and fail-closed credit cap.
"""
from __future__ import annotations
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path

from dms_evidence import SOURCE, validate_tsv, get_pinned_dataset
from probe import TARGET_ROWAN_FOLDER_UUID
from budget_gate import finite_nonnegative
from rowan_bbc1_autopilot import existing_by_name,BudgetHold

ROOT=Path(__file__).resolve().parent
QUANTILES=(5,15,30,45,60,75,90,98)
CREDIT_CAP=10
WEEKLY_FLOOR=30
MAX_PILOT_JOBS=len(QUANTILES)
MAX_PILOT_CREDITS=MAX_PILOT_JOBS*CREDIT_CAP

def select(blob:bytes,manifest:dict)->list[dict]:
    # K5A already validated in the same immutable file: no alternate data.
    source=validate_tsv(blob,manifest)
    if source["status"]!="EXPERIMENTAL_DERIVATIVE_ROW_VERIFIED":
        raise BudgetHold("PINNED_PUBLIC_ASSAY_NOT_VERIFIED")
    wt=manifest["independent_control"]["wildtype"]
    k5a=manifest["independent_control"]["mutant"]
    reader=csv.DictReader(io.StringIO(blob.decode("utf-8-sig")),delimiter="\t")
    values=[]
    seen=set()
    for row in reader:
        seq=(row.get("sequences") or "").strip()
        if len(seq)!=len(wt) or seq in seen:
            raise BudgetHold("DMS_ROW_LENGTH_OR_DUPLICATE")
        seen.add(seq)
        changes=[i for i,(a,b) in enumerate(zip(wt,seq)) if a!=b]
        if len(changes)!=1:
            raise BudgetHold("DMS_NOT_SINGLE_SUBSTITUTION")
        score=float(row["labels"])
        if not math.isfinite(score):
            raise BudgetHold("DMS_NONFINITE_SCORE")
        pos=changes[0]
        if seq==k5a:
            continue
        mutation=wt[pos]+str(pos+1)+seq[pos]
        values.append((score,seq,pos,mutation))
    if len(seen)!=1084:
        raise BudgetHold("DMS_POPULATION_CHANGED")
    values.sort(key=lambda x:(x[0],x[1]))
    chosen=[]
    occupied=set()
    for pct in QUANTILES:
        pivot=round((len(values)-1)*(pct/100))
        selected=None
        for delta in range(len(values)):
            indices=(pivot,) if delta==0 else (pivot-delta,pivot+delta)
            for idx in indices:
                if 0<=idx<len(values) and values[idx][2] not in occupied:
                    selected=values[idx];break
            if selected is not None:break
        if selected is None:
            raise BudgetHold("NOT_ENOUGH_DISTINCT_MUTATION_POSITIONS")
        score,seq,pos,mutation=selected
        occupied.add(pos)
        chosen.append({
            "case_name":f"JANUS-ROWAN-BBC1-S2-Q{pct:02d}-{mutation}",
            "quantile_target_percent":pct,
            "mutation":mutation,
            "position_1_based":pos+1,
            "sequence":seq,
            "sequence_sha256":hashlib.sha256(seq.encode()).hexdigest(),
            "processed_label":score,
            "published_processed_not_raw":True,
        })
    if len(chosen)!=MAX_PILOT_JOBS or len({v["case_name"] for v in chosen})!=MAX_PILOT_JOBS:
        raise BudgetHold("SAMPLE_SELECTION_NOT_UNIQUE")
    return chosen

def credit_budget(u):
    org=getattr(u,"organization",None)
    if org is None:
        raise BudgetHold("NO_ORGANIZATION")
    if getattr(u,"individual_subscription",None) is not None:
        raise BudgetHold("SUBSCRIPTION_NEEDS_REVIEW")
    weekly=finite_nonnegative(getattr(org,"weekly_credits",None))
    if weekly is None or weekly<WEEKLY_FLOOR:
        raise BudgetHold("WEEKLY_BALANCE_TOO_LOW")
    return "ORG_WEEKLY_GREATER_THAN_30"

def step(api,manifest,blob,*,approval:bool):
    if not approval:
        raise BudgetHold("EXPLICIT_WEEKLY_ONLY_AUTH_REQUIRED")
    candidates=select(blob,manifest)
    credit_budget(api.whoami())
    folder=api.retrieve_folder(TARGET_ROWAN_FOLDER_UUID)
    if str(getattr(folder,"uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
        raise BudgetHold("FOLDER_NOT_APPROVED")
    matches=existing_by_name(api,TARGET_ROWAN_FOLDER_UUID,[c["case_name"] for c in candidates])
    for c in candidates:
        if matches[c["case_name"]]:
            continue
        wf=api.submit_protein_cofolding_workflow(
            initial_protein_sequences=[c["sequence"]],
            model=api.CofoldingModel.BOLTZ_2,
            num_samples=1,
            use_msa_server=True,
            max_credits=CREDIT_CAP,
            folder_uuid=TARGET_ROWAN_FOLDER_UUID,
            name=c["case_name"],
            is_draft=False,
        )
        if not str(getattr(wf,"uuid","")):
            raise BudgetHold("SUBMISSION_UUID_NOT_ATTESTED")
        return {"status":"ONE_PILOT_MUTANT_SUBMITTED","mutation":c["mutation"],
                "anchor_percentile":c["quantile_target_percent"],
                "max_credits":CREDIT_CAP,"pilot_max_credits":MAX_PILOT_CREDITS,
                "no_fitness_claim":True}
    return {"status":"ALL_EIGHT_CASES_EXIST_NO_MORE_SUBMISSIONS",
            "pilot_max_credits":MAX_PILOT_CREDITS,"no_fitness_claim":True}

def main()->int:
    if not os.environ.get("ROWAN_API_KEY") or os.environ.get("JANUS_ROWAN_S2_APPROVED")!="TRUE":
        print("ROWAN_S2=AUTHORIZATION_SWITCH_OR_SECRET_MISSING");return 2
    try:
        manifest=json.loads((ROOT/"discovery_manifest.json").read_text(encoding="utf-8"))
        blob=get_pinned_dataset()
        api=importlib.import_module("rowan")
        outcome=step(api,manifest,blob,approval=True)
    except BudgetHold as exc:
        print("ROWAN_S2_HOLD="+str(exc))
        return 0
    except Exception:
        print("ROWAN_S2_PROVIDER_OR_DOWNLOAD_FAILURE_MANUAL_REVIEW")
        return 3
    print("ROWAN_S2_STATE="+outcome["status"])
    if outcome.get("mutation"):
        print("ROWAN_S2_CASE="+outcome["mutation"])
        print("ROWAN_S2_MAX_CREDITS="+str(outcome["max_credits"]))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
