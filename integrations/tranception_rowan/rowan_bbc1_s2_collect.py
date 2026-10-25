#!/usr/bin/env python3
"""Read-only results for the frozen eight-mutant BBC1 stratified pilot.

Spearman computed solely within the preregistered deliberately stratified
sample: cannot estimate unbiased full-assay correlation or causal fitness.
"""
from __future__ import annotations
import importlib
import json
import math
import os
from pathlib import Path

from dms_evidence import get_pinned_dataset
from probe import prepare,TARGET_ROWAN_FOLDER_UUID
from budget_gate import finite_nonnegative
from rowan_bbc1_autopilot import existing_by_name,BudgetHold
from rowan_bbc1_s2_pilot import select

ROOT=Path(__file__).resolve().parent

def corr_spearman(a,b):
    if len(a)!=len(b) or len(a)<3:
        return None
    def ranks(v):
        order=sorted(range(len(v)),key=lambda i:(v[i],i))
        z=[0.]*len(v);i=0
        while i<len(v):
            j=i+1
            while j<len(v) and v[order[j]]==v[order[i]]:
                j+=1
            r=(i+1+j)/2
            for k in range(i,j):z[order[k]]=r
            i=j
        return z
    x,y=ranks(a),ranks(b)
    mx=sum(x)/len(x);my=sum(y)/len(y)
    xx=sum((v-mx)**2 for v in x)
    yy=sum((v-my)**2 for v in y)
    if xx==0 or yy==0:return None
    r=sum((u-mx)*(v-my) for u,v in zip(x,y))/math.sqrt(xx*yy)
    return round(max(-1,min(1,r)),6)

def score(w):
    w=w.fetch_latest()
    if str(getattr(w,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
        raise BudgetHold("FOLDER_DRIFT")
    if not w.done():
        return {"state":"PENDING_OR_FAILED"}
    try:
        result=w.result(wait=False)
        scores=getattr(result,"scores",None)
        ptm=finite_nonnegative(getattr(scores,"ptm",None))
        lddt=finite_nonnegative(getattr(scores,"avg_lddt",None))
        if ptm is not None and not 0<=ptm<=1:ptm=None
        if lddt is not None and not 0<=lddt<=1:lddt=None
        credit=finite_nonnegative(getattr(w,"credits_charged",None))
        return {"state":"PROVIDER_RESULT_RETURNED","ptm":ptm,"avg_lddt":lddt,
                "credits_charged":credit}
    except Exception:
        return {"state":"RESULT_UNREADABLE"}

def collect(api,manifest,blob):
    samples=select(blob,manifest)
    wt=prepare(manifest,"wildtype")
    api.whoami()
    folder=api.retrieve_folder(TARGET_ROWAN_FOLDER_UUID)
    if str(getattr(folder,"uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
        raise BudgetHold("FOLDER_DRIFT")
    allnames=[wt["id"]]+[c["case_name"] for c in samples]
    jobs=existing_by_name(api,TARGET_ROWAN_FOLDER_UUID,allnames)
    if len(jobs[wt["id"]])!=1:
        raise BudgetHold("WT_BASELINE_NOT_PRESENT")
    wt_score=score(jobs[wt["id"]][0])
    receipts=[]
    for c in samples:
        item={"quantile":c["quantile_target_percent"],"mutation":c["mutation"],
              "processed_label":c["processed_label"],"provider_state":"NOT_SUBMITTED"}
        if len(jobs[c["case_name"]])==1:
            v=score(jobs[c["case_name"]][0])
            item.update({"provider_state":v.pop("state"),**v})
        receipts.append(item)
    paired=[x for x in receipts if x["provider_state"]=="PROVIDER_RESULT_RETURNED"
            and x.get("ptm") is not None and x.get("avg_lddt") is not None]
    ready=len(paired)==len(samples) and wt_score.get("state")=="PROVIDER_RESULT_RETURNED"
    assoc={"ptm":None,"avg_lddt":None}
    if ready:
        assoc={name:corr_spearman([x["processed_label"] for x in paired],
                [x[name]-wt_score[name] for x in paired]) for name in ("ptm","avg_lddt")}
    return {
        "schema_id":"janus.rowan.bbc1.stratified_8_mutant_pilot.v1",
        "source_commit":"cc6b0b56553e694dbf27dc84c6b37a8686e64b85",
        "source_sha256":"ca355a655381c95fe899d61a406251e0cd17d68bbabe287651d526fdeb821f87",
        "sampling_scheme":"PRESELECTED_PROCESSED_LABEL_QUANTILES_DISTINCT_POSITIONS",
        "population_interpretation":"DELIBERATELY_STRATIFIED_NOT_UNBIASED_ESTIMATE",
        "wildtype":wt_score,
        "pilot_cases":receipts,
        "all_eight_ready_for_descriptive_association":ready,
        "spearman_processed_label_vs_delta_structure_confidence":assoc,
        "signed_label_direction_verified_against_original_raw_assay":False,
        "protein_stability_or_fitness_inferred":False,
        "topa_scientific_review":"OPEN",
        "new_provider_jobs_by_this_collector":0,
    }

def main():
    if not os.getenv("ROWAN_API_KEY"):
        print("JANUS_ROWAN_S2_COLLECT=NO_KEY");return 2
    try:
        manifest=json.loads((ROOT/"discovery_manifest.json").read_text())
        blob=get_pinned_dataset()
        api=importlib.import_module("rowan")
        r=collect(api,manifest,blob)
    except Exception:
        print("JANUS_ROWAN_S2_COLLECT=READ_OR_INTEGRITY_BLOCKED");return 3
    completed=sum(x["provider_state"]=="PROVIDER_RESULT_RETURNED" for x in r["pilot_cases"])
    print("JANUS_ROWAN_S2_COLLECT_COMPLETED="+str(completed)+"/8")
    print("JANUS_ROWAN_S2_ASSOCIATION_READY="+str(r["all_eight_ready_for_descriptive_association"]))
    print("JANUS_ROWAN_S2_CAUSAL_FITNESS_PROVEN=FALSE")
    output=os.getenv("JANUS_ROWAN_S2_RECEIPT")
    if output:
        path=Path(output)
        if path.exists():return 3
        with path.open("x",encoding="utf-8") as f:
            json.dump(r,f,indent=2,allow_nan=False);f.write("\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
