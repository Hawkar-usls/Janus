#!/usr/bin/env python3
"""Offline source-bound Rowan -> Lapis -> Fundamentum evidence admission.

No API client, no provider token and no computation submission. GitHub Actions
must independently establish the origin run identity and artifact uniqueness.
The Lapis IR is annotation only and the Fundamentum receiver is structurally
valid but not a scientific proof or external wetlab attestation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA="ca355a655381c95fe899d61a406251e0cd17d68bbabe287651d526fdeb821f87"
SOURCE_COMMIT="cc6b0b56553e694dbf27dc84c6b37a8686e64b85"
FROZEN=((5,"G53A"),(15,"G48D"),(30,"E27F"),(45,"W38L"),
        (60,"K13N"),(75,"G64Q"),(90,"Q44V"),(98,"E42H"))
SCHEMA="janus.rowan.autonomous_lapis_receiver_admission.v1"

class Hold(ValueError):pass

def bounded(value,lo,hi):
    if type(value) not in (int,float) or not lo<=value<=hi:
        raise Hold("INVALID_PROVIDER_CONFIDENCE")
    return float(value)

def validate_collector(data):
    if not isinstance(data,dict) or data.get("schema_id")!="janus.rowan.bbc1.stratified_8_mutant_pilot.v1":
        raise Hold("COLLECTOR_SCHEMA_UNTRUSTED")
    if data.get("source_sha256")!=SOURCE_SHA or data.get("source_commit")!=SOURCE_COMMIT:
        raise Hold("FROZEN_CORPUS_IDENTITY_DRIFT")
    samples=data.get("pilot_cases")
    if not isinstance(samples,list) or len(samples)!=8:
        raise Hold("FROZEN_EIGHT_CASE_COUNT_CHANGED")
    for (q,mutation),item in zip(FROZEN,samples):
        if not isinstance(item,dict) or item.get("quantile")!=q or item.get("mutation")!=mutation:
            raise Hold("FROZEN_MUTATION_OR_ORDER_CHANGED")
        if item.get("provider_state")!="PROVIDER_RESULT_RETURNED":
            raise Hold("ROWAN_PROVIDER_INCOMPLETE_HOLD")
        bounded(item.get("ptm"),0,1)
        bounded(item.get("avg_lddt"),0,1)
        charged=item.get("credits_charged")
        if charged is not None and bounded(charged,0,100)>10:
            raise Hold("PROVIDER_CREDIT_CAP_ALERT")
    wt=data.get("wildtype")
    if not isinstance(wt,dict) or wt.get("state")!="PROVIDER_RESULT_RETURNED":
        raise Hold("WILDTYPE_NOT_ATTESTED")
    bounded(wt.get("ptm"),0,1)
    bounded(wt.get("avg_lddt"),0,1)
    if (data.get("all_eight_ready_for_descriptive_association") is not True
        or data.get("topa_scientific_review")!="OPEN"
        or data.get("protein_stability_or_fitness_inferred") is not False):
        raise Hold("UNAUTHORIZED_SCIENTIFIC_PROMOTION")
    return True

def validate_lapis(source,bundle):
    c=json.loads((bundle/"candidate.json").read_text(encoding="utf-8"))
    ir=json.loads((bundle/"algorithm_ir.json").read_text(encoding="utf-8"))
    mf=json.loads((bundle/"manifest.json").read_text(encoding="utf-8"))
    if (c.get("facts",{}).get("source_class")!="ROWAN_BBC1_STRATIFIED"
        or c.get("facts",{}).get("ready_count")!=8
        or c.get("for_fundamentum",{}).get("proof_or_exact_sat_witness") is not False
        or c.get("for_fundamentum",{}).get("P_VS_NP")!="OPEN"):
        raise Hold("LAPIS_CANDIDATE_PROOF_ESCALATION")
    samples=c.get("facts",{}).get("samples")
    if not isinstance(samples,list) or len(samples)!=8:
        raise Hold("LAPIS_SAMPLE_COUNT_CHANGED")
    for original,converted in zip(source["pilot_cases"],samples):
        for key in ("mutation","ptm","avg_lddt"):
            if converted.get(key)!=original.get(key):
                raise Hold("LAPIS_PROVIDER_DATA_CHANGED")
    if (ir.get("algorithm",{}).get("mode")!="explicit_contract"
        or not isinstance(ir.get("algorithm",{}).get("steps"),list)
        or not ir["algorithm"]["steps"]
        or any(step.get("operator")!="ANNOTATE" for step in ir["algorithm"]["steps"])):
        raise Hold("LAPIS_MATHEMATICAL_AUTHORITY_ESCALATION")
    if not isinstance(mf,dict):
        raise Hold("LAPIS_MANIFEST_INVALID")
    return c,ir

def check(source_path:Path,bundle:Path,run_id:int):
    if type(run_id) is not int or run_id<=0:raise Hold("RUN_ID_MISSING")
    if source_path.stat().st_size>80_000:raise Hold("SOURCE_OVERSIZED")
    raw=source_path.read_bytes()
    source=json.loads(raw.decode("utf-8"))
    validate_collector(source)
    c,ir=validate_lapis(source,bundle)
    digest=hashlib.sha256(raw).hexdigest()
    return {
      "schema":SCHEMA,
      "status":"SOURCE_BOUND_ROWAN_TO_LAPIS_CANDIDATE_VALIDATED_UNAUTHENTICATED",
      "provider_source_run_id":run_id,
      "provider_source_artifact_sha256":digest,
      "rowan_eight_mutants":8,
      "lapis_candidate_ready_count":c["facts"]["ready_count"],
      "ir_step_count":len(ir["algorithm"]["steps"]),
      "lapis_proof_authority":False,
      "fundamentum_independent_receiver_must_pass_separately":True,
      "rowan_provider_jobs_started":0,
      "credits_spent":0,
      "production_models_promoted":False,
      "wetlab_independent_validation":"OPEN",
      "p_vs_np":"OPEN",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--run-id",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    receipt=check(a.source,a.bundle,a.run_id)
    a.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("ROWAN_LAPIS_AUTONOMOUS_ADMISSION=SOURCE_BOUND_ZERO_SPEND_NOT_SAT_PROOF")
    print("ROWAN_LAPIS_UPSTREAM_RUN="+str(a.run_id))

if __name__=="__main__":
    main()
