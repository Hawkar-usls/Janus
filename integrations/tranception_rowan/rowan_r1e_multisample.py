"""R1E one frozen 1EY0 WT five-diffusion-sample cloud job; never auto retry."""
import hashlib, inspect, json, os, statistics
from pathlib import Path
from rowan_1ey0_weekly_api import WT, WT_SHA256, budget
from rowan_r1_cloud_pair import find, Hold
from budget_gate import finite_nonnegative

NAME="JANUS-ROWAN-1EY0-R1E-WT-5S-20261119-R0"
CAP=10
N=5

def score_fields(s):
    r={k:finite_nonnegative(getattr(s,k,None)) for k in ("ptm","avg_lddt")}
    if any(v is None or v>1 for v in r.values()):
        raise Hold("SAMPLE_SCORE_INVALID")
    return r

def collect(job,receipt):
    latest=job.fetch_latest()
    if getattr(latest,"name",None)!=NAME or str(getattr(latest,"parent_uuid",None))!=receipt["folder_uuid"]:
        raise Hold("COLLECT_IDENTITY")
    receipt["provider_state"]=str(getattr(latest,"status",None))
    cost=finite_nonnegative(getattr(latest,"credits_charged",None))
    receipt["credits_charged"]=cost
    receipt["status"]="EXISTING_NO_SPEND"
    if cost is not None and cost>CAP:
        receipt["status"]="HOLD_CAP_EXCEEDED";return receipt
    if not latest.done():return receipt
    if receipt["provider_state"]!="Status.COMPLETED_OK" or cost is None:
        receipt["status"]="HOLD_TERMINAL_OR_COST";return receipt
    rs=latest.result(wait=False)
    samples=list(rs.predictions)
    receipt["returned_sample_count"]=len(samples)
    receipt["primary_scores"]=score_fields(rs.scores)
    if len(samples)!=N:
        receipt["status"]="HOLD_SAMPLE_COUNT_MISMATCH";return receipt
    seen=set();rows=[]
    for i,pred in enumerate(samples):
        uid=str(getattr(pred,"predicted_structure_uuid",None) or "")
        if not uid or uid in seen:
            raise Hold("MISSING_OR_DUPLICATE_STRUCTURE")
        seen.add(uid)
        rows.append({"index":i,"structure_uuid_sha256":hashlib.sha256(uid.encode()).hexdigest(),**score_fields(pred.scores)})
    receipt["samples"]=rows
    receipt["summary"]={k:{"mean":statistics.mean([x[k] for x in rows]),"sd":statistics.stdev([x[k] for x in rows]),"min":min(x[k] for x in rows),"max":max(x[k] for x in rows)} for k in ("ptm","avg_lddt")}
    receipt["credits_per_returned_sample"]=cost/N
    receipt["single_WT_comparator_credits"]=5.05
    receipt["cost_ratio_vs_five_repeated_single_WT_reference"]=cost/(N*5.05)
    receipt["status"]="RESULT_COLLECTED_ALL_FIVE"
    return receipt

def operation(api, action, approved, attempt):
    from probe import TARGET_ROWAN_FOLDER_UUID
    if action not in ("submit","collect") or (action=="collect" and approved):
        raise Hold("ACTION_INVALID")
    if action=="submit" and (not approved or attempt!="1"):
        raise Hold("EXPLICIT_FIRST_ATTEMPT_REQUIRED")
    if len(WT)!=136 or hashlib.sha256(WT.encode()).hexdigest()!=WT_SHA256:
        raise Hold("SEQUENCE_DRIFT")
    if "num_samples" not in inspect.signature(api.submit_protein_cofolding_workflow).parameters:
        raise Hold("SDK_NO_NUM_SAMPLES")
    receipt={"schema_id":"janus.rowan.r1e.multisample_wt.v1","case":NAME,"folder_uuid":TARGET_ROWAN_FOLDER_UUID,"sequence_sha256":WT_SHA256,"model":"BOLTZ_2","requested_samples":N,"use_msa_server":True,"max_credits":CAP,"new_jobs":0,"independent_seeds_verified":False,"provider_equivalence":False,"wetlab_validation":False,"weights_changed":False,"label_access":False}
    job=find(api,NAME)
    if job is not None:
        return collect(job,receipt)
    if action=="collect":
        receipt["status"]="NOT_FOUND_NO_SPEND";return receipt
    receipt["credit_gate"]=budget(api.whoami())
    job=api.submit_protein_cofolding_workflow(initial_protein_sequences=[WT],model=api.CofoldingModel.BOLTZ_2,num_samples=N,use_msa_server=True,max_credits=CAP,folder_uuid=TARGET_ROWAN_FOLDER_UUID,name=NAME,is_draft=False)
    if not getattr(job,"uuid",None):raise Hold("AMBIGUOUS_NO_AUTO_RETRY")
    receipt.update(status="SUBMITTED",new_jobs=1,provider_state=str(getattr(job,"status",None)),provider_uuid_sha256=hashlib.sha256(str(job.uuid).encode()).hexdigest())
    return receipt

def main():
    try:
        import rowan
        from importlib.metadata import version
        if version("rowan-python")!="3.2.3" or version("stjames")!="0.0.289":raise Hold("SDK_DRIFT")
        if not hasattr(rowan.ProteinCofoldingResult,"predictions"):raise Hold("SDK_NO_PREDICTIONS")
        out=Path(os.environ["R1E_RECEIPT"])
        if out.exists():raise Hold("IMMUTABLE_OUTPUT_EXISTS")
        r=operation(rowan,os.environ["R1E_ACTION"],os.environ.get("R1E_APPROVED")=="TRUE",os.environ.get("GITHUB_RUN_ATTEMPT",""))
        r["github_run_id"]=os.environ.get("GITHUB_RUN_ID")
        with out.open("x",encoding="utf8") as f:json.dump(r,f,indent=2,allow_nan=False)
        print("ROWAN_R1E_STATUS="+r["status"])
        print("ROWAN_R1E_NEW_JOBS="+str(r["new_jobs"]))
        print("ROWAN_R1E_REQUESTED_SAMPLES="+str(r["requested_samples"]))
        return 2 if r["status"].startswith("HOLD") else 0
    except Exception:
        print("ROWAN_R1E_HOLD_PROVIDER_OR_GATE_NO_AUTO_RETRY");return 2
if __name__=="__main__":raise SystemExit(main())
