"""R1 cloud control: two explicitly approved repeat identities; one job per dispatch."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from rowan_1ey0_weekly_api import WT, WT_SHA256, budget, check_folder
from rowan_1ey0_T8A_weekly_pair import mutant, MUT_SHA256
from budget_gate import finite_nonnegative
from probe import TARGET_ROWAN_FOLDER_UUID

CAP=10
ROLES=("WT","T8A")
PREFIX="JANUS-ROWAN-1EY0-R1C-PAIRED-R1-20261119-"
class Hold(RuntimeError):
    pass

def identity(role):
    if role not in ROLES:
        raise Hold("ROLE_INVALID")
    seq, expected = (WT,WT_SHA256) if role=="WT" else (mutant(),MUT_SHA256)
    if len(seq)!=136 or hashlib.sha256(seq.encode("ascii")).hexdigest()!=expected:
        raise Hold("SEQUENCE_DRIFT")
    return PREFIX+role,seq,expected

def find(api,name):
    check_folder(api)
    matches=[]
    for page in range(15):
        batch=api.list_workflows(parent_uuid=TARGET_ROWAN_FOLDER_UUID,page=page,size=100)
        if len(batch)>100:
            raise Hold("PAGE_OVERFLOW")
        for job in batch:
            if getattr(job,"name",None)==name:
                if str(getattr(job,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID or str(getattr(job,"workflow_type",None))!="protein_cofolding":
                    raise Hold("WRONG_IDENTITY")
                matches.append(job)
        if len(batch)<100:
            break
    else:
        raise Hold("PAGINATION_BOUND")
    if len(matches)>1:
        raise Hold("DUPLICATE_REVIEW_REQUIRED")
    return matches[0] if matches else None

def operation(api, role, action, approved, attempt):
    if action not in ("submit","collect") or (action=="collect" and approved):
        raise Hold("ACTION_INVALID")
    if action=="submit" and (not approved or attempt!="1"):
        raise Hold("EXPLICIT_FIRST_ATTEMPT_REQUIRED")
    name,seq,digest=identity(role)
    job=find(api,name)
    receipt={"schema_id":"janus.rowan.r1.cloud_pair.v1","case":name,"role":role,"sequence_sha256":digest,"new_jobs":0,"max_credits_per_job":CAP,"model":"BOLTZ_2","num_samples":1,"use_msa_server":True,"scientific_scope":"technical_repeat_only","external_lab_validated":False,"provider_equivalence":False,"weights_changed":False,"label_access":False}
    if job is not None:
        receipt["status"]="EXISTING_NO_SPEND"
        latest=job.fetch_latest()
        if str(getattr(latest,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID or getattr(latest,"name",None)!=name:
            raise Hold("COLLECT_IDENTITY_DRIFT")
        receipt["provider_state"]=str(getattr(latest,"status",None))
        receipt["credits_charged"]=finite_nonnegative(getattr(latest,"credits_charged",None))
        if latest.done():
            data=latest.result(wait=False)
            scores=getattr(data,"scores",None)
            ptm=finite_nonnegative(getattr(scores,"ptm",None))
            lddt=finite_nonnegative(getattr(scores,"avg_lddt",None))
            if ptm is not None and lddt is not None and ptm<=1 and lddt<=1:
                receipt.update(ptm=ptm,avg_lddt=lddt,status="RESULT_COLLECTED")
        if receipt["credits_charged"] is not None and receipt["credits_charged"]>CAP:
            receipt["status"]="HOLD_CREDIT_CAP_EXCEEDED"
        return receipt
    if action=="collect":
        receipt["status"]="NOT_FOUND_NO_SPEND"
        return receipt
    receipt["credit_gate"]=budget(api.whoami())
    job=api.submit_protein_cofolding_workflow(initial_protein_sequences=[seq],model=api.CofoldingModel.BOLTZ_2,num_samples=1,use_msa_server=True,max_credits=CAP,folder_uuid=TARGET_ROWAN_FOLDER_UUID,name=name,is_draft=False)
    if not getattr(job,"uuid",None):
        raise Hold("AMBIGUOUS_SUBMISSION_NO_RETRY")
    receipt.update(status="SUBMITTED",new_jobs=1,provider_state=str(getattr(job,"status",None)))
    return receipt

def main():
    out=Path(os.environ["R1_RECEIPT"])
    if out.exists():
        raise Hold("IMMUTABLE_RECEIPT_EXISTS")
    try:
        import rowan
        r=operation(rowan,os.environ["R1_ROLE"],os.environ["R1_ACTION"],os.environ.get("R1_APPROVED")=="TRUE",os.environ.get("GITHUB_RUN_ATTEMPT",""))
        r["github_run_id"]=os.environ.get("GITHUB_RUN_ID")
        out.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print("ROWAN_R1_STATUS="+r["status"])
        print("ROWAN_R1_NEW_JOBS="+str(r["new_jobs"]))
        return 2 if r["status"].startswith("HOLD") else 0
    except Exception:
        print("ROWAN_R1_HOLD_PROVIDER_OR_GATE_NO_AUTO_RETRY")
        return 2

if __name__=="__main__":
    raise SystemExit(main())
