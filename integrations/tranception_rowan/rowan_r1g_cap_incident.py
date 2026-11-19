"""R1G credit-cap incident witness. Read-only, one existing case, no provider writes."""
import json,os,statistics
from pathlib import Path
from rowan_r1g_panel import identity,CAP
from rowan_r1_cloud_pair import find,Hold
from budget_gate import finite_nonnegative
from rowan_r1e_multisample import score_fields
from probe import TARGET_ROWAN_FOLDER_UUID
import hashlib

def inspect_existing(api):
    case,name,seq=identity("BBC1","WT",5)
    r={"schema":"janus.rowan.r1g.cap_incident.v1","case":case,"source_submit_run":"44565022896","source_hold_run":"44571163182","max_credits_requested":CAP,"sequence_sha256":hashlib.sha256(seq.encode()).hexdigest(),"new_jobs":0,"credit_spending_attempted":False,"status":"NOT_FOUND"}
    j=find(api,name)
    if j is None:return r
    j=j.fetch_latest()
    if j.name!=name or str(j.parent_uuid)!=TARGET_ROWAN_FOLDER_UUID:raise Hold("IDENTITY")
    cost=finite_nonnegative(getattr(j,"credits_charged",None))
    r.update(provider_state=str(j.status),provider_reported_credits=cost,provider_done=bool(j.done()),cap_exceeded=cost is not None and cost>CAP,status="HOLD_CAP_EXCEEDED" if cost is not None and cost>CAP else "READ_ONLY")
    if j.done() and str(j.status)=="Status.COMPLETED_OK":
        p=list(j.result(wait=False).predictions)
        rows=[]
        seen=set()
        for i,x in enumerate(p):
            uid=str(x.predicted_structure_uuid or "")
            if not uid or uid in seen:raise Hold("STRUCTURE_ID")
            seen.add(uid);rows.append({"index":i,"id_sha256":hashlib.sha256(uid.encode()).hexdigest(),**score_fields(x.scores)})
        r["returned_samples"]=len(rows);r["samples"]=rows
        if len(rows)==5:r["summary"]={k:{"mean":statistics.mean(z[k] for z in rows),"sample_sd":statistics.stdev(z[k] for z in rows)} for k in ("ptm","avg_lddt")}
    return r
def main():
    import rowan
    out=Path(os.environ["R1G_INCIDENT_RECEIPT"])
    r=inspect_existing(rowan)
    r["github_run_id"]=os.environ.get("GITHUB_RUN_ID")
    with out.open("x",encoding="utf8") as f:json.dump(r,f,indent=2,allow_nan=False)
    print("R1G_INCIDENT_STATUS="+r["status"])
    print("R1G_INCIDENT_NEW_JOBS=0")
    return 0
if __name__=="__main__":raise SystemExit(main())
