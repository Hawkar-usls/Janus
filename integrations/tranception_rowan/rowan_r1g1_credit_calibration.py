"""R1G1 read-only hardware/runtime/cap provenance. Never submits provider jobs."""
import contextlib,hashlib,json,math,os,re
from pathlib import Path
from rowan_r1_cloud_pair import find,Hold
from rowan_r1g_panel import identity
from rowan_r1e_multisample import NAME as WT5
from rowan_r1f_multisample import NAME as MUT5
from budget_gate import finite_nonnegative
from probe import TARGET_ROWAN_FOLDER_UUID
CASE_NAMES={"BBC1_WT_5":identity("BBC1","WT",5)[1],"1EY0_WT_5":WT5,"1EY0_T8A_5":MUT5,"1EY0_WT_1":"JANUS-ROWAN-1EY0-R1C-PAIRED-R1-20261119-WT","1EY0_T8A_1":"JANUS-ROWAN-1EY0-R1C-PAIRED-R1-20261119-T8A"}
BBC_SEQ=identity("BBC1","WT",5)[2]
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def settings(j):
    d=getattr(j,"data",None)
    if not isinstance(d,dict):return {"available":False}
    seq=d.get("initial_protein_sequences")
    if isinstance(seq,list):
        seqs=[v if isinstance(v,str) else v.get("sequence") if isinstance(v,dict) else None for v in seq]
        seqhash=[digest(v) if isinstance(v,str) else None for v in seqs]
    else:seqhash=None
    model=d.get("model")
    return {"available":True,"sequence_sha256":seqhash,"num_samples":d.get("num_samples"),"model":model if isinstance(model,str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,40}",model) else None,"use_msa_server":d.get("use_msa_server"),"use_potentials":d.get("use_potentials"),"do_pose_refinement":d.get("do_pose_refinement"),"compute_strain":d.get("compute_strain"),"use_templates_server":d.get("use_templates_server")}
RULES={"budget_stop":r"(?:credit|budget).{0,100}(?:exceed|limit|insufficient|stop|exhaust)|(?:exceed|insufficient|stop|exhaust).{0,100}(?:credit|budget)","msa":r"\bmsa\b|multiple.sequence.alignment|colabfold|mmseqs","gpu_oom":r"out of memory|cuda oom","retry":r"\bretry\b|retrying|retried","diffusion":r"diffusion|sampling","hardware":r"\b(?:h100|h200|a100|a10|l40s|l4|t4|cuda)\b"}
def classify(log):
    return {k:len(re.findall(p,log,re.I)) for k,p in RULES.items()}
def interval(a,b):
    return (b-a).total_seconds() if a is not None and b is not None else None
def summarize(j):
    log=getattr(j,"logfile",None) or ""
    if not isinstance(log,str):log=""
    hardware=getattr(j,"compute_hardware",None)
    if not isinstance(hardware,str) or not re.fullmatch(r"[\w .:/+()\-]{1,80}",hardware):hardware=None
    # Only scalar metadata and keyword counts, never raw provider logs or identifiers.
    return {"state":str(j.status),"requested_max_credits":getattr(j,"max_credits",None),"credits_charged":finite_nonnegative(getattr(j,"credits_charged",None)),"elapsed_api_raw":finite_nonnegative(getattr(j,"elapsed",None)),"wall_execution_seconds":interval(getattr(j,"started_at",None),getattr(j,"completed_at",None)),"queue_seconds":interval(getattr(j,"created_at",None),getattr(j,"started_at",None)),"compute_hardware":hardware,"settings":settings(j),"log_available":bool(log),"log_sha256":digest(log) if log else None,"log_marker_counts":classify(log)}
def witness(api):
    records={}
    for k,name in CASE_NAMES.items():
        j=find(api,name)
        records[k]=summarize(j.fetch_latest()) if j is not None else {"state":"NOT_FOUND"}
    # Historical BBC1 singleton controls: inspect only exact sequence hashes, no unrelated jobs.
    matches=[]
    for page in range(15):
        batch=api.list_workflows(parent_uuid=TARGET_ROWAN_FOLDER_UUID,page=page,size=100)
        for j in batch:
            name=getattr(j,"name","")
            if "BBC1" not in name or "R1G" in name or getattr(j,"workflow_type",None)!="protein_cofolding":continue
            j=j.fetch_latest()
            if str(j.parent_uuid)!=TARGET_ROWAN_FOLDER_UUID:raise Hold("WRONG_FOLDER")
            s=settings(j)
            if s.get("sequence_sha256")==[digest(BBC_SEQ)]:
                if s.get("num_samples")!=1:continue
                matches.append({"name_sha256":digest(name),**summarize(j)})
        if len(batch)<100:break
    else:raise Hold("PAGINATION_BOUND")
    return {"schema":"janus.rowan.r1g1.credit_calibration.v1","status":"READ_ONLY_METADATA_CAPTURED","cases":records,"historical_bbc1_exact_single":matches,"new_jobs":0,"new_credit_spending_attempted":False,"cap_policy":"HOLD_ALL_NEW_R1G_SUBMISSIONS_PENDING_PROVIDER_CLARIFICATION","scope":"elapsed field units not assumed; keyword counts are not causal log evidence; provider hardware and runtime are reported metadata"}
def main():
    import rowan
    from importlib.metadata import version
    if version("rowan-python")!="3.2.3" or version("stjames")!="0.0.289":raise Hold("SDK_DRIFT")
    out=Path(os.environ["R1G1_RECEIPT"])
    result=witness(rowan);result["github_run_id"]=os.environ.get("GITHUB_RUN_ID")
    with out.open("x",encoding="utf8") as f:json.dump(result,f,indent=2,allow_nan=False)
    print("R1G1=READ_ONLY_METADATA_CAPTURED");print("R1G1_NEW_JOBS=0")
if __name__=="__main__":raise SystemExit(main())
