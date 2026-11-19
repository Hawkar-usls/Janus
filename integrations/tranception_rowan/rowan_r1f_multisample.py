"""R1F one frozen 1EY0 T8A five-diffusion-sample cloud job; never auto retry."""
import hashlib, inspect, json, os, statistics
from pathlib import Path
from rowan_1ey0_weekly_api import budget
from rowan_1ey0_T8A_weekly_pair import mutant, MUT_SHA256
SEQ=mutant()
from rowan_r1_cloud_pair import find, Hold
from budget_gate import finite_nonnegative

NAME="JANUS-ROWAN-1EY0-R1F-T8A-5S-20261119-R0"
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
    receipt["single_T8A_comparator_credits"]=3.61
    receipt["cost_ratio_vs_five_repeated_single_T8A_reference"]=cost/(N*3.61)
    receipt["status"]="RESULT_COLLECTED_ALL_FIVE"
    return receipt

def operation(api, action, approved, attempt):
    from probe import TARGET_ROWAN_FOLDER_UUID
    if action not in ("submit","collect") or (action=="collect" and approved):
        raise Hold("ACTION_INVALID")
    if action=="submit" and (not approved or attempt!="1"):
        raise Hold("EXPLICIT_FIRST_ATTEMPT_REQUIRED")
    if len(SEQ)!=136 or hashlib.sha256(SEQ.encode()).hexdigest()!=MUT_SHA256:
        raise Hold("SEQUENCE_DRIFT")
    if "num_samples" not in inspect.signature(api.submit_protein_cofolding_workflow).parameters:
        raise Hold("SDK_NO_NUM_SAMPLES")
    receipt={"schema_id":"janus.rowan.r1f.multisample_t8a.v1","case":NAME,"folder_uuid":TARGET_ROWAN_FOLDER_UUID,"sequence_sha256":MUT_SHA256,"model":"BOLTZ_2","requested_samples":N,"use_msa_server":True,"max_credits":CAP,"new_jobs":0,"independent_seeds_verified":False,"provider_equivalence":False,"wetlab_validation":False,"weights_changed":False,"label_access":False}
    job=find(api,NAME)
    if job is not None:
        return collect(job,receipt)
    if action=="collect":
        receipt["status"]="NOT_FOUND_NO_SPEND";return receipt
    receipt["credit_gate"]=budget(api.whoami())
    job=api.submit_protein_cofolding_workflow(initial_protein_sequences=[SEQ],model=api.CofoldingModel.BOLTZ_2,num_samples=N,use_msa_server=True,max_credits=CAP,folder_uuid=TARGET_ROWAN_FOLDER_UUID,name=NAME,is_draft=False)
    if not getattr(job,"uuid",None):raise Hold("AMBIGUOUS_NO_AUTO_RETRY")
    receipt.update(status="SUBMITTED",new_jobs=1,provider_state=str(getattr(job,"status",None)),provider_uuid_sha256=hashlib.sha256(str(job.uuid).encode()).hexdigest())
    return receipt

def export_coordinates(api, receipt, out):
    from rowan_r1e_export_structures import AA, sha
    import math
    job=find(api,NAME)
    if job is None:raise Hold("EXPORT_JOB_MISSING")
    job=job.fetch_latest()
    if job.name!=NAME or str(job.parent_uuid)!=receipt["folder_uuid"] or str(job.status)!="Status.COMPLETED_OK" or not job.done():raise Hold("EXPORT_IDENTITY")
    if float(job.credits_charged)!=receipt["credits_charged"]:raise Hold("EXPORT_COST_CHANGED")
    expected={x["structure_uuid_sha256"]:x for x in receipt["samples"]}
    predictions=list(job.result(wait=False).predictions)
    if len(predictions)!=N:raise Hold("EXPORT_COUNT")
    by_hash={}
    for p in predictions:
        uid=str(p.predicted_structure_uuid or "")
        h=sha(uid.encode())
        if not uid or h in by_hash or h not in expected:raise Hold("EXPORT_UUID")
        if any(score_fields(p.scores)[k]!=expected[h][k] for k in ("ptm","avg_lddt")):raise Hold("EXPORT_SCORE_DRIFT")
        by_hash[h]=uid
    out.mkdir(exist_ok=False)
    temp=out.parent/"r1f-raw-pdb";temp.mkdir(exist_ok=False)
    rows=[]
    for x in receipt["samples"]:
        h=x["structure_uuid_sha256"];uid=by_hash[h]
        name=f"1EY0_T8A_R1F_sample_{x['index']+1:02}"
        protein=api.retrieve_protein(uid,workflow_uuid=str(job.uuid))
        path=protein.download_structure(path=temp,name=name,workflow_uuid=str(job.uuid),file_format="pdb")
        if Path(path).resolve()!=(temp/(name+".pdb")).resolve():raise Hold("EXPORT_PATH")
        atoms=[s for s in Path(path).read_text(encoding="utf8").splitlines() if s.startswith("ATOM  ")]
        residues={};ca={}
        for s in atoms:
            if len(s)<54 or s[17:20] not in AA:raise Hold("PDB_FORMAT")
            if not all(math.isfinite(float(s[i:i+8])) for i in (30,38,46)):raise Hold("PDB_NONFINITE")
            key=(s[21],s[22:27]);aa=AA[s[17:20]]
            if key in residues and residues[key]!=aa:raise Hold("RESIDUE_CONFLICT")
            residues[key]=aa
            if s[12:16].strip()=="CA":ca[key]=ca.get(key,0)+1
        if len(set(k[0] for k in residues))!=1 or "".join(residues.values())!=SEQ or set(ca)!=set(residues) or any(n!=1 for n in ca.values()):raise Hold("PDB_SEQUENCE")
        cleaned=("\n".join(atoms)+"\nEND\n").encode("ascii")
        dest=out/(name+".pdb")
        with dest.open("xb") as f:f.write(cleaned)
        rows.append({**x,"filename":dest.name,"pdb_sha256":sha(cleaned),"atom_count":len(atoms),"residue_count":len(residues)})
    if len(set(x["pdb_sha256"] for x in rows))!=N:raise Hold("PDB_IDENTICAL")
    (out/"manifest.json").write_text(json.dumps({"status":"PASS_FIVE_EXPORTED_PDBS","case":NAME,"sequence_sha256":MUT_SHA256,"samples":rows,"new_jobs":0,"existing_job_credits":receipt["credits_charged"],"collection_run":receipt["github_run_id"]},indent=2,allow_nan=False),encoding="utf8")
    print("R1F_PDB_EXPORT=PASS_FIVE_EXPORTED_PDBS")

def main():
    try:
        import rowan
        from importlib.metadata import version
        if version("rowan-python")!="3.2.3" or version("stjames")!="0.0.289":raise Hold("SDK_DRIFT")
        if not hasattr(rowan.ProteinCofoldingResult,"predictions"):raise Hold("SDK_NO_PREDICTIONS")
        out=Path(os.environ["R1F_RECEIPT"])
        if out.exists():raise Hold("IMMUTABLE_OUTPUT_EXISTS")
        r=operation(rowan,os.environ["R1F_ACTION"],os.environ.get("R1F_APPROVED")=="TRUE",os.environ.get("GITHUB_RUN_ATTEMPT",""))
        r["github_run_id"]=os.environ.get("GITHUB_RUN_ID")
        with out.open("x",encoding="utf8") as f:json.dump(r,f,indent=2,allow_nan=False)
        print("ROWAN_R1F_STATUS="+r["status"])
        print("ROWAN_R1F_NEW_JOBS="+str(r["new_jobs"]))
        print("ROWAN_R1F_REQUESTED_SAMPLES="+str(r["requested_samples"]))
        if os.environ["R1F_ACTION"]=="collect" and r["status"]=="RESULT_COLLECTED_ALL_FIVE":
            export_coordinates(rowan,r,Path(os.environ["R1F_EXPORT_DIR"]))
        return 2 if r["status"].startswith("HOLD") else 0
    except Exception:
        print("ROWAN_R1F_HOLD_PROVIDER_OR_GATE_NO_AUTO_RETRY");return 2
if __name__=="__main__":raise SystemExit(main())
