"""R1G frozen seven-case engineering pilot; manual weekly credits only, no retry."""
from pathlib import Path
import hashlib,json,os,time,statistics,math
from rowan_r1_cloud_pair import find,Hold
from rowan_r1e_multisample import score_fields
from budget_gate import finite_nonnegative
from probe import TARGET_ROWAN_FOLDER_UUID

WT={
 "BBC1":"EVPFKVVAQFPYKSDYEDDLNFEKDQEIIVTSVEDAEWYFGEYQDSNGDVIEGIFPKSFVAVQG",
 "2LZM":"MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
}
CASES=(("BBC1","WT",5),("BBC1","E27F",5),("2LZM","WT",5),("2LZM","I3T",5),("BBC1","E42H",5),("2LZM","WT",1),("2LZM","I3T",1))
CAP=10
FLOOR=40
RESERVE=30
WAIT=150
def sha(s):return hashlib.sha256(s.encode("ascii")).hexdigest()
def identity(protein,mut,n):
    if (protein,mut,n) not in CASES:raise Hold("CASE_OUTSIDE_PLAN")
    seq=WT[protein]
    if len(seq)!={"BBC1":64,"2LZM":164}[protein]:raise Hold("LENGTH")
    if mut!="WT":
        pos=int(mut[1:-1])-1
        if seq[pos]!=mut[0]:raise Hold("MUTATION_SOURCE_MISMATCH")
        seq=seq[:pos]+mut[-1]+seq[pos+1:]
    if protein=="2LZM":
        expected={"WT":"0619a8bdf08b15085930db9c8a635510324e67bfd558bb2f8d1654b896355133","I3T":"c5280e3fa313d52b1c12658e9b1118f8b65fab06887361f30c9181e78b7e1383"}[mut]
        if sha(seq)!=expected:raise Hold("SOURCE_HASH")
    return f"{protein}_{mut}_{n}",f"JANUS-ROWAN-R1G-{protein}-{mut}-{n}S-20261119-R0",seq
def weekly(api):
    u=api.whoami()
    org=getattr(u,"organization",None)
    if org is None or getattr(u,"individual_subscription",None) is not None:raise Hold("ACCOUNT_NOT_WEEKLY_ORG")
    n=finite_nonnegative(getattr(org,"weekly_credits",None))
    if n is None:raise Hold("WEEKLY_BALANCE_UNKNOWN")
    return n
def read_case(api,job,case,name,seq,n):
    job=job.fetch_latest()
    if job.name!=name or str(job.parent_uuid)!=TARGET_ROWAN_FOLDER_UUID:raise Hold("CASE_IDENTITY")
    cost=finite_nonnegative(getattr(job,"credits_charged",None))
    out={"case":case,"sequence_sha256":sha(seq),"sequence_length":len(seq),"requested_samples":n,"provider_state":str(job.status),"credits_charged":cost,"new_jobs":0,"status":"PENDING"}
    if cost is not None and cost>CAP:raise Hold("CAP_EXCEEDED")
    if not job.done():return out
    if str(job.status)!="Status.COMPLETED_OK" or cost is None:raise Hold("TERMINAL_OR_COST_UNKNOWN")
    rs=job.result(wait=False); samples=list(rs.predictions)
    if len(samples)!=n:raise Hold("SAMPLE_COUNT")
    seen=set();rows=[]
    for i,pred in enumerate(samples):
        uid=str(getattr(pred,"predicted_structure_uuid",None) or "")
        if not uid or uid in seen:raise Hold("DUPLICATE_OR_MISSING_STRUCTURE")
        seen.add(uid);rows.append({"index":i,"structure_uuid_sha256":sha(uid),**score_fields(pred.scores)})
    out.update(status="COMPLETE",samples=rows,primary_scores=score_fields(rs.scores),credits_per_sample=cost/n)
    out["summary"]={}
    for k in ("ptm","avg_lddt"):
        vals=[x[k] for x in rows]
        loo=[statistics.mean(vals[:i]+vals[i+1:]) for i in range(n)] if n>1 else []
        out["summary"][k]={"mean":statistics.mean(vals),"sd":statistics.stdev(vals) if n>1 else None,"min":min(vals),"max":max(vals),"loo_min":min(loo) if loo else None,"loo_max":max(loo) if loo else None}
    return out
def run(api,action,approved,attempt,save,sleep=time.sleep):
    if action not in ("submit","collect") or (action=="collect" and approved):raise Hold("ACTION")
    if action=="submit" and (not approved or attempt!="1"):raise Hold("EXPLICIT_FIRST_ATTEMPT_REQUIRED")
    if action=="submit":raise Hold("R1G2_HOLD_PROVIDER_CAP_UNBOUNDED")
    state={"schema":"janus.rowan.r1g.panel.receipt.v1","status":"RUNNING","model":"BOLTZ_2","use_msa_server":True,"max_total_cap":70,"new_jobs":0,"new_jobs_confirmed_complete":0,"cases":[],"provider_equivalence":False,"wetlab_validation":False,"label_access":False,"weights_changed":False,"seed_independence_verified":False,"github_run_id":os.getenv("GITHUB_RUN_ID")}
    save(state)
    charged_new=0
    initial=weekly(api) if action=="submit" else None
    try:
        # Complete identity scan before spending; pending old jobs block any submission.
        old={}
        for p,m,n in CASES:
            key,name,seq=identity(p,m,n);old[key]=find(api,name)
        if action=="submit":
            for p,m,n in CASES:
                key,name,seq=identity(p,m,n)
                if old[key] is not None and read_case(api,old[key],key,name,seq,n)["status"]!="COMPLETE":
                    state["status"]="HOLD_PREEXISTING_PENDING";save(state);return state
        for p,m,n in CASES:
            key,name,seq=identity(p,m,n);job=old[key];new=0
            if job is None and action=="collect":
                state["cases"].append({"case":key,"status":"NOT_FOUND","new_jobs":0});save(state);continue
            if job is None:
                # Reserve 30 weekly credits even if the next job reaches its cap.
                if min(weekly(api),initial-charged_new)<FLOOR:
                    state["status"]="HOLD_WEEKLY_RESERVE";save(state);return state
                if state["new_jobs"]>=7 or charged_new+CAP>70:raise Hold("TOTAL_BUDGET")
                # Durable intent before any provider write. Never retry ambiguous submissions.
                state["pending_intent"]=key;save(state)
                job=api.submit_protein_cofolding_workflow(initial_protein_sequences=[seq],model=api.CofoldingModel.BOLTZ_2,num_samples=n,use_msa_server=True,max_credits=CAP,folder_uuid=TARGET_ROWAN_FOLDER_UUID,name=name,is_draft=False)
                if not getattr(job,"uuid",None):raise Hold("AMBIGUOUS_SUBMISSION")
                new=1;state["new_jobs"]+=1
                state["pending_intent"]=None;state["active_case"]=key;save(state)
                print("R1G_SUBMITTED="+key,flush=True)
            result=read_case(api,job,key,name,seq,n)
            if new:
                for _ in range(WAIT//10):
                    if result["status"]=="COMPLETE":break
                    sleep(10);result=read_case(api,job,key,name,seq,n)
                if result["status"]!="COMPLETE":
                    result["new_jobs"]=1;state["cases"].append(result);state["status"]="HOLD_PENDING_NO_FURTHER_SPEND";save(state);return state
                charged_new+=result["credits_charged"];state["new_jobs_confirmed_complete"]+=1
            result["new_jobs"]=new;state["cases"].append(result);state["active_case"]=None
            save(state);print("R1G_CASE="+key+":"+result["status"],flush=True)
        state["status"]="COMPLETE_ALL_CASES" if all(x["status"]=="COMPLETE" for x in state["cases"]) else "PARTIAL_READ_ONLY"
    except Exception as e:
        state["status"]="HOLD_NO_AUTO_RETRY"
        state["error_class"]=type(e).__name__
        if isinstance(e,Hold):state["reason"]=str(e)
    save(state);return state
def main():
    import rowan
    from importlib.metadata import version
    if version("rowan-python")!="3.2.3" or version("stjames")!="0.0.289":raise Hold("SDK_DRIFT")
    p=Path(os.environ["R1G_RECEIPT"])
    if p.exists():raise Hold("OUTPUT_EXISTS")
    def write(s):
        temp=p.with_suffix(".tmp");temp.write_text(json.dumps(s,indent=2,allow_nan=False),encoding="utf8");temp.replace(p)
    s=run(rowan,os.environ["R1G_ACTION"],os.environ.get("R1G_APPROVED")=="TRUE",os.environ.get("GITHUB_RUN_ATTEMPT",""),write)
    print("R1G_STATUS="+s["status"]);print("R1G_NEW_JOBS="+str(s["new_jobs"]))
    return 2 if s["status"].startswith("HOLD") else 0
if __name__=="__main__":raise SystemExit(main())
