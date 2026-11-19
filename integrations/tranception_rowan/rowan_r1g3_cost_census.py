"""R1G3 bounded read-only cost census of already public source-bound controls."""
import hashlib,json,os,statistics
from pathlib import Path
from rowan_r1g1_credit_calibration import summarize
from rowan_1ey0_weekly_api import WT,check_folder
from rowan_1ey0_T8A_weekly_pair import mutant
from rowan_r1g_panel import WT as PANEL_WT
from rowan_r1g2_credit_guard import FEATURES
from rowan_scoby_placenta_two_protein_pilots import TARGETS
from rowan_scoby_milk_lalba_single import SEQUENCE as LALBA
from probe import TARGET_ROWAN_FOLDER_UUID
from rowan_r1_cloud_pair import Hold
def sha(s):return hashlib.sha256(s.encode("ascii")).hexdigest()
def allowlist():
    cases=[("1EY0","WT",WT),("1EY0","T8A",mutant()),("BBC1","WT",PANEL_WT["BBC1"]),("LALBA","WT",LALBA),("ACSD","WT",TARGETS["scoby_acsD"]["sequence"]),("FN1","WT",TARGETS["placenta_fn1"]["sequence"])]
    for m in ("G53A","G48D","E27F","W38L","K13N","G64Q","Q44V","E42H","K5A"):
        s=PANEL_WT["BBC1"];i=int(m[1:-1])-1
        if s[i]!=m[0]:raise Hold("PUBLIC_VARIANT_IDENTITY")
        cases.append(("BBC1",m,s[:i]+m[-1]+s[i+1:]))
    return {sha(s):(p,m,len(s)) for p,m,s in cases}

def census(api):
    check_folder(api); allowed=allowlist();rows=[];seen=set();rejected={};selected=0
    def reject(k):rejected[k]=rejected.get(k,0)+1
    for page in range(15):
        batch=api.list_workflows(parent_uuid=TARGET_ROWAN_FOLDER_UUID,page=page,size=100)
        if len(batch)>100:raise Hold("PAGE_OVERFLOW")
        for j in batch:
            if getattr(j,"workflow_type",None)!="protein_cofolding":continue
            # Only known JANUS public control namespace. Never expose unrelated jobs.
            name=getattr(j,"name","")
            if not isinstance(name,str) or not name.startswith("JANUS-") or not any(t in name for t in ("BBC1","1EY0","LALBA","ACSD","FN1")):continue
            selected+=1
            if selected>180:raise Hold("READ_BOUND")
            j=j.fetch_latest()
            if str(j.parent_uuid)!=TARGET_ROWAN_FOLDER_UUID:raise Hold("FOLDER")
            a=summarize(j);s=a["settings"];seq=s.get("sequence_sha256")
            if not isinstance(seq,list) or len(seq)!=1 or seq[0] not in allowed:reject("sequence_not_allowlisted");continue
            if not all(s.get(k)==v for k,v in FEATURES.items()):reject("settings_not_matched");continue
            if a["compute_hardware"]!="A100_80GB":reject("hardware_not_matched");continue
            n=s.get("num_samples")
            if type(n)!=int or n not in (1,5):reject("sample_count_unknown");continue
            if a["state"] not in ("Status.COMPLETED_OK","Status.STOPPED"):reject("not_terminal_control");continue
            t=a["elapsed_api_raw"];c=a["credits_charged"]
            if t is None or t<=0 or c is None or abs(c-t*.05)>.011:reject("billing_metadata_mismatch");continue
            if a["state"]=="Status.COMPLETED_OK" and (a["wall_execution_seconds"] is None or abs(a["wall_execution_seconds"]-t)>.03):reject("time_metadata_mismatch");continue
            uid=sha(str(j.uuid))
            if uid in seen:raise Hold("DUPLICATE_PROVIDER_ID")
            seen.add(uid);p,m,length=allowed[seq[0]]
            rows.append({"job_sha256":uid,"protein_group":p,"variant":m,"sequence_sha256":seq[0],"length":length,"samples":n,"hardware":a["compute_hardware"],"state":a["state"],"elapsed_seconds":t,"credits":c,"requested_cap":a["requested_max_credits"],"wall_seconds":a["wall_execution_seconds"]})
        if len(batch)<100:break
    else:raise Hold("PAGINATION_BOUND")
    return {"schema":"janus.rowan.r1g3.public_control_cost_census.v1","status":"READ_ONLY_CENSUS","records":sorted(rows,key=lambda x:(x["protein_group"],x["variant"],x["samples"],x["job_sha256"])),"n_records":len(rows),"n_selected_for_read":selected,"rejected":rejected,"new_jobs":0,"new_credits":0,"scope":"Historical observational records only, clustered by protein; no provider writes, labels, raw logs or sequence bytes exported."}
def qualify(c):
    rows=c["records"];done=[r for r in rows if r["state"]=="Status.COMPLETED_OK"]
    # Fit and test single-sample runtime only; protein-level split, equal family weighting.
    groups=sorted({r["protein_group"] for r in done if r["samples"]==1})
    folds=[]
    for g in groups:
        train=[r for r in done if r["samples"]==1 and r["protein_group"]!=g];test=[r for r in done if r["samples"]==1 and r["protein_group"]==g]
        if not train:continue
        tg=sorted({r["protein_group"] for r in train})
        med=statistics.median([statistics.median([r["elapsed_seconds"] for r in train if r["protein_group"]==q]) for q in tg])
        # proportional length comparator with no intercept; no fitted multivariate model
        rate=statistics.median([statistics.median([r["elapsed_seconds"]/r["length"] for r in train if r["protein_group"]==q]) for q in tg])
        err0=statistics.mean(abs(r["elapsed_seconds"]-med) for r in test);err1=statistics.mean(abs(r["elapsed_seconds"]-r["length"]*rate) for r in test)
        folds.append({"heldout_protein":g,"n_train_groups":len(tg),"n_test":len(test),"constant_seconds":med,"length_seconds_per_residue":rate,"constant_mae_seconds":err0,"length_mae_seconds":err1,"constant_125_undercoverage":sum(r["elapsed_seconds"]>med*1.25 for r in test),"length_125_undercoverage":sum(r["elapsed_seconds"]>r["length"]*rate*1.25 for r in test)})
    five=[r for r in rows if r["samples"]==5]
    return {"status":"HOLD_COST_MODEL_QUALIFICATION","counts":{"records":len(rows),"completed":len(done),"stopped":sum(r["state"]=="Status.STOPPED" for r in rows),"protein_groups":len({r["protein_group"] for r in rows}),"single_completed":sum(r["samples"]==1 for r in done),"five_completed":sum(r["samples"]==5 for r in done),"five_completed_groups":len({r["protein_group"] for r in done if r["samples"]==5})},"single_sample_leave_one_protein_out":folds,"macro_mae_seconds":{k:statistics.mean(f[k] for f in folds) if folds else None for k in ("constant_mae_seconds","length_mae_seconds")},"five_sample_records":five,"qualified_risk_bound":False,"paid_submission_authorized":False,"limitations":"Few protein clusters; retrospective, non-independent jobs within groups; no guaranteed cap; no qualified five-sample out-of-protein predictor."}
def main():
    import rowan
    from importlib.metadata import version
    if version("rowan-python")!="3.2.3" or version("stjames")!="0.0.289":raise Hold("SDK_DRIFT")
    c=census(rowan);c["github_run_id"]=os.environ.get("GITHUB_RUN_ID");c["qualification"]=qualify(c)
    with Path(os.environ["R1G3_RECEIPT"]).open("x",encoding="utf8") as f:json.dump(c,f,indent=2,allow_nan=False)
    print("R1G3_RECORDS="+str(c["n_records"]));print("R1G3=HOLD_COST_MODEL_QUALIFICATION");print("R1G3_NEW_JOBS=0")
if __name__=="__main__":main()
