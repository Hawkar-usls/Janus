"""Offline credit risk triage. Estimates are exploratory, not financial bounds."""
from pathlib import Path
from decimal import Decimal as D
import hashlib,json,math
SOURCE=Path(__file__).with_name("rowan_r1g2_cost_history.json")
FEATURES={"model":"boltz_2","use_msa_server":True,"use_potentials":False,"do_pose_refinement":False,"compute_strain":False,"use_templates_server":False}
MARGIN=D("1.25")
def require(x,msg):
    if not x:raise ValueError(msg)
def load():
    x=json.loads(SOURCE.read_text())
    require(x["source_receipt_sha256"]=="f6a739f92edb16fcd63a62142a483a261159c3f6caa983519758ac48f6f63086","source")
    require(x["safety"]=={"provider_cap_is_hard":False,"provider_verified_overshoot_bound":None,"submissions_allowed":False},"policy")
    require(len(x["records"])==6,"count")
    return x
def dec(x):
    d=D(str(x));require(d.is_finite() and d>=0,"invalid numeric field");return d
def sequence_hash(seq):return hashlib.sha256(seq.encode("ascii")).hexdigest()
def settings_match(row,hardware,features):
    return row["hardware"]==hardware and features==FEATURES and all(row["settings"].get(k)==v for k,v in FEATURES.items())
def report(seq,n,hardware,features,cap=10,history=None):
    require(isinstance(seq,str) and bool(seq) and set(seq)<=set("ACDEFGHIKLMNPQRSTVWY"),"sequence")
    require(type(n)==int and n>0,"samples")
    cap=dec(cap);require(cap>0,"cap")
    source=load() if history is None else history
    key=sequence_hash(seq);rate=source["tariff_credits_per_minute"].get(hardware)
    out={"sequence_sha256":key,"length":len(seq),"samples":n,"hardware":hardware,"settings":features,"cap":str(cap),"admission":"HOLD_PROVIDER_CAP_UNBOUNDED","new_credit_authorization":False,"estimated_cost":None,"planning_envelope_not_bound":None,"stopped_cost_lower_bound":None,"risk":"UNKNOWN"}
    if rate is None or features!=FEATURES:return out
    rate=dec(rate)
    out["runtime_threshold_seconds"]=str(cap*60/rate)
    matches=[r for r in source["records"] if settings_match(r,hardware,features) and r["settings"]["sequence_sha256"]==[key] and r["settings"]["num_samples"]==n]
    if not matches:return out
    complete=[r for r in matches if r["status"]=="Status.COMPLETED_OK"]
    stopped=[r for r in matches if r["status"]=="Status.STOPPED"]
    if complete:
        t=sorted(dec(r["elapsed_seconds"]) for r in complete)
        mean=sum(t)/len(t)
        out["estimated_cost"]=str(mean*rate/60)
        out["planning_envelope_not_bound"]=str(max(t)*rate/60*MARGIN)
        out["risk"]="HEURISTIC_HIGH" if dec(out["planning_envelope_not_bound"])>=cap else "OBSERVED_BELOW_CAP_NOT_CALIBRATED"
    if stopped:
        lower=max(max(dec(r["credits"]),dec(r["elapsed_seconds"])*rate/60) for r in stopped)
        out["stopped_cost_lower_bound"]=str(lower)
        out["risk"]="OBSERVED_CAP_BREACH" if lower>=cap else "CENSORED_UNKNOWN"
    out["matching_completed_records"]=len(complete)
    out["matching_stopped_records"]=len(stopped)
    return out
def transfer_falsification(source):
    # Held-out protein BBC1: 1EY0 completed two ratios; never treat censored cost as a target.
    rows={r["id"]:r for r in source["records"]}
    factors=[dec(rows[p+"_5"]["elapsed_seconds"])/dec(rows[p+"_1"]["elapsed_seconds"]) for p in ("1EY0_WT","1EY0_T8A")]
    pred=max(factors)*dec(rows["BBC1_WT_1_HISTORICAL_0"]["elapsed_seconds"])
    lower=dec(rows["BBC1_WT_5"]["elapsed_seconds"])
    return {"train_completed_ratios":[str(x) for x in factors],"heldout_bbc1_predicted_seconds_upper_from_training_max":str(pred),"observed_stopped_elapsed_lower_bound":str(lower),"transfer_upper_estimate_falsified":pred<lower,"predicted_cost":str(pred*D("0.05")),"observed_cost_lower_bound":str(lower*D("0.05")),"conclusion":"Do not extrapolate from short sequence or 1EY0 sample factor; stopped duration is a lower bound, not completed runtime."}
def panel():
    from rowan_r1g_panel import CASES,identity
    source=load();out=[]
    for p,m,n in CASES:
        key,_,seq=identity(p,m,n)
        out.append({"case":key,**report(seq,n,"A100_80GB",FEATURES,history=source)})
    return {"schema":"janus.rowan.r1g2.shadow_risk.v1","status":"HOLD_ALL_PAID_SUBMISSIONS","cases":out,"transfer_falsification":transfer_falsification(source),"n_historical":6,"n_completed":5,"n_right_censored":1,"calibrated_upper_bound":False,"new_jobs":0,"credits_spent":0}
if __name__=="__main__":print(json.dumps(panel(),indent=2,allow_nan=False))
