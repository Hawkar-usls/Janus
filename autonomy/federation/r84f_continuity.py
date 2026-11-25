"""R84F finite neural lineage experiment, inherited R84E tensors, no production."""
import json,random,math,time,sys
from pathlib import Path
import r84e_neural as e
import jfp_gateway as g
import r8_learning as r
HERE=Path(__file__).parent
EXAM="6f2131a2935c51d4636b9e855ba54c28bc7447ac"
DATA="417b5b65dc6c3c8ff4b13c2d1cff71e1a35601a6b92a8aa2a65ede4277535da1"
ARMS=("replay","joint")
PINS={"replay":["bfe866cb965d44a5b2857d8fd2cd72c25cfa150a094b1c17baf06de9fb7ead21","25b61206381996be0d2ad7cd4a15e74c145234294dda03da7def5fa24002f0e3","e0f6185bdfeb080b8935887250e96f0c165ee94779f6c9d8cc09069b9d48c19d"],"joint":["978e3981c9c5a4f58571b0bae278123d4242cc1e385501edf8704036d687e19b","ee901a97af94b427f5fa9821c3a99f208b94f231fd64bdfec6d0fcd558d5dae9","8de01662162d51b2bc0fdd086c42e973395855c6ec6915957957b69364bb44a7"]}
TRAIN=[
{"f_hold_source":["source missing origin not verified","unknown provenance hold the source","dataset lacks traceable source identity","no provenance available for evidence","unverified origin of research material"],
"f_hold_rights":["verified source but license permission missing","unknown copyright permission for reuse","rights unclear hold publication material","license not permitting redistribution","source confirmed but reuse rights absent"],
"f_hold_conflict":["verified measurements conflict with each other","trusted evidence disagrees across controls","independent results contradict one another","reconcile conflicting measurements before acceptance","valid observations show contradictory outcomes"],
"f_accept_review":["verified origin and license with reproducible evidence","source permission and replication all confirmed","independent reproduction source rights passed","provenance rights verified and result reproduced","all checks passed with reproducible measurements"]},
{"f_retry_transient":["temporary endpoint failure and retries available","network timeout remaining attempts permitted","retry after transient connection outage","short service outage within retry budget","temporary connection error with attempts remaining"],
"f_stop_budget":["budget exhausted halt further attempts","compute resource limit reached stop","maximum attempts reached no retries","resource cap reached before execution","remaining compute budget is zero"],
"f_stop_operator":["owner manually paused current run","operator stop flag enabled","manual disable command received","human authorization withdrawn stop work","operator explicitly halted research execution"],
"f_restore_checkpoint":["corrupted model restore verified parent","checksum mismatch recover last valid checkpoint","candidate state corrupt revert stable version","failed generation restore known good state","recover previous verified model after corruption"]}]
def ck(v,s):e.ck(v,s)
def log(t,x):e.log(t,x)
def groups():return [[["gate practice "+p,y] for y,ps in group.items() for p in ps] for group in TRAIN]
def exam():
    raw=(HERE/"R84F_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");z=json.loads(raw);ck(len(z["cohorts"])==2 and all(len(x)==16 for x in z["cohorts"]) and len(z["legacy_new"])==10,"EXAM")
    return z
def orig(prep):
    d=e.read(prep/"data.json");ck(g.digest(d)==DATA and d["production"] is False,"DATA_PIN");return d
def data(d,cycle):
    ck(cycle in (1,2),"CYCLE")
    z={**d,"vocab":dict(d["vocab"]),"labels":dict(d["labels"]),"old":d["old"]+d["new"]+sum(groups()[:cycle-1],[]),"new":groups()[cycle-1],"legacy_vocab":dict(d["vocab"]),"f_cycle":cycle,"f_exam":EXAM}
    for j in range(cycle):
        for w in sorted({w for x,y in groups()[j] for w in e.tokenize(x)}-set(z["vocab"])):
            z["vocab"][w]=max(z["vocab"].values())+1
        for y in sorted(set(TRAIN[j])-set(z["labels"])):
            z["labels"][y]=max(z["labels"].values())+1
    ck(max(z["vocab"].values())<200 and max(z["labels"].values())<258,"VOCAB_CAP")
    return z
def configure(d):
    # Retain original tokenizer exactly for legacy queries, including unknown words.
    def encode(z,s):
        vocab=z["vocab"] if s.startswith("gate ") else z.get("legacy_vocab",z["vocab"])
        # Both practice/exam are formatting prefixes, not label hints.
        if s.startswith("gate practice "):s="gate "+s[len("gate practice "):]
        if s.startswith("gate exam "):s="gate "+s[len("gate exam "):]
        ids=[0]+[vocab.get(w,1) for w in e.tokenize(s)]+[0]
        ck(len(ids)<=24,"LENGTH");return ids
    e.encode=encode
def pin_parent(arm,seed,p):
    ck(p["model_sha256"]==PINS[arm][e.SEEDS.index(seed)] and p["seed"]==seed and p["arm"]==arm and p["phase"]==2,"E_PARENT_PIN")
def train(prep,parent,out,arm,cycle):
    ck(arm in ARMS and cycle in (1,2),"SCOPE")
    L=e.lib(prep);d0=orig(prep);d=data(d0,cycle);prev=d0 if cycle==1 else data(d0,cycle-1)
    configure(d);out.mkdir(parents=True,exist_ok=False)
    # Fixed finite 240 updates, 12 examples per step, 25% replay. Old and new token IDs unchanged.
    e.NEW_STEPS=240
    for seed in e.SEEDS:
        m,p=e.load(L,prev,parent/str(seed))
        if cycle==1:pin_parent(arm,seed,p)
        else:ck(p["seed"]==seed and p["arm"]==arm and p["cycle"]==cycle-1 and p["root_sha256"]==PINS[arm][e.SEEDS.index(seed)],"F_PARENT_PIN")
        q=e.optimize(L,m,d,seed+100*cycle,2,arm)
        ck(q["steps"]==240 and q["examples"]==2880 and q["old_examples"]==720,"BUDGET")
        rec={"seed":seed,"arm":arm,"cycle":cycle,"parent_sha256":p["model_sha256"],"root_sha256":PINS[arm][e.SEEDS.index(seed)],**q}
        e.save(L,m,d,out/str(seed),rec);log("R84F_TRAIN",rec)
def evaluate(prep,replay0,joint0,replay1,joint1,replay2,joint2,out):
    L=e.lib(prep);d0=orig(prep);ds=[d0,data(d0,1),data(d0,2)];z=exam();old,facts,_=e.exam()
    all_train={t for t,y in d0["old"]+d0["new"]+sum(groups(),[])}
    cohorts=[old,facts,z["legacy_new"],*z["cohorts"]]
    ck(not all_train&{t for grp in cohorts for t,y in grp},"RAW_OVERLAP")
    norm=lambda s:s.replace("gate practice ","gate ").replace("gate exam ","gate ")
    ck(not {norm(t) for t in all_train}&{norm(t) for grp in cohorts for t,y in grp},"NORMALIZED_OVERLAP")
    ck(len({norm(t) for grp in z["cohorts"] for t,y in grp})==32,"UNIQUE_EXAM")
    configure(ds[-1]);results=[]
    for arm,folders in zip(ARMS,[(replay0,replay1,replay2),(joint0,joint1,joint2)]):
        for seed in e.SEEDS:
            last=None;last_preds=None
            for cycle,folder in enumerate(folders):
                m,p=e.load(L,ds[cycle],folder/str(seed))
                if cycle==0:pin_parent(arm,seed,p)
                else:ck(p["cycle"]==cycle and p["arm"]==arm and p["seed"]==seed and p["parent_sha256"]==last and p["root_sha256"]==PINS[arm][e.SEEDS.index(seed)] and p["steps"]==240,"LINEAGE")
                # Score every generation with the same final input vocabulary for NEW prompts only;
                # legacy queries use original vocabulary. Unseen labels already have unused slots.
                score_d=ds[-1];scores=[e.score(L,m,score_d,grp) for grp in cohorts]
                exact=[s["exact"] for s in scores];retained=None
                if last_preds is not None:
                    retained=[sum(x==score_d["labels"][y] and q==score_d["labels"][y] for (t,y),x,q in zip(grp,prev,s["preds"])) for grp,prev,s in zip(cohorts,last_preds,scores)]
                res={"arm":arm,"seed":seed,"cycle":cycle,"scores":exact,"n":[len(grp) for grp in cohorts],"retained_from_previous":retained,"inactive":[s["inactive_fraction"] for s in scores],"model_sha256":p["model_sha256"],"parent_sha256":p.get("parent_sha256"),"training_seconds":p["seconds"],"production":False}
                log("R84F_RESULT",res);results.append(res);last=p["model_sha256"];last_preds=[s["preds"] for s in scores]
            # Load immutable original again; no production rollback.
            root,_=e.load(L,d0,folders[0]/str(seed));ck(e.state_hash(root)==PINS[arm][e.SEEDS.index(seed)],"ROOT_REOPEN")
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84f.result.v1","results":results,"cohorts":["legacy_regression","source_facts_regression","new_legacy_phrasing","synthetic_evidence_gate","synthetic_recovery_gate"],"root_reopen":True,"independent_external_exam":False,"production":False,"scope":"two fixed continuation cycles; synthetic typed tasks; inherited arms differ at root, not causal regularizer ablation"})
if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="train":train(*[Path(x) for x in sys.argv[2:5]],sys.argv[5],int(sys.argv[6]))
    elif mode=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
