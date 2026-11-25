"""R84F2: resume only cycle 2 with an append-only input embedding extension."""
import sys
from pathlib import Path
import r84f_continuity as f
import r84e_neural as e
import jfp_gateway as g
PINS1={"replay":["deb29e5c9c397701c1ef5c49f30f86236d50eea2660dbd96ddeeb33276645a9e","139fc98ef2233d74d004e05e4c020bdc784eb1aef4e1e3e94f9d8ed468947965","f919e0a2cbd8dabcd4cea5341a4c1e65732eea88b70b3563b6c62dd1af8eee00"],"joint":["6d10de154bd513deed2397852c9160435de2591153c711a2f53bd4ade43df7e2","dc4e300e1d5696f975059c1f464e4bb73275c4309540017bd630400d2df2c689","97f1bbb7dbab3a04da7c37174e1269244867ab8908019ff736895a2ce746ded8"]}
DATA1="4854a3e14624d446e1dc541635c280ead972f252b0223b66be31b2488c8c5ad8"
def ck(v,s):e.ck(v,s)
def log(t,x):e.log(t,x)
def datasets(d0):
    d1=f.data(d0,1);ck(g.digest(d1)==DATA1,"EXACT_CYCLE1_DATA")
    d2={**d1,"vocab":dict(d1["vocab"]),"labels":dict(d1["labels"]),"old":d1["old"]+d1["new"],"new":f.groups()[1],"f_cycle":2,"f2_extension":"append_input_only_v1"}
    new=sorted({w for t,y in d2["new"] for w in e.tokenize(t)}-set(d1["vocab"]))
    for i,w in enumerate(new,258):d2["vocab"][w]=i
    for y in sorted(set(f.TRAIN[1])-set(d2["labels"])):d2["labels"][y]=max(d2["labels"].values())+1
    size=258+len(new);ck(258<size<=386 and max(d2["labels"].values())<258,"FINITE_CAP")
    ck(all(d2["vocab"][w]==v for w,v in d1["vocab"].items()) and all(d2["labels"][w]==v for w,v in d1["labels"].items()) and not set(d2["vocab"].values())&set(d2["labels"].values()),"ID_STABILITY")
    d2["input_embedding_size"]=size
    return [d0,d1,d2]
def expand(L,m,size,seed):
    torch,_=L;old=m.wte;ck(old.num_embeddings==258 and size>258,"EXPAND_SCOPE")
    new=torch.nn.Embedding(size,old.embedding_dim)
    gen=torch.Generator().manual_seed(seed+840200)
    with torch.no_grad():
        new.weight[:258].copy_(old.weight)
        new.weight[258:].copy_(torch.randn((size-258,old.embedding_dim),generator=gen)*.02)
    m.wte=new;return m
def migrate(L,m,d1,d2,seed):
    torch,_=L;before={k:v.detach().clone() for k,v in m.state_dict().items()}
    rows=d1["old"]+d1["new"];x,l,_=e.batch(L,d1,rows)
    m.eval()
    with torch.no_grad():z=e.forward(L,m,x,l).clone()
    expand(L,m,d2["input_embedding_size"],seed)
    after=m.state_dict();ck(set(before)==set(after),"KEYS")
    ck(all(torch.equal(v,after[k][:258] if k=="wte.weight" else after[k]) for k,v in before.items()),"TENSOR_PREFIX")
    x2,l2,_=e.batch(L,d2,rows);ck(torch.equal(x,x2) and torch.equal(l,l2),"INPUT_STABILITY")
    m.eval()
    with torch.no_grad():z2=e.forward(L,m,x2,l2)
    ck(torch.equal(z,z2),"LOGIT_STABILITY")
    return {"old_embedding":258,"new_embedding":d2["input_embedding_size"],"output_vocab":m.lm_head.out_features,"old_parameters":sum(v.numel() for v in before.values()),"new_parameters":sum(p.numel() for p in m.parameters()),"all_old_tensors_exact":True,"all_old_training_logits_exact":True,"logit_rows":len(rows)}
def load2(L,d,folder,seed):
    torch,_=L;p=e.read(folder/"receipt.json");path=folder/"shadow.pt"
    ck(p["file_sha256"]==e.sha(path.read_bytes()) and p["data_sha256"]==g.digest(d) and p["production"] is False,"CHECKPOINT")
    m=expand(L,e.model(L),d["input_embedding_size"],seed)
    m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
    ck(e.state_hash(m)==p["model_sha256"] and all(torch.isfinite(x).all() for x in m.state_dict().values()),"MODEL_HASH")
    return m,p
def train(prep,parent,out,arm):
    ck(arm in f.ARMS,"ARM")
    L=e.lib(prep);d0,d1,d2=datasets(f.orig(prep));f.configure(d2);e.NEW_STEPS=240
    out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        m,p=e.load(L,d1,parent/str(seed))
        ck(p["model_sha256"]==PINS1[arm][i] and p["root_sha256"]==f.PINS[arm][i] and p["seed"]==seed and p["arm"]==arm and p["cycle"]==1,"CYCLE1_PIN")
        mig=migrate(L,m,d1,d2,seed);log("R84F2_MIGRATION",{"arm":arm,"seed":seed,**mig})
        migration_hash=e.state_hash(m)
        q=e.optimize(L,m,d2,seed+200,2,arm)
        ck(q["steps"]==240 and q["examples"]==2880 and q["old_examples"]==720,"BUDGET")
        rec={"seed":seed,"arm":arm,"cycle":2,"parent_sha256":p["model_sha256"],"root_sha256":f.PINS[arm][i],"migration_sha256":migration_hash,"migration":mig,**q}
        e.save(L,m,d2,out/str(seed),rec);log("R84F2_TRAIN",rec)
def evaluate(prep,replay0,joint0,replay1,joint1,replay2,joint2,out):
    L=e.lib(prep);ds=datasets(f.orig(prep));d0,d1,d2=ds;z=f.exam();old,facts,_=e.exam()
    cohorts=[old,facts,z["legacy_new"],*z["cohorts"]]
    f.configure(d2)
    train=d2["old"]+d2["new"];norm=lambda s:s.replace("gate practice ","gate ").replace("gate exam ","gate ")
    ck(not {norm(t) for t,y in train}&{norm(t) for grp in cohorts for t,y in grp},"NORMALIZED_OVERLAP")
    encoded={tuple(e.encode(d2,t)) for t,y in train}
    overlap=[sum(tuple(e.encode(d2,t)) in encoded for t,y in grp) for grp in cohorts]
    log("R84F2_EXAM",{"raw_overlap":0,"encoded_overlap_by_cohort":overlap,"n":[len(x) for x in cohorts],"stage_specific_input_vocabulary":True})
    results=[]
    for arm,folders in zip(f.ARMS,[(replay0,replay1,replay2),(joint0,joint1,joint2)]):
        for i,seed in enumerate(e.SEEDS):
            last=None;last_preds=None
            for cycle,folder in enumerate(folders):
                m,p=load2(L,d2,folder/str(seed),seed) if cycle==2 else e.load(L,ds[cycle],folder/str(seed))
                if cycle==0:f.pin_parent(arm,seed,p)
                else:
                    ck(p["cycle"]==cycle and p["arm"]==arm and p["seed"]==seed and p["parent_sha256"]==last and p["root_sha256"]==f.PINS[arm][i] and p["steps"]==240,"LINEAGE")
                    if cycle==1:ck(p["model_sha256"]==PINS1[arm][i],"CYCLE1_PIN")
                    if cycle==2:ck(p["migration"]["all_old_tensors_exact"] and p["migration"]["all_old_training_logits_exact"],"MIGRATION")
                # Earlier models have no embedding for future words. Use their own vocabulary,
                # while preserving output label IDs for comparable scoring.
                score_d={**ds[cycle],"labels":d2["labels"],"legacy_vocab":d0["vocab"]}
                scores=[e.score(L,m,score_d,grp) for grp in cohorts]
                retained=None if last_preds is None else [sum(x==score_d["labels"][y] and q==score_d["labels"][y] for (t,y),x,q in zip(grp,prev,s["preds"])) for grp,prev,s in zip(cohorts,last_preds,scores)]
                res={"arm":arm,"seed":seed,"cycle":cycle,"scores":[s["exact"] for s in scores],"n":[len(grp) for grp in cohorts],"retained_from_previous":retained,"inactive":[s["inactive_fraction"] for s in scores],"model_sha256":p["model_sha256"],"parent_sha256":p.get("parent_sha256"),"parameters":sum(q.numel() for q in m.parameters()),"production":False}
                log("R84F2_RESULT",res);results.append(res);last=p["model_sha256"];last_preds=[s["preds"] for s in scores]
            root,_=e.load(L,d0,folders[0]/str(seed));ck(e.state_hash(root)==f.PINS[arm][i],"ROOT_REOPEN")
            first,_=e.load(L,d1,folders[1]/str(seed));ck(e.state_hash(first)==PINS1[arm][i],"FIRST_REOPEN")
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84f2.result.v1","results":results,"cohorts":["legacy_regression","source_facts_regression","new_legacy_phrasing","synthetic_evidence_gate","synthetic_recovery_gate"],"encoded_overlap":overlap,"root_reopen":True,"cycle1_reopen":True,"cycle1_retrained":False,"independent_external_exam":False,"production":False})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:5]],sys.argv[5])
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
