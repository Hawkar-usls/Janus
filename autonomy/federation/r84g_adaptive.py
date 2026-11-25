"""R84G bounded paired replay ablation. No autonomous task execution or promotion."""
import math,random,sys,time,json
from pathlib import Path
import r84e_neural as e
import r84f_continuity as f
import r84f2_vocab as v
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="56012e10a47e7d4e276a7215a0ee2d4ff3f28a1a"
ROOTS={"replay":["46042d7572f11a56f4ea8a3464b86ab7272919df8af41138b44540e43f660feb","29d9a86712964b39e1e5f159d47b154d23b54de152ef9d17e619ecacd30b4560","c12e355041694c6e7e60a464f0a10acf657e530175cf891bccf38d86cf9da5bf"],"joint":["2f543c1593802caf55200eabf303d54ac61cace8e320529adc5123436a6d00d9","8d425ff7ee528cd04ce7f5fab0157f1819de7879e772d3ca05df764b4e37898e","728f085398c838f4b953bbddb8250ef7f63e5e8b8dc363d17cd2f79a5600a9d2"]}
MODES=("fixed","adaptive")
STEPS=240
BATCH=12
BLOCK=20
NEW={"g_missing":["required observation has no measurement","missing mandatory data value","record lacks required numeric entry","absent measurement in sample table","incomplete row missing required field"],"g_units":["measurement units differ without conversion","mixed units require conversion before comparing","centimeter and meter values not normalized","incompatible units for recorded quantities","inconsistent units in numeric measurements"],"g_duplicate":["duplicate sample identifier in table","same data record appears twice","repeated identical observation entry","duplicated measurement rows with same identifier","identical sample records listed repeatedly"],"g_outlier":["extreme value outside expected distribution","outlier measurement needs investigation","unusual numeric observation far from typical values","abnormal deviation investigate before exclusion","suspicious extreme measurement verify validity"]}
MON_ROUTE=[["show action run progress","status"],["locate project file source","locate"],["verify evidence source attribution","provenance"],["review copyright permission terms","license"],["compare previous results for regression","regression"],["recover prior checkpoint","rollback"],["calibration check against standard","calibration"],["replicate result using fresh sample","replication"],["measure benchmark response timing","latency"],["inspect memory consumption peak","memory"]]
MON_EVIDENCE=[["gate probe provenance absent for source","f_hold_source"],["gate probe no reuse permission for known origin","f_hold_rights"],["gate probe valid measurements contradict","f_hold_conflict"],["gate probe reproducible evidence source rights confirmed","f_accept_review"]]
MON_RECOVERY=[["gate probe transient timeout retry available","f_retry_transient"],["gate probe stop because compute budget exhausted","f_stop_budget"],["gate probe operator manual stop received","f_stop_operator"],["gate probe restore stable checkpoint after corruption","f_restore_checkpoint"]]
def ck(v,s):e.ck(v,s)
def log(t,z):e.log(t,z)
def norm(s):
    for p in ("gate practice ","gate exam ","gate probe "):
        if s.startswith(p):return "gate "+s[len(p):]
    return s
def facts(d0):
    templates=[(k,q) for k,qs in e.TEMPLATES.items() for q in qs[:1]]
    out=[]
    for t,y in d0["new"]:
        for k,q in templates:
            left,right=q.split("{id}")
            if t.startswith(left) and t.endswith(right):
                i=t[len(left):len(t)-len(right) if right else None]
                if i.startswith("2611.") and i.endswith("v1"):out.append([i,k,y])
    ck(len(out)==17 and len({tuple(x) for x in out})==17,"FACTS")
    return out
def configure(d):
    def encode(z,s):
        vocab=z["vocab"] if s.startswith("gate ") else z.get("legacy_vocab",z["vocab"])
        ids=[0]+[vocab.get(w,1) for w in e.tokenize(norm(s))]+[0]
        ck(len(ids)<=24,"LENGTH");return ids
    e.encode=encode
def datasets(prep):
    d0,d1,d2=v.datasets(f.orig(prep))
    pools=[d0["old"],d0["new"],f.groups()[0],f.groups()[1]]
    new=[["gate practice "+x,y] for y,ts in NEW.items() for x in ts]
    d3={**d2,"old":sum(pools,[]),"new":new,"vocab":dict(d2["vocab"]),"labels":dict(d2["labels"]),"g_scope":"bounded_paired_replay_v1"}
    newwords=sorted({w for t,y in new for w in e.tokenize(t)}-set(d3["vocab"]))
    for i,w in enumerate(newwords,d2["input_embedding_size"]):d3["vocab"][w]=i
    for y in sorted(NEW):d3["labels"][y]=max(d3["labels"].values())+1
    d3["input_embedding_size"]=d2["input_embedding_size"]+len(newwords)
    ck(d3["input_embedding_size"]<=386 and max(d3["labels"].values())<258 and not set(d3["vocab"].values())&set(d3["labels"].values()),"CAP")
    ft=facts(d0);mt={"primary":"state {id} primary category","author_count":"state {id} author count","license":"state {id} license type","claim":"state {id} reported topic"}
    monitor=[MON_ROUTE,[[mt[k].format(id=i),y] for i,k,y in ft],MON_EVIDENCE,MON_RECOVERY]
    configure(d3)
    tr={tuple(e.encode(d3,t)) for t,y in d3["old"]+new}
    # Exclude monitor collisions by deterministic pre-training rule, not by model performance.
    monitor=[[row for row in grp if tuple(e.encode(d3,row[0])) not in tr] for grp in monitor]
    ck(all(len(x)>=2 for x in monitor),"MONITOR_TOO_SMALL")
    d3["monitor"]=monitor
    return d0,d2,d3,pools
def expand(L,m,size,seed):
    torch,_=L;old=m.wte;n=old.num_embeddings
    ck(n<=size<=386,"EMBEDDING_CAP")
    if n==size:return m
    new=torch.nn.Embedding(size,old.embedding_dim)
    gen=torch.Generator().manual_seed(seed+840700)
    with torch.no_grad():
        new.weight[:n].copy_(old.weight)
        new.weight[n:].copy_(torch.randn((size-n,old.embedding_dim),generator=gen)*.02)
    m.wte=new;return m
def migrate(L,m,d2,d3,seed):
    torch,_=L;prior={k:x.detach().clone() for k,x in m.state_dict().items()}
    rows=d3["old"];x,l,_=e.batch(L,d2,rows);m.eval()
    with torch.no_grad():before=e.forward(L,m,x,l).clone()
    expand(L,m,d3["input_embedding_size"],seed)
    n=d2["input_embedding_size"];after=m.state_dict()
    ck(all(torch.equal(x,after[k][:n] if k=="wte.weight" else after[k]) for k,x in prior.items()),"OLD_TENSORS")
    xx,ll,_=e.batch(L,d3,rows)
    ck(torch.equal(x,xx) and torch.equal(l,ll),"OLD_IDS")
    m.eval()
    with torch.no_grad():ck(torch.equal(before,e.forward(L,m,xx,ll)),"OLD_LOGITS")
    return e.state_hash(m)
def root(L,d2,folder,arm,seed):
    m,p=v.load2(L,d2,folder/str(seed),seed)
    ck(p["model_sha256"]==ROOTS[arm][e.SEEDS.index(seed)] and p["arm"]==arm and p["cycle"]==2,"ROOT_PIN")
    return m,p
def monitor_loss(L,m,d):
    torch,_=L;m.eval();loss=[]
    with torch.no_grad():
        for group in d["monitor"]:
            x,l,y=e.batch(L,d,group);loss.append(float(torch.nn.functional.cross_entropy(e.forward(L,m,x,l),y)))
    ck(all(math.isfinite(x) for x in loss),"MONITOR_FINITE")
    return loss
def weights(loss,base,sizes):
    # Fixed floor + bounded excess-loss priority. No test labels or external inputs.
    q=[n*math.exp(min(3.,max(0.,a-b))+.25*min(4.,a)) for n,a,b in zip(sizes,loss,base)]
    return [.5*n/sum(sizes)+.5*x/sum(q) for n,x in zip(sizes,q)]
def optimize(L,m,d,pools,seed,arm,mode):
    torch,_=L;torch.manual_seed(seed+30700);rng=random.Random(seed+30700);rr=random.Random(seed+40700)
    opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=.01)
    cap=[];hooks=[layer.linear1.register_forward_hook(lambda mod,inp,out:cap.append(out)) for layer in m.transformer_encoder.layers]
    sizes=list(map(len,pools));prior=[n/sum(sizes) for n in sizes];base=monitor_loss(L,m,d);counts=[0]*4;history=[];start=time.monotonic()
    try:
        for j in range(STEPS):
            if j%BLOCK==0:
                loss=monitor_loss(L,m,d);w=weights(loss,base,sizes) if mode=="adaptive" else prior
                history.append({"step":j,"monitor_loss":loss,"probabilities":w})
            nr=rng.choices(d["new"],k=9);old=[]
            for k in rr.choices(range(4),weights=w,k=3):counts[k]+=1;old.append(rr.choice(pools[k]))
            m.train();cap.clear();opt.zero_grad(set_to_none=True)
            x,l,y=e.batch(L,d,nr+old);z=e.forward(L,m,x,l);ce=torch.nn.functional.cross_entropy(z,y)
            ck(len(cap)==2,"HOOKS")
            mask=(torch.arange(x.shape[1])[None,:]<=l[:,None]).unsqueeze(-1)
            rates=[(torch.sigmoid(a/.1)*mask).sum((0,1))/mask.sum() for a in cap]
            reg=sum(((a-.30)**2).mean() for a in rates)/len(rates)
            loss=ce+(.2*reg if arm=="joint" else 0.);ck(math.isfinite(float(loss.detach())),"LOSS")
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    finally:
        for h in hooks:h.remove()
    ck(sum(counts)==720,"REPLAY_BUDGET")
    return {"steps":STEPS,"examples":2880,"old_examples":720,"new_examples":2160,"replay_counts":counts,"monitor_sizes":list(map(len,d["monitor"])),"monitor_before":base,"monitor_after":monitor_loss(L,m,d),"history":history,"seconds":time.monotonic()-start}
def train(prep,parent,out,arm,mode):
    ck(arm in ROOTS and mode in MODES,"SCOPE")
    L=e.lib(prep);d0,d2,d3,pools=datasets(prep);out.mkdir(parents=True,exist_ok=False)
    for seed in e.SEEDS:
        m,p=root(L,d2,parent,arm,seed);mig=migrate(L,m,d2,d3,seed);rec=optimize(L,m,d3,pools,seed,arm,mode)
        rec.update({"arm":arm,"mode":mode,"seed":seed,"parent_sha256":p["model_sha256"],"migration_sha256":mig,"exact_old_tensors_logits":True})
        e.save(L,m,d3,out/str(seed),rec);log("R84G_TRAIN",{k:x for k,x in rec.items() if k!="history"})
def load(L,d,folder,seed):
    torch,_=L;p=e.read(folder/"receipt.json");path=folder/"shadow.pt"
    ck(p["file_sha256"]==e.sha(path.read_bytes()) and p["data_sha256"]==g.digest(d) and p["production"] is False,"FILE")
    m=expand(L,e.model(L),d["input_embedding_size"],seed);m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
    ck(e.state_hash(m)==p["model_sha256"] and all(torch.isfinite(x).all() for x in m.state_dict().values()),"TENSORS");return m,p
def evaluate(prep,replay,joint,out,*folders):
    L=e.lib(prep);d0,d2,d3,pools=datasets(prep)
    raw=(HERE/"R84G_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");z=json.loads(raw)
    old,fact_old,_=e.exam();previous=f.exam()
    reg=[old,fact_old,previous["legacy_new"],*previous["cohorts"]]
    fresh=[z["old_routing"],[[q.format(id=i),y] for i,k,y in facts(d0) for q in z["facts_templates"][k]],z["evidence"],z["recovery"],z["new"]]
    tr={tuple(e.encode(d3,t)) for t,y in d3["old"]+d3["new"]};mon={tuple(e.encode(d3,t)) for grp in d3["monitor"] for t,y in grp}
    rawtrain={norm(t) for t,y in d3["old"]+d3["new"]+sum(d3["monitor"],[])}
    # Predetermined collision filter for primary fresh cohort; full scores also reported.
    clean=[[row for row in grp if norm(row[0]) not in rawtrain and tuple(e.encode(d3,row[0])) not in tr|mon] for grp in fresh]
    ck(all(len(grp)>=4 for grp in clean),"FRESH_EXAM_TOO_SMALL")
    log("R84G_EXAM",{"fresh_full_n":list(map(len,fresh)),"fresh_clean_n":list(map(len,clean)),"excluded_collisions":[len(a)-len(b) for a,b in zip(fresh,clean)],"monitor_n":list(map(len,d3["monitor"])),"legacy_n":list(map(len,reg)),"input_embedding_size":d3["input_embedding_size"]})
    results=[]
    def scored(m,grps):return [e.score(L,m,d3,grp) for grp in grps]
    for ai,(arm,parent) in enumerate(zip(("replay","joint"),(replay,joint))):
        for seed in e.SEEDS:
            m,p=root(L,d2,parent,arm,seed);mig=migrate(L,m,d2,d3,seed)
            base=scored(m,reg+clean);base_pred=[x["preds"] for x in base]
            for mi,mode in enumerate(MODES):
                q,rec=load(L,d3,folders[ai*2+mi]/str(seed),seed)
                ck(rec["arm"]==arm and rec["mode"]==mode and rec["seed"]==seed and rec["parent_sha256"]==p["model_sha256"] and rec["migration_sha256"]==mig and rec["steps"]==STEPS and rec["old_examples"]==720 and rec["new_examples"]==2160,"PAIR")
                a=scored(q,reg+clean);full=scored(q,fresh)
                retained=[sum(x==d3["labels"][y] and k==d3["labels"][y] for (t,y),x,k in zip(grp,b,s["preds"])) for grp,b,s in zip(reg+clean,base_pred,a)]
                row={"arm":arm,"mode":mode,"seed":seed,"before":[x["exact"] for x in base],"after":[x["exact"] for x in a],"n":[x["n"] for x in a],"retained":retained,"fresh_full":[x["exact"] for x in full],"replay_counts":rec["replay_counts"],"seconds":rec["seconds"],"model_sha256":rec["model_sha256"],"parent_sha256":p["model_sha256"]}
                log("R84G_RESULT",row);results.append(row)
            root(L,d2,parent,arm,seed)
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84g.result.v1","results":results,"paired":True,"production":False,"external_validation":False,"scope":"one finite synthetic shadow cycle; monitor-only prioritization; equal updates and replay slots; no production model"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:5]],sys.argv[5],sys.argv[6])
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
