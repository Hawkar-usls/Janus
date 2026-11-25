"""R84H2: two bounded shadow generations with paired replay and measured CPU cost."""
import json,math,random,sys,time
from pathlib import Path
import r84g2_selective as s
import r84g_adaptive as a
import r84e_neural as e
import r84f_continuity as f
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="53bc4ac6f771d929c8747e2233ac6b1e1255f53a"
ROOTS=("27fe73e5ea50175a9206cd7eb973128e8e0a914077c2c289ae207cde24f94412","d3aa71efadaa3e86ed42d55a71ed26461b894a3f26d44fc81e947f309a512673","fe071c521098db7a1925bd636542af6b4e7574ea0387e1785ef17d316287e6c3")
TRAIN=[
{"h_schema":["record violates specified data schema","field type fails structural validation","invalid format for required record fields","schema constraint mismatch in input","check structure matches declared schema"],"h_time":["timestamps use different clock references","convert local timestamps to common time","inconsistent timezone offset in observations","align time reference before comparison","normalize time zones for recorded dates"],"h_uncertainty":["report measurement error and uncertainty interval","estimated value lacks uncertainty bounds","measurement precision unknown include confidence interval","quantify uncertainty of numerical estimate","state measurement error alongside result"],"h_version":["dataset revision changed from pinned version","source version checksum no longer matches","verify release version of source data","different dataset revision detected","record exact version identifier for input"]},
{"h_leak":["test labels exposed during training","evaluation examples included in learning set","held out data used to tune threshold","training contaminated by final exam records","detect leakage from test into development"],"h_balance":["unequal samples per class","rare class has few observations","class imbalance requires per class metrics","majority category dominates sample counts","report minority class accuracy separately"],"h_seed":["repeat training with several random seeds","change random initialization for reproducibility","multiple independent seed runs compare","repeat protocol with different random draws","test variation across initial model weights"],"h_cost":["measure processor time and elapsed runtime","include monitoring cost in compute budget","compare learning gain per processor second","account for extra forward inference cost","record total computational cost of experiment"]}]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def datasets(prep):
    d0,d2,d3,_=a.datasets(prep);ds=[d3]
    for cycle,grp in enumerate(TRAIN,1):
        p=ds[-1];new=[["gate practice "+t,y] for y,ts in grp.items() for t in ts]
        d={**p,"old":p["old"]+p["new"],"new":new,"vocab":dict(p["vocab"]),"labels":dict(p["labels"]),"h2_cycle":cycle}
        words=sorted({w for t,y in new for w in e.tokenize(t)}-set(p["vocab"]))
        for i,w in enumerate(words,p["input_embedding_size"]):d["vocab"][w]=i
        for y in sorted(grp):d["labels"][y]=max(d["labels"].values())+1
        d["input_embedding_size"]=p["input_embedding_size"]+len(words)
        ck(d["input_embedding_size"]<=512 and max(d["labels"].values())<258 and not set(d["vocab"].values())&set(d["labels"].values()),"CAP")
        ds.append(d)
    a.configure(ds[-1]);return d0,ds
def model(L,size,seed):
    torch,_=L;m=e.model(L)
    return extend(L,m,size,seed)
def extend(L,m,size,seed):
    torch,_=L;old=m.wte;n=old.num_embeddings;ck(n<=size<=512,"EMBED")
    if n==size:return m
    new=torch.nn.Embedding(size,old.embedding_dim);gen=torch.Generator().manual_seed(seed+842000)
    with torch.no_grad():
        new.weight[:n].copy_(old.weight);new.weight[n:].copy_(torch.randn((size-n,old.embedding_dim),generator=gen)*.02)
    m.wte=new;return m
def load(L,d,folder,seed):
    torch,_=L;p=e.read(folder/"receipt.json");path=folder/"shadow.pt"
    ck(p["file_sha256"]==e.sha(path.read_bytes()) and p["data_sha256"]==g.digest(d) and p["production"] is False,"RECEIPT")
    m=model(L,d["input_embedding_size"],seed);m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
    ck(e.state_hash(m)==p["model_sha256"] and all(torch.isfinite(x).all() for x in m.state_dict().values()),"TENSORS")
    return m,p
def migrate(L,m,p,d,seed):
    torch,_=L;prior={k:v.detach().clone() for k,v in m.state_dict().items()};x,l,_=e.batch(L,p,d["old"]);m.eval()
    with torch.no_grad():z=e.forward(L,m,x,l).clone()
    extend(L,m,d["input_embedding_size"],seed);n=p["input_embedding_size"]
    ck(all(torch.equal(v,m.state_dict()[k][:n] if k=="wte.weight" else m.state_dict()[k]) for k,v in prior.items()),"OLD_TENSORS")
    xx,ll,_=e.batch(L,d,d["old"]);ck(torch.equal(x,xx) and torch.equal(l,ll),"OLD_IDS")
    m.eval()
    with torch.no_grad():ck(torch.equal(z,e.forward(L,m,xx,ll)),"OLD_LOGITS")
    return e.state_hash(m)
def fixed(L,m,d,seed):
    torch,_=L;torch.manual_seed(seed+30700);rng=random.Random(seed+30700);rr=random.Random(seed+50700)
    opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=.01)
    for _ in range(240):
        rows=rng.choices(d["new"],k=9)+rr.choices(d["old"],k=3);m.train();opt.zero_grad(set_to_none=True)
        x,l,y=e.batch(L,d,rows);loss=torch.nn.functional.cross_entropy(e.forward(L,m,x,l),y);ck(math.isfinite(float(loss.detach())),"LOSS")
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    return {"steps":240,"examples":2880,"old_examples":720,"new_examples":2160,"sample_probe_forward_rows":0}
def train(prep,parent,out,cycle):
    ck(cycle in (1,2),"CYCLE");L=e.lib(prep);_,ds=datasets(prep);p,d=ds[cycle-1:cycle+1]
    out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        modes=("fixed","selective") if (i+cycle)%2 else ("selective","fixed")
        for mode in modes:
            folder=parent/str(seed) if cycle==1 else parent/mode/str(seed)
            m,rec=load(L,p,folder,seed)
            if cycle==1:ck(rec["model_sha256"]==ROOTS[i],"ROOT_PIN")
            else:ck(rec["mode"]==mode and rec["cycle"]==1 and rec["seed"]==seed and rec["root_sha256"]==ROOTS[i],"PARENT")
            start=time.perf_counter();cpu=time.process_time();mig=migrate(L,m,p,d,seed+100*cycle)
            z=fixed(L,m,d,seed+100*cycle) if mode=="fixed" else s.optimize(L,m,d,seed+100*cycle,"replay")
            z.update({"cycle":cycle,"seed":seed,"mode":mode,"root_sha256":ROOTS[i],"parent_sha256":rec["model_sha256"],"migration_sha256":mig,"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-start})
            e.save(L,m,d,out/mode/str(seed),z);log("R84H2_TRAIN",{k:v for k,v in z.items() if k not in ("history","sample_counts","per_class_replays")})
def exams(d0,ds):
    raw=(HERE/"R84H2_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");z=json.loads(raw)
    old,fo,_=e.exam();prev=f.exam();zg=json.loads((HERE/"R84G_FROZEN_EXAM.json").read_text());z2=json.loads((HERE/"R84G2_FROZEN_EXAM.json").read_text());ft=a.facts(d0)
    prior=[old,fo,prev["legacy_new"],*prev["cohorts"],zg["old_routing"],[[q.format(id=i),y] for i,k,y in ft for q in zg["facts_templates"][k]],zg["evidence"],zg["recovery"],zg["new"]]
    for name,rows in z2["cohorts"]:
        prior.append([[q.format(id=i),y] for i,k,y in ft for q in rows[k]] if name=="facts_templates" else rows)
    ck(sum(map(len,prior))==243,"PRIOR")
    # Keep prior prompts on their historic vocabulary. New prompts use the current stage vocabulary.
    alltrain=ds[-1]["old"]+ds[-1]["new"]+sum(ds[0]["monitor"],[]);seen={a.norm(t) for t,y in alltrain+sum(prior,[])}
    clean=[]
    for i,grp in enumerate(z["cohorts"],1):
        tr={tuple(e.encode(ds[i],t)) for t,y in alltrain+sum(prior,[])}
        q=[row for row in grp if a.norm(row[0]) not in seen and tuple(e.encode(ds[i],row[0])) not in tr]
        ck(len(q)>=8,"FRESH_COHORT");clean.append(q)
    log("R84H2_EXAM",{"prior_n":243,"fresh_n":list(map(len,clean)),"fresh_full_n":list(map(len,z["cohorts"]))})
    return [sum(prior,[]),*clean]
def evaluate(prep,root,c1,c2,out):
    L=e.lib(prep);d0,ds=datasets(prep);coh=exams(d0,ds);res=[]
    def score(m,cycle):
        # Prior cohort uses unchanged historical token IDs; new cohorts use their
        # own stage vocabulary when available, otherwise current vocabulary.
        ans=[]
        for i,grp in enumerate(coh):
            d={**ds[min(i,cycle)],"labels":ds[-1]["labels"]}
            ans.append(e.score(L,m,d,grp))
        return ans
    for i,seed in enumerate(e.SEEDS):
        m,p=load(L,ds[0],root/str(seed),seed);ck(p["model_sha256"]==ROOTS[i],"ROOT")
        base=score(m,0)
        for mode in ("fixed","selective"):
            prev=base;last=p["model_sha256"]
            for cycle,folder in enumerate((c1,c2),1):
                q,rec=load(L,ds[cycle],folder/mode/str(seed),seed);ck(rec["cycle"]==cycle and rec["mode"]==mode and rec["parent_sha256"]==last and rec["root_sha256"]==ROOTS[i] and rec["steps"]==240,"LINEAGE")
                ans=score(q,cycle);before=sum(x["exact"] for x in prev);after=sum(x["exact"] for x in ans)
                kept=sum(sum(b==ds[-1]["labels"][y] and c==ds[-1]["labels"][y] for (_,y),b,c in zip(grp,u["preds"],v["preds"])) for grp,u,v in zip(coh,prev,ans))
                row={"mode":mode,"seed":seed,"cycle":cycle,"before":before,"dep":after,"n":sum(map(len,coh)),"scores":[x["exact"] for x in ans],"lost":before-kept,"risk":(before-kept)/before if before else None,"net_gain":after-before,"cpu_seconds":rec["cpu_seconds"],"wall_seconds":rec["wall_seconds"],"profit_cpu":(after-before)/rec["cpu_seconds"],"parent_sha256":last,"model_sha256":rec["model_sha256"]}
                res.append(row);log("R84H2_RESULT",row);prev=ans;last=rec["model_sha256"]
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h2.result.v1","results":res,"production":False,"external_validation":False,"scope":"two fixed synthetic shadow cycles; prior exams fixed historic vocab; CPU process time includes migration and training probes but excludes installation, artifact IO and exam; no automatic admission"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:5]],int(sys.argv[5]))
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
