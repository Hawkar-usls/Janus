"""R84H3: finite statistical-text classification ablation, diagnostic and two continuations."""
import json,math,random,sys,time
from pathlib import Path
import r84h2_growth as h
import r84g2_selective as s
import r84g_adaptive as a
import r84e_neural as e
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="6742a281c91c335307f682fd8bc78ef8e83c2fc0"
ROOTS=("27b508ccad0c2b8a325dfa0c93cf4bc6c400e0a667b70d8e2cfa166f443dd7c8","3666e6bae5161935f2c31efb5c8a86a072ef00b68096bcc596d8a5a6e9a3be35","6f2d9b45d593d8109f691c24494435c136a09fcf6e3d3ec0e6192065f27bc205")
H2FIRST={"fixed":("f5253bb4614533631a3febd87c32df253151d03828ed7e817af81316b7902d11","daf940654059cff49eca544cddb6bc680df1f84a4507062c4bb5a4fef3805c05","f7edad794ed3d2edaee18a298ef9fd10dc142f38de034962ca78b1d7f85868ee"),"selective":("b418e4483edbe53de3562eb8f9bc34c9197b5b59f8d667aa8ded0db660bdc169","3dc2d741c674c9047accf423873d81c0d45157f52b58b4f437c41d4af6a56d91","aa20703433424302ae9caa1cc85a3cfcacf782406158873758d3b126f15982b1")}
MODES=("fixed","coverage","selective")
TRAIN=[
{"h3_mean":["calculate mean as sum over count","arithmetic average of observations","mean value summarizes repeated measurements","compute average of numerical readings","sum measurements and divide by sample size"],"h3_median":["sort observations select middle value","median robust against extreme readings","central value in ordered data","choose middle observation for skewed distribution","report median of ranked values"],"h3_spread":["standard deviation measures numerical spread","describe dispersion around mean","quantify variability across observations","measure spread of repeated readings","report standard deviation of dataset"],"h3_size":["record total number of samples","count independent measured observations","report sample size for estimate","number of records used in analysis","specify count of measurements"]},
{"h3_assoc":["correlation measures paired variable association","two variables change together without causal proof","describe strength of statistical relationship","assess correlation of paired numerical data","measure association between quantities"],"h3_confound":["third factor influences both variables","confounding variable explains apparent association","consider alternative common cause","hidden factor affects observed relationship","control confounders before causal inference"],"h3_resample":["bootstrap sampling with replacement","resample observations to estimate uncertainty","draw repeated samples from existing dataset","estimate sampling variability by bootstrap","repeat random resampling with replacement"],"h3_transfer":["evaluate on different independent population","test generalization to new measurement distribution","assess transfer on external dataset","compare performance on unseen population","validate method with another data source"]}]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def data(prep):
    d0,old=h.datasets(prep);ds=[old[-1]]
    for cycle,grp in enumerate(TRAIN,1):
        p=ds[-1];new=[["gate practice "+t,y] for y,ts in grp.items() for t in ts]
        d={**p,"old":p["old"]+p["new"],"new":new,"labels":dict(p["labels"]),"vocab":dict(p["vocab"]),"h3_cycle":cycle}
        words=sorted({w for t,y in new for w in e.tokenize(t)}-set(p["vocab"]))
        for i,w in enumerate(words,p["input_embedding_size"]):d["vocab"][w]=i
        for y in sorted(grp):d["labels"][y]=max(d["labels"].values())+1
        d["input_embedding_size"]=p["input_embedding_size"]+len(words)
        ck(d["input_embedding_size"]<=640 and max(d["labels"].values())<258 and not set(d["labels"].values())&set(d["vocab"].values()),"CAP")
        ds.append(d)
    a.configure(ds[-1]);return d0,old,ds
def extend(L,m,size,seed):
    torch,_=L;old=m.wte;n=old.num_embeddings;ck(n<=size<=640,"EMBED_CAP")
    if n==size:return m
    new=torch.nn.Embedding(size,old.embedding_dim);gen=torch.Generator().manual_seed(seed+842000)
    with torch.no_grad():
        new.weight[:n].copy_(old.weight);new.weight[n:].copy_(torch.randn((size-n,old.embedding_dim),generator=gen)*.02)
    m.wte=new;return m
def load(L,d,folder,seed):
    torch,_=L;p=e.read(folder/"receipt.json");path=folder/"shadow.pt"
    ck(p["file_sha256"]==e.sha(path.read_bytes()) and p["data_sha256"]==g.digest(d) and p["production"] is False,"RECEIPT")
    m=extend(L,e.model(L),d["input_embedding_size"],seed);m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
    ck(e.state_hash(m)==p["model_sha256"] and all(torch.isfinite(x).all() for x in m.state_dict().values()),"TENSORS");return m,p
def migrate(L,m,p,d,seed):
    torch,_=L;old={k:v.detach().clone() for k,v in m.state_dict().items()};x,l,_=e.batch(L,p,d["old"]);m.eval()
    with torch.no_grad():before=e.forward(L,m,x,l).clone()
    extend(L,m,d["input_embedding_size"],seed);n=p["input_embedding_size"]
    ck(all(torch.equal(v,m.state_dict()[k][:n] if k=="wte.weight" else m.state_dict()[k]) for k,v in old.items()),"OLD_TENSORS")
    xx,ll,_=e.batch(L,d,d["old"]);ck(torch.equal(x,xx) and torch.equal(l,ll),"OLD_IDS")
    m.eval()
    with torch.no_grad():ck(torch.equal(before,e.forward(L,m,xx,ll)),"OLD_LOGITS")
    return e.state_hash(m)
def coverage(L,m,d,seed):
    torch,_=L;torch.manual_seed(seed+30700);rng=random.Random(seed+30700);rr=random.Random(seed+50700)
    opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=.01);old=d["old"];by={};counts=[0]*len(old)
    for i,(_,y) in enumerate(old):by.setdefault(y,[]).append(i)
    ck(len(by)<=60 and len(old)<=720,"COVERAGE_CAP")
    for block in range(12):
        selected=[]
        def pick(i):selected.append(i);counts[i]+=1
        for label,ids in sorted(by.items()):
            low=min(counts[i] for i in ids);pick(rr.choice([i for i in ids if counts[i]==low]))
        while len(selected)<60:
            unseen=[i for i,c in enumerate(counts) if c==0];pick(rr.choice(unseen) if unseen else rr.randrange(len(old)))
        rr.shuffle(selected)
        for j in range(20):
            rows=rng.choices(d["new"],k=9)+[old[i] for i in selected[3*j:3*j+3]]
            m.train();opt.zero_grad(set_to_none=True);x,l,y=e.batch(L,d,rows);loss=torch.nn.functional.cross_entropy(e.forward(L,m,x,l),y)
            ck(math.isfinite(float(loss.detach())),"LOSS");loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    ck(sum(counts)==720 and min(counts)>0,"COVERAGE")
    return {"steps":240,"examples":2880,"old_examples":720,"new_examples":2160,"covered_examples":sum(c>0 for c in counts),"old_pool":len(old),"min_class_replays":min(sum(counts[i] for i in ids) for ids in by.values()),"sample_probe_forward_rows":0,"sample_counts":counts}
def fit(L,m,p,d,seed,mode):
    start=time.perf_counter();cpu=time.process_time();mig=migrate(L,m,p,d,seed)
    z=h.fixed(L,m,d,seed) if mode=="fixed" else coverage(L,m,d,seed) if mode=="coverage" else s.optimize(L,m,d,seed,"replay")
    ck(z["steps"]==240 and z["old_examples"]==720 and z["new_examples"]==2160,"BUDGET")
    z.update({"migration_sha256":mig,"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-start});return z
def train(prep,parent,out,cycle):
    ck(cycle in (1,2),"CYCLE");L=e.lib(prep);_,old,ds=data(prep);p,d=ds[cycle-1:cycle+1];out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        order=MODES[(i+cycle)%3:]+MODES[:(i+cycle)%3]
        for mode in order:
            folder=parent/"selective"/str(seed) if cycle==1 else parent/mode/str(seed)
            m,rec=load(L,p,folder,seed)
            if cycle==1:ck(rec["model_sha256"]==ROOTS[i],"ROOT_PIN")
            else:ck(rec["mode"]==mode and rec["cycle"]==1 and rec["seed"]==seed and rec["root_sha256"]==ROOTS[i],"PARENT")
            z=fit(L,m,p,d,seed+300+100*cycle,mode)
            z.update({"mode":mode,"cycle":cycle,"seed":seed,"root_sha256":ROOTS[i],"parent_sha256":rec["model_sha256"]})
            e.save(L,m,d,out/mode/str(seed),z);log("R84H3_TRAIN",{k:v for k,v in z.items() if k not in ("history","sample_counts","per_class_replays")})
def diagnostic(prep,parent,out):
    L=e.lib(prep);_,ds=h.datasets(prep);p,d=ds[:2];out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        m,rec=load(L,p,parent/str(seed),seed);ck(rec["model_sha256"]==h.ROOTS[i],"DIAG_ROOT")
        z=fit(L,m,p,d,seed+100,"coverage");z.update({"mode":"coverage","seed":seed,"parent_sha256":rec["model_sha256"]})
        e.save(L,m,d,out/str(seed),z);log("R84H3_DIAG_TRAIN",{k:v for k,v in z.items() if k!="sample_counts"})
def score(L,m,groups,dicts,labels):
    return [e.score(L,m,{**d,"labels":labels},rows) for d,rows in zip(dicts,groups)]
def compare(base,ans,groups,labels):
    before=sum(x["exact"] for x in base);dep=sum(x["exact"] for x in ans);lost=[];by={}
    for gi,(u,v,grp) in enumerate(zip(base,ans,groups)):
        ids=[]
        for j,((t,y),b,c) in enumerate(zip(grp,u["preds"],v["preds"])):
            b=b==labels[y];c=c==labels[y];z=by.setdefault(y,{"n":0,"before":0,"after":0,"lost":0});z["n"]+=1;z["before"]+=b;z["after"]+=c;z["lost"]+=b and not c
            if b and not c:ids.append(j)
        lost.append(ids)
    n=sum(map(len,groups));l=sum(map(len,lost))
    return {"before":before,"dep":dep,"n":n,"lost":l,"risk":l/before if before else None,"net_gain":dep-before,"scores":[x["exact"] for x in ans],"lost_indices":lost,"per_class":by}
def evaluate(prep,diagroot,h2first,diag,h2final,c1,c2,out):
    L=e.lib(prep);d0,old,ds=data(prep);prior=h.exams(d0,old)
    raw=(HERE/"R84H3_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");ex=json.loads(raw)
    alltrain=ds[-1]["old"]+ds[-1]["new"]+sum(ds[0]["monitor"],[]);used=alltrain+sum(prior,[])
    fresh=[]
    for i,grp in enumerate(ex["cohorts"],1):
        enc={tuple(e.encode(ds[i],t)) for t,y in used};norm={a.norm(t) for t,y in used}
        clean=[row for row in grp if a.norm(row[0]) not in norm and tuple(e.encode(ds[i],row[0])) not in enc]
        ck(len(clean)>=8,"FRESH_COHORT");fresh.append(clean)
    groups=prior+fresh;log("R84H3_EXAM",{"prior_n":list(map(len,prior)),"fresh_n":list(map(len,fresh)),"fresh_full_n":list(map(len,ex["cohorts"]))})
    diagnostics=[];results=[]
    for i,seed in enumerate(e.SEEDS):
        m,p=load(L,old[0],diagroot/str(seed),seed);ck(p["model_sha256"]==h.ROOTS[i],"DIAG_ROOT")
        base=score(L,m,prior,[old[0]]*3,old[-1]["labels"])
        for mode in MODES:
            folder=diag/str(seed) if mode=="coverage" else h2first/mode/str(seed)
            q,rec=load(L,old[1],folder,seed);ck(rec["parent_sha256"]==p["model_sha256"],"DIAG_PAIR")
            if mode!="coverage":ck(rec["model_sha256"]==H2FIRST[mode][i],"DIAG_PIN")
            ans=score(L,q,prior,[old[0],old[1],old[1]],old[-1]["labels"]);row=compare(base,ans,prior,old[-1]["labels"])
            row.update({"mode":mode,"seed":seed,"cpu_seconds":rec["cpu_seconds"],"profit_cpu":row["net_gain"]/rec["cpu_seconds"]})
            diagnostics.append(row);log("R84H3_DIAG",{k:v for k,v in row.items() if k not in ("lost_indices","per_class")})
        m,p=load(L,ds[0],h2final/"selective"/str(seed),seed);ck(p["model_sha256"]==ROOTS[i],"ROOT_PIN")
        def stage(m,cy):
            return score(L,m,groups,old+[ds[min(1,cy)],ds[min(2,cy)]],ds[-1]["labels"])
        base=stage(m,0)
        for mode in MODES:
            prev=base;last=p["model_sha256"]
            for cycle,folder in enumerate((c1,c2),1):
                q,rec=load(L,ds[cycle],folder/mode/str(seed),seed);ck(rec["parent_sha256"]==last and rec["cycle"]==cycle and rec["mode"]==mode and rec["root_sha256"]==ROOTS[i] and rec["seed"]==seed,"LINEAGE")
                ans=stage(q,cycle);row=compare(prev,ans,groups,ds[-1]["labels"]);row.update({"mode":mode,"seed":seed,"cycle":cycle,"cpu_seconds":rec["cpu_seconds"],"profit_cpu":row["net_gain"]/rec["cpu_seconds"],"model_sha256":rec["model_sha256"],"parent_sha256":last})
                results.append(row);log("R84H3_RESULT",{k:v for k,v in row.items() if k not in ("lost_indices","per_class")});prev=ans;last=rec["model_sha256"]
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h3.result.v1","diagnostics":diagnostics,"results":results,"production":False,"external_validation":False,"scope":"finite synthetic classification only; diagnostic third arm coverage vs historical fixed and selective; two continuation cycles from identical roots; process CPU includes migration and training, excludes CI and exam; no promotion"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:5]],int(sys.argv[5]))
    elif sys.argv[1]=="diagnostic":diagnostic(*[Path(x) for x in sys.argv[2:]])
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
