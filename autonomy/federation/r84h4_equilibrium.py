"""R84H4 finite paired statistical concept classification: coverage, selective, mixed."""
import json,math,random,sys,time
from pathlib import Path
import r84h3_stability as h3
import r84h2_growth as h2
import r84g2_selective as s
import r84g_adaptive as a
import r84e_neural as e
import r8_learning as r
HERE=Path(__file__).parent
EXAM="1ceb8b689b846a9957e449571131c12863a83fd2"
ROOTS=("28ce4b19c69be81119512c43c69a0d703c9e28d7a5a219e7404acf9d23b5d420","0661c92367dd7bfccfb5b89fea50898ed5b61739e9569ee5dcb6a7ad5f728f04","d8e0b678c4de81c7e3975aed65788526f1c6fc4c47e565fca4c5d0cc4c091152")
MODES=("coverage","selective","mixed")
TRAIN=[
{"h4_min":["select minimum observation","smallest value in sample","least numerical reading","report minimum measured value","identify lowest number in records","find smallest recorded measurement"],"h4_max":["select maximum observation","greatest value in sample","largest numerical reading","report maximum measured value","identify highest number in records","find largest recorded measurement"]},
{"h4_range":["maximum minus minimum is range","span between lowest and highest measurement","subtract smallest value from largest","report numerical range of sample","measure distance between extrema","calculate observed range"],"h4_fraction":["proportion equals selected count over total","fraction of records satisfying criterion","divide matching observations by sample size","relative frequency of selected values","report proportion meeting condition","share of matching observations"]},
{"h4_quartile":["quartiles split sorted values into quarters","report first and third quartile","find quarter positions in ordered data","summarize lower and upper quartiles","identify twenty fifth and seventy fifth percentile","quartile cut points in distribution"],"h4_iqr":["interquartile range is third minus first quartile","spread of central half of observations","measure middle fifty percent width","difference between upper and lower quartile","report interquartile range of sample","central half dispersion using quartiles"]}]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def data(prep):
    d0,old,hist=h3.data(prep);ds=[hist[-1]]
    for cycle,grp in enumerate(TRAIN,1):
        p=ds[-1];new=[["gate practice "+t,y] for y,ts in grp.items() for t in ts]
        d={**p,"old":p["old"]+p["new"],"new":new,"labels":dict(p["labels"]),"vocab":dict(p["vocab"]),"h4_cycle":cycle}
        words=sorted({w for t,y in new for w in e.tokenize(t)}-set(p["vocab"]))
        for i,w in enumerate(words,p["input_embedding_size"]):d["vocab"][w]=i
        for y in sorted(grp):d["labels"][y]=max(d["labels"].values())+1
        d["input_embedding_size"]=p["input_embedding_size"]+len(words)
        ck(d["input_embedding_size"]<=640 and max(d["labels"].values())<258 and not set(d["labels"].values())&set(d["vocab"].values()),"CAP")
        ds.append(d)
    a.configure(ds[-1]);return d0,old,hist,ds
def mixed(L,m,d,seed):
    torch,_=L;torch.manual_seed(seed+30700);rng=random.Random(seed+30700);rr=random.Random(seed+50700)
    opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=.01)
    old=d["old"];n=len(old);by={};counts=[0]*n
    for i,(_,y) in enumerate(old):by.setdefault(y,[]).append(i)
    ck(len(by)<=60 and n<=720,"COVERAGE_CAP")
    base=s.losses(L,m,d,old);prev=base;last=[0]*n;utility=[0.]*n;weighted=0
    for block in range(12):
        now=base if block==0 else s.losses(L,m,d,old)
        if block:
            for i,c in enumerate(last):
                if c:utility[i]=.5*utility[i]+.5*max(-1.,min(1.,(prev[i]-now[i])/c))
        selected=[];bc=[0]*n
        def pick(i):selected.append(i);counts[i]+=1;bc[i]+=1
        for label,ids in sorted(by.items()):
            low=min(counts[i] for i in ids);pick(rr.choice([i for i in ids if counts[i]==low]))
        while len(selected)<60:
            unseen=[i for i in range(n) if counts[i]==0]
            if unseen:pick(rr.choice(unseen));continue
            if rr.random()<.5:
                w=[(0.1+min(3.,max(0.,now[i]-base[i]))+.35*min(4.,now[i]))*(1.+.5*max(-.5,min(1.,utility[i])))/(1.+bc[i]) for i in range(n)]
                pick(rr.choices(range(n),weights=w,k=1)[0]);weighted+=1
            else:pick(rr.randrange(n))
        rr.shuffle(selected);ck(all(any(bc[i] for i in ids) for ids in by.values()),"CLASS_FLOOR")
        for j in range(20):
            rows=rng.choices(d["new"],k=9)+[old[i] for i in selected[3*j:3*j+3]]
            m.train();opt.zero_grad(set_to_none=True);x,l,y=e.batch(L,d,rows);loss=torch.nn.functional.cross_entropy(e.forward(L,m,x,l),y)
            ck(math.isfinite(float(loss.detach())),"LOSS");loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        prev=now;last=bc
    ck(sum(counts)==720 and min(counts)>0,"COVERAGE")
    return {"steps":240,"examples":2880,"old_examples":720,"new_examples":2160,"covered_examples":n,"old_pool":n,"min_class_replays":min(sum(counts[i] for i in ids) for ids in by.values()),"sample_probe_forward_rows":12*n,"weighted_slots":weighted,"sample_counts":counts}
def train(prep,parent,out,cycle):
    ck(cycle in (1,2,3),"CYCLE");L=e.lib(prep);_,_,_,ds=data(prep);p,d=ds[cycle-1:cycle+1];out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        order=MODES[(i+cycle)%3:]+MODES[:(i+cycle)%3]
        for mode in order:
            folder=parent/"coverage"/str(seed) if cycle==1 else parent/mode/str(seed)
            m,rec=h3.load(L,p,folder,seed)
            if cycle==1:ck(rec["model_sha256"]==ROOTS[i],"ROOT")
            else:ck(rec["mode"]==mode and rec["cycle"]==cycle-1 and rec["seed"]==seed and rec["root_sha256"]==ROOTS[i],"PARENT")
            ss=seed+700+100*cycle;start=time.perf_counter();cpu=time.process_time();mig=h3.migrate(L,m,p,d,ss)
            z=h3.coverage(L,m,d,ss) if mode=="coverage" else s.optimize(L,m,d,ss,"replay") if mode=="selective" else mixed(L,m,d,ss)
            ck(z["steps"]==240 and z["old_examples"]==720 and z["new_examples"]==2160 and z["covered_examples"]==len(d["old"]) and z["min_class_replays"]>=12,"BUDGET")
            z.update({"mode":mode,"cycle":cycle,"seed":seed,"root_sha256":ROOTS[i],"parent_sha256":rec["model_sha256"],"migration_sha256":mig,"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-start})
            e.save(L,m,d,out/mode/str(seed),z);log("R84H4_TRAIN",{k:v for k,v in z.items() if k not in ("history","sample_counts","per_class_replays")})
def evaluate(prep,root,c1,c2,c3,out):
    L=e.lib(prep);d0,old,hist,ds=data(prep);prior=h2.exams(d0,old)
    raw=(HERE/"R84H3_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==h3.EXAM,"H3_EXAM");ex=json.loads(raw)
    used=hist[-1]["old"]+hist[-1]["new"]+sum(hist[0]["monitor"],[])+sum(prior,[])
    for i,grp in enumerate(ex["cohorts"],1):
        enc={tuple(e.encode(hist[i],t)) for t,y in used};norm={a.norm(t) for t,y in used}
        prior.append([row for row in grp if a.norm(row[0]) not in norm and tuple(e.encode(hist[i],row[0])) not in enc])
    ck(sum(map(len,prior))==307,"PRIOR")
    raw=(HERE/"R84H4_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM");ex=json.loads(raw)
    used=ds[-1]["old"]+ds[-1]["new"]+sum(ds[0]["monitor"],[])+sum(prior,[]);fresh=[]
    for i,grp in enumerate(ex["cohorts"],1):
        enc={tuple(e.encode(ds[i],t)) for t,y in used};norm={a.norm(t) for t,y in used}
        clean=[row for row in grp if a.norm(row[0]) not in norm and tuple(e.encode(ds[i],row[0])) not in enc]
        ck(len(clean)>=4,"FRESH");fresh.append(clean)
    groups=prior+fresh;log("R84H4_EXAM",{"prior_n":list(map(len,prior)),"fresh_n":list(map(len,fresh)),"fresh_full_n":list(map(len,ex["cohorts"]))})
    results=[]
    for i,seed in enumerate(e.SEEDS):
        m,p=h3.load(L,ds[0],root/"coverage"/str(seed),seed);ck(p["model_sha256"]==ROOTS[i],"ROOT")
        def stage(q,cy):return h3.score(L,q,groups,old+[hist[1],hist[2]]+[ds[min(j,cy)] for j in (1,2,3)],ds[-1]["labels"])
        base=stage(m,0)
        for mode in MODES:
            prev=base;last=p["model_sha256"]
            for cycle,folder in enumerate((c1,c2,c3),1):
                q,rec=h3.load(L,ds[cycle],folder/mode/str(seed),seed)
                ck(rec["parent_sha256"]==last and rec["root_sha256"]==ROOTS[i] and rec["mode"]==mode and rec["seed"]==seed and rec["cycle"]==cycle,"LINEAGE")
                ans=stage(q,cycle);row=h3.compare(prev,ans,groups,ds[-1]["labels"])
                row.update({"mode":mode,"seed":seed,"cycle":cycle,"cpu_seconds":rec["cpu_seconds"],"profit_cpu":row["net_gain"]/rec["cpu_seconds"],"model_sha256":rec["model_sha256"],"parent_sha256":last})
                results.append(row);log("R84H4_RESULT",{k:v for k,v in row.items() if k not in ("lost_indices","per_class")});prev=ans;last=rec["model_sha256"]
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h4.result.v1","results":results,"production":False,"external_validation":False,"scope":"three finite synthetic concept classification cycles; mixed residual slots 50% uniform 50% priority after coverage; no exam feedback or automatic promotion; CPU process time includes migration and probes"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:5]],int(sys.argv[5]))
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
