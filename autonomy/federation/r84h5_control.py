"""R84H5 finite statistical concept classification stabilization comparison."""
import json,math,random,sys,time
from pathlib import Path
import r84h4_equilibrium as h4
import r84h3_stability as h3
import r84h2_growth as h2
import r84g2_selective as s
import r84g_adaptive as a
import r84e_neural as e
import r8_learning as r
HERE=Path(__file__).parent
EXAM="9446986e30708d25df90750d7cfec924cd2f0974"
ROOTS=("2863e87b4219fe8e96791d4fe90190f9c02a40ff631d91477bd5204d645bec7f","f92e01b19693601591ea1bc4b5d780e785c77c0c0f847c374032697e0f58a0b0","e7edbcf2b3a18ca3d25f938ed163772a09ead7978ea01abc7beb3af0ecd9e372")
MODES=("selective","low_lr","ewc","distill")
TRAIN=[
{"h5_mode":["mode is most common value","find value with highest frequency","most frequent observation in sample","report mode of distribution","identify frequently repeated value","select observation with greatest count"],"h5_variance":["variance is average squared deviation","square differences from mean then average","dispersion measured in squared units","variance equals standard deviation squared","report variance of observations","quantify mean squared deviation"]},
{"h5_cv":["coefficient of variation equals spread divided by mean","relative variability ratio for positive mean","standard deviation over arithmetic mean","normalized spread relative to average","compare coefficient of variation across samples","report relative dispersion ratio"],"h5_zscore":["z score equals centered value over standard deviation","subtract mean divide by spread","standardized numerical observation","distance from mean in standard deviation units","calculate standardized position of value","normalize observation by mean and standard deviation"]}]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def data(prep):
    d0,old,hist,prior=h4.data(prep);ds=[prior[-1]]
    for cycle,grp in enumerate(TRAIN,1):
        p=ds[-1];new=[["gate practice "+t,y] for y,ts in grp.items() for t in ts]
        d={**p,"old":p["old"]+p["new"],"new":new,"labels":dict(p["labels"]),"vocab":dict(p["vocab"]),"h5_cycle":cycle}
        words=sorted({w for t,y in new for w in e.tokenize(t)}-set(p["vocab"]))
        for i,w in enumerate(words,p["input_embedding_size"]):d["vocab"][w]=i
        for y in sorted(grp):d["labels"][y]=max(d["labels"].values())+1
        d["input_embedding_size"]=p["input_embedding_size"]+len(words)
        ck(d["input_embedding_size"]<=640 and max(d["labels"].values())<258 and not set(d["labels"].values())&set(d["vocab"].values()),"CAP")
        ds.append(d)
    a.configure(ds[-1]);return d0,old,hist,prior,ds
def optimize(L,m,d,seed,mode):
    torch,_=L;torch.manual_seed(seed+30700);rng=random.Random(seed+30700);rr=random.Random(seed+50700)
    opt=torch.optim.AdamW(m.parameters(),lr=4e-4 if mode=='low_lr' else 8e-4,weight_decay=.01)
    cap=[];hooks=[layer.linear1.register_forward_hook(lambda mod,inp,out:cap.append(out)) for layer in m.transformer_encoder.layers]
    old=d["old"];n=len(old);by={}
    for i,(_,y) in enumerate(old):by.setdefault(y,[]).append(i)
    ck(len(by)<=60 and n<=720,"COVERAGE_CAP")
    base=s.losses(L,m,d,old);previous=base;counts=[0]*n;last=[0]*n;utility=[0.]*n;history=[];start=time.monotonic()
    anchor={k:p.detach().clone() for k,p in m.named_parameters()}
    fisher={};teacher=None;extra=0;penalty_total=0.
    m.eval()
    if mode=="ewc":
        params=list(m.named_parameters());fisher={k:torch.zeros_like(p) for k,p in params}
        sample=random.Random(seed+990).sample(range(n),min(32,n))
        for i in sample:
            x,l,y=e.batch(L,d,[old[i]]);loss=torch.nn.functional.cross_entropy(e.forward(L,m,x,l),y)
            grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
            for (k,p),v in zip(params,grads):
                if v is not None:fisher[k]+=v.detach().square()/len(sample)
        extra=len(sample)
    elif mode=="distill":
        with torch.no_grad():
            teacher=torch.cat([e.forward(L,m,*e.batch(L,d,old[j:j+32])[:2]).detach() for j in range(0,n,32)])
        extra=n
    try:
        for block in range(12):
            now=base if block==0 else s.losses(L,m,d,old)
            if block:
                for i,c in enumerate(last):
                    if c:utility[i]=.5*utility[i]+.5*max(-1.,min(1.,(previous[i]-now[i])/c))
            # One guaranteed replay for each old class per block. Choose a least-seen
            # example, then favor larger forgetting. Spend remaining slots on unseen
            # examples first, then bounded sample-level loss and observed utility.
            selected=[];blockcounts=[0]*n
            def pick(i):
                selected.append(i);blockcounts[i]+=1;counts[i]+=1
            for label,ids in sorted(by.items()):
                pick(min(ids,key=lambda i:(counts[i],-max(0.,now[i]-base[i]),-now[i],i)))
            while len(selected)<60:
                unseen=[i for i in range(n) if counts[i]==0]
                if unseen:pick(rr.choice(unseen));continue
                p=[(0.1+min(3.,max(0.,now[i]-base[i]))+.35*min(4.,now[i]))*(1.+.5*max(-.5,min(1.,utility[i])))/(1.+blockcounts[i]) for i in range(n)]
                pick(rr.choices(range(n),weights=p,k=1)[0])
            rr.shuffle(selected)
            ck(all(any(blockcounts[i]>0 for i in ids) for ids in by.values()),"CLASS_FLOOR")
            for j in range(20):
                nr=rng.choices(d["new"],k=9);batch=nr+[old[i] for i in selected[3*j:3*j+3]]
                m.train();cap.clear();opt.zero_grad(set_to_none=True)
                x,l,y=e.batch(L,d,batch);z=e.forward(L,m,x,l);ce=torch.nn.functional.cross_entropy(z,y)
                ck(len(cap)==2,"HOOKS")
                mask=(torch.arange(x.shape[1])[None,:]<=l[:,None]).unsqueeze(-1)
                rates=[(torch.sigmoid(t/.1)*mask).sum((0,1))/mask.sum() for t in cap]
                reg=sum(((t-.30)**2).mean() for t in rates)/len(rates)
                penalty=z.new_zeros(())
                if mode=="ewc":
                    penalty=50.*sum((fisher[k]*(p-anchor[k]).square()).sum() for k,p in m.named_parameters())
                elif mode=="distill":
                    ids=selected[3*j:3*j+3];T=2.
                    penalty=.5*T*T*torch.nn.functional.kl_div(torch.nn.functional.log_softmax(z[-3:]/T,dim=-1),torch.nn.functional.softmax(teacher[ids]/T,dim=-1),reduction="batchmean")
                loss=ce+penalty;penalty_total+=float(penalty.detach());ck(math.isfinite(float(loss.detach())),"LOSS")
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            history.append({"block":block,"class_min":min(sum(blockcounts[i] for i in ids) for ids in by.values()),"covered":sum(c>0 for c in counts),"mean_loss":sum(now)/n,"utility_nonzero":sum(u!=0 for u in utility)})
            previous=now;last=blockcounts
    finally:
        for h in hooks:h.remove()
    final=s.losses(L,m,d,old)
    ck(sum(counts)==720 and min(counts)>=1,"FULL_REPLAY_COVERAGE")
    return {"steps":240,"extra_probes":extra,"mean_penalty":penalty_total/240,"examples":2880,"old_examples":720,"new_examples":2160,"old_pool":n,"covered_examples":sum(c>0 for c in counts),"old_classes":len(by),"min_class_replays":min(sum(counts[i] for i in ids) for ids in by.values()),"per_class_replays":{y:sum(counts[i] for i in ids) for y,ids in by.items()},"sample_probe_forward_rows":13*n,"history":history,"sample_counts":counts,"mean_train_loss_before":sum(base)/n,"mean_train_loss_after":sum(final)/n,"seconds":time.monotonic()-start}
def train(prep,parent,out,cycle):
    ck(cycle in (1,2),"CYCLE");L=e.lib(prep);_,_,_,_,ds=data(prep);p,d=ds[cycle-1:cycle+1];out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        order=MODES[(i+cycle)%4:]+MODES[:(i+cycle)%4]
        for mode in order:
            folder=parent/"selective"/str(seed) if cycle==1 else parent/mode/str(seed)
            m,rec=h3.load(L,p,folder,seed)
            if cycle==1:ck(rec["model_sha256"]==ROOTS[i],"ROOT")
            else:ck(rec["mode"]==mode and rec["cycle"]==1 and rec["seed"]==seed and rec["root_sha256"]==ROOTS[i],"PARENT")
            ss=seed+900+100*cycle;start=time.perf_counter();cpu=time.process_time();mig=h3.migrate(L,m,p,d,ss)
            z=optimize(L,m,d,ss,mode)
            ck(z["steps"]==240 and z["old_examples"]==720 and z["new_examples"]==2160 and z["covered_examples"]==len(d["old"]) and z["min_class_replays"]>=12,"BUDGET")
            z.update({"mode":mode,"cycle":cycle,"seed":seed,"root_sha256":ROOTS[i],"parent_sha256":rec["model_sha256"],"migration_sha256":mig,"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-start})
            e.save(L,m,d,out/mode/str(seed),z);log("R84H5_TRAIN",{k:v for k,v in z.items() if k not in ("history","sample_counts","per_class_replays")})
def clean_exam(filename,pin,ds,prior):
    raw=(HERE/filename).read_bytes();ck(r.blob(raw)==pin,"EXAM_PIN");ex=json.loads(raw)
    used=ds[-1]["old"]+ds[-1]["new"]+sum(ds[0]["monitor"],[])+sum(prior,[])
    norm={a.norm(t) for t,y in used};fresh=[]
    for i,grp in enumerate(ex["cohorts"],1):
        enc={tuple(e.encode(ds[i],t)) for t,y in used}
        clean=[row for row in grp if a.norm(row[0]) not in norm and tuple(e.encode(ds[i],row[0])) not in enc]
        ck(len(clean)>=4,"FRESH");fresh.append(clean)
    return fresh
def evaluate(prep,root,c1,c2,out):
    L=e.lib(prep);d0,old,hist,prior,ds=data(prep);groups=h2.exams(d0,old)
    groups+=clean_exam("R84H3_FROZEN_EXAM.json",h3.EXAM,hist,groups)
    groups+=clean_exam("R84H4_FROZEN_EXAM.json",h4.EXAM,prior,groups)
    ck(sum(map(len,groups))==331,"PRIOR")
    fresh=clean_exam("R84H5_FROZEN_EXAM.json",EXAM,ds,groups);groups+=fresh
    log("R84H5_EXAM",{"prior_n":331,"fresh_n":list(map(len,fresh)),"fresh_full_n":[8,8]})
    results=[]
    for i,seed in enumerate(e.SEEDS):
        m,p=h3.load(L,ds[0],root/"selective"/str(seed),seed);ck(p["model_sha256"]==ROOTS[i],"ROOT")
        def stage(q,cy):return h3.score(L,q,groups,old+[hist[1],hist[2]]+prior[1:]+[ds[min(j,cy)] for j in (1,2)],ds[-1]["labels"])
        base=stage(m,0)
        for mode in MODES:
            prev=base;last=p["model_sha256"]
            for cycle,folder in enumerate((c1,c2),1):
                q,rec=h3.load(L,ds[cycle],folder/mode/str(seed),seed)
                ck(rec["parent_sha256"]==last and rec["root_sha256"]==ROOTS[i] and rec["mode"]==mode and rec["seed"]==seed and rec["cycle"]==cycle,"LINEAGE")
                ans=stage(q,cycle);row=h3.compare(prev,ans,groups,ds[-1]["labels"])
                row.update({"mode":mode,"seed":seed,"cycle":cycle,"cpu_seconds":rec["cpu_seconds"],"profit_cpu":row["net_gain"]/rec["cpu_seconds"],"model_sha256":rec["model_sha256"],"parent_sha256":last})
                results.append(row);log("R84H5_RESULT",{k:v for k,v in row.items() if k not in ("lost_indices","per_class")});prev=ans;last=rec["model_sha256"]
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h5.result.v1","results":results,"production":False,"external_validation":False,"scope":"two finite synthetic statistical concept classification cycles; EWC empirical diagonal Fisher on 32 old training examples; distillation on replayed training examples only; no automatic promotion; CPU includes regularizers"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:5]],int(sys.argv[5]))
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
