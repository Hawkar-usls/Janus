"""R84H6: finite classification-only heritage ablation, three matched arms."""
import json,math,random,sys,time
from pathlib import Path
import r84h5_control as h5
import r84h4_equilibrium as h4
import r84h3_stability as h3
import r84h2_growth as h2
import r84g2_selective as s
import r84g_adaptive as a
import r84e_neural as e
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="42d13b597cc479c3443f33dc07d029a9825bd1c7"
ROOTS=("52759c49c4188930c6f8126141852e004d4dce6caf7282eed14f9dd02343ecb0","4d7766a99b007677f442a91171faf101d0e985a0f8ba612f0f1eb4e42018e071","5c4d2e4693d6549ad53ed6db40e49feeba14948cfc65c6c5eb87abf9a14ca7b2")
MODES=("distill","low_lr","heritage")
TRAIN=[
{"h6_skew":["skewness measures distribution asymmetry","third standardized moment describes skew","positive skew has longer right tail","negative skew has longer left tail","measure asymmetric shape of data","report skewness of distribution"],"h6_kurt":["kurtosis measures tail heaviness","fourth standardized moment of distribution","describe extreme value tendency using kurtosis","fourth central moment over variance squared","compare tail weight against normal distribution","report kurtosis of numerical sample"]},
{"h6_weighted":["weighted mean is sum of weighted values over weight sum","average observations with different weights","assign relative importance before averaging","multiply each observation by its weight","compute weighted average of measurements","normalize weighted sum by total weights"],"h6_trimmed":["trimmed mean excludes both extreme tails","remove smallest and largest equal proportions","average remaining values after trimming","reduce outlier influence using trimmed mean","discard upper and lower extremes then average","report trimmed arithmetic mean"]},
{"h6_geometric":["geometric mean is root of product","average positive multiplicative factors","take exponential of average logarithms","geometric average of growth rates","multiply positive values then take nth root","report geometric mean of positive sample"],"h6_harmonic":["harmonic mean is reciprocal of average reciprocals","sample size divided by sum of inverses","average rates over equal distances","use harmonic mean for equal distance speeds","invert mean of reciprocal observations","report harmonic mean of positive rates"]}]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def data(prep):
    d0,old,hist,prior,pre=h5.data(prep);ds=[pre[-1]]
    for cycle,grp in enumerate(TRAIN,1):
        p=ds[-1];new=[["gate practice "+t,y] for y,ts in grp.items() for t in ts]
        d={**p,"old":p["old"]+p["new"],"new":new,"labels":dict(p["labels"]),"vocab":dict(p["vocab"]),"h6_cycle":cycle}
        words=sorted({w for t,y in new for w in e.tokenize(t)}-set(p["vocab"]))
        for i,w in enumerate(words,p["input_embedding_size"]):d["vocab"][w]=i
        for y in sorted(grp):d["labels"][y]=max(d["labels"].values())+1
        d["input_embedding_size"]=p["input_embedding_size"]+len(words)
        d["output_size"]=max(258,max(d["labels"].values())+1)
        log("R84H6_CAPACITY",{"cycle":cycle,"input_size":d["input_embedding_size"],"output_size":d["output_size"],"old_output_size":p.get("output_size",258),"classes":len(d["labels"]),"namespace_overlap":len(set(d["labels"].values())&set(d["vocab"].values()))})
        ck(d["input_embedding_size"]<=768 and d["output_size"]<=272 and len(set(d["vocab"].values()))==len(d["vocab"]) and len(set(d["labels"].values()))==len(d["labels"]),"CAPACITY")
        ck(all(d["vocab"][k]==v for k,v in p["vocab"].items()) and all(d["labels"][k]==v for k,v in p["labels"].items()),"ID_STABILITY")
        ds.append(d)
    a.configure(ds[-1]);return d0,old,hist,prior,pre,ds
def expand(L,m,d,seed):
    torch,_=L;nin=m.wte.num_embeddings;nout=m.lm_head.out_features
    si=d["input_embedding_size"];so=d.get("output_size",258)
    ck(nin<=si<=768 and nout<=so<=272,"SHAPE_CAP")
    if si>nin:
        new=torch.nn.Embedding(si,m.wte.embedding_dim);gen=torch.Generator().manual_seed(seed+842000)
        with torch.no_grad():
            new.weight[:nin].copy_(m.wte.weight);new.weight[nin:].copy_(torch.randn((si-nin,m.wte.embedding_dim),generator=gen)*.02)
        m.wte=new
    if so>nout:
        ck(m.lm_head.bias is None,"HEAD_BIAS")
        new=torch.nn.Linear(m.lm_head.in_features,so,bias=False)
        with torch.no_grad():
            new.weight[:nout].copy_(m.lm_head.weight);new.weight[nout:].zero_()
        m.lm_head=new
    return m
def load(L,d,folder,seed):
    torch,_=L;p=e.read(folder/"receipt.json");path=folder/"shadow.pt"
    ck(p["file_sha256"]==e.sha(path.read_bytes()) and p["data_sha256"]==g.digest(d) and p["production"] is False,"RECEIPT")
    m=expand(L,e.model(L),d,seed);m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
    ck(e.state_hash(m)==p["model_sha256"] and all(torch.isfinite(x).all() for x in m.state_dict().values()),"TENSORS")
    return m,p
def migrate(L,m,p,d,seed):
    torch,_=L;old={k:v.detach().clone() for k,v in m.state_dict().items()}
    x,l,y=e.batch(L,p,d["old"]);m.eval();nout=m.lm_head.out_features
    with torch.no_grad():before=e.forward(L,m,x,l).clone()
    expand(L,m,d,seed);nin=p["input_embedding_size"]
    for k,v in old.items():
        after=m.state_dict()[k]
        if k=="wte.weight":after=after[:nin]
        elif k=="lm_head.weight":after=after[:nout]
        ck(torch.equal(v,after),"OLD_TENSORS:"+k)
    xx,ll,yy=e.batch(L,d,d["old"]);ck(torch.equal(x,xx) and torch.equal(l,ll) and torch.equal(y,yy),"OLD_IDS")
    m.eval()
    with torch.no_grad():
        z=e.forward(L,m,xx,ll);delta=float((before-z[:,:nout]).abs().max())
        ck(torch.allclose(before,z[:,:nout],atol=2e-5,rtol=2e-5),"OLD_LOGITS")
        ck(torch.equal(before.argmax(-1),z[:,:nout].argmax(-1)),"OLD_ARGMAX")
        changed=int((before.argmax(-1)!=z.argmax(-1)).sum())
        ck(changed==0,"NEW_HEAD_CHANGED_OLD_PREDICTIONS")
    log("R84H6_MIGRATION",{"old_input":nin,"new_input":m.wte.num_embeddings,"old_output":nout,"new_output":m.lm_head.out_features,"old_rows":len(d["old"]),"max_old_logit_delta":delta,"changed_predictions":changed,"exact_old_tensors":True})
    return e.state_hash(m)
def optimize(L,m,d,seed,mode,old_output):
    torch,_=L;torch.manual_seed(seed+30700);rng=random.Random(seed+30700);rr=random.Random(seed+50700)
    opt=torch.optim.AdamW(m.parameters(),lr=4e-4 if mode in ('low_lr','heritage') else 8e-4,weight_decay=.01)
    cap=[];hooks=[layer.linear1.register_forward_hook(lambda mod,inp,out:cap.append(out)) for layer in m.transformer_encoder.layers]
    old=d["old"];n=len(old);by={}
    for i,(_,y) in enumerate(old):by.setdefault(y,[]).append(i)
    ck(len(by)<=120 and n<=720,"COVERAGE_CAP")
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
    elif mode in ("distill","heritage"):
        with torch.no_grad():
            teacher=torch.cat([e.forward(L,m,*e.batch(L,d,old[j:j+32])[:2]).detach() for j in range(0,n,32)])
        extra=n
    try:
        for block in range(6):
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
            while len(selected)<120:
                unseen=[i for i in range(n) if counts[i]==0]
                if unseen:pick(rr.choice(unseen));continue
                p=[(0.1+min(3.,max(0.,now[i]-base[i]))+.35*min(4.,now[i]))*(1.+.5*max(-.5,min(1.,utility[i])))/(1.+blockcounts[i]) for i in range(n)]
                pick(rr.choices(range(n),weights=p,k=1)[0])
            rr.shuffle(selected)
            ck(all(any(blockcounts[i]>0 for i in ids) for ids in by.values()),"CLASS_FLOOR")
            for j in range(40):
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
                elif mode in ("distill","heritage"):
                    ids=selected[3*j:3*j+3];T=2.
                    penalty=.5*T*T*torch.nn.functional.kl_div(torch.nn.functional.log_softmax(z[-3:,:old_output]/T,dim=-1),torch.nn.functional.softmax(teacher[ids,:old_output]/T,dim=-1),reduction="batchmean")
                loss=ce+penalty;penalty_total+=float(penalty.detach());ck(math.isfinite(float(loss.detach())),"LOSS")
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            history.append({"block":block,"class_min":min(sum(blockcounts[i] for i in ids) for ids in by.values()),"covered":sum(c>0 for c in counts),"mean_loss":sum(now)/n,"utility_nonzero":sum(u!=0 for u in utility)})
            previous=now;last=blockcounts
    finally:
        for h in hooks:h.remove()
    final=s.losses(L,m,d,old)
    ck(sum(counts)==720 and min(counts)>=1,"FULL_REPLAY_COVERAGE")
    return {"steps":240,"extra_probes":extra,"mean_penalty":penalty_total/240,"examples":2880,"old_examples":720,"new_examples":2160,"old_pool":n,"covered_examples":sum(c>0 for c in counts),"old_classes":len(by),"min_class_replays":min(sum(counts[i] for i in ids) for ids in by.values()),"per_class_replays":{y:sum(counts[i] for i in ids) for y,ids in by.items()},"sample_probe_forward_rows":7*n,"history":history,"sample_counts":counts,"mean_train_loss_before":sum(base)/n,"mean_train_loss_after":sum(final)/n,"seconds":time.monotonic()-start}

def train(prep,parent,out,cycle):
    ck(cycle in (1,2,3),"CYCLE");L=e.lib(prep);_,_,_,_,_,ds=data(prep);p,d=ds[cycle-1:cycle+1];out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        order=MODES[(i+cycle)%3:]+MODES[:(i+cycle)%3]
        for mode in order:
            folder=parent/"distill"/str(seed) if cycle==1 else parent/mode/str(seed)
            m,rec=load(L,p,folder,seed)
            if cycle==1:ck(rec["model_sha256"]==ROOTS[i],"ROOT")
            else:ck(rec["mode"]==mode and rec["cycle"]==cycle-1 and rec["seed"]==seed and rec["root_sha256"]==ROOTS[i],"PARENT")
            ss=seed+1200+100*cycle;start=time.perf_counter();cpu=time.process_time();mig=migrate(L,m,p,d,ss)
            z=optimize(L,m,d,ss,mode,p.get('output_size',258))
            ck(z["steps"]==240 and z["old_examples"]==720 and z["new_examples"]==2160 and z["covered_examples"]==len(d["old"]) and z["min_class_replays"]>=6,"BUDGET")
            ck(z["extra_probes"]==(len(d["old"]) if mode in ("distill","heritage") else 0),"TEACHER")
            z.update({"mode":mode,"cycle":cycle,"seed":seed,"root_sha256":ROOTS[i],"parent_sha256":rec["model_sha256"],"migration_sha256":mig,"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-start,"lr":4e-4 if mode in ("low_lr","heritage") else 8e-4,"blocks":6})
            e.save(L,m,d,out/mode/str(seed),z);log("R84H6_TRAIN",{k:v for k,v in z.items() if k not in ("history","sample_counts","per_class_replays")})
def evaluate(prep,root,c1,c2,c3,out):
    L=e.lib(prep);d0,old,hist,prior,pre,ds=data(prep);groups=h2.exams(d0,old)
    groups+=h5.clean_exam("R84H3_FROZEN_EXAM.json",h3.EXAM,hist,groups)
    groups+=h5.clean_exam("R84H4_FROZEN_EXAM.json",h4.EXAM,prior,groups)
    groups+=h5.clean_exam("R84H5_FROZEN_EXAM.json",h5.EXAM,pre,groups)
    ck(sum(map(len,groups))==347,"PRIOR")
    fresh=h5.clean_exam("R84H6_FROZEN_EXAM.json",EXAM,ds,groups);groups+=fresh
    log("R84H6_EXAM",{"prior_n":347,"fresh_n":list(map(len,fresh)),"fresh_full_n":[8,8,8]})
    results=[]
    for i,seed in enumerate(e.SEEDS):
        m,p=load(L,ds[0],root/"distill"/str(seed),seed);ck(p["model_sha256"]==ROOTS[i],"ROOT")
        def stage(q,cy):return h3.score(L,q,groups,old+hist[1:]+prior[1:]+pre[1:]+[ds[min(j,cy)] for j in (1,2,3)],ds[-1]["labels"])
        base=stage(m,0)
        for mode in MODES:
            prev=base;last=p["model_sha256"]
            for cycle,folder in enumerate((c1,c2,c3),1):
                q,rec=load(L,ds[cycle],folder/mode/str(seed),seed)
                ck(rec["parent_sha256"]==last and rec["root_sha256"]==ROOTS[i] and rec["mode"]==mode and rec["seed"]==seed and rec["cycle"]==cycle,"LINEAGE")
                ans=stage(q,cycle);row=h3.compare(prev,ans,groups,ds[-1]["labels"])
                cumulative=h3.compare(base,ans,groups,ds[-1]["labels"])
                row.update({"mode":mode,"seed":seed,"cycle":cycle,"cpu_seconds":rec["cpu_seconds"],"profit_cpu":row["net_gain"]/rec["cpu_seconds"],"root_lost":cumulative["lost"],"root_before":cumulative["before"],"model_sha256":rec["model_sha256"],"parent_sha256":last})
                results.append(row);log("R84H6_RESULT",{k:v for k,v in row.items() if k not in ("lost_indices","per_class")});prev=ans;last=rec["model_sha256"]
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h6.result.v1","results":results,"production":False,"external_validation":False,"scope":"three finite synthetic concept classification cycles; matched 6 blocks of 40 updates to allow >60 classes; KL only old training rows and pre-expansion outputs; separate input/output ID namespaces; zero-initialized new head rows; no automatic promotion; CPU includes teacher evaluation"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:5]],int(sys.argv[5]))
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
