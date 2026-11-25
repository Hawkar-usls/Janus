"""R84H7: finite classification-only factorial memory experiment."""
import json,math,random,sys,time
from pathlib import Path
import r84h6_heritage as h6
import r84h5_control as h5
import r84h4_equilibrium as h4
import r84h3_stability as h3
import r84h2_growth as h2
import r84g2_selective as s
import r84g_adaptive as a
import r84e_neural as e
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="dff0e04addaf318c949cb1ff56e20809b368745a"
ROOTS=("0e170f92dd8785d522a467e71e108e4cf0510f69ddb8db39e677a223048829d6","b6ce0285340635deaffce50501359e706853d181f32230d01b17eb3a31e7743e","d3e7454023c8c3e256999d9e62c70f338959f8e0da0f694ee6eecb219c68aa9d")
MODES=("single25","dual25","single50","dual50")
TRAIN=[
{"h7_random":["simple random sample gives equal chance","select units randomly from population","probability sample without selection preference","random draw from complete sampling frame","use random selection for observations","sample each population member with equal probability"],"h7_strata":["stratified sample draws from each subgroup","divide population into strata before sampling","sample within every population stratum","represent subgroups through separate samples","use stratified random selection","select units from each defined category"]},
{"h7_cluster":["cluster sampling selects existing groups","randomly choose clusters of population members","sample groups instead of individual units","choose entire selected neighborhoods for sampling","use cluster selection of schools","draw clusters then inspect their members"],"h7_systematic":["systematic sample selects every fixed interval","choose every tenth item after random start","regular interval sampling from ordered list","sample every kth unit in sequence","use fixed spacing in selection process","systematic selection from sampling frame"]},
{"h7_selection":["selection bias favors particular population units","recruitment process produces unrepresentative sample","inclusion probability systematically differs","biased selection distorts population estimates","sample differs because of inclusion mechanism","nonrandom recruitment introduces selection bias"],"h7_nonresponse":["nonresponse bias from missing survey answers","nonrespondents differ from respondents","people who decline survey affect estimate","missing participants create biased survey results","response rate affects representativeness","systematic refusal introduces nonresponse bias"]},
{"h7_bias":["systematic measurement bias shifts readings","instrument calibration offset persists","consistent directional measurement error","readings consistently above true reference","fixed error does not average away","measure systematic offset from reference"],"h7_randomerror":["random measurement error varies unpredictably","unpredictable deviations between repeated readings","random fluctuations around true measurement","averaging reduces independent random errors","irregular variation in measurement process","random noise causes scattered measurements"]},
{"h7_repeat":["repeatability under same operator and apparatus","measure repeatedly under identical conditions","same setup repeated readings agree","within laboratory consistency under fixed conditions","assess repeatability of same instrument","repeat measurements without changing conditions"],"h7_reproduce":["reproducibility across different laboratories","compare measurements by different operators","agreement under changed measurement conditions","test reproducibility using independent equipment","cross laboratory measurement consistency","reproduce result with another measurement team"]}]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def data(prep):
    d0,old,hist,prior,pre,h6ds=h6.data(prep);ds=[h6ds[-1]]
    for cycle,grp in enumerate(TRAIN,1):
        p=ds[-1];new=[["gate practice "+t,y] for y,ts in grp.items() for t in ts]
        d={**p,"old":p["old"]+p["new"],"new":new,"labels":dict(p["labels"]),"vocab":dict(p["vocab"]),"h7_cycle":cycle}
        words=sorted({w for t,y in new for w in e.tokenize(t)}-set(p["vocab"]))
        for i,w in enumerate(words,p["input_embedding_size"]):d["vocab"][w]=i
        for y in sorted(grp):d["labels"][y]=max(d["labels"].values())+1
        d["input_embedding_size"]=p["input_embedding_size"]+len(words);d["output_size"]=max(258,max(d["labels"].values())+1)
        ck(d["input_embedding_size"]<=896 and d["output_size"]<=288 and len(set(d["vocab"].values()))==len(d["vocab"]) and len(set(d["labels"].values()))==len(d["labels"]),"CAP")
        ck(all(d["vocab"][k]==v for k,v in p["vocab"].items()) and all(d["labels"][k]==v for k,v in p["labels"].items()),"ID_STABILITY")
        log("R84H7_CAPACITY",{"cycle":cycle,"input_size":d["input_embedding_size"],"output_size":d["output_size"],"classes":len(d["labels"])})
        ds.append(d)
    a.configure(ds[-1]);return d0,old,hist,prior,pre,h6ds,ds
def expand(L,m,d,seed):
    torch,_=L;nin=m.wte.num_embeddings;nout=m.lm_head.out_features
    si=d["input_embedding_size"];so=d.get("output_size",258)
    ck(nin<=si<=896 and nout<=so<=288,"SHAPE_CAP")
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
    log("R84H7_MIGRATION",{"old_input":nin,"new_input":m.wte.num_embeddings,"old_output":nout,"new_output":m.lm_head.out_features,"old_rows":len(d["old"]),"max_old_logit_delta":delta,"changed_predictions":changed,"exact_old_tensors":True})
    return e.state_hash(m)

def logits(L,m,d,rows):
    torch,_=L;m.eval()
    with torch.no_grad():
        z=torch.cat([e.forward(L,m,*e.batch(L,d,rows[j:j+32])[:2]).detach() for j in range(0,len(rows),32)])
    ck(torch.isfinite(z).all(),"TEACHER_FINITE");return z
def optimize(L,m,d,seed,mode,old_output,ancestor,anchor_d):
    torch,_=L;torch.manual_seed(seed+30700);rng=random.Random(seed+30700);rr=random.Random(seed+50700)
    opt=torch.optim.AdamW(m.parameters(),lr=4e-4,weight_decay=.01)
    old=d["old"];n=len(old);k=6 if mode.endswith("50") else 3;slots=40*k;by={};counts=[0]*n
    for i,(_,y) in enumerate(old):by.setdefault(y,[]).append(i)
    ck(len(by)<=slots and n<=6*slots,"COVERAGE_CAP")
    teacher=logits(L,m,d,old)[:,:old_output];base=s.losses(L,m,d,old);prev=base;last=[0]*n;utility=[0.]*n;anc=None;na=0;anc_sha=None
    if mode.startswith("dual"):
        ck(ancestor is not None,"ANCESTOR");rows=anchor_d["old"]+anchor_d["new"];na=len(rows)
        ck(old[:na]==rows and all(d["labels"][y]==anchor_d["labels"][y] for _,y in rows),"ANCESTOR_ROWS")
        anc_sha=e.state_hash(ancestor);anc=logits(L,ancestor,anchor_d,rows);ck(anc.shape[1]==anchor_d.get("output_size",258),"ANCESTOR_HEAD")
    else:ck(ancestor is None,"SINGLE")
    T=2.;total_kl=0.;anchored=0
    for block in range(6):
        now=base if block==0 else s.losses(L,m,d,old)
        if block:
            for i,c in enumerate(last):
                if c:utility[i]=.5*utility[i]+.5*max(-1.,min(1.,(prev[i]-now[i])/c))
        selected=[];bc=[0]*n
        def pick(i):selected.append(i);counts[i]+=1;bc[i]+=1
        for label,ids in sorted(by.items()):
            pick(min(ids,key=lambda i:(counts[i],-max(0.,now[i]-base[i]),-now[i],i)))
        while len(selected)<slots:
            unseen=[i for i in range(n) if counts[i]==0]
            if unseen:pick(rr.choice(unseen));continue
            p=[(0.1+min(3.,max(0.,now[i]-base[i]))+.35*min(4.,now[i]))*(1.+.5*max(-.5,min(1.,utility[i])))/(1.+bc[i]) for i in range(n)]
            pick(rr.choices(range(n),weights=p,k=1)[0])
        rr.shuffle(selected);ck(all(any(bc[i] for i in ids) for ids in by.values()),"CLASS_FLOOR")
        for j in range(40):
            ids=selected[k*j:k*(j+1)];rows=rng.choices(d["new"],k=12-k)+[old[i] for i in ids]
            m.train();opt.zero_grad(set_to_none=True);x,l,y=e.batch(L,d,rows);z=e.forward(L,m,x,l);ce=torch.nn.functional.cross_entropy(z,y)
            student=z[-k:];kl=torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student[:,:old_output]/T,dim=-1),torch.nn.functional.softmax(teacher[ids]/T,dim=-1),reduction="none").sum(-1)
            penalty=.5*T*T*kl.mean()
            if anc is not None:
                pos=[j for j,i in enumerate(ids) if i<na]
                if pos:
                    ak=torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student[pos,:anc.shape[1]]/T,dim=-1),torch.nn.functional.softmax(anc[[ids[j] for j in pos]]/T,dim=-1),reduction="none").sum(-1)
                    # Keep total KL weight identical: replace half current-teacher weight on anchor-eligible rows.
                    penalty=penalty+.25*T*T*(ak-kl[pos]).sum()/k;anchored+=len(pos)
            loss=ce+penalty;total_kl+=float(penalty.detach());ck(math.isfinite(float(loss.detach())),"LOSS")
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        prev=now;last=bc
    final=s.losses(L,m,d,old);ck(sum(counts)==240*k and min(counts)>0,"REPLAY")
    if anc is not None:ck(anchored>0 and e.state_hash(ancestor)==anc_sha,"ANCESTOR_IMMUTABLE")
    return {"steps":240,"examples":2880,"old_examples":240*k,"new_examples":240*(12-k),"covered_examples":n,"old_pool":n,"old_classes":len(by),"min_class_replays":min(sum(counts[i] for i in ids) for ids in by.values()),"parent_probe_rows":n,"ancestor_probe_rows":na,"ancestor_replay_slots":anchored,"mean_penalty":total_kl/240,"mean_train_loss_before":sum(base)/n,"mean_train_loss_after":sum(final)/n}
def train(prep,parent,anchor,out,cycle):
    ck(cycle in range(1,6),"CYCLE");L=e.lib(prep);_,_,_,_,_,hist,ds=data(prep);p,d=ds[cycle-1:cycle+1];out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        order=MODES[(i+cycle)%4:]+MODES[:(i+cycle)%4]
        for mode in order:
            folder=parent/"heritage"/str(seed) if cycle==1 else parent/mode/str(seed)
            m,rec=load(L,p,folder,seed)
            if cycle==1:ck(rec["model_sha256"]==ROOTS[i],"ROOT")
            else:ck(rec["mode"]==mode and rec["cycle"]==cycle-1 and rec["seed"]==seed and rec["root_sha256"]==ROOTS[i],"PARENT")
            ss=seed+1700+100*cycle;start=time.perf_counter();cpu=time.process_time();mig=migrate(L,m,p,d,ss)
            teacher=None
            if mode.startswith("dual"):
                teacher,tr=load(L,hist[0],anchor/"distill"/str(seed),seed);ck(tr["model_sha256"]==h6.ROOTS[i],"ANCHOR")
            z=optimize(L,m,d,ss,mode,p.get("output_size",258),teacher,hist[0])
            ck(z["steps"]==240 and z["examples"]==2880 and z["covered_examples"]==len(d["old"]) and z["min_class_replays"]>=6,"BUDGET")
            z.update({"mode":mode,"cycle":cycle,"seed":seed,"root_sha256":ROOTS[i],"ancestor_sha256":h6.ROOTS[i] if teacher is not None else None,"parent_sha256":rec["model_sha256"],"migration_sha256":mig,"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-start})
            e.save(L,m,d,out/mode/str(seed),z);log("R84H7_TRAIN",z)
def evaluate(prep,root,anchor,c1,c2,c3,c4,c5,out):
    L=e.lib(prep);d0,old,hist,prior,pre,h6ds,ds=data(prep);groups=h2.exams(d0,old)
    for mod,dd,name in [(h3,hist,"H3"),(h4,prior,"H4"),(h5,pre,"H5"),(h6,h6ds,"H6")]:
        groups+=h5.clean_exam("R84"+name+"_FROZEN_EXAM.json",mod.EXAM,dd,groups)
    ck(sum(map(len,groups))==371,"PRIOR");fresh=h5.clean_exam("R84H7_FROZEN_EXAM.json",EXAM,ds,groups);groups+=fresh
    log("R84H7_EXAM",{"prior_n":371,"fresh_n":list(map(len,fresh)),"fresh_full_n":[6]*5})
    results=[];historic=old+hist[1:]+prior[1:]+pre[1:]+h6ds[1:]
    for i,seed in enumerate(e.SEEDS):
        m,p=load(L,ds[0],root/"heritage"/str(seed),seed);ck(p["model_sha256"]==ROOTS[i],"ROOT")
        anc,ar=load(L,h6ds[0],anchor/"distill"/str(seed),seed);ck(ar["model_sha256"]==h6.ROOTS[i],"ANCHOR")
        # Anchor exam uses only cohorts that existed at R84H5; preserve exact historic encodings.
        ag=groups[:10];ad=historic[:10];ab=h3.score(L,anc,ag,ad,h6ds[0]["labels"])
        def stage(q,cy):return h3.score(L,q,groups,historic+[ds[min(j,cy)] for j in range(1,6)],ds[-1]["labels"])
        base=stage(m,0)
        for mode in MODES:
            prev=base;last=p["model_sha256"]
            for cycle,folder in enumerate((c1,c2,c3,c4,c5),1):
                q,rec=load(L,ds[cycle],folder/mode/str(seed),seed)
                ck(rec["parent_sha256"]==last and rec["root_sha256"]==ROOTS[i] and rec["mode"]==mode and rec["seed"]==seed and rec["cycle"]==cycle,"LINEAGE")
                ans=stage(q,cycle);row=h3.compare(prev,ans,groups,ds[-1]["labels"]);cum=h3.compare(base,ans,groups,ds[-1]["labels"])
                ac=h3.compare(ab,ans[:10],ag,h6ds[0]["labels"])
                row.update({"mode":mode,"seed":seed,"cycle":cycle,"cpu_seconds":rec["cpu_seconds"],"profit_cpu":row["net_gain"]/rec["cpu_seconds"],"root_lost":cum["lost"],"root_before":cum["before"],"ancestor_lost":ac["lost"],"ancestor_before":ac["before"],"model_sha256":rec["model_sha256"],"parent_sha256":last})
                results.append(row);log("R84H7_RESULT",{k:v for k,v in row.items() if k not in ("lost_indices","per_class")});prev=ans;last=rec["model_sha256"]
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h7.result.v1","results":results,"production":False,"external_validation":False,"scope":"finite factorial classification-only experiment; matched 240 steps and 2880 presentations; 25/50 percent old rows; fixed R84H5 anchor on eligible old training rows; matched total KL weight; no automatic promotion"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:6]],int(sys.argv[6]))
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
