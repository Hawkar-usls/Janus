"""R84H8 bounded frozen-core modular vs monolithic classification."""
import sys,random,time,math
from pathlib import Path
import r84h7_immortal as h7
import r84h6_heritage as h6
import r84h5_control as h5
import r84h4_equilibrium as h4
import r84h3_stability as h3
import r84h2_growth as h2
import r84g_adaptive as a
import r84e_neural as e
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="10b58c9fdcc99debe2eb2d613e1e349c54fbf970"
ROOTS=("07dec5f7692e2d6b48452623933263d70804f03af47496fcda76a4703d1de712","9560dce854922a5093b536b2c4c9c49381250239dfa1a38cbd89252da74eeb17","d51f3381a1ac3638efd92c05fb1320213aae53966e758d1f0f58c9577df4d485")
TRAIN=[
{"h8_sensitivity":["sensitivity is true positive rate","detect actual positives among all actual positives","recall measures proportion of real positives caught","measure sensitivity of binary test","true positive count divided by actual positives","identify detection probability when condition exists"],"h8_specificity":["specificity is true negative rate","correctly reject actual negatives","measure specificity of binary classifier","true negatives divided by all actual negatives","identify exclusion probability when condition absent","report proportion of true negatives detected"]},
{"h8_precision":["precision is positive predictive value","correct positives among predicted positives","divide true positives by positive predictions","measure fraction of positive calls correct","evaluate positive prediction accuracy","report precision of binary decisions"],"h8_f1":["f1 score harmonic mean of precision and recall","combine precision and recall harmonically","two times product over sum of precision and recall","report f1 classification metric","balance recall and precision using harmonic mean","calculate f1 from positive predictive value and recall"]},
{"h8_calibration":["calibration aligns confidence with observed outcomes","compare probabilities with empirical event rates","measure reliability of probability estimates","predicted confidence matches actual frequency","check probabilistic calibration of classifier","assess predicted versus observed probabilities"],"h8_discrimination":["discrimination separates positive and negative cases","rank positive cases above negative cases","assess ordering by prediction score","measure discrimination independent of calibration","classifier distinguishes classes using score ranking","evaluate separation ability of prediction model"]}]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def data(prep):
    d0,old,hist,prior,pre,h6ds,h7ds=h7.data(prep);ds=[h7ds[-1]]
    for cycle,grp in enumerate(TRAIN,1):
        p=ds[-1];new=[["gate practice "+t,y] for y,ts in grp.items() for t in ts]
        d={**p,"old":p["old"]+p["new"],"new":new,"labels":dict(p["labels"]),"vocab":dict(p["vocab"]),"h8_cycle":cycle}
        words=sorted({w for t,y in new for w in e.tokenize(t)}-set(p["vocab"]))
        for i,w in enumerate(words,p["input_embedding_size"]):d["vocab"][w]=i
        for y in sorted(grp):d["labels"][y]=max(d["labels"].values())+1
        d["input_embedding_size"]=p["input_embedding_size"]+len(words);d["output_size"]=max(258,max(d["labels"].values())+1)
        ck(d["input_embedding_size"]<=896 and d["output_size"]<=288,"CAP")
        ck(all(d["vocab"][k]==v for k,v in p["vocab"].items()) and all(d["labels"][k]==v for k,v in p["labels"].items()),"ID_STABILITY")
        ds.append(d);log("R84H8_CAPACITY",{"cycle":cycle,"input":d["input_embedding_size"],"output":d["output_size"],"classes":len(d["labels"])})
    a.configure(ds[-1]);return d0,old,hist,prior,pre,h6ds,h7ds,ds
def routes(root,ds,cycle):
    m={y:0 for y in root["labels"]}
    for i in range(1,cycle+1):
        m.update({y:i for y in TRAIN[i-1]})
    return m
def features(L,core,root,d,rows):
    torch,_=L
    # Text only at inference; labels supplied solely to satisfy legacy batch API.
    rr=[[t,next(iter(root["labels"]))] for t,_ in rows]
    z=h7.logits(L,core,root,rr)
    bag=torch.zeros(len(rows),896)
    for i,(t,_) in enumerate(rr):
        ids=[v for v in e.encode(d,t) if v>1]
        if ids:
            for v in ids:bag[i,v]+=1.
            bag[i]/=math.sqrt(len(ids))
    return torch.cat([bag,z/10.],dim=1),z.argmax(-1)
def bundle(L,root,cycle,seed):
    torch,_=L;torch.manual_seed(seed+840800)
    dim=896+root["output_size"]
    router=torch.nn.Sequential(torch.nn.Linear(dim,48),torch.nn.Tanh(),torch.nn.Linear(48,4))
    mods=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(dim,32),torch.nn.Tanh(),torch.nn.Linear(32,2)) for _ in range(cycle)])
    return torch.nn.ModuleDict({"router":router,"modules_list":mods})
def load_bundle(L,root,d,folder,cycle,seed):
    torch,_=L;rec=e.read(folder/"receipt.json");p=folder/"shadow.pt"
    ck(rec["file_sha256"]==e.sha(p.read_bytes()) and rec["data_sha256"]==g.digest(d) and rec["production"] is False,"BUNDLE_RECEIPT")
    m=bundle(L,root,cycle,seed);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True)
    ck(e.state_hash(m)==rec["model_sha256"],"BUNDLE_HASH");return m,rec
def train_module(L,core,root,d,ds,cycle,seed,previous):
    torch,_=L;m=bundle(L,root,cycle,seed);old_hashes=[]
    if previous is not None:
        m["router"].load_state_dict(previous["router"].state_dict())
        for i,q in enumerate(previous["modules_list"]):
            m["modules_list"][i].load_state_dict(q.state_dict());old_hashes.append(e.state_hash(q))
    for q in m["modules_list"][:-1]:q.requires_grad_(False)
    core.requires_grad_(False);sha=e.state_hash(core);rows=d["old"]+d["new"];n=len(d["old"])
    x,_=features(L,core,root,d,rows);mapping=routes(root,ds,cycle)
    y=torch.tensor([mapping[v] for _,v in rows]);labels=sorted(TRAIN[cycle-1]);local=torch.tensor([labels.index(v) for _,v in d["new"]])
    opt=torch.optim.AdamW(list(m["router"].parameters())+list(m["modules_list"][-1].parameters()),lr=4e-4,weight_decay=.01)
    rng=random.Random(seed+840801);old_ids=list(range(n));schedule=[]
    while len(schedule)<1440:
        rng.shuffle(old_ids);schedule.extend(old_ids)
    schedule=schedule[:1440];ck(len(set(schedule))==n,"OLD_COVERAGE")
    counts=torch.bincount(y,minlength=4).float();weights=(counts[:cycle+1].sum()/counts[:cycle+1].clamp(min=1)).sqrt();weights=weights/weights.mean()
    for j in range(240):
        fresh=rng.choices(range(len(d["new"])),k=6);ids=schedule[6*j:6*j+6]+[n+i for i in fresh]
        opt.zero_grad(set_to_none=True);r=m["router"](x[ids])[:,:cycle+1]
        ce=torch.nn.functional.cross_entropy(r,y[ids],weight=weights)
        ce2=torch.nn.functional.cross_entropy(m["modules_list"][-1](x[[n+i for i in fresh]]),local[fresh])
        loss=ce+ce2;ck(math.isfinite(float(loss.detach())),"LOSS");loss.backward();opt.step()
    ck(e.state_hash(core)==sha and all(e.state_hash(q)==old_hashes[i] for i,q in enumerate(m["modules_list"][:-1])),"FROZEN_WEIGHTS")
    return m,{"steps":240,"old_examples":1440,"new_examples":1440,"old_pool":n,"covered_examples":len(set(schedule)),"core_sha256":sha,"frozen_modules":len(old_hashes),"module_parameters":sum(p.numel() for p in m.parameters()),"trainable_parameters":sum(p.numel() for p in m.parameters() if p.requires_grad)}
def train(prep,parent,root_path,anchor_path,out,cycle):
    ck(cycle in (1,2,3),"CYCLE");L=e.lib(prep);_,_,_,_,_,h6ds,_,ds=data(prep);root=ds[0];p,d=ds[cycle-1:cycle+1];out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        core,rec=h7.load(L,root,root_path/"dual50"/str(seed),seed);ck(rec["model_sha256"]==ROOTS[i],"ROOT")
        for mode in (("modular","monolith") if (i+cycle)%2 else ("monolith","modular")):
            cpu=time.process_time();ss=seed+1800+cycle*100
            if mode=="modular":
                prev=None;prevsha=ROOTS[i]
                if cycle>1:
                    prev,pr=load_bundle(L,root,p,parent/mode/str(seed),cycle-1,seed);ck(pr["root_sha256"]==ROOTS[i] and pr["cycle"]==cycle-1 and pr["seed"]==seed and pr["mode"]==mode,"MODULE_LINEAGE");prevsha=pr["model_sha256"]
                m,z=train_module(L,core,root,d,ds,cycle,ss,prev)
                z["parent_sha256"]=prevsha
            else:
                m,pr=h7.load(L,p,(root_path/"dual50"/str(seed)) if cycle==1 else parent/mode/str(seed),seed)
                if cycle==1:ck(pr["model_sha256"]==ROOTS[i],"MONO_ROOT")
                else:ck(pr["root_sha256"]==ROOTS[i] and pr["cycle"]==cycle-1 and pr["mode"]==mode and pr["seed"]==seed,"MONO_LINEAGE")
                h7.migrate(L,m,p,d,ss);anc,ar=h7.load(L,h6ds[0],anchor_path/"distill"/str(seed),seed);ck(ar["model_sha256"]==h6.ROOTS[i],"ANCHOR")
                z=h7.optimize(L,m,d,ss,"dual50",p["output_size"],anc,h6ds[0]);z["parent_sha256"]=pr["model_sha256"]
            ck(z["steps"]==240 and z["old_examples"]==1440 and z["new_examples"]==1440,"BUDGET")
            z.update({"mode":mode,"cycle":cycle,"seed":seed,"root_sha256":ROOTS[i],"cpu_seconds":time.process_time()-cpu})
            e.save(L,m,d,out/mode/str(seed),z);log("R84H8_TRAIN",z)
def predictions(L,core,root,d,rows):
    _,pred=features(L,core,root,d,rows);return pred.tolist()
def modular_predict(L,core,root,d,ds,cycle,m,rows):
    torch,_=L;xx,base=features(L,core,root,d,rows);m.eval();mapping=routes(root,ds,cycle)
    with torch.no_grad():
        route=m["router"](xx)[:,:cycle+1].argmax(-1)
        candidates=[base]
        for i,q in enumerate(m["modules_list"]):
            labels=sorted(TRAIN[i]);v=torch.tensor([d["labels"][y] for y in labels]);candidates.append(v[q(xx).argmax(-1)])
        cand=torch.stack(candidates,dim=1);pred=cand[torch.arange(len(rows)),route]
        # Oracle labels only for evaluator; never fed to actual route.
        truth=torch.tensor([mapping.get(y,-1) for _,y in rows]);valid=truth>=0
        oracle=torch.full_like(pred,-1);oracle[valid]=cand[torch.arange(len(rows))[valid],truth[valid]]
    return pred.tolist(),oracle.tolist(),route.tolist(),truth.tolist()
def scores(pred,groups,labels):
    out=[];at=0
    for grp in groups:
        p=pred[at:at+len(grp)];at+=len(grp)
        out.append({"preds":p,"exact":sum(v==labels[y] for v,(_,y) in zip(p,grp)),"n":len(grp)})
    ck(at==len(pred),"PRED_COUNT");return out
def evaluate(prep,root_path,c1,c2,c3,out):
    L=e.lib(prep);d0,old,hist,prior,pre,h6ds,h7ds,ds=data(prep);root=ds[0];groups=h2.exams(d0,old)
    for mod,dd,name in [(h3,hist,"H3"),(h4,prior,"H4"),(h5,pre,"H5"),(h6,h6ds,"H6"),(h7,h7ds,"H7")]:
        groups+=h5.clean_exam("R84"+name+"_FROZEN_EXAM.json",mod.EXAM,dd,groups)
    ck(sum(map(len,groups))==401,"PRIOR");fresh=h5.clean_exam("R84H8_FROZEN_EXAM.json",EXAM,ds,groups);groups+=fresh;rows=sum(groups,[]);labels=ds[-1]["labels"]
    log("R84H8_EXAM",{"prior_n":401,"fresh_n":list(map(len,fresh)),"uniform_inference":True})
    results=[]
    for i,seed in enumerate(e.SEEDS):
        core,cr=h7.load(L,root,root_path/"dual50"/str(seed),seed);ck(cr["model_sha256"]==ROOTS[i],"ROOT")
        bp=predictions(L,core,root,root,rows);base=scores(bp,groups,labels)
        for mode in ("modular","monolith"):
            prev=base;last=ROOTS[i]
            for cycle,folder in enumerate((c1,c2,c3),1):
                d=ds[cycle];start=time.process_time()
                if mode=="modular":
                    m,rec=load_bundle(L,root,d,folder/mode/str(seed),cycle,seed)
                    pred,oracle,route,truth=modular_predict(L,core,root,d,ds,cycle,m,rows)
                    eligible=sum(t>=0 for t in truth);route_correct=sum(r==t and t>=0 for r,t in zip(route,truth))
                    oc=sum(v==labels[y] for v,(_,y) in zip(oracle,rows));old_n=sum(map(len,groups[:-3]))
                    extra={"oracle_dep":oc,"route_correct":route_correct,"route_eligible":eligible,"old_misroute":sum(r!=0 for r in route[:old_n]),"core_unchanged":e.state_hash(core)==ROOTS[i],"parameters":sum(p.numel() for p in core.parameters())+sum(p.numel() for p in m.parameters())}
                else:
                    m,rec=h7.load(L,d,folder/mode/str(seed),seed)
                    z=h7.logits(L,m,{**d,"labels":labels},rows);pred=z.argmax(-1).tolist()
                    extra={"parameters":sum(p.numel() for p in m.parameters())}
                ck(rec["parent_sha256"]==last and rec["root_sha256"]==ROOTS[i] and rec["cycle"]==cycle and rec["seed"]==seed and rec["mode"]==mode,"LINEAGE")
                ans=scores(pred,groups,labels);row=h3.compare(prev,ans,groups,labels);cum=h3.compare(base,ans,groups,labels)
                row.update(extra);row.update({"mode":mode,"seed":seed,"cycle":cycle,"root_lost":cum["lost"],"root_before":cum["before"],"cpu_seconds":rec["cpu_seconds"],"eval_cpu_seconds":time.process_time()-start,"profit_cpu":row["net_gain"]/rec["cpu_seconds"],"model_sha256":rec["model_sha256"]})
                results.append(row);log("R84H8_RESULT",{k:v for k,v in row.items() if k not in ("lost_indices","per_class")});prev=ans;last=rec["model_sha256"]
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h8.result.v1","results":results,"production":False,"external_validation":False,"scope":"frozen core with append-only small MLP heads and learned text-only router; same 240 steps and 2880 sampled rows; distinct parameter counts and losses; uniform inference dictionaries not historical cohort-specific evaluation; oracle diagnostic only"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:7]],int(sys.argv[7]))
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
