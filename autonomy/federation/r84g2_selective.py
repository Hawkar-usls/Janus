"""R84G2: bounded sample-level replay with deterministic coverage and utility estimate."""
import math,random,time,json,sys
from collections import Counter
from pathlib import Path
import r84g_adaptive as a
import r84e_neural as e
import r84f_continuity as f
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="4cb064a302d248a3d0465625d5dbf63f8e43c4f0"
FIXED={"replay":["c3bb58e7ff89d529b982c9020d212bdadb97162669e5d5c15820600e092ba7e8","630b344c61cbd03890c1686737bf34d6557ea498b8e97c198bb5056e27ba9931","1b044fccd0d4bb4c6c6b0b3e6d4ac7e3c7d8f5222b32dd28a82007aba8d5d338"],"joint":["f76666f49573457cc499c47c8488dd0406954ac26d6bc80f4298bdc762d8bbfc","619f15c29303d5cf7c73636c4398e32a5ddbc8ec69988de86a08eeec90e5caf9","4d0c6ba9c54d7061ebd4791a04c3952aabbf229bc9647e3848806d64b090196b"]}
def ck(v,s):e.ck(v,s)
def log(t,x):e.log(t,x)
def losses(L,m,d,rows):
    torch,_=L;m.eval();out=[]
    with torch.no_grad():
        for j in range(0,len(rows),32):
            x,l,y=e.batch(L,d,rows[j:j+32]);z=e.forward(L,m,x,l)
            out.extend(torch.nn.functional.cross_entropy(z,y,reduction="none").tolist())
    ck(len(out)==len(rows) and all(math.isfinite(x) for x in out),"LOSS_FINITE")
    return out
def optimize(L,m,d,seed,arm):
    torch,_=L;torch.manual_seed(seed+30700);rng=random.Random(seed+30700);rr=random.Random(seed+50700)
    opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=.01)
    cap=[];hooks=[layer.linear1.register_forward_hook(lambda mod,inp,out:cap.append(out)) for layer in m.transformer_encoder.layers]
    old=d["old"];n=len(old);by={}
    for i,(_,y) in enumerate(old):by.setdefault(y,[]).append(i)
    ck(len(by)<=60 and n<=720,"COVERAGE_CAP")
    base=losses(L,m,d,old);previous=base;counts=[0]*n;last=[0]*n;utility=[0.]*n;history=[];start=time.monotonic()
    try:
        for block in range(12):
            now=base if block==0 else losses(L,m,d,old)
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
                loss=ce+(.2*reg if arm=="joint" else 0.);ck(math.isfinite(float(loss.detach())),"LOSS")
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            history.append({"block":block,"class_min":min(sum(blockcounts[i] for i in ids) for ids in by.values()),"covered":sum(c>0 for c in counts),"mean_loss":sum(now)/n,"utility_nonzero":sum(u!=0 for u in utility)})
            previous=now;last=blockcounts
    finally:
        for h in hooks:h.remove()
    final=losses(L,m,d,old)
    ck(sum(counts)==720 and min(counts)>=1,"FULL_REPLAY_COVERAGE")
    return {"steps":240,"examples":2880,"old_examples":720,"new_examples":2160,"old_pool":n,"covered_examples":sum(c>0 for c in counts),"old_classes":len(by),"min_class_replays":min(sum(counts[i] for i in ids) for ids in by.values()),"per_class_replays":{y:sum(counts[i] for i in ids) for y,ids in by.items()},"sample_probe_forward_rows":13*n,"history":history,"sample_counts":counts,"mean_train_loss_before":sum(base)/n,"mean_train_loss_after":sum(final)/n,"seconds":time.monotonic()-start}
def train(prep,parent,out,arm):
    ck(arm in a.ROOTS,"ARM");L=e.lib(prep);d0,d2,d3,_=a.datasets(prep);out.mkdir(parents=True,exist_ok=False)
    for seed in e.SEEDS:
        m,p=a.root(L,d2,parent,arm,seed);mig=a.migrate(L,m,d2,d3,seed);rec=optimize(L,m,d3,seed,arm)
        rec.update({"arm":arm,"mode":"selective","seed":seed,"parent_sha256":p["model_sha256"],"migration_sha256":mig,"exact_old_tensors_logits":True})
        e.save(L,m,d3,out/str(seed),rec);log("R84G2_TRAIN",{k:v for k,v in rec.items() if k not in ("history","sample_counts","per_class_replays")})
def evaluate(prep,replay,joint,out,rf,jf,rs,js):
    L=e.lib(prep);d0,d2,d3,_=a.datasets(prep)
    raw=(HERE/"R84G2_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");z=json.loads(raw)
    rawg=(HERE/"R84G_FROZEN_EXAM.json").read_bytes();ck(r.blob(rawg)==a.EXAM,"G_EXAM_PIN");zg=json.loads(rawg)
    old,fo,_=e.exam();prev=f.exam();reg=[old,fo,prev["legacy_new"],*prev["cohorts"]]
    fact=a.facts(d0);prior=reg+[zg["old_routing"],[[q.format(id=i),y] for i,k,y in fact for q in zg["facts_templates"][k]],zg["evidence"],zg["recovery"],zg["new"]]
    fresh=[]
    for name,rows in z["cohorts"]:
        if name=="facts_templates":rows=[[q.format(id=i),y] for i,k,y in fact for q in rows[k]]
        fresh.append(rows)
    used=d3["old"]+d3["new"]+sum(d3["monitor"],[])+sum(prior,[])
    codes={tuple(e.encode(d3,t)) for t,y in used};texts={a.norm(t) for t,y in used}
    clean=[[row for row in grp if a.norm(row[0]) not in texts and tuple(e.encode(d3,row[0])) not in codes] for grp in fresh]
    ck(all(len(x)>0 for x in clean),"FRESH_COHORT_EMPTY")
    log("R84G2_EXAM",{"prior_n":list(map(len,prior)),"fresh_n":list(map(len,clean)),"fresh_full_n":list(map(len,fresh))})
    allrows=prior+clean;result=[]
    for arm,parent,fix,sel in (("replay",replay,rf,rs),("joint",joint,jf,js)):
        for idx,seed in enumerate(e.SEEDS):
            m,p=a.root(L,d2,parent,arm,seed);mig=a.migrate(L,m,d2,d3,seed);base=[e.score(L,m,d3,x) for x in allrows]
            for mode,folder in (("fixed",fix),("selective",sel)):
                q,rec=a.load(L,d3,folder/str(seed),seed)
                ck(rec["arm"]==arm and rec["mode"]==mode and rec["seed"]==seed and rec["parent_sha256"]==p["model_sha256"] and rec["migration_sha256"]==mig and rec["steps"]==240 and rec["old_examples"]==720 and rec["new_examples"]==2160,"PAIR")
                if mode=="fixed":ck(rec["model_sha256"]==FIXED[arm][idx],"FIXED_PIN")
                else:ck(rec["covered_examples"]==len(d3["old"]) and rec["min_class_replays"]>=12,"COVERAGE")
                scores=[e.score(L,q,d3,x) for x in allrows];ret=[];lost=[]
                for j,(grp,b,s) in enumerate(zip(allrows,base,scores)):
                    good=[i for i,((t,y),v) in enumerate(zip(grp,b["preds"])) if v==d3["labels"][y]]
                    bad=[i for i in good if s["preds"][i]!=d3["labels"][grp[i][1]]]
                    ret.append(len(good)-len(bad));lost.append(bad)
                row={"arm":arm,"mode":mode,"seed":seed,"before":[s["exact"] for s in base],"after":[s["exact"] for s in scores],"n":[len(x) for x in allrows],"retained":ret,"lost_case_indices":lost,"seconds":rec["seconds"],"model_sha256":rec["model_sha256"],"parent_sha256":p["model_sha256"]}
                result.append(row);log("R84G2_RESULT",{k:v for k,v in row.items() if k!="lost_case_indices"})
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84g2.result.v1","results":result,"paired":True,"production":False,"external_validation":False,"scope":"same prior roots and training rows; fixed historical checkpoints; equal gradient budget, extra per-sample forward probes; utility is observational not causal"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:5]],sys.argv[5])
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
