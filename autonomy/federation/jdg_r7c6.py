"""R7C6: staged native CPU foundation trial. Research-only."""
import json, math, random, sys, tempfile, urllib.request, hashlib
from pathlib import Path
from dataclasses import dataclass
import jdg_r7c3 as evo
import jdg_r7c4 as school
import jfp_gateway as g

HERE=Path(__file__).resolve().parent
EXAM="393a6490bba4f0e4f41cf8484729407ceaeb91d7"
OLD="84c033c7bf38824249255cdc90f44db2a475f1af"
DOVOD="5f7de85ea89ae3ffcee7c9cf13d6eb653e43ccd0"
SEED=76
BLOCK=200
BLOCKS=8
def check(x,s):evo.check(x,s)
def read(p):return json.loads(Path(p).read_text())
def write(p,o):Path(p).write_bytes(g.canonical(o))
def key(r):
    s,k,a,b,y=r
    return (s,a,0) if s==1 else (s,min(a,b),max(a,b))
def difficulty(r):
    s,k,a,b,y=r
    return int(k=="ones") if s==1 else int((a+b>=5) if s==0 else (a+b>=15) if s==2 else (a%10+b%10>=10))
def prompt(r):
    s,k,a,b,y=r
    if k=="tens":return f"What is the tens digit of {a}? Reply with the integer only."
    if k=="ones":return f"What is the ones digit of {a}? Reply with the integer only."
    return f"Compute {a} + {b}. Reply with the integer only."
def exam():
    raw=(HERE/"R7C6_FROZEN_FOUNDATION_EXAM.json").read_bytes()
    check(evo.blob(raw)==EXAM,"EXAM_PIN")
    rows=json.loads(raw)["cases"]
    check(len(rows)==48 and [sum(r[0]==i for r in rows) for i in range(4)]==[8,16,8,16],"EXAM_COUNT")
    check(all(y==(a//10 if k=="tens" else a%10 if k=="ones" else a+b) for s,k,a,b,y in rows),"REFERENCE")
    return rows
def prepare(demi,out):
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(demi))
    from tools import trinity_r14_verified_oracle as oracle
    check(evo.blob(Path(oracle.__file__).read_bytes())=="7792bae07705e040504052bcf673342b661858f7","ORACLE")
    # Verify the original DOVOD source as provenance only; do not execute it.
    with urllib.request.urlopen("https://raw.githubusercontent.com/Hawkar-usls/janus-meta-registry/main/DovoD.py",timeout=30) as f:raw=f.read(262145)
    check(evo.blob(raw)==DOVOD,"DOVOD_SOURCE_PIN")
    old=(HERE/"R7C5_FROZEN_INTERNAL_EXAM.json").read_bytes()
    check(evo.blob(old)==OLD,"OLD_EXAM_PIN")
    blocked={tuple(sorted((a,b))) for a,b,y,c in json.loads(old)["cases"]}
    blocked|={tuple(sorted((11+7*i%38,13+11*i%39))) for i in range(24)}
    ex=exam();excluded={key(r) for r in ex};pools=[[],[],[],[]]
    for a in range(10):
        for b in range(a,10):
            s=0 if a+b<10 else 2
            pools[s].append([[s,"add",a,b,a+b]])
    for a in range(10,100):
        pools[1].append([[1,"tens",a,0,a//10],[1,"ones",a,0,a%10]])
        for b in range(a,100):
            if (a,b) not in blocked:pools[3].append([[3,"add",a,b,a+b]])
    rng=random.Random(SEED);train=[];dev=[]
    for s,p in enumerate(pools):
        p=[x for x in p if key(x[0]) not in excluded]
        rng.shuffle(p);n=[6,8,6,16][s]
        # Stratify by difficulty, except place-value groups contain both kinds.
        if s!=1:
            lo=[x for x in p if difficulty(x[0])==0];hi=[x for x in p if difficulty(x[0])==1]
            nlo=min(n//2,len(lo));nhi=min(n-nlo,len(hi));nlo=n-nhi
            chosen=lo[:nlo]+hi[:nhi];chosenkeys={key(x[0]) for x in chosen}
        else:chosen=p[:n];chosenkeys={key(x[0]) for x in chosen}
        rest=[x for x in p if key(x[0]) not in chosenkeys]
        if s==3:rest=rest[:300]
        dev.extend(r for x in chosen for r in x);train.extend(r for x in rest for r in x)
    check(not ({key(r) for r in train}&{key(r) for r in dev}) and not ({key(r) for r in train+dev}&excluded),"GROUP_LEAK")
    for s,k,a,b,y in train+dev:
        if k=="add":
            q=f"Calculate {a} + {b}. Answer with one integer."
            check(oracle.verify(q,str(y))["verdict"]=="APPROVE" and oracle.verify(q,str(y+1))["verdict"]=="VETO_WRONG","LABEL")
        else:
            t=0;u=a
            while u>=10:u-=10;t+=1
            check(y==(t if k=="tens" else u),"PLACE_VALUE_LABEL")
    for s in range(4):
        check({difficulty(r) for r in train if r[0]==s}=={0,1},"TRAIN_STRATA")
    data={"train":train,"dev":dev,"exam_blob":EXAM,"dovod_blob":DOVOD,"promotion":False}
    write(out/"data.json",data)
    print("R7C6_PREPARED",json.dumps({"train":len(train),"dev":len(dev),"exam":48,"data_sha256":g.digest(data),"dovod_provenance":DOVOD,"overlap":0,"promotion":False}),flush=True)
@dataclass
class Allocation:
    hard:int=50
    def run(self):return {"hard":self.hard}
def optimize(stage,stats,block,root):
    from evoagentx.optimizers.map_elites_optimizer import MapElitesOptimizer
    from evoagentx.optimizers.engine.registry import ParamRegistry
    low,high=stats
    # Laplace smoothing prevents starvation. Score is allocation proxy, not learning gain.
    e0=(low[0]+1)/(low[1]+2);e1=(high[0]+1)/(high[1]+2)
    target=100*e1/(e0+e1)
    def metric(cfg):
        return {"score":100-abs(cfg["hard"]-target),"hard":float(cfg["hard"])}
    p=Allocation();reg=ParamRegistry();reg.track(p,"hard",name="hard")
    opt=MapElitesOptimizer(registry=reg,evaluator=metric,program=p.run,
        search_space={"hard":[25,50,75]},feature_dimensions=["hard"],feature_ranges={"hard":(25.,75.)},
        feature_bins=3,n_iterations=18,exploration_ratio=.6,random_seed=SEED+block)
    cfg,res=opt.optimize()
    check(cfg["hard"] in (25,50,75),"OPTIMIZER")
    return cfg["hard"],{"stage":stage,"target":target,"selected":cfg["hard"],"error_counts":stats,"proxy":metric(cfg)["score"]}
def train(demi,prepared,out,arm):
    import torch
    sys.path.insert(0,str(demi))
    from janus_model.model import ByteTokenizer
    from janus_model.train_registry import load_checkpoint,save_checkpoint,sha256_file
    from tools.trinity_parental_cpu import student_answer
    check(arm in ("control","adaptive"),"ARM")
    data=read(prepared/"data.json");check(data["exam_blob"]==EXAM and not data["promotion"],"DATA")
    trainrows,dev=data["train"],data["dev"]
    check(not ({key(r) for r in trainrows}&{key(r) for r in dev}),"DEV_LEAK")
    torch.manual_seed(SEED);torch.set_num_threads(4)
    rng=random.Random(SEED);base=demi/"janus_model/checkpoints/promoted.pt"
    model,_=load_checkpoint(base,expected_sha256=school.CKPT)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01)
    out.mkdir(parents=True,exist_ok=False)
    def tensor(r):
        pre=ByteTokenizer.encode("Q:"+prompt(r)+"\nJANUS:",bos=True)
        ans=ByteTokenizer.encode(str(r[4]),eos=True);seq=pre+ans
        check(len(seq)-1<=128,"LENGTH")
        return torch.tensor([seq[:-1]],dtype=torch.long),torch.tensor([[-100]*(len(pre)-1)+ans],dtype=torch.long)
    def score(rows):
        model.eval();acc=[];losses=[]
        for r in rows:
            with torch.inference_mode():
                x,y=tensor(r);_,loss=model(x,y)
            losses.append(float(loss.item()))
            ans=student_answer(model,prompt(r));acc.append((r,ans==str(r[4]),ans))
        return acc,sum(losses)/len(losses)
    stage=0;history=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        if arm=="adaptive":evo.obtain(root);sys.path.insert(0,str(root))
        for block in range(BLOCKS):
            allocation=None;hard=50
            if arm=="adaptive":
                current=[r for r in dev if r[0]==stage];acc,_=score(current)
                stats=[(sum(not ok for r,ok,a in acc if difficulty(r)==d),sum(difficulty(r)==d for r,ok,a in acc)) for d in (0,1)]
                hard,allocation=optimize(stage,stats,block,root)
            losses=[];used=[0]*4
            for i in range(BLOCK):
                if arm=="control":
                    s=rng.randrange(4);pool=[r for r in trainrows if r[0]==s]
                else:
                    s=rng.randrange(stage) if stage>0 and rng.random()<.2 else stage
                    d=int(rng.randrange(100)<hard) if s==stage else rng.randrange(2)
                    pool=[r for r in trainrows if r[0]==s and difficulty(r)==d]
                r=rng.choice(pool);used[r[0]]+=1
                model.train();x,y=tensor(r);opt.zero_grad(set_to_none=True);_,loss=model(x,y)
                check(math.isfinite(loss.item()),"LOSS");loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.item()))
            acc,devloss=score([r for r in dev if arm=="control" or r[0]<=stage])
            # Matched post-block sample: dev loss minus train loss, not last-step comparison.
            _,trainloss=score(rng.sample([r for r in trainrows if arm=="control" or r[0]<=stage],min(12,len([r for r in trainrows if arm=="control" or r[0]<=stage]))))
            metrics={str(s):{"correct":sum(ok for r,ok,a in acc if r[0]==s),"n":sum(r[0]==s for r,ok,a in acc)} for s in range(4) if any(r[0]==s for r,ok,a in acc)}
            passed=all(v["correct"]>=math.ceil(.8*v["n"]) for v in metrics.values())
            event={"block":block,"stage":stage if arm=="adaptive" else None,"train_steps_by_stage":used,"allocation":allocation,
                "dev":metrics,"gate":passed,"mean_update_loss":sum(losses)/len(losses),"post_train_loss":trainloss,"dev_loss":devloss,"generalization_gap":devloss-trainloss}
            if arm=="adaptive" and passed and stage<3:stage+=1
            event["next_stage"]=stage if arm=="adaptive" else None
            history.append(event);print("R7C6_BLOCK",json.dumps({"arm":arm,**event}),flush=True)
    target=out/"shadow.pt";save_checkpoint(target,model,{"shadow_only":True,"arm":arm,"production_promotion":False})
    check(sha256_file(base)==school.CKPT and sha256_file(target)!=school.CKPT,"BASE")
    receipt={"arm":arm,"steps":BLOCK*BLOCKS,"data_sha256":g.digest(data),"shadow_sha256":sha256_file(target),"base_sha256":school.CKPT,"history":history,"promotion":False}
    write(out/"receipt.json",receipt);print("R7C6_TRAINED",json.dumps({"arm":arm,"steps":receipt["steps"],"shadow_sha256":receipt["shadow_sha256"],"last":history[-1]}),flush=True)
def evaluate(demi,prepared,control,adaptive,out):
    import torch
    sys.path.insert(0,str(demi))
    from janus_model.train_registry import load_checkpoint
    from tools.trinity_parental_cpu import student_answer
    torch.set_num_threads(4);data=read(prepared/"data.json");rows=exam()
    check(not ({key(r) for r in data["train"]+data["dev"]}&{key(r) for r in rows}),"EXAM_OVERLAP")
    models={"original":(demi/"janus_model/checkpoints/promoted.pt",school.CKPT)}
    for arm,folder in (("control",control),("adaptive",adaptive)):
        r=read(folder/"receipt.json")
        check(r["arm"]==arm and r["data_sha256"]==g.digest(data) and r["steps"]==BLOCK*BLOCKS and r["base_sha256"]==school.CKPT and not r["promotion"],"RECEIPT")
        models[arm]=(folder/"shadow.pt",r["shadow_sha256"])
    results={}
    for name,(path,pin) in models.items():
        model,_=load_checkpoint(path,expected_sha256=pin);model.eval()
        answers=[]
        for r in rows:
            torch.manual_seed(SEED);answers.append(student_answer(model,prompt(r)))
        ok=[a==str(r[4]) for a,r in zip(answers,rows)]
        results[name]={"exact":sum(ok),"correct":ok,"answers":answers,"by_stage":[sum(c for c,r in zip(ok,rows) if r[0]==s) for s in range(4)]}
    a,b,c=[results[k]["correct"] for k in ("adaptive","original","control")]
    loss=sum(x and not y for x,y in zip(b,a));gain=sum(y and not x for x,y in zip(b,a))
    status="INTERNAL_GAIN_ONLY" if sum(a)>max(sum(b),sum(c)) and loss==0 else "HOLD_NO_ADAPTIVE_SUPERIORITY"
    report={"status":status,"results":results,"new_vs_original":gain,"lost_vs_original":loss,"exam_blob":EXAM,"external_validation":False,"broader_regression":"NOT_RUN","promotion":False}
    out.mkdir(parents=True,exist_ok=False);write(out/"result.json",report)
    print("R7C6_EVALUATED",json.dumps({**report,"results":{k:{x:y for x,y in v.items() if x not in ("answers","correct")} for k,v in results.items()}}),flush=True)
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x).resolve() for x in sys.argv[2:]]
    if mode=="prepare":prepare(*args)
    elif mode=="train":train(*args[:3],sys.argv[5])
    elif mode=="evaluate":evaluate(*args)
    else:raise ValueError("MODE")
