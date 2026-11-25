"""R7C7: anti-overfit trial; shadow weights, three arms and three fixed seeds."""
import json, math, random, sys, tempfile, urllib.request
from pathlib import Path
import jdg_r7c3 as evo
import jdg_r7c4 as school
import jdg_r7c6 as foundation
import jfp_gateway as g

HERE=Path(__file__).resolve().parent
EXAM="bec690ffa67c085fa997a15ed6558e847852efd9"
OLD6="393a6490bba4f0e4f41cf8484729407ceaeb91d7"
OLD5="84c033c7bf38824249255cdc90f44db2a475f1af"
SEEDS=(77,78,79)
BLOCK=100
MAX_BLOCKS=16
PATIENCE=3
ARMS=("fixed","early","enriched")
def check(ok,why):evo.check(ok,why)
def read(p):return json.loads(Path(p).read_text())
def write(p,o):Path(p).write_bytes(g.canonical(o))
def pair(a,b):return tuple(sorted((a,b)))
def pinned(path,pin):
    raw=Path(path).read_bytes();check(evo.blob(raw)==pin,"PIN:"+str(path));return json.loads(raw)
def frozen():
    x=pinned(HERE/"R7C7_FROZEN_ANTI_OVERFIT_EXAM.json",EXAM)
    a=x["cohorts"]["digit_pairs"];b=x["cohorts"]["transfer_pairs"]
    check(len(a)==12 and len(b)==24 and all(v==x+y for x,y,v in a+b),"EXAM")
    check(len({pair(x,y) for x,y,v in a+b})==36,"EXAM_UNIQUE")
    return x
def standard(a,b):return f"Compute {a} + {b}. Reply with the integer only."
def alt(a,b):return f"What is {b} plus {a}? Answer with one integer."
def templates(a,b):
    w=("zero","one","two","three","four","five","six","seven","eight","nine")
    return [standard(a,b),alt(a,b),f"Add {w[a]} and {w[b]}. Reply with the integer only.",f"{a}+{b}=? Reply with the integer only."]
def task(q,y):return [q,str(y)]
def main_tasks(rows):
    return [task(q,y) for a,b,y in rows for q in (standard(a,b),alt(a,b))]
def prepare(demi,out):
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(demi))
    from tools import trinity_r14_verified_oracle as oracle
    check(evo.blob(Path(oracle.__file__).read_bytes())=="7792bae07705e040504052bcf673342b661858f7","ORACLE")
    with urllib.request.urlopen("https://raw.githubusercontent.com/Hawkar-usls/janus-meta-registry/main/DovoD.py",timeout=30) as f:raw=f.read(262145)
    check(evo.blob(raw)==foundation.DOVOD,"DOVOD_PIN")
    ex=frozen();old6=pinned(HERE/"R7C6_FROZEN_FOUNDATION_EXAM.json",OLD6)
    old5=pinned(HERE/"R7C5_FROZEN_INTERNAL_EXAM.json",OLD5)
    excluded={pair(a,b) for a,b,y in ex["cohorts"]["digit_pairs"]+ex["cohorts"]["transfer_pairs"]}
    excluded|={pair(a,b) for s,k,a,b,y in old6["cases"] if k=="add"}
    excluded|={pair(a,b) for a,b,y,c in old5["cases"]}
    rng=random.Random(7700);train=[];dev=[]
    for hard in (False,True):
        p=[[a,b,a+b] for a in range(10) for b in range(a,10) if (a+b>=10)==hard and pair(a,b) not in excluded]
        rng.shuffle(p);check(len(p)>5,"POOL")
        dev+=p[:5];train+=p[5:]
    tr={pair(a,b) for a,b,y in train};dv={pair(a,b) for a,b,y in dev}
    check(not tr&dv and not (tr|dv)&excluded,"LEAK")
    for a,b,y in train+dev:
        q=f"Calculate {a} + {b}. Answer with one integer."
        check(oracle.verify(q,str(y))["verdict"]=="APPROVE" and oracle.verify(q,str(y+1))["verdict"]=="VETO_WRONG","LABEL")
    # Intermediate tasks use only training operand pairs, never dev or exam pairs.
    # Labels are checked using iterative counting and independent divmod.
    aux=[]
    for a,b,y in train:
        total=a
        for _ in range(b):total+=1
        tens,ones=divmod(total,10)
        check(total==y and 10*tens+ones==y,"DECOMPOSITION")
        aux.extend([task(f"Add {a} and {b}. How many tens in the sum? Reply with the integer only.",tens),
                    task(f"Add {a} and {b}. How many ones in the sum? Reply with the integer only.",ones),
                    task(f"Does {a} + {b} need a carry? Reply 1 for yes or 0 for no.",int(tens>0))])
    data={"train":train,"dev":dev,"aux":aux,"exam_blob":EXAM,"dovod_blob":foundation.DOVOD,"promotion":False}
    write(out/"data.json",data)
    print("R7C7_PREPARED",json.dumps({"train_pairs":len(train),"dev_pairs":len(dev),"dev_prompts":2*len(dev),"aux":len(aux),"final_prompts":48,"overlap":0,"data_sha256":g.digest(data),"promotion":False}),flush=True)
def train(demi,prepared,out,arm):
    import torch
    sys.path.insert(0,str(demi))
    from janus_model.model import ByteTokenizer
    from janus_model.train_registry import load_checkpoint,save_checkpoint,sha256_file
    from tools.trinity_parental_cpu import student_answer
    check(arm in ARMS,"ARM");data=read(prepared/"data.json");check(data["exam_blob"]==EXAM and not data["promotion"],"DATA")
    check(not {pair(a,b) for a,b,y in data["train"]}&{pair(a,b) for a,b,y in data["dev"]},"DEV_LEAK")
    torch.set_num_threads(4);base=demi/"janus_model/checkpoints/promoted.pt"
    out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);evo.obtain(root);sys.path.insert(0,str(root))
        for seed in SEEDS:
            torch.manual_seed(seed);rng=random.Random(seed)
            model,_=load_checkpoint(base,expected_sha256=school.CKPT)
            opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01)
            folder=out/str(seed);folder.mkdir()
            def tensor(q,y):
                pre=ByteTokenizer.encode("Q:"+q+"\nJANUS:",bos=True);ans=ByteTokenizer.encode(y,eos=True);seq=pre+ans
                check(len(seq)-1<=128,"LENGTH")
                return torch.tensor([seq[:-1]],dtype=torch.long),torch.tensor([[-100]*(len(pre)-1)+ans],dtype=torch.long)
            def score(rows):
                model.eval();results=[];losses=[]
                for a,b,y in rows:
                    for q in (standard(a,b),alt(a,b)):
                        with torch.inference_mode():
                            x,t=tensor(q,str(y));_,loss=model(x,t)
                            prediction=student_answer(model,q)
                        losses.append(float(loss.item()));results.append([a,b,q,str(y),prediction,prediction==str(y)])
                return {"correct":sum(r[5] for r in results),"n":len(results),"loss":sum(losses)/len(losses),"rows":results}
            def state():return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            def rank(m):return (m["correct"],-m["loss"])
            dev=score(data["dev"]);tr=score(data["train"]);best=dev;best_step=0;best_state=state()
            stale=0;worse=0;history=[];stopped="budget"
            for block in range(MAX_BLOCKS):
                counts=[(sum(not r[5] for r in dev["rows"] if (r[0]+r[1]>=10)==bool(d)),sum((r[0]+r[1]>=10)==bool(d) for r in dev["rows"])) for d in (0,1)]
                hard,allocation=foundation.optimize(0,counts,block+seed*MAX_BLOCKS,root)
                losses=[];used={"direct":0,"variant":0,"intermediate":0}
                for step in range(BLOCK):
                    d=rng.randrange(100)<hard
                    pool=[r for r in data["train"] if (r[0]+r[1]>=10)==d]
                    a,b,y=rng.choice(pool)
                    if arm=="enriched" and step%4==0:
                        q,target=rng.choice(data["aux"]);used["intermediate"]+=1
                    elif arm=="enriched" and step%4==1:
                        q=rng.choice(templates(a,b)[1:]);target=str(y);used["variant"]+=1
                    else:q=standard(a,b);target=str(y);used["direct"]+=1
                    model.train();x,t=tensor(q,target);opt.zero_grad(set_to_none=True);_,loss=model(x,t)
                    check(math.isfinite(loss.item()),"LOSS");loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.item()))
                dev=score(data["dev"]);tr=score(data["train"]);steps=(block+1)*BLOCK
                improved=rank(dev)>rank(best)
                if improved:best=dev;best_step=steps;best_state=state();stale=0;worse=0
                else:
                    stale+=1
                    worse=worse+1 if dev["loss"]>best["loss"]+.1 and dev["correct"]<=best["correct"] else 0
                entry={"arm":arm,"seed":seed,"steps":steps,"dev_exact":dev["correct"],"dev_n":dev["n"],
                    "dev_loss":dev["loss"],"train_loss":tr["loss"],"gap":dev["loss"]-tr["loss"],
                    "best_step":best_step,"best_dev_exact":best["correct"],"allocation":allocation,"used":used}
                history.append(entry);print("R7C7_BLOCK",json.dumps(entry),flush=True)
                if arm!="fixed" and (stale>=PATIENCE or worse>=2):
                    stopped="generalization_worse" if worse>=2 else "no_dev_improvement";break
            actual=steps;final_dev=dev
            if arm!="fixed":model.load_state_dict(best_state)
            selected=actual if arm=="fixed" else best_step
            selected_dev=score(data["dev"])
            target=folder/"shadow.pt";save_checkpoint(target,model,{"shadow_only":True,"arm":arm,"seed":seed,"selected_step":selected,"production_promotion":False})
            check(sha256_file(base)==school.CKPT,"BASE")
            receipt={"arm":arm,"seed":seed,"steps":actual,"selected_step":selected,"stop":stopped,"best_step":best_step,
                "best_dev_exact":best["correct"],"final_dev_exact":final_dev["correct"],"selected_dev_exact":selected_dev["correct"],
                "base_sha256":school.CKPT,"shadow_sha256":sha256_file(target),"data_sha256":g.digest(data),
                "history":history,"promotion":False}
            write(folder/"receipt.json",receipt)
            print("R7C7_TRAINED",json.dumps({k:v for k,v in receipt.items() if k!="history"}),flush=True)
def evaluate(demi,prepared,fixed,early,enriched,out):
    import torch
    sys.path.insert(0,str(demi))
    from janus_model.train_registry import load_checkpoint
    from tools.trinity_parental_cpu import student_answer
    torch.set_num_threads(4);data=read(prepared/"data.json");ex=frozen()
    allpairs={pair(a,b) for a,b,y in ex["cohorts"]["digit_pairs"]+ex["cohorts"]["transfer_pairs"]}
    check(not allpairs&{pair(a,b) for a,b,y in data["train"]+data["dev"]},"EXAM_LEAK")
    old6=pinned(HERE/"R7C6_FROZEN_FOUNDATION_EXAM.json",OLD6);old5=pinned(HERE/"R7C5_FROZEN_INTERNAL_EXAM.json",OLD5)
    cohorts={"digits":main_tasks(ex["cohorts"]["digit_pairs"]),
             "transfer":[task(standard(a,b),y) for a,b,y in ex["cohorts"]["transfer_pairs"]],
             "regression_r7c6":[task(foundation.prompt(r),r[4]) for r in old6["cases"]],
             "regression_r7c5":[task(standard(a,b),y) for a,b,y,c in old5["cases"]]}
    models=[("original",0,demi/"janus_model/checkpoints/promoted.pt",school.CKPT,None)]
    for arm,folder in (("fixed",fixed),("early",early),("enriched",enriched)):
        for seed in SEEDS:
            r=read(folder/str(seed)/"receipt.json")
            check(r["arm"]==arm and r["seed"]==seed and r["data_sha256"]==g.digest(data) and r["base_sha256"]==school.CKPT and not r["promotion"],"RECEIPT")
            models.append((arm,seed,folder/str(seed)/"shadow.pt",r["shadow_sha256"],r))
    results=[]
    for arm,seed,path,pin,receipt in models:
        model,_=load_checkpoint(path,expected_sha256=pin);model.eval();metrics={}
        for name,rows in cohorts.items():
            answers=[]
            for q,y in rows:
                torch.manual_seed(77)
                a=student_answer(model,q);answers.append({"prompt":q,"expected":y,"answer":a,"correct":a==y})
            metrics[name]={"n":len(rows),"exact":sum(r["correct"] for r in answers),"answers":answers}
        rec={"arm":arm,"seed":seed,"selected_step":receipt["selected_step"] if receipt else 0,"metrics":metrics}
        results.append(rec)
        print("R7C7_MODEL",json.dumps({**rec,"metrics":{k:{x:y for x,y in v.items() if x!="answers"} for k,v in metrics.items()}}),flush=True)
    out.mkdir(parents=True,exist_ok=False)
    report={"status":"INTERNAL_MEASUREMENT_ONLY","exam_blob":EXAM,"results":results,"external_validation":False,"promotion":False}
    write(out/"result.json",report)
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x).resolve() for x in sys.argv[2:]]
    if mode=="prepare":prepare(*args)
    elif mode=="train":train(*args[:3],sys.argv[5])
    elif mode=="evaluate":evaluate(*args)
    else:raise ValueError("MODE")
