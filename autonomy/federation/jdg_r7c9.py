"""R7C9: matched DOVOD number representations; no production writes."""
import hashlib, importlib.util, json, math, random, sys, urllib.request
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
EXAM="9c01c15a4749094d4da08cff42fb759310832a29"
DOVOD="5f7de85ea89ae3ffcee7c9cf13d6eb653e43ccd0"
ARMS=("byte","place","vector")
SEEDS=(91,92,93)
STEPS=2000
TASKS=("inc","dec","carry","borrow","compare")
def check(ok,why):
    if not ok:raise RuntimeError(why)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,o):Path(p).write_text(json.dumps(o,sort_keys=True,separators=(",",":"))+"\n")
def digest(o):return sha(json.dumps(o,sort_keys=True,separators=(",",":")).encode())
def trail(n,d):
    s=str(n);return len(s)-len(s.rstrip(str(d)))
def label(k,a,b):
    if k=="inc":return a+1
    if k=="dec":check(a>0,"DEC");return a-1
    if k=="carry":return trail(a,9)
    if k=="borrow":check(a>0,"BORROW");return trail(a,0)
    if k=="compare":return int(a>b)
    raise ValueError(k)
def exam():
    raw=(HERE/"R7C9_FROZEN_NUMBER_SENSE_EXAM.json").read_bytes();check(blob(raw)==EXAM,"EXAM_PIN")
    obj=json.loads(raw);rows=obj["rows"]
    check(len(rows)==278 and len({tuple(r[:4]) for r in rows})==278,"EXAM")
    check(all(y==label(k,a,b) and a in obj["groups"][c] and (k!="compare" or b in obj["groups"][c]) for c,k,a,b,y in rows),"LABEL")
    check(sum(map(len,obj["groups"].values()))==len(set(n for ns in obj["groups"].values() for n in ns)),"GROUP")
    return obj
def make_rows(ns,seed):
    rows=[];rng=random.Random(seed)
    for a in ns:
        for k in TASKS[:4]:
            if a==0 and k in ("dec","borrow"):continue
            rows.append([k,a,0,label(k,a,0)])
    # Matched paired comparison directions. Avoid constant-label accuracy illusion.
    pairs=set()
    for a in ns:
        b=rng.choice([n for n in ns if n!=a]);pairs.add(tuple(sorted((a,b))))
    for a,b in sorted(pairs):
        rows.extend([["compare",a,b,0],["compare",b,a,1]])
    return rows
def inputs(rows):return {n for k,a,b,y in rows for n in ((a,b) if k=="compare" else (a,))}
def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    with urllib.request.urlopen("https://raw.githubusercontent.com/Hawkar-usls/janus-meta-registry/main/DovoD.py",timeout=30) as f:raw=f.read(262145)
    check(blob(raw)==DOVOD,"SOURCE_PIN");(out/"dovod.py").write_bytes(raw)
    ex=exam();excluded=set(n for ns in ex["groups"].values() for n in ns)
    pool=[n for n in range(200) if n not in excluded];rng=random.Random(9100)
    boundary=[n for n in pool if n%10 in (0,9)];other=[n for n in pool if n%10 not in (0,9)]
    rng.shuffle(boundary);rng.shuffle(other);dv=sorted(boundary[:8]+other[:16]);tr=sorted(set(pool)-set(dv))
    train=make_rows(tr,9101);dev=make_rows(dv,9102)
    check(not inputs(train)&inputs(dev) and not (inputs(train)|inputs(dev))&excluded,"NUMBER_LEAK")
    # independent label audit via integer arithmetic, including carry-chain length.
    for k,a,b,y in train+dev:
        if k=="inc":check(y-a==1,"INC")
        elif k=="dec":check(a-y==1,"DEC")
        elif k=="compare":check((a-b>0)==bool(y),"CMP")
        elif k=="carry":
            j=0;v=a
            while v%10==9:j+=1;v//=10
            check(j==y,"CARRY")
        elif k=="borrow":
            j=0;v=a
            while v>0 and v%10==0:j+=1;v//=10
            check(j==y,"BORROW")
    data={"train":train,"dev":dev,"train_numbers":tr,"dev_numbers":dv,"exam_blob":EXAM,"source_blob":DOVOD,"promotion":False}
    write(out/"data.json",data)
    print("R7C9_PREPARED",json.dumps({"train_numbers":len(tr),"dev_numbers":len(dv),"train_rows":len(train),"dev_rows":len(dev),"exam":len(ex["rows"]),"input_overlap":0,"data_sha256":digest(data)}),flush=True)
def lib(prepared):
    import torch
    p=prepared/"dovod.py";check(blob(p.read_bytes())==DOVOD,"SOURCE_PIN")
    spec=importlib.util.spec_from_file_location("dovod_original",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return torch,m.HierarchicalTransformer
def count(m):return sum(p.numel() for p in m.parameters())
def make(L,arm):
    torch,D=L;m=D(vocab_size=258,n_embd=124,n_head=4,n_layer=2,block_size=128)
    if arm=="vector":
        # deterministic input-only decimal features; no answer/operation-dependent values.
        # Same trainable parameter count and output byte alphabet as other arms.
        def hook(module,args,out):
            ids=args[0];mask=(ids>=128)&(ids<178);z=(ids-128).clamp(0,49)
            d=(z%10).to(out.dtype)/9.;p=(z//10).to(out.dtype)/4.
            f=torch.stack((d,p,d*p,torch.sin(d*math.pi),torch.cos(d*math.pi),torch.sin(p*math.pi),torch.cos(p*math.pi),torch.ones_like(d)),dim=-1)
            extra=torch.zeros_like(out);extra[...,:8]=.1*f*mask.unsqueeze(-1)
            return out+extra
        m.wte.register_forward_hook(hook)
    return m
def number(n,arm):
    s=str(n);check(len(s)<=5 and n>=0,"NUMBER")
    if arm=="byte":return list(s.encode())
    return [128+10*(len(s)-1-i)+int(d) for i,d in enumerate(s)]
def prefix(r,arm):
    k,a,b,y=r
    op={"inc":"I","dec":"D","carry":"C","borrow":"B","compare":"G"}[k]
    return [256,ord(op),ord(":")]+number(a,arm)+([ord(">")]+number(b,arm) if k=="compare" else [])+[ord("=")]
def operations(L,m,arm):
    torch,_=L
    def loss(r):
        pre=prefix(r,arm);ans=list(str(r[3]).encode())+[257];seq=pre+ans
        x=torch.tensor(seq[:-1]);y=torch.tensor([-100]*(len(pre)-1)+ans)
        return torch.nn.functional.cross_entropy(m(x),y)
    @torch.no_grad()
    def answer(r):
        ids=prefix(r,arm);out=[]
        for _ in range(7):
            x=int(m(torch.tensor(ids))[-1].argmax())
            if x==257:return bytes(out).decode() if out and all(48<=i<=57 for i in out) else None
            if not 48<=x<=57:return None
            out.append(x);ids.append(x)
        return None
    @torch.no_grad()
    def score(rows,with_loss=True):
        m.eval();a=[answer(r) for r in rows];ok=[v==str(r[3]) for v,r in zip(a,rows)]
        return {"n":len(rows),"exact":sum(ok),"loss":sum(float(loss(r)) for r in rows)/len(rows) if with_loss else None,"answers":a,"ok":ok}
    return loss,score
def train(prepared,out,arm):
    check(arm in ARMS,"ARM");L=lib(prepared);torch,_=L;torch.set_num_threads(4);data=read(prepared/"data.json")
    check(data["exam_blob"]==EXAM and not data["promotion"] and not inputs(data["train"])&inputs(data["dev"]),"DATA")
    out.mkdir(parents=True,exist_ok=False)
    for seed in SEEDS:
        torch.manual_seed(seed);rng=random.Random(seed);m=make(L,arm);params=count(m);check(params==452352,"PARAMS")
        loss,score=operations(L,m,arm);opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=.01)
        def state():return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        def rank(s):return (s["exact"],-s["loss"])
        best=score(data["dev"]);best_state=state();best_step=0;history=[]
        pools={k:[r for r in data["train"] if r[0]==k] for k in TASKS}
        for step in range(1,STEPS+1):
            r=rng.choice(pools[TASKS[(step-1)%len(TASKS)]]);m.train();opt.zero_grad(set_to_none=True);l=loss(r);check(math.isfinite(l.item()),"LOSS");l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
            if step%250==0:
                s=score(data["dev"])
                if rank(s)>rank(best):best=s;best_state=state();best_step=step
                e={"arm":arm,"seed":seed,"steps":step,"dev_exact":s["exact"],"dev_n":s["n"],"dev_loss":s["loss"],"best_step":best_step}
                history.append(e);print("R7C9_BLOCK",json.dumps(e),flush=True)
        folder=out/str(seed);folder.mkdir();p=folder/"shadow.pt";torch.save(best_state,p)
        r={"arm":arm,"seed":seed,"steps":STEPS,"selected_step":best_step,"dev_exact":best["exact"],"params":params,"data_sha256":digest(data),"shadow_sha256":sha(p.read_bytes()),"history":history,"promotion":False}
        write(folder/"receipt.json",r);print("R7C9_TRAINED",json.dumps({k:v for k,v in r.items() if k!="history"}),flush=True)
def probe(L,m,arm,train_numbers,groups):
    torch,_=L;m.eval()
    # Fit a ridge linear readout only on training input numbers. Not used for solving.
    cache=[]
    h=m.ln_f.register_forward_hook(lambda mod,inp,out:cache.append(out[-1].detach().cpu()))
    @torch.no_grad()
    def features(ns):
        x=[]
        for n in ns:
            cache.clear();m(torch.tensor(prefix(["inc",n,0,0],arm)));check(len(cache)==1,"PROBE")
            x.append(cache[0])
        return torch.stack(x).double()
    try:
        X=features(train_numbers);mean=X.mean(0);X=X-mean;Y=torch.tensor(train_numbers,dtype=torch.double);mu=Y.mean()
        w=torch.linalg.solve(X.T@X+torch.eye(X.shape[1],dtype=torch.double)*1.,X.T@(Y-mu))
        results={}
        for name,ns in groups.items():
            y=torch.tensor(ns,dtype=torch.double);pred=(features(ns)-mean)@w+mu
            correct=sum(float(pred[i])<float(pred[j]) for i in range(len(ns)) for j in range(i+1,len(ns)) if ns[i]<ns[j])
            total=sum(ns[i]<ns[j] for i in range(len(ns)) for j in range(i+1,len(ns)))
            results[name]={"mae":float((pred-y).abs().mean()),"order_correct":correct,"order_n":total}
        return results
    finally:h.remove()
def evaluate(prepared,out,*folders):
    L=lib(prepared);torch,_=L;torch.set_num_threads(4);data=read(prepared/"data.json");ex=exam()
    excluded=set(n for ns in ex["groups"].values() for n in ns)
    check(not (inputs(data["train"])|inputs(data["dev"]))&excluded,"EXAM_LEAK")
    majority={k:str(Counter(r[3] for r in data["train"] if r[0]==k).most_common(1)[0][0]) for k in TASKS}
    metrics=[]
    def summarize(answers):
        res={}
        for c in ex["groups"]:
            for k in TASKS:
                idx=[i for i,r in enumerate(ex["rows"]) if r[0]==c and r[1]==k]
                res[c+"/"+k]={"exact":sum(answers[i]==str(ex["rows"][i][4]) for i in idx),"n":len(idx)}
            idx=[i for i,r in enumerate(ex["rows"]) if r[0]==c and ((r[1]=="inc" and r[2]%10==9) or (r[1]=="dec" and r[2]%10==0))]
            res[c+"/boundary"]={"exact":sum(answers[i]==str(ex["rows"][i][4]) for i in idx),"n":len(idx)}
            pairs=[(i,i+1) for i,r in enumerate(ex["rows"]) if r[0]==c and r[1]=="compare" and r[2]<r[3]]
            res[c+"/paired_compare"]={"exact":sum(answers[i]=="0" and answers[j]=="1" for i,j in pairs),"n":len(pairs)}
        return res
    baseline=summarize([majority[k] for c,k,a,b,y in ex["rows"]])
    print("R7C9_BASELINE",json.dumps({"constant_by_task":majority,"metrics":baseline}),flush=True)
    for arm,folder in zip(ARMS,folders):
        for seed in SEEDS:
            r=read(folder/str(seed)/"receipt.json");p=folder/str(seed)/"shadow.pt"
            check(r["arm"]==arm and r["seed"]==seed and r["steps"]==STEPS and r["data_sha256"]==digest(data) and not r["promotion"] and sha(p.read_bytes())==r["shadow_sha256"],"RECEIPT")
            m=make(L,arm);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True);m.eval()
            with torch.no_grad():
                a=m(torch.tensor([256,ord("I"),ord(":"),*number(9,arm)]));b=m(torch.tensor([256,ord("I"),ord(":"),*number(8,arm)]))
                check(torch.allclose(a[:-1],b[:-1],atol=1e-5,rtol=1e-5),"CAUSAL")
            _,score=operations(L,m,arm);s=score([r[1:] for r in ex["rows"]],False)
            result={"arm":arm,"seed":seed,"params":count(m),"selected_step":r["selected_step"],"metrics":summarize(s["answers"]),"probe":probe(L,m,arm,data["train_numbers"],ex["groups"]),"answers":s["answers"]}
            metrics.append(result);print("R7C9_MODEL",json.dumps({k:v for k,v in result.items() if k!="answers"}),flush=True)
    out.mkdir(parents=True,exist_ok=False);write(out/"result.json",{"status":"INTERNAL_ONLY","exam_blob":EXAM,"baseline":baseline,"results":metrics,"promotion":False,"external_validation":False,"broad_regression":"NOT_RUN"})
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x).resolve() for x in sys.argv[2:]]
    if mode=="prepare":prepare(*args)
    elif mode=="train":train(*args[:2],sys.argv[4])
    elif mode=="evaluate":evaluate(*args)
    else:raise ValueError("MODE")
