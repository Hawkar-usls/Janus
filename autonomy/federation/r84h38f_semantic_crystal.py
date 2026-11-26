"""H38F factorized semantic crystal, matched joint comparator, research only."""
import sys,json,time,hashlib,re
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import r84h38e_semantic_invariance as e
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
MODES=("joint","factor")
HOLD=16
def ck(x,s):g.ck(x,s)
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,sort_keys=True)+"\n")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parts(y):return (y//4,(y%4)//2,y%2,1) if y<16 else (0,0,0,0)
class Net(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.body=nn.Sequential(nn.Linear(e.B,64),nn.GELU())
        self.heads=nn.ModuleList([nn.Linear(64,n) for n in ([17] if mode=="joint" else [4,2,2,2])])
    def forward(self,x):
        z=self.body(x);return [h(z) for h in self.heads]
def loss(m,x,ys):
    out=m(x)
    if m.mode=="joint":return nn.functional.cross_entropy(out[0],ys)
    p=torch.tensor([parts(int(y)) for y in ys]);known=ys<16
    return nn.functional.cross_entropy(out[3],p[:,3])+sum(nn.functional.cross_entropy(o[known],p[known,i]) for i,o in enumerate(out[:3]))
def prob(ms,texts):
    with torch.inference_mode():
        xs=e.batch(texts);raw=[m(xs) for m in ms]
        return [torch.stack([r[i].softmax(-1) for r in raw]).min(0).values.tolist() for i in range(len(raw[0]))]
def unpack(p,mode):
    js=[max(range(len(a)),key=lambda i:a[i]) for a in p]
    if mode=="joint":return js[0],p[0][js[0]],p[0][js[0]]-max(v for i,v in enumerate(p[0]) if i!=js[0]),None
    label=4*js[0]+2*js[1]+js[2] if js[3]==1 else HOLD
    conf=min(a[j] for a,j in zip(p,js));margin=min(a[j]-max(v for i,v in enumerate(a) if i!=j) for a,j in zip(p,js))
    return label,conf,margin,js
def select(p,mode,rule):
    j,c,m,_=unpack(p,mode);return j if j!=HOLD and c>=rule[0] and m>=rule[1] else None
def perrow(ps,i):return [p[i] for p in ps]
def train(bank,out):
    g.witness();torch.set_num_threads(2);out.mkdir();data=read(bank);tr=data["train"];dev=data["dev"]
    ck(len({z["text"] for z in tr+dev})==len(tr+dev),"DISJOINT")
    withheld=set(data["withheld"]);ck(not any(z["y"] in withheld for z in tr+dev),"WITHHELD")
    x=e.batch([z["text"] for z in tr]);y=torch.tensor([z["y"] for z in tr]);runs={};rules={}
    for mode in MODES:
        ms=[];runs[mode]=[]
        for seed in (387,388):
            torch.manual_seed(seed);m=Net(mode);opt=torch.optim.AdamW(m.parameters(),lr=.006);start=time.process_time()
            for _ in range(240):
                l=loss(m,x,y);opt.zero_grad();l.backward();opt.step()
            p=out/f"{mode}-{seed}.safetensors";save_file(m.state_dict(),str(p));ms.append(m.eval());runs[mode].append({"seed":seed,"sha":sha(p),"parameters":sum(v.numel() for v in m.parameters()),"cpu":time.process_time()-start})
        ps=prob(ms,[z["text"] for z in dev]);choices=[]
        for c in (.35,.5,.65,.8,.9,.95,1.01):
            for margin in (0,.05,.1,.2,.4):
                r=(c,margin);pred=[select(perrow(ps,i),mode,r) for i in range(len(dev))]
                good=sum(a==z["y"] and a is not None for a,z in zip(pred,dev));bad=sum(a is not None and a!=z["y"] for a,z in zip(pred,dev))
                if not bad:choices.append((good,r))
        good,r=max(choices);rules[mode]={"threshold":r,"dev_correct":good}
    (out/"bank.json").write_bytes(bank.read_bytes())
    rec={"schema":"h38f.train.v1","bank_sha":sha(bank),"train_n":len(tr),"dev_n":len(dev),"withheld":sorted(withheld),"rules":rules,"runs":runs,"exam_read":False,"teacher_labels":False,"production":False}
    write(out/"receipt.json",rec);print("R84H38F_TRAIN",json.dumps(rec,sort_keys=True),flush=True)
def evaluate(trained,exam,oldc,oldd,olde,out):
    g.witness();torch.set_num_threads(2);out.mkdir();rec=read(trained/"receipt.json");bank=read(trained/"bank.json");ck(sha(trained/"bank.json")==rec["bank_sha"],"BANK")
    nets={}
    for mode in MODES:
        nets[mode]=[]
        for z in rec["runs"][mode]:
            p=trained/f"{mode}-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"WEIGHT");m=Net(mode);m.load_state_dict(load_file(str(p)));nets[mode].append(m.eval())
    ex=read(exam);refs={z["text"] for z in bank["train"]+bank["dev"]}
    # Exact normalized overlap check against previous exams and current training.
    old=read(oldd)["cases"];oldE=read(olde)
    for z in oldE["rows"]:
        a=e.norm(z["text"],z["names"])
        if a:refs.add(a[0])
    for z in old:
        names=sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")})
        for t in z["sentences"]+[z["claim"]]:
            a=e.norm(t,names)
            if a:refs.add(a[0])
    cache={}
    def pred(t,names,mode,raw=False):
        z=e.norm(t,names)
        if z is None:return (None,None)
        text,ns=z;k=(text,mode)
        if k not in cache:
            p=perrow(prob(nets[mode],[text]),0);cache[k]=(select(p,mode,rec["rules"][mode]["threshold"]),unpack(p,mode))
        j,diagnostic=cache[k]
        return (e.decode(j,ns) if j is not None else None,diagnostic)
    def parse(t,names,mode):
        base=d.parse(t)
        if mode=="baseline":return base
        if mode=="guarded" and base is not None:return base
        return pred(t,names,"factor" if mode=="guarded" else mode)[0]
    modes=("baseline","joint","factor","guarded")
    details=[]
    for z in ex["rows"]:
        n=e.norm(z["text"],z["names"]);ck(n is None or n[0] not in refs,"OVERLAP");ck(z["y"] in rec["withheld"] or z["y"]==HOLD or z["y"] in set(v["y"] for v in bank["train"]),"LABEL")
        if z["y"]<16:ck(n is not None and e.decode(z["y"],n[1])==z["gold"],"GOLD_DIRECTION")
        details.append({**z,"pred":{m:parse(z["text"],z["names"],m) for m in modes},"diagnostics":{m:pred(z["text"],z["names"],m)[1] for m in MODES}})
    stats={}
    for mode in modes:
        stats[mode]={"correct":0,"wrong":0,"unknown_false":0,"withheld_correct":0,"withheld_n":0,"groups":0,"component_raw_correct":[0,0,0,0],"component_raw_n":[0,0,0,0]}
        s=stats[mode]
        for z in details:
            a=z["pred"][mode];gold=z["gold"];y=z["y"]
            s["correct"]+=gold is not None and a==gold;s["wrong"]+=gold is not None and a is not None and a!=gold;s["unknown_false"]+=gold is None and a is not None
            if y in rec["withheld"]:s["withheld_n"]+=1;s["withheld_correct"]+=a==gold
            if mode in MODES and z["diagnostics"][mode] is not None:
                j,_,_,raw=z["diagnostics"][mode];raw=raw if raw is not None else parts(j);target=parts(y)
                for i in range(4):
                    if i==3 or y!=HOLD:s["component_raw_n"][i]+=1;s["component_raw_correct"][i]+=raw[i]==target[i]
        s["groups"]=sum(all(details[i]["gold"] is not None and details[i]["pred"][mode]==details[i]["gold"] and details[i]["gold"]==details[group[0]]["gold"] for i in group) for group in ex["groups"])
    def solve(s,q,names,m):
        fs=[parse(t,names,m) for t in s];c=parse(q,names,m)
        if c is None or any(a is None for a in fs):return "HOLD_UNPARSED"
        return g.solve({"schema":"janus.event.graph.v1","facts":[{"id":f"s{i}",**a} for i,a in enumerate(fs)],"claim":c})["status"]
    graphstats={m:{"correct":0,"false_certainty":0} for m in modes}
    for z in ex["graphs"]:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,z["names"]);ck(n is None or n[0] not in refs,"GRAPH_OVERLAP")
        for m in modes:
            a=solve(z["sentences"],z["claim"],z["names"],m);graphstats[m]["correct"]+=a==z["expected"];graphstats[m]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"]
    replay={m:{"correct":0,"false_certainty":0,"baseline_lost":0} for m in ("baseline","guarded")}
    for z in old:
        names=sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")});base=solve(z["sentences"],z["claim"],names,"baseline")
        for m in replay:
            a=solve(z["sentences"],z["claim"],names,m);replay[m]["correct"]+=a==z["expected"];replay[m]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"];replay[m]["baseline_lost"]+=base==z["expected"] and a!=z["expected"]
    er={}
    for m in modes:er[m]={"correct":sum(z["gold"] is not None and parse(z["text"],z["names"],m)==z["gold"] for z in oldE["rows"]),"false":sum((a:=parse(z["text"],z["names"],m)) is not None and a!=z["gold"] for z in oldE["rows"])}
    prior=read(oldc)["cases"];ck(len(prior)==76 and all(g.solve(z["request"])["status"]==z["expected"] for z in prior),"H38C_REPLAY");ck(replay["baseline"]["correct"]==41,"H38D_REPLAY")
    report={"schema":"h38f.result.v1","exam_sha":sha(exam),"known_n":sum(z["gold"] is not None for z in details),"unknown_n":sum(z["gold"] is None for z in details),"groups_n":len(ex["groups"]),"graph_n":len(ex["graphs"]),"routes":stats,"graphs":graphstats,"h38d_replay":replay,"h38e_disclosed":er,"h38c_replay":76,"quality_gate":"HOLD","production":False,"scope":"Same-author supplied event inventory; matched train data, features, seeds, steps; joint and factorized heads differ in parameter count; held-out joint labels absent from train/dev; historical H38E weights not loaded, not a preservation proof; no independent external exam, no deployment."}
    write(out/"result.json",report);write(out/"cases.json",details);print("R84H38F_RESULT",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":{"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
