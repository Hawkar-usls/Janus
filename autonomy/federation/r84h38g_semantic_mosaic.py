"""H38G: separate learned tiles, component masks and a conservative compatibility gate."""
import json,sys,re,time
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import r84h38f_semantic_crystal as f
import r84h38e_semantic_invariance as e
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
ck=f.ck;read=f.read;write=f.write;sha=f.sha
SIZES=(4,2,2,2);HOLD=16
# Explicit linguistic priors (not learned evidence): polarity marker and
# argument reversal marker. Relation tile masks these nuisance features.
NEG={"not","never"}
DIR1={"after","later","subsequent","following","behind"}
DIR0={"before","earlier","prior","ahead","preceding"}
def cues(t):
    w=t.split();n=sum(x in NEG for x in w)
    if n>1:return None,None
    neg=int(n==1)
    rev=bool(set(w)&DIR1 or "by" in w or "for" in w)
    forward=bool(set(w)&DIR0)
    if rev and forward:return neg,None
    return neg,int(rev)
def views(t):
    w=t.split()
    # Factor-specific input transforms, predeclared and identical across seeds.
    rel=" ".join(x for x in w if x not in NEG and x not in {"did","does","was","is","were","by","for"})
    return [rel,t,t,t]
class Mosaic(nn.Module):
    def __init__(self):
        super().__init__();self.tiles=nn.ModuleList([nn.Sequential(nn.Linear(e.B,16),nn.GELU(),nn.Linear(16,n)) for n in SIZES])
    def forward(self,x):return [m(v) for m,v in zip(self.tiles,x)]
def batch(ts):return [e.batch([views(t)[i] for t in ts]) for i in range(4)]
def probs(ms,ts):
    with torch.inference_mode():
        x=batch(ts);o=[m(x) for m in ms]
        return [torch.stack([r[i].softmax(-1) for r in o]).min(0).values.tolist() for i in range(4)]
def raw(p):
    js=[max(range(len(a)),key=lambda j:a[j]) for a in p]
    conf=[a[j] for a,j in zip(p,js)]
    margin=[a[j]-max(v for i,v in enumerate(a) if i!=j) for a,j in zip(p,js)]
    return js,conf,margin
def select(t,p,rule,compat):
    js,c,m=raw(p);j=4*js[0]+2*js[1]+js[2]
    if js[3]!=1 or any(v<r for v,r in zip(c,rule[0])) or min(m)<rule[1]:return None
    if compat:
        pol,di=cues(t)
        if pol is None or di is None or js[1]!=pol or js[2]!=di:return None
    return j
def train(bank,out):
    g.witness();torch.set_num_threads(2);out.mkdir();data=read(bank);tr=data["train"];dev=data["dev"];withheld=set(data["withheld"])
    ck(len({z["text"] for z in tr+dev})==len(tr+dev) and not any(z["y"] in withheld for z in tr+dev),"SPLIT")
    x=batch([z["text"] for z in tr]);y=torch.tensor([f.parts(z["y"]) for z in tr]);known=y[:,3]==1;ms=[];runs=[]
    for seed in (389,390):
        torch.manual_seed(seed);m=Mosaic();opts=[torch.optim.AdamW(tile.parameters(),lr=.008) for tile in m.tiles];start=time.process_time()
        for _ in range(240):
            for i,(tile,opt) in enumerate(zip(m.tiles,opts)):
                mask=known if i<3 else torch.ones_like(known)
                l=nn.functional.cross_entropy(tile(x[i][mask]),y[mask,i]);opt.zero_grad();l.backward();opt.step()
        p=out/f"mosaic-{seed}.safetensors";save_file(m.state_dict(),str(p));ms.append(m.eval());runs.append({"seed":seed,"sha":sha(p),"parameters":sum(v.numel() for v in m.parameters()),"cpu":time.process_time()-start})
    ps=probs(ms,[z["text"] for z in dev]);rules={}
    for mode in ("tiles","compatible"):
        choices=[]
        for a in (.4,.6,.8,.95,1.01):
            for b in (.5,.7,.9):
                for c in (.5,.7,.9):
                    for margin in (0,.1,.25):
                        r=((a,b,b,c),margin);pred=[select(z["text"],f.perrow(ps,i),r,mode=="compatible") for i,z in enumerate(dev)]
                        good=sum(p==z["y"] and p is not None for p,z in zip(pred,dev));bad=sum(p is not None and p!=z["y"] for p,z in zip(pred,dev))
                        if not bad:choices.append((good,r))
        good,r=max(choices);rules[mode]={"rule":r,"dev_correct":good}
    (out/"bank.json").write_bytes(bank.read_bytes());rec={"schema":"h38g.train.v1","bank_sha":sha(bank),"train_n":len(tr),"dev_n":len(dev),"withheld":sorted(withheld),"runs":runs,"rules":rules,"exam_read":False,"teacher_labels":False,"production":False}
    write(out/"receipt.json",rec);print("R84H38G_TRAIN",json.dumps(rec,sort_keys=True),flush=True)
def evaluate(trained,oldF,oldE,demi,exam,priorc,priord,priore,priorf,out):
    g.witness();torch.set_num_threads(2);out.mkdir();r=read(trained/"receipt.json");bank=read(trained/"bank.json");ck(sha(trained/"bank.json")==r["bank_sha"],"BANK")
    ms=[]
    for z in r["runs"]:
        p=trained/f"mosaic-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"WEIGHT");m=Mosaic();m.load_state_dict(load_file(str(p)));ms.append(m.eval())
    fr=read(oldF/"receipt.json");ck(sha(oldF/"bank.json")==fr["bank_sha"],"F_BANK");fm=[]
    for z in fr["runs"]["factor"]:
        p=oldF/f"factor-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"F_WEIGHT");m=f.Net("factor");m.load_state_dict(load_file(str(p)));fm.append(m.eval())
    er=read(oldE/"receipt.json");eb=read(oldE/"bank.json");ck(sha(oldE/"bank.json")==er["bank_sha"],"E_BANK");em=[]
    for z in er["runs"]:
        p=oldE/f"model-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"E_WEIGHT");m=e.Net();m.load_state_dict(load_file(str(p)));em.append(m.eval())
    di=e.Dictionary(e.cortex(demi),oldE/"cortex",eb["train"]);cache={}
    def scores(t):
        if t not in cache:
            p=f.perrow(probs(ms,[t]),0);fp=f.perrow(f.prob(fm,[t]),0);ep=e.probs(em,[t])[0];lex=di.search(t)
            cache[t]=(p,fp,e.pick(ep,lex,er["rules"]["agreement"],"agreement"))
        return cache[t]
    def parse(t,names,mode):
        base=d.parse(t)
        if mode=="baseline":return base
        if mode in ("h38e","stack") and base is not None:return base
        n=e.norm(t,names)
        if n is None:return None
        text,ns=n;p,fp,ej=scores(text)
        if mode=="h38f":j=f.select(fp,"factor",fr["rules"]["factor"]["threshold"])
        elif mode=="h38e":j=ej
        elif mode=="stack" and ej is not None:j=ej
        else:
            k="compatible" if mode=="stack" else mode
            j=select(text,p,r["rules"][k]["rule"],k=="compatible")
        return e.decode(j,ns) if j is not None else None
    ex=read(exam);oldD=read(priord)["cases"];oldEex=read(priore);oldFex=read(priorf)
    refs={z["text"] for z in bank["train"]+bank["dev"]+read(oldF/"bank.json")["train"]+read(oldF/"bank.json")["dev"]+eb["train"]+eb["dev"]}
    for old in (oldEex,oldFex):
        for z in old["rows"]:
            n=e.norm(z["text"],z["names"])
            if n:refs.add(n[0])
    def namesD(z):return sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")})
    for z in oldD:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,namesD(z))
            if n:refs.add(n[0])
    modes=("baseline","h38e","h38f","tiles","compatible","stack")
    details=[]
    for z in ex["rows"]:
        n=e.norm(z["text"],z["names"])
        ck(n is None or n[0] not in refs,"FRESH_OVERLAP")
        if z["y"]<16:ck(n is not None and e.decode(z["y"],n[1])==z["gold"],"GOLD")
        details.append({**z,"pred":{m:parse(z["text"],z["names"],m) for m in modes}})
    def metrics(rows,m,groups=()):
        s={"correct":0,"wrong":0,"unknown_false":0,"withheld_correct":0,"withheld_n":0}
        for z in rows:
            a=z["pred"][m];y=z["gold"];s["correct"]+=y is not None and a==y;s["wrong"]+=y is not None and a is not None and a!=y;s["unknown_false"]+=y is None and a is not None
            if z.get("y") in r["withheld"]:s["withheld_n"]+=1;s["withheld_correct"]+=a==y
        s["groups"]=sum(all(rows[i]["gold"] is not None and rows[i]["pred"][m]==rows[i]["gold"] for i in group) for group in groups)
        return s
    stats={m:metrics(details,m,ex["groups"]) for m in modes}
    # Disclosed H38F failure decomposition: raw components vs threshold.
    diagnostic={"known":0,"raw_all_correct":0,"correct_rejected":0,"raw_wrong":0,"wrong_accepted":0,"component_errors":[0,0,0,0],"withheld_raw_correct":0,"withheld_n":0}
    for z in oldFex["rows"]:
        if z["y"]==HOLD:continue
        n=e.norm(z["text"],z["names"]);ck(n is not None,"F_NORM");_,fp,_=scores(n[0]);j,_,_,parts=f.unpack(fp,"factor");a=f.select(fp,"factor",fr["rules"]["factor"]["threshold"]);y=z["y"]
        diagnostic["known"]+=1;diagnostic["raw_all_correct"]+=j==y;diagnostic["correct_rejected"]+=j==y and a is None;diagnostic["raw_wrong"]+=j!=y;diagnostic["wrong_accepted"]+=a is not None and a!=y
        diagnostic["component_errors"]=[v+int(x!=q) for v,x,q in zip(diagnostic["component_errors"],parts,f.parts(y))]
        if y in fr["withheld"]:diagnostic["withheld_n"]+=1;diagnostic["withheld_raw_correct"]+=j==y
    def solve(ss,q,ns,m):
        fs=[parse(t,ns,m) for t in ss];c=parse(q,ns,m)
        if c is None or any(a is None for a in fs):return "HOLD_UNPARSED"
        return g.solve({"schema":"janus.event.graph.v1","facts":[{"id":f"s{i}",**a} for i,a in enumerate(fs)],"claim":c})["status"]
    graphstats={m:{"correct":0,"false_certainty":0} for m in modes}
    for z in ex["graphs"]:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,z["names"]);ck(n is None or n[0] not in refs,"GRAPH_OVERLAP")
        for m in modes:
            a=solve(z["sentences"],z["claim"],z["names"],m);graphstats[m]["correct"]+=a==z["expected"];graphstats[m]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"]
    replay={m:{"correct":0,"false_certainty":0,"h38e_correct_lost":0} for m in ("baseline","h38e","stack")}
    for z in oldD:
        ns=namesD(z);base=solve(z["sentences"],z["claim"],ns,"h38e")
        for m in replay:
            a=solve(z["sentences"],z["claim"],ns,m);replay[m]["correct"]+=a==z["expected"];replay[m]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"];replay[m]["h38e_correct_lost"]+=base==z["expected"] and a!=z["expected"]
    ck(replay["baseline"]["correct"]==41 and replay["h38e"]["correct"]==46,"H38E_FROZEN_REPLAY")
    old=read(priorc)["cases"];ck(len(old)==76 and all(g.solve(z["request"])["status"]==z["expected"] for z in old),"H38C_REPLAY")
    report={"schema":"h38g.result.v1","exam_sha":sha(exam),"known_n":sum(z["gold"] is not None for z in details),"unknown_n":sum(z["gold"] is None for z in details),"groups_n":len(ex["groups"]),"graph_n":len(ex["graphs"]),"routes":stats,"graphs":graphstats,"h38f_diagnostic":diagnostic,"h38d_replay":replay,"h38c_replay":76,"cortex_queries":di.calls,"quality_gate":"HOLD","production":False,"scope":"Authored synthetic English clauses with supplied event IDs, separate learned tiles and explicit compatibility cues, frozen H38E/H38F artifacts, no external exam, no matched-data comparison against frozen F (different training banks), no deployment."}
    write(out/"result.json",report);write(out/"cases.json",details);print("R84H38G_RESULT",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":{"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
