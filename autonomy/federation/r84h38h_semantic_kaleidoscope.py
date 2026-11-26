"""H38H: relation resonance using reversible feature views and provenance-gated mosaic.
Not a recovered 12D pyramid; only source-bound, bounded adaptations.
"""
import sys,json,re,time,math,hashlib
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import r84h38g_semantic_mosaic as m
import r84h38f_semantic_crystal as f
import r84h38e_semantic_invariance as e
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
ck=f.ck;read=f.read;write=f.write;sha=f.sha
MODES=("direct","mosaic","mirror","source")
PINS={
"data/JANUS-DEEP-ANALYSIS-5D-SPIRAL-ORIGIN-PRIME-ORACLE-PROTOCOL-2026-08-21-v2.1.json":"b4922c58fde2254fa625f80d5e0de07d9aec3293",
"data/JANUS-JSON-FORWARD-REVERSE-MIRROR-PROTOCOL-2026-08-21-v1.0.json":"5d8df65fb43bd2dc176d809e3211c469c2ffaa0b",
"data/JANUS-DEEP-ANALYSIS-5D-RECURSIVE-TRANCEPTION-HRAIN-INAIHR-PROTOCOL-2026-08-21-v2.0.json":"87f97a8ff52162e3d3f5645713930ab7e774f2a7",
"data/JANUS-P-VS-NP-R5-E9-THREE-TRANSLATE-TILING-NORMAL-FORM-2026-09-24-v1.0.json":"74d6af8ecb07da97da9ea27035d9236236d2c68c",
"data/JANUS-WEDJAT-FULL-EYE-COMPOSITION-CONTROL-RESULT-2026-08-15-v0.6.json":"21d83efba94d9fe10b9c3bad5a934e7c4dd3cccc",
}
def verify_sources(reg,cos):
    for p,h in PINS.items():ck(e.blob(reg/p)==h,"SOURCE_PIN:"+p)
    p=cos/"research/seti/retina_5d_mosaic_protocol.json"
    q=read(p);ck(q["origins"][0]["blob"]==PINS[next(iter(PINS))] and q["input_contract"]["baseline_alert_always_preserved"] and e.blob(p)=="c80b6eb38eb3ac0ed856bfb0d5d66b11e9c2180e" and not q["production_adoption"],"COSMOS_PROTOCOL")
    return {"registry_blobs":PINS,"cosmos_protocol_sha256":sha(p),"pyramid_12d":"BLOCKED_ORIGINAL_NOT_LOCATED"}
DROP={"evone","evtwo","not","never","did","does","is","was","were","be","been","by","for","to","the","a","an","of","than","directly","effectively"}
# Remove role/polarity markers only for coarse relation classification; keep
# full original input for independent direction/polarity/certainty tiles.
def view(t):return " ".join(w for w in t.split() if w not in DROP)
def reverse(t):return " ".join(reversed(t.split()))
def features(t):
    # Character features make inflection variants less brittle; exact same
    # hashed feature function and network used by both experimental models.
    w=t.split();z=e.features(t)
    for token in w:
        padded="^"+token+"$"
        for n in (3,4):
            for i in range(len(padded)-n+1):z["c:"+padded[i:i+n]]+=1
    x=torch.zeros(e.B)
    for k,v in z.items():x[int.from_bytes(hashlib.blake2s(k.encode(),digest_size=4).digest(),"little")%e.B]+=v
    return x/x.norm().clamp_min(1e-8)
def batch(ts):return torch.stack([features(t) for t in ts])
class Net(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(e.B,32),nn.GELU(),nn.Linear(32,5))
    def forward(self,x):return self.net(x)
def rel(y):return y//4 if y<16 else 4
def probs(ms,ts):
    with torch.inference_mode():return torch.stack([n(batch(ts)).softmax(-1) for n in ms]).min(0).values
def top(p):
    j=int(p.argmax());return j,float(p[j]),float(p[j]-max(float(v) for i,v in enumerate(p) if i!=j))
def anchor(t,bank):
    v=features(view(t));scores=[(float(v@features(view(z["text"]))),rel(z["y"]),i) for i,z in enumerate(bank)]
    scores.sort(reverse=True);return scores[0],scores[0][0]-max([s for s,y,i in scores if y!=scores[0][1]]+[0])
def candidates(t,ns,bank):
    a=probs(ns["direct"],[t])[0];b,c=probs(ns["views"],[view(t),reverse(view(t))]);s,gap=anchor(t,bank)
    return {"direct":top(a),"mosaic":top(b),"mirror":top(torch.minimum(b,c)),"source":top(torch.minimum(b,c)),"anchor":s,"anchor_gap":gap,"view_agree":int(b.argmax())==int(c.argmax())}
def decide(c,mode,r):
    j,p,margin=c[mode]
    if j==4 or p<r[0] or margin<r[1]:return None
    if mode in ("mirror","source") and not c["view_agree"]:return None
    if mode=="source" and (c["anchor"][1]!=j or c["anchor"][0]<r[2] or c["anchor_gap"]<r[3]):return None
    return j
def train(bank,reg,cos,out):
    g.witness();torch.set_num_threads(2);out.mkdir();sources=verify_sources(reg,cos);data=read(bank);tr=data["train"];dev=data["dev"]
    ck(not any(z["y"] in data["withheld"] for z in tr+dev),"WITHHELD")
    ck(len({z["text"] for z in tr+dev})==len(tr+dev),"SPLIT")
    ns={};runs={}
    for mode in ("direct","views"):
        ns[mode]=[];runs[mode]=[]
        # Equal optimizer updates and two views per input in both arms:
        # direct duplicates the original, views adds reversible representation.
        ts=[];ys=[]
        for z in tr:
            v=z["text"] if mode=="direct" else view(z["text"])
            ts.extend([v,v if mode=="direct" else reverse(v)]);ys.extend([rel(z["y"])]*2)
        x=batch(ts);y=torch.tensor(ys)
        for seed in (391,392):
            torch.manual_seed(seed);net=Net();opt=torch.optim.AdamW(net.parameters(),lr=.008);start=time.process_time()
            for _ in range(240):
                loss=nn.functional.cross_entropy(net(x),y);opt.zero_grad();loss.backward();opt.step()
            p=out/f"{mode}-{seed}.safetensors";save_file(net.state_dict(),str(p));ns[mode].append(net.eval());runs[mode].append({"seed":seed,"sha":sha(p),"parameters":sum(v.numel() for v in net.parameters()),"cpu":time.process_time()-start})
    cs=[candidates(z["text"],ns,tr) for z in dev];rules={}
    for mode in MODES:
        choices=[]
        for p in (.35,.5,.65,.8,.9,.95,1.01):
            for mar in (0,.1,.25,.4):
                for sim in (.2,.35,.5,.65):
                    for gap in (0,.05,.15):
                        r=(p,mar,sim,gap);ps=[decide(c,mode,r) for c in cs]
                        good=sum(a is not None and a==rel(z["y"]) for a,z in zip(ps,dev));bad=sum(a is not None and a!=rel(z["y"]) for a,z in zip(ps,dev))
                        if not bad:choices.append((good,r))
        good,r=max(choices);rules[mode]={"rule":r,"dev_correct":good}
    (out/"bank.json").write_bytes(bank.read_bytes());receipt={"schema":"h38h.train.v1","bank_sha":sha(bank),"train_n":len(tr),"dev_n":len(dev),"rules":rules,"runs":runs,"source_pins":sources,"exam_read":False,"teacher_labels":False,"production":False}
    write(out/"receipt.json",receipt);print("R84H38H_TRAIN",json.dumps(receipt,sort_keys=True),flush=True)
def evaluate(trained,oldE,oldG,demi,reg,cos,exam,priorc,priord,priore,priorf,priorg,out):
    g.witness();torch.set_num_threads(2);out.mkdir();sources=verify_sources(reg,cos);r=read(trained/"receipt.json");ck(sources==r["source_pins"],"SOURCES");bank=read(trained/"bank.json");ck(sha(trained/"bank.json")==r["bank_sha"],"BANK")
    ns={}
    for mode,rows in r["runs"].items():
        ns[mode]=[]
        for z in rows:
            p=trained/f"{mode}-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"WEIGHT");n=Net();n.load_state_dict(load_file(str(p)));ns[mode].append(n.eval())
    gr=read(oldG/"receipt.json");ck(sha(oldG/"bank.json")==gr["bank_sha"]==r["bank_sha"],"MATCHED_G_BANK")
    gm=[]
    for z in gr["runs"]:
        p=oldG/f"mosaic-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"G_WEIGHT");net=m.Mosaic();net.load_state_dict(load_file(str(p)));gm.append(net.eval())
    er=read(oldE/"receipt.json");eb=read(oldE/"bank.json");ck(sha(oldE/"bank.json")==er["bank_sha"],"E_BANK");em=[]
    for z in er["runs"]:
        p=oldE/f"model-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"E_WEIGHT");net=e.Net();net.load_state_dict(load_file(str(p)));em.append(net.eval())
    di=e.Dictionary(e.cortex(demi),oldE/"cortex",eb["train"])
    oldD=read(priord)["cases"];ex=read(exam);refs={z["text"] for z in bank["train"]+bank["dev"]+eb["train"]+eb["dev"]}
    for p in (priore,priorf,priorg):
        for z in read(p)["rows"]:
            n=e.norm(z["text"],z["names"])
            if n:refs.add(n[0])
    def namesD(z):return sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")})
    for z in oldD:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,namesD(z))
            if n:refs.add(n[0])
    cache={}
    def score(t):
        if t not in cache:
            c=candidates(t,ns,bank["train"]);gp=f.perrow(m.probs(gm,[t]),0);ej=e.pick(e.probs(em,[t])[0],di.search(t),er["rules"]["agreement"],"agreement")
            cache[t]=(c,gp,ej)
        return cache[t]
    def old_parse(t,names):
        base=d.parse(t)
        if base is not None:return base
        n=e.norm(t,names)
        if n is None:return None
        _,_,j=score(n[0]);return e.decode(j,n[1]) if j is not None else None
    # 5D-inspired bounded micro-pass: preserve direct candidate, compare
    # forward/mirror, abstract relation, inspect competing candidates,
    # source-bound anchor. No repeated correlated view is an independent vote.
    def crystal(t,names,mode):
        if mode=="h38e":return old_parse(t,names)
        if mode=="stack":
            old=old_parse(t,names)
            if old is not None:return old
        n=e.norm(t,names)
        if n is None:return None
        text,events=n;c,gp,_=score(text);js,conf,margin=m.raw(gp)
        k="source" if mode in ("stack","fractal") else mode
        relation=decide(c,k,r["rules"][k]["rule"])
        if relation is None:return None
        if mode=="fractal":
            # Four recursive children of the same shape: split the normalized
            # evidence into two role-delimited halves, retain reversible views,
            # and return their candidate constraints to the parent. Unknown
            # child constraints do not become new facts.
            def tree(tokens,depth):
                st=" ".join(tokens)
                if not tokens:return {"candidates":set(range(4)),"leaves":0}
                a=top(probs(ns["views"],[st])[0])
                b=top(probs(ns["views"],[reverse(st)])[0])
                support={a[0]} if a[0]==b[0] and a[0]<4 and min(a[1],b[1])>=.95 else set(range(4))
                if depth==0 or len(tokens)<4:return {"candidates":support,"leaves":1}
                mid=len(tokens)//2
                chunks=[tokens[:mid],tokens[mid:],list(reversed(tokens[:mid])),list(reversed(tokens[mid:]))]
                kids=[tree(v,depth-1) for v in chunks]
                return {"candidates":support.intersection(*(z["candidates"] for z in kids)),"leaves":sum(z["leaves"] for z in kids)}
            proof=tree(view(text).split(),2)
            if relation not in proof["candidates"]:return None
        pol,di_=m.cues(text)
        # Conservatively require old component agreement and calibrated
        # minimum confidence for polarity/direction/certainty; no relation
        # threshold from disabled H38G reused.
        if js[3]!=1 or pol is None or di_ is None or js[1]!=pol or js[2]!=di_ or min(conf[1:])<.9 or min(margin[1:])<.25:return None
        return e.decode(4*relation+2*js[1]+js[2],events)
    modes=MODES+("fractal","h38e","stack")
    stats={k:{"relation_correct":0,"relation_wrong":0,"unknown_false":0,"raw_relation_correct":0,"crystal_correct":0,"crystal_wrong":0,"crystal_unknown_false":0,"withheld_correct":0,"groups":0} for k in modes}
    details=[]
    for z in ex["rows"]:
        n=e.norm(z["text"],z["names"]);ck(n is None or n[0] not in refs,"OVERLAP")
        gold=z["gold"];y=z["y"];known=y<16
        if known:ck(n is not None and e.decode(y,n[1])==gold,"GOLD")
        c=score(n[0])[0] if n else None
        preds={k:crystal(z["text"],z["names"],k) for k in modes}
        for k in modes:
            a=preds[k];v=stats[k];v["crystal_correct"]+=known and a==gold;v["crystal_wrong"]+=known and a is not None and a!=gold;v["crystal_unknown_false"]+=not known and a is not None
            if y in bank["withheld"]:v["withheld_correct"]+=a==gold
            if k in MODES and c is not None:
                j=decide(c,k,r["rules"][k]["rule"]);v["relation_correct"]+=known and j==rel(y);v["relation_wrong"]+=known and j is not None and j!=rel(y);v["unknown_false"]+=not known and j is not None;v["raw_relation_correct"]+=known and c[k][0]==rel(y)
        details.append({**z,"pred":preds,"micro_pass":c,"source_anchor_text":bank["train"][c["anchor"][2]]["text"] if c else None})
    for k in modes:stats[k]["groups"]=sum(all(details[i]["gold"] is not None and details[i]["pred"][k]==details[i]["gold"] for i in group) for group in ex["groups"])
    def solve(ss,q,names,k):
        fs=[crystal(t,names,k) for t in ss];c=crystal(q,names,k)
        if c is None or any(a is None for a in fs):return "HOLD_UNPARSED"
        return g.solve({"schema":"janus.event.graph.v1","facts":[{"id":f"s{i}",**a} for i,a in enumerate(fs)],"claim":c})["status"]
    graphs={k:{"correct":0,"false_certainty":0} for k in modes}
    for z in ex["graphs"]:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,z["names"]);ck(n is None or n[0] not in refs,"GRAPH_OVERLAP")
        for k in modes:
            a=solve(z["sentences"],z["claim"],z["names"],k);graphs[k]["correct"]+=a==z["expected"];graphs[k]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"]
    replay={k:{"correct":0,"false_certainty":0,"lost":0} for k in ("h38e","stack")}
    for z in oldD:
        ns_=namesD(z);base=solve(z["sentences"],z["claim"],ns_,"h38e")
        for k in replay:
            a=solve(z["sentences"],z["claim"],ns_,k);replay[k]["correct"]+=a==z["expected"];replay[k]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"];replay[k]["lost"]+=base==z["expected"] and a!=z["expected"]
    ck(replay["h38e"]["correct"]==46,"H38E_REPLAY")
    old=read(priorc)["cases"];ck(len(old)==76 and all(g.solve(z["request"])["status"]==z["expected"] for z in old),"H38C_REPLAY")
    result={"schema":"h38h.result.v1","exam_sha":sha(exam),"known_n":sum(z["y"]<16 for z in details),"unknown_n":sum(z["y"]==16 for z in details),"groups_n":len(ex["groups"]),"graphs_n":len(ex["graphs"]),"routes":stats,"graphs":graphs,"h38d_replay":replay,"h38c_replay":76,"cortex_queries":di.calls,"source_pins":sources,"quality_gate":"HOLD","production":False,"scope":"Same-author bounded English with supplied event inventory. Matched data and equal updates direct vs view models. Reversible feature views are correlated, not independent evidence. Source gate is authored correspondence, not external truth. Fixed mosaic slot checks and route ablations are inspired by tiling/Wedjat. Fractal mode is a new depth-two four-child constraint intersection, not a recovered canonical pyramid or a 12D model. No independent external exam."}
    write(out/"result.json",result);write(out/"cases.json",details);print("R84H38H_RESULT",json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":{"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
