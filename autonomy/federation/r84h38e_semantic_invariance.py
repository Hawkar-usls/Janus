"""H38E: isolated Cortex dictionary + own learned relation normalizer, shadow."""
import sys,re,json,hashlib,time,math
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import r84h38d_world_understanding as d
import r84h38c_semantic_world_graph as g
REL=["BEFORE","CAUSES","REQUIRES","PREVENTS"];HOLD=16;B=1024
def ck(x,s):g.ck(x,s)
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,sort_keys=True)+"\n")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blob(p):
    b=p.read_bytes();return hashlib.sha1(("blob %d\0"%len(b)).encode()+b).hexdigest()
def key(t):return " ".join(re.findall(r"[a-z0-9_-]+",t.lower()))
def norm(t,names):
    if not isinstance(t,str) or not 1<=len(t)<=220:return None
    t=t.lower().strip()
    if re.search(r"\b(if|unless|might|may|could|would|whether|perhaps|possibly|claims?|allegedly|reports?|and|or|but|either|both|it|they|not necessarily)\b|[?;]",t):return None
    if not re.fullmatch(r"[a-z0-9_ ,.'-]+",t):return None
    hits=[]
    for n in names:
        if not g.ident(n):return None
        for m in re.finditer(r"(?<![\w-])"+re.escape(n.lower())+r"(?![\w-])",t):hits.append((m.start(),m.end(),n.lower()))
    hits.sort()
    if len(hits)!=2 or hits[0][2]==hits[1][2] or hits[0][1]>hits[1][0]:return None
    a,b=hits
    return key(t[:a[0]]+" evone "+t[a[1]:b[0]]+" evtwo "+t[b[1]:]),[a[2],b[2]]
def decode(j,n):
    r,v,s=j//4,(j%4)//2,j%2
    return {"left":n[s],"rel":REL[r],"right":n[1-s],"positive":v==0}
def features(t):
    w=t.split();f=["w:"+x for x in w]+["b:"+a+"_"+b for a,b in zip(w,w[1:])]+["t:"+a+"_"+b+"_"+c for a,b,c in zip(w,w[1:],w[2:])]
    f += ["p:"+str(i)+":"+x for i,x in enumerate(w)]
    return Counter(f)
def vector(t):
    x=torch.zeros(B)
    for k,v in features(t).items():x[int.from_bytes(hashlib.blake2s(k.encode(),digest_size=4).digest(),"little")%B]+=v
    return x/x.norm().clamp_min(1e-8)
def batch(ts):return torch.stack([vector(t) for t in ts])
class Net(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(B,64),nn.GELU(),nn.Linear(64,17))
    def forward(self,x):return self.net(x)
def probs(ms,ts):
    with torch.inference_mode():return torch.stack([m(batch(ts)).softmax(-1) for m in ms]).min(0).values.tolist()
def cortex(demi):
    for name,pin in [("memory_cortex_growth.py","626b74379e195d2e72d7c5b727198f6a3e297f04"),("memory_512_audit_index_r3.py","a01fe718018cd10bf4b0e5c1ae3b83dd90364ff3")]:
        ck(blob(demi/"janus_model"/name)==pin,"CORTEX_PIN")
    sys.path.insert(0,str(demi.resolve()))
    from janus_model import memory_cortex_growth as m
    return m
class Dictionary:
    def __init__(self,m,root,rows):
        self.m=m;self.root=root;self.rows={hashlib.sha256(z["text"].encode()).hexdigest():z for z in rows};self.cache={};self.calls=0
        ck(set(m.load_state(root)["active"])==set(self.rows),"CORTEX_STATE")
    def search(self,t):
        if t in self.cache:return self.cache[t]
        query=" ".join(w for w in t.split() if w not in ("evone","evtwo"))[:120]
        vals=[]
        if query:
            result=self.m.query(self.root,query,limit=8);self.m.checked(result);self.calls+=1
            f=features(t)
            for hit in result["citations"]:
                z=self.rows[hit["sha256"]];raw=self.m.read_object(self.root,hit["sha256"],hit["verified_bytes"])
                ck(raw.decode()==z["text"] and hit["origin"]["label"]==z["y"] and hit["content_is_untrusted_data"],"CORTEX_PROVENANCE")
                v=features(z["text"]);score=sum(c*v.get(k,0) for k,c in f.items())/math.sqrt(sum(c*c for c in f.values())*sum(c*c for c in v.values()))
                vals.append((score,z["y"],hit["sha256"]))
        vals.sort(reverse=True);best=vals[0] if vals else (0,HOLD,None);margin=best[0]-max([s for s,y,_ in vals if y!=best[1]]+[0])
        self.cache[t]=(best,margin);return self.cache[t]
def pick(p,lex,rule,mode):
    j=max(range(17),key=lambda i:p[i]);best,margin=lex
    if mode=="dictionary":return best[1] if best[1]!=HOLD and best[0]>=rule[2] and margin>=rule[3] else None
    if j==HOLD or p[j]<rule[0] or p[j]-max(p[i] for i in range(17) if i!=j)<rule[1]:return None
    if mode=="agreement" and (best[1]!=j or best[0]<rule[2] or margin<rule[3]):return None
    return j
def train(demi,bank,out):
    g.witness();torch.set_num_threads(2);out.mkdir();data=read(bank);tr=data["train"];dev=data["dev"]
    ck(len({z["text"] for z in tr+dev})==len(tr+dev),"OVERLAP")
    m=cortex(demi);root=out/"cortex";root.mkdir();state=m.new_state()
    for z in tr:
        raw=z["text"].encode();origin={"kind":"SYNTHETIC_RELATION_CORRESPONDENCE","label":z["y"],"training_allowed":True,"external_truth_verified":False}
        ck(m.add_text(root,state,raw,origin) and not m.add_text(root,state,raw,origin),"DEDUP")
    manifest=m.publish(root,state,"H38E_DISPOSABLE_RELATION_DICTIONARY");m.write_json(root/"STATE.json",m.sealed(state));di=Dictionary(m,root,tr)
    x=batch([z["text"] for z in tr]);y=torch.tensor([z["y"] for z in tr]);ms=[];runs=[]
    for seed in (385,386):
        torch.manual_seed(seed);net=Net();opt=torch.optim.AdamW(net.parameters(),lr=.008);start=time.process_time()
        for _ in range(240):
            loss=nn.functional.cross_entropy(net(x),y);opt.zero_grad();loss.backward();opt.step()
        p=out/f"model-{seed}.safetensors";save_file(net.state_dict(),str(p));ms.append(net.eval());runs.append({"seed":seed,"sha":sha(p),"parameters":sum(v.numel() for v in net.parameters()),"cpu":time.process_time()-start})
    ps=probs(ms,[z["text"] for z in dev]);ls=[di.search(z["text"]) for z in dev];rules={}
    for mode in ("learned","dictionary","agreement"):
        choices=[]
        for c in (.5,.65,.8,.9,1.01):
            for margin in (.1,.25,.45):
                for lex in (.3,.5,.7,1.01):
                    for gap in (.02,.1,.2):
                        r=(c,margin,lex,gap);pred=[pick(p,l,r,mode) for p,l in zip(ps,ls)]
                        good=sum(a==z["y"] and a is not None for a,z in zip(pred,dev));bad=sum(a is not None and a!=z["y"] for a,z in zip(pred,dev))
                        if not bad:choices.append((good,r))
        rules[mode]=max(choices)[1]
    receipt={"schema":"h38e.train.v1","train_n":len(tr),"dev_n":len(dev),"bank_sha":sha(bank),"cortex_documents":manifest["indexed_documents"],"cortex_queries":di.calls,"runs":runs,"rules":rules,"exam_read":False,"teacher_labels":False,"production":False}
    write(out/"receipt.json",receipt);(out/"bank.json").write_bytes(bank.read_bytes());print("R84H38E_TRAIN",json.dumps(receipt,sort_keys=True),flush=True)
def evaluate(demi,trained,exam,oldc,oldd,out):
    g.witness();torch.set_num_threads(2);out.mkdir();r=read(trained/"receipt.json");bank=read(trained/"bank.json");ck(sha(trained/"bank.json")==r["bank_sha"],"BANK")
    ms=[]
    for z in r["runs"]:
        p=trained/f"model-{z['seed']}.safetensors";ck(sha(p)==z["sha"],"WEIGHT");m=Net();m.load_state_dict(load_file(str(p)));ms.append(m.eval())
    di=Dictionary(cortex(demi),trained/"cortex",bank["train"]);ex=read(exam);refs={z["text"] for z in bank["train"]+bank["dev"]}
    modes=("baseline","learned","dictionary","agreement","guarded")
    def parse(t,names,mode):
        base=d.parse(t)
        if mode=="baseline" or (mode=="guarded" and base is not None):return base
        z=norm(t,names)
        if z is None:return None
        text,ns=z;p=probs(ms,[text])[0];lex=di.search(text);k="agreement" if mode=="guarded" else mode
        j=pick(p,lex,r["rules"][k],k)
        return decode(j,ns) if j is not None else None
    for z in ex["rows"]:
        q=norm(z["text"],z["names"])
        ck(q is None or q[0] not in refs,"EXAM_OVERLAP")
    # Also reject exact normalized overlap with disclosed H38D text.
    prior=read(oldd)["cases"]
    prior_norm=set()
    for z in prior:
        names=sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")})
        for t in z["sentences"]+[z["claim"]]:
            q=norm(t,names)
            if q:prior_norm.add(q[0])
    for z in ex["rows"]:
        q=norm(z["text"],z["names"]);ck(q is None or q[0] not in prior_norm,"PRIOR_EXAM_OVERLAP")
    details=[]
    for z in ex["rows"]:details.append({**z,"pred":{m:parse(z["text"],z["names"],m) for m in modes}})
    stats={}
    for m in modes:
        stats[m]={"known_correct":sum(z["gold"] is not None and z["pred"][m]==z["gold"] for z in details),"wrong":sum(z["pred"][m] is not None and z["pred"][m]!=z["gold"] for z in details),"unknown_false":sum(z["gold"] is None and z["pred"][m] is not None for z in details),"invariant_groups":sum(all(details[i]["pred"][m]==details[i]["gold"] and details[i]["gold"]==details[group[0]]["gold"] and details[i]["gold"] is not None for i in group) for group in ex["groups"]),"baseline_lost":sum(z["gold"] is not None and z["pred"]["baseline"]==z["gold"] and z["pred"][m]!=z["gold"] for z in details)}
    def solve(s,q,names,m):
        facts=[parse(t,names,m) for t in s];claim=parse(q,names,m)
        if any(a is None for a in facts) or claim is None:return "HOLD_UNPARSED"
        return g.solve({"schema":"janus.event.graph.v1","facts":[{"id":f"s{i}",**a} for i,a in enumerate(facts)],"claim":claim})["status"]
    graphstats={m:{"correct":0,"false_certainty":0} for m in modes}
    for z in ex["graphs"]:
        for m in modes:
            a=solve(z["sentences"],z["claim"],z["names"],m);graphstats[m]["correct"]+=a==z["expected"];graphstats[m]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"]
    replay={m:{"correct":0,"false_certainty":0,"baseline_lost":0} for m in ("baseline","guarded")}
    for z in prior:
        names=sorted({a[k] for a in z["atoms"]+[z["claim_atom"]] if a for k in ("left","right")})
        base=solve(z["sentences"],z["claim"],names,"baseline")
        for m in replay:
            a=solve(z["sentences"],z["claim"],names,m);replay[m]["correct"]+=a==z["expected"];replay[m]["false_certainty"]+=a in ("ENTAILED","CONTRADICTED") and a!=z["expected"];replay[m]["baseline_lost"]+=base==z["expected"] and a!=z["expected"]
    old=read(oldc)["cases"];ck(len(old)==76 and all(g.solve(z["request"])["status"]==z["expected"] for z in old),"H38C_REPLAY");ck(replay["baseline"]["correct"]==41,"H38D_REPLAY")
    result={"schema":"h38e.result.v1","exam_sha":sha(exam),"known_n":sum(z["gold"] is not None for z in details),"unknown_n":sum(z["gold"] is None for z in details),"groups_n":len(ex["groups"]),"graph_n":len(ex["graphs"]),"routes":stats,"graphs":graphstats,"h38d_replay":replay,"h38c_replay":76,"cortex_queries":di.calls,"quality_gate":"HOLD","production":False,"scope":"Same-author English relation clauses, supplied event inventory, disposable native Cortex dictionary, own learned model, no external independent exam or MiniLM comparison; no deployment."}
    write(out/"result.json",result);write(out/"cases.json",details);print("R84H38E_RESULT",json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":{"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
