"""H38B bounded event relations: cache access and persistence order only."""
import sys,re,time
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file,save_file
import r84h37_polarity_independence as h
from r84h38a_polarity_diagnostic import load_models

# State labels: cache miss/hit, acknowledgement first/persistence first, HOLD.
ST=[("cached_available",0),("cached_available",1),("persist_first",0),("persist_first",1)]
HOLD=4
# General event lexicon, shared by the grammar and the neural representation.
ROLE={
"cache":r"\b(?:cache|cached|local (?:store|entry|copy)|nearby store)\b",
"origin":r"\b(?:origin|source|backing service|authoritative service|remote (?:read|fetch|service))\b",
"persist":r"\b(?:persist(?:ed|ence|s)?|durab(?:le|ility)|permanent write|disk write|disk commit|saved to disk)\b",
"ack":r"\b(?:acknowledg(?:e|ed|ement|es)|confirmation|confirmed|success reply|success response|success is reported)\b",
}
def roles(t):
    spans=[]
    for k,rx in ROLE.items():
        spans.extend((m.start(),m.end(),k) for m in re.finditer(rx,t))
    return sorted(spans)
def shape(t):
    for k,rx in ROLE.items():t=re.sub(rx," "+k+" ",t)
    return h.tokens(t)
def relation(t):
    t=t.lower().strip(" .")
    if not h.scope(t) or re.search(r"\b(?:if|unless|would|could|question|perhaps|or|both|not necessarily|not clear|no evidence|not sure)\b",t):return None
    rs=roles(t)
    # Exact ordering of unique event types on either side of before/after.
    if {x[2] for x in rs}=={"persist","ack"} and len(rs)==2:
        a,b=rs
        if re.search(r"\b(?:not|never|no)\b",t):return None
        middle=t[a[1]:b[0]]
        prefix=t[:a[0]]
        if re.search(r"\bonly after\b|\bafter\b",prefix):
            first=a[2]
        elif re.search(r"\bbefore\b|\bprecedes\b|\bprior to\b",middle):
            first=a[2]
        elif re.search(r"\bafter\b|\bfollows\b",middle):
            first=b[2]
        else:return None
        return 3 if first=="persist" else 2
    # For cache, only consistent explicit evidence per clause; negation tied
    # to its event rather than globally flipping the sentence.
    if {x[2] for x in rs}<= {"cache","origin"} and any(x[2]=="cache" for x in rs):
        clauses=re.split(r";|,\s*(?:so|therefore|and)\s+|\b(?:because|therefore|so)\b",t)
        evidence=[]
        for c in clauses:
            kinds={x[2] for x in roles(c)}
            if kinds=={"cache"}:
                if re.search(r"\b(?:miss|empty|absent|lacks|unavailable|does not contain|not present)\b",c):evidence.append(0)
                elif re.search(r"\b(?:hit|contains|present|available|supplies|satisfies)\b",c) and not re.search(r"\b(?:not|no|never)\b",c):evidence.append(1)
            elif kinds=={"origin"}:
                if re.search(r"\b(?:no|not|never|without)\b",c) and re.search(r"\b(?:fetch|read|contacted|called|needed|required|accessed|queried)\b",c):evidence.append(1)
                elif re.search(r"\b(?:fetch|read|contacted|called|needed|required|accessed|queried)\b",c) and not re.search(r"\b(?:no|not|never|without)\b",c):evidence.append(0)
        if len(evidence)>=2 and len(set(evidence))==1:return evidence[0]
        # Two agreeing observations required: no guesses from lone cues.
    return None

# Separate authored training and calibration families; no prior exam text.
TRAIN=[
["entity has a cache miss; the origin is queried","entity has a cache hit; the origin is not queried"],
["the local store lacks entity; the source is called","the local store contains entity; the source is not called"],
["entity is absent from the cache; a remote read is required","entity is present in the cache; no remote read is required"],
["the cache is empty for entity; an origin fetch is needed","the cache contains entity; no origin fetch is needed"],
["the cache is unavailable for entity; the source is accessed","the cache is available for entity; the source is not accessed"],
["the origin is contacted; the local copy is absent for entity","the origin is not contacted; the local copy is present for entity"],
["confirmation for entity precedes persistence","persistence for entity precedes confirmation"],
["entity is acknowledged before the disk commit","the disk commit for entity occurs before it is acknowledged"],
["durability for entity follows confirmation","confirmation for entity follows durability"],
["after acknowledgement entity is persisted","after persistence entity is acknowledged"],
["entity is confirmed prior to the permanent write","the permanent write for entity occurs prior to confirmation"],
["the success response for entity comes after acknowledgement","the success response for entity comes after persistence"], # first is HOLD; corrected below
]
TRAIN[-1]=["persistence for entity occurs after the success response","the success response for entity occurs after persistence"]
DEV=[
["the cache lacks entity; a source fetch is required","the cache contains entity; no source fetch is required"],
["a remote read is needed; entity is absent from the local copy","no remote read is needed; entity is present in the local copy"],
["the permanent write follows acknowledgement for entity","acknowledgement follows the permanent write for entity"],
["only after confirmation is entity persisted","only after persistence is entity confirmed"],
]
UNK_T=["entity has a cache capacity","entity has a disk write policy","the origin has a hostname","the cache is present but the source is queried","whether entity has a local copy is unknown","confirmation and persistence are both recorded for entity","entity is persisted or acknowledged first","entity does not have a known confirmation time","entity allows multiple holders","entity is fully replicated on every node","if the cache contains entity the source is not called","entity has a success reply identifier"]
UNK_D=["entity has a source address","entity has a confirmation timestamp","entity has both a cache hit and a cache miss","entity may persist before or after confirmation","the cache stores an identifier for entity","entity is partitioned across nodes"]
def records(pairs,unknown,split):
    out=[]
    cut=len(pairs)//2
    for i,pair in enumerate(pairs):
        for v,t in enumerate(pair):
            out.append({"text":t,"y":v+(0 if i<cut else 2),"family":split+str(i)})
    return out+[{"text":t,"y":HOLD,"family":split+"u"+str(i)} for i,t in enumerate(unknown)]
# Ordered token roles + original text words, learned compact sequence classifier.
B=1024;L=64
def enc(t):
    w=shape(t)
    return [1+h.hid(x)%(B-1) for x in w[:L]]+[0]*max(0,L-len(w))
def batch(ts):return torch.tensor([enc(t) for t in ts],dtype=torch.long)
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(B,24,padding_idx=0);self.gru=nn.GRU(24,32,batch_first=True,bidirectional=True);self.head=nn.Linear(64,5)
    def forward(self,x):
        a,_=self.gru(self.emb(x));mask=x.ne(0).unsqueeze(-1);return self.head((a*mask).sum(1)/mask.sum(1).clamp_min(1))
def probs(ms,ts):
    with torch.inference_mode():return torch.stack([m(batch(ts)).softmax(-1) for m in ms]).min(0).values.tolist()
def pred(p,r):
    j=max(range(5),key=lambda i:p[i]);return j if j!=HOLD and p[j]>=r else None
def train(out):
    h.offline();out.mkdir(parents=True,exist_ok=False)
    tr=records(TRAIN,UNK_T,"train");dev=records(DEV,UNK_D,"dev")
    h.ck(not ({h.key(z["text"]) for z in tr}&{h.key(z["text"]) for z in dev}),"SPLIT")
    ms=[];runs=[];x=batch([z["text"] for z in tr]);y=torch.tensor([z["y"] for z in tr])
    for seed in (381,382):
        torch.manual_seed(seed);m=Net();opt=torch.optim.AdamW(m.parameters(),lr=.006);start=time.process_time()
        for _ in range(160):
            loss=nn.functional.cross_entropy(m(x),y);opt.zero_grad();loss.backward();opt.step()
        p=out/("event-%s.safetensors"%seed);save_file(m.state_dict(),str(p));ms.append(m.eval());runs.append({"seed":seed,"sha":h.sha(p),"params":sum(z.numel() for z in m.parameters()),"cpu":time.process_time()-start})
    ps=probs(ms,[z["text"] for z in dev]);choices=[]
    for r in (.5,.6,.7,.8,.9,.95,.98,1.01):
        g=[pred(p,r) for p in ps];bad=sum(a is not None and a!=z["y"] for a,z in zip(g,dev));good=sum(a==z["y"] for a,z in zip(g,dev))
        if not bad:choices.append((good,r))
    good,threshold=max(choices)
    result={"schema":"h38b.train.v1","train_n":len(tr),"dev_n":len(dev),"train_sha":h.digest(tr),"dev_sha":h.digest(dev),"runs":runs,"threshold":threshold,"dev_correct":good,"exam_read":False,"teacher_labels":False,"production":False}
    h.write(out/"train.json",tr);h.write(out/"dev.json",dev);h.write(out/"receipt.json",result);h.log("R84H38B_TRAIN",result)
def evaluate(own,old,out):
    h.offline();out.mkdir(parents=True,exist_ok=False)
    r=h.read(own/"receipt.json");tr=h.read(own/"train.json");dev=h.read(own/"dev.json");h.ck(h.digest(tr)==r["train_sha"] and h.digest(dev)==r["dev_sha"] and not r["exam_read"],"TRAIN_PIN")
    ms=[]
    for z in r["runs"]:
        p=own/("event-%s.safetensors"%z["seed"]);h.ck(h.sha(p)==z["sha"],"WEIGHT_PIN");m=Net();m.load_state_dict(load_file(str(p)));ms.append(m.eval())
    oldrec,older=load_models(old,h.V37)
    path=Path(__file__).parent/"R84H38B_FROZEN_EXAM.json";ex=h.read(path)
    refs={h.key(z["text"]) for z in tr+dev}
    prior={}
    for n,pin in [(33,"02e40d9360d717b0c8f3e5e8f420bdc5003e272b"),(34,"582ea0e9aa40c545af77964042121f7a95eb0b0a"),(36,"69823b8919ae31c23fae2f1f150feeaefdd28027"),(37,h.EXAM)]:
        p=path.parent/("R84H%d_FROZEN_EXAM.json"%n);h.ck(h.blob(p)==pin,"PRIOR_PIN");e=h.read(p);norm=h.corpus(e)
        refs|={h.key(z[1]) for z in norm.values() if z}
        rows=[]
        for z in e["statements"]:
            if z["fact"]["attribute"] in {a for a,v in ST} and norm[z["text"]]:
                rows.append({"text":norm[z["text"]][1],"y":ST.index((z["fact"]["attribute"],z["fact"]["value"]))})
        rows += [{"text":norm[t][1],"y":HOLD} for t in e["unknown"] if norm[t]]
        prior[str(n)]=rows
    h.ck(all(h.key(z["text"]) not in refs for z in ex["rows"]),"FRESH_OVERLAP")
    def assess(rows):
        ts=[z["text"] for z in rows];ps=probs(ms,ts);op,_=h.prob37(older,ts);details=[];metrics={}
        for z,p,q in zip(rows,ps,op):
            t=z["text"];a=h.accepted(q,oldrec["rule"],True,t);a=a[0] if len(a)==1 else None
            oldval=None if a is None else ST.index(h.STATES[a]) if h.STATES[a] in ST else 5
            g=relation(t);n=pred(p,r["threshold"]);agree=g if g is not None and g==n else None
            routes={"h37":oldval,"grammar":g,"learned":n,"agreement":agree,"guarded_union":g if g is not None else oldval}
            details.append({**z,"predictions":routes})
        for k in details[0]["predictions"]:
            metrics[k]={"correct":sum(z["predictions"][k]==z["y"] and z["y"]!=HOLD for z in details),"wrong":sum(z["predictions"][k] is not None and z["predictions"][k]!=z["y"] and z["y"]!=HOLD for z in details),"unknown_false":sum(z["predictions"][k] is not None and z["y"]==HOLD for z in details),"h37_correct_lost":sum(z["predictions"]["h37"]==z["y"] and z["y"]!=HOLD and z["predictions"][k]!=z["y"] for z in details)}
        return {"known_n":sum(z["y"]!=HOLD for z in rows),"unknown_n":sum(z["y"]==HOLD for z in rows),"routes":metrics},details
    fresh,details=assess(ex["rows"]);replay={n:assess(z)[0] for n,z in prior.items()}
    # Paired contrast accuracy requires both members correct.
    for k,m in fresh["routes"].items():
        m["paired_correct"]=sum(all(details[i]["predictions"][k]==details[i]["y"] for i in pair) for pair in ex["pairs"])
    report={"schema":"h38b.result.v1","exam_blob":h.blob(path),"exam_n":len(details),"pairs_n":len(ex["pairs"]),"fresh":fresh,"regression":replay,"production":False,"teacher_loaded":False,"quality_gate":"HOLD","scope":"Two-property authored synthetic benchmark, supplied entity placeholder, same-author data and grammar; no MiniLM comparison, no integrated production regression or independent external exam. No promotion."}
    h.write(out/"result.json",report);h.write(out/"cases.json",details);h.log("R84H38B_RESULT",report)
if __name__=="__main__":{"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
