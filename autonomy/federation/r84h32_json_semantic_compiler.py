"""H32 bounded text-to-fact compiler; exact JSON relation downstream, no promotion."""
import sys,re,json,time,random,hashlib,math
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
import r84h31_semantic_evolution as h31
import r84h30_semantic_inheritance as h30
import r84h29_semantic_resonance as h29
import r84h19_exploration as h19
import r8_learning as r
import verified_capabilities as exact
HERE=Path(__file__).parent
EXAM="4cb34144b048e5408d57ddf0a388df7a30f201e5"
SOLVER="1b131dc27b0fb5524761473e3b27c6c4db51e759"
ATTR=tuple(sorted(h31.BANK));STATES=[(a,v) for a in ATTR for v in (0,1)]+[("HOLD",0)]
HOLD=len(STATES)-1
MOTTO="Мы должны узнать"
ck=h30.ck;read=h30.read;write=h30.write;log=h30.log;digest=h30.digest
# Single explicitly named subject from supplied inventory, not an arbitrary NER.
def normalize(text,subjects):
    if not isinstance(text,str) or not 1<=len(text)<=600 or not 1<=len(subjects)<=32:return None
    if any(type(s) is not str or not 1<=len(s)<=80 for s in subjects) or len({s.casefold() for s in subjects})!=len(subjects):return None
    hits=[]
    for s in subjects:
        rx=r"(?<!\w)"+re.escape(s)+r"(?!\w)"
        if re.search(rx,text,re.I):hits.append((s,rx))
    if len(hits)!=1:return None
    s,rx=hits[0];return s," ".join(re.sub(rx," ENTITY ",text,flags=re.I).lower().split())
def key(t):return " ".join(re.findall(r"[a-z0-9]+",t.lower()))
def row(text,subject,attribute,value):
    norm=normalize(text,[subject]);ck(norm is not None,"SUBJECT")
    return {"text":norm[1],"label":STATES.index((attribute,value)),"attribute":attribute,"value":value}
# New authored variants, distinct from H31 and H32 holdout. Explicit state
# labels come from the structured domain, not from MiniLM predictions.
EXTRA={
"cached_available":{0:["The request for {s} misses the fast storage and reads the value from the original database.","No local entry matches {s}, so an upstream service must return it.","A lookup cannot find {s} in the buffer; another store supplies the record."],1:["The local store finds {s} and answers without accessing the backing database.","A previously saved local value for {s} is used to answer the lookup.","The request obtains {s} directly from its resident cached entry."]},
"persist_first":{0:["The writer reports success for {s} before copying the changed data to persistent storage.","The memory buffer receives the edit to {s}; the application gets an acknowledgement and a later flush writes it to disk.","The write of {s} finishes for the caller before its modification reaches stable media."],1:["The caller gets confirmation for {s} only after both cache and backing disk are updated.","Persistence of {s} is synchronous with the original cached write.","The service commits {s} to persistent storage and then confirms the operation."]},
"multi_holder":{0:["The lock for {s} permits exactly one owner, with all competing workers excluded.","The controller of {s} prohibits concurrent entry by two distinct holders.","One exclusive token gives access to {s}, and nobody else can acquire it while held."],1:["A finite pool of tokens lets multiple callers enter {s} together.","The counter for {s} permits three concurrent owners and increments when each releases its permit.","Four holders can simultaneously access {s} by acquiring individual permits."]},
"full_replica":{0:["The nodes store distinct slices of {s}, rather than complete copies.","The collection {s} is separated into disjoint shards placed on different servers.","Each machine holds only its assigned part of {s}."],1:["Each of several machines retains all records of {s}.","Identical complete versions of {s} are stored on redundant servers.","The entire dataset {s} is copied onto each database node."]},
"newer_recovery":{0:["The failed upgrade of {s} is undone by reinstalling the last stable release.","Recovery returns {s} to the earlier software version.","The previous build of {s} replaces the unsuccessful new deployment."],1:["The failed deployment of {s} is repaired by applying a newer patch.","The service advances {s} to a corrected release rather than restoring its old version.","Recovery installs a later build of {s} that corrects the defect."]},
"all_or_none":{0:["The linked operations on {s} may commit independently even if one fails.","A failure may leave a subset of the changes to {s} committed.","One adjustment of {s} can persist alone while its companion is discarded."],1:["All changes to {s} must commit as a unit or be discarded as a unit.","The linked edits of {s} cannot leave a partially committed result.","The transaction on {s} commits every modification together or commits nothing."]}}
DEV={
"cached_available":{0:"An unsuccessful search in the fast tier for {s} requires obtaining it from the original source.",1:"The saved local entry answers the request for {s} without an origin lookup."},
"persist_first":{0:"The response confirming {s} arrives while the modification is still in RAM and is persisted later.",1:"The service sends the confirmation of {s} after the backing storage has received the change."},
"multi_holder":{0:"The gate for {s} admits a sole exclusive owner and excludes a second concurrent holder.",1:"A limited permit supply allows more than one task to occupy {s} concurrently."},
"full_replica":{0:"Different nonoverlapping portions of {s} reside on separate nodes.",1:"Redundant nodes store matching complete copies of {s}."},
"newer_recovery":{0:"The system reverses the deployment of {s} to a preceding working build.",1:"A newer repaired revision of {s} is installed to recover the service."},
"all_or_none":{0:"A partial result of the paired modifications of {s} can remain committed.",1:"Neither linked change to {s} can commit without the other."}}
NEG=["The storage of {s} uses compression.","The cache for {s} records its eviction statistics.","The permit documentation for {s} is printed in a manual.","The storage device for {s} uses a flash controller.","The transaction for {s} has a timestamp.","The backup of {s} is encrypted.","The metadata of {s} is sorted alphabetically.","The network connection for {s} has low latency.","The cache policy for {s} selects the oldest entry for eviction.","The record {s} is indexed by a unique key.","The version of {s} is stored in a log.","The scheduler for {s} uses priority queues.","The storage of {s} may be volatile or durable; no order is specified.","It is unknown whether {s} is cached.","The resource {s} might allow one holder or more; nothing is established.","The status of {s} has not been determined.","The collection {s} may be copied or partitioned.","No conclusion is stated about whether changes to {s} commit together.","The description of {s} specifies neither earlier nor later recovery.","The state of {s} is ambiguous."]
DEVNEG=["The cache metrics for {s} are plotted on a chart.","The service {s} stores its logs on disk.","The semaphore identifier of {s} is a string.","The release of {s} was built yesterday.","The transaction ID for {s} is a number.","There is insufficient information about the local availability of {s}.","The description does not establish the persistence order of {s}.","The ownership count for {s} is unspecified."]
def unique(rows):
    out={}
    for z in rows:
        k=key(z["text"])
        if k in out:ck(out[k]["label"]==z["label"],"CONFLICT_LABEL")
        out[k]=z
    return list(out.values())
def extract(rows):
    out=[]
    for z in rows:
        if "frame" not in z:continue
        f=z["frame"];ck(exact.relation(f)==z["label"],"H31_FRAME")
        for text,claim in [(z["premise"],f["facts"][0]),(z["hypothesis"],f["claim"])]:
            if claim["attribute"] not in ATTR:continue
            out.append(row(text,claim["subject"],claim["attribute"],claim["value"]))
    return unique(out)
def dataset(trained):
    receipt=read(trained/"receipt.json");tr=read(trained/"train.json");dv=read(trained/"dev.json")
    ck(receipt["train_sha"]==digest(tr) and receipt["dev_sha"]==digest(dv) and receipt["new_json_n"]==864 and not receipt["exam_read"],"H31_PIN")
    train=extract(tr);dev=extract(dv)
    for a,values in EXTRA.items():
        for v,forms in values.items():
            train += [row(t.format(s="trainitem"),"trainitem",a,v) for t in forms]
    train += [row(t.format(s="trainitem"),"trainitem","HOLD",0) for t in NEG]
    for a,values in DEV.items():
        for v,t in values.items():dev.append(row(t.format(s="devitem"),"devitem",a,v))
    dev += [row(t.format(s="devitem"),"devitem","HOLD",0) for t in DEVNEG]
    train=unique(train);dev=[z for z in unique(dev) if key(z["text"]) not in {key(x["text"]) for x in train}]
    ck(set(z["label"] for z in train)==set(range(len(STATES))) and set(z["label"] for z in dev)==set(range(len(STATES))),"COVERAGE")
    return train,dev
BUCKETS=4096;LIMIT=96;DIM=40
def hid(s):return 1+int.from_bytes(hashlib.blake2s(s.encode(),digest_size=4).digest(),"little")%(BUCKETS-1)
def tokens(t):return re.findall(r"[a-z0-9]+",t.lower())
def fids(t,mode):
    w=tokens(t);w=["<num>" if x.isdigit() else x for x in w]
    if mode=="ordered":f=["w:"+x for x in w]
    else:f=["w:"+x for x in w]+["b:"+a+"_"+b for a,b in zip(w,w[1:])]+["p:"+x[:4] for x in w if len(x)>4]
    ids=[hid(x) for x in f[:LIMIT]];return ids+[0]*(LIMIT-len(ids))
def inputs(rows,mode):return torch.tensor([fids(z,mode) for z in rows],dtype=torch.long)
class Compiler(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.word=nn.Embedding(BUCKETS,DIM,padding_idx=0)
        if mode=="ordered":self.gru=nn.GRU(DIM,32,batch_first=True,bidirectional=True)
        width=64 if mode=="ordered" else DIM
        self.head=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,len(STATES)))
    def forward(self,x):
        m=x.ne(0).unsqueeze(-1);z=self.word(x)
        if self.mode=="ordered":
            # Packed sequence prevents padding from altering backward state.
            lengths=m.squeeze(-1).sum(1).clamp_min(1).cpu();p=nn.utils.rnn.pack_padded_sequence(z,lengths,batch_first=True,enforce_sorted=False);_,h=self.gru(p);v=torch.cat((h[0],h[1]),-1)
        else:v=(z*m).sum(1)/m.sum(1).clamp_min(1)
        return self.head(v)
def predict(m,mode,texts):
    start=time.process_time();x=inputs(texts,mode);out=[]
    with torch.inference_mode():
        for i in range(0,len(x),32):out+=m(x[i:i+32]).softmax(-1).tolist()
    return out,time.process_time()-start
def lexical_features(t):
    w=tokens(t);return Counter(["w:"+x for x in w]+["b:"+a+"_"+b for a,b in zip(w,w[1:])]+["c:"+x[i:i+3] for x in w for i in range(max(0,len(x)-2))])
def lexical(train,texts):
    base=[(lexical_features(z["text"]),z["label"]) for z in train];out=[];start=time.process_time()
    for t in texts:
        f=lexical_features(t);den=math.sqrt(sum(v*v for v in f.values()));scores=[0.]*len(STATES)
        for a,k in base:
            sim=sum(v*a.get(w,0) for w,v in f.items())/max(1e-12,den*math.sqrt(sum(v*v for v in a.values())))
            scores[k]=max(scores[k],sim)
        out.append(scores)
    return out,time.process_time()-start
def candidate(p):
    order=sorted(range(len(p)),key=lambda i:(-p[i],i));j,k=order[:2];return j,p[j],p[j]-p[k]
def evalrows(rows,probs,policy):
    result={"known_n":0,"raw_exact":0,"raw_attribute":0,"raw_value_given_attribute":0,"accepted":0,"accepted_correct":0,"accepted_wrong":0,"unknown_n":0,"unknown_accepted":0,"unknown_raw_hold":0}
    for z,p in zip(rows,probs):
        truth=z["label"];j,c,g=candidate(p);use=j!=HOLD and policy is not None and c>=policy["confidence"] and g>=policy["margin"]
        if truth==HOLD:result["unknown_n"]+=1;result["unknown_accepted"]+=use;result["unknown_raw_hold"]+=j==HOLD
        else:
            result["known_n"]+=1;result["raw_exact"]+=j==truth;result["raw_attribute"]+=STATES[j][0]==STATES[truth][0];result["raw_value_given_attribute"]+=j==truth
            if use:result["accepted"]+=1;result["accepted_correct"]+=j==truth;result["accepted_wrong"]+=j!=truth
    return result
def calibrate(rows,p):
    opts=[]
    for c in (.40,.50,.60,.70,.80,.90,.95,.99):
        for g in (.05,.10,.20,.35):
            rule={"confidence":c,"margin":g};m=evalrows(rows,p,rule)
            if m["accepted_wrong"]==0 and m["unknown_accepted"]==0 and m["accepted_correct"]>=3:opts.append((m["accepted_correct"],c,g,rule))
    return max(opts,key=lambda z:z[:3])[3] if opts else None
def train(h31trained,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);ck(r.blob((HERE/"verified_capabilities.py").read_bytes())==SOLVER,"SOLVER_PIN")
    tr,dv=dataset(h31trained);runs=[];texts=[z["text"] for z in tr];truth=torch.tensor([z["label"] for z in tr])
    lp,lc=lexical(tr,[z["text"] for z in dv]);lr=calibrate(dv,lp)
    for mode in ("bag","ordered"):
        x=inputs(texts,mode)
        for seed in (321,322):
            torch.manual_seed(seed);m=Compiler(mode);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001);gen=torch.Generator().manual_seed(seed+1)
            # Balance the 13 classes despite deduplication, never use exam.
            by=[torch.where(truth==i)[0] for i in range(len(STATES))];start=time.process_time()
            for step in range(360):
                ix=torch.cat([z[torch.randint(len(z),(4,),generator=gen)] for z in by]);o=m(x[ix]);loss=F.cross_entropy(o,truth[ix]);ck(bool(torch.isfinite(loss)),"LOSS");opt.zero_grad();loss.backward();opt.step()
            cpu=time.process_time()-start;m.eval();p,pc=predict(m,mode,[z["text"] for z in dv]);rule=calibrate(dv,p);metric=evalrows(dv,p,rule)
            path=out/("%s-%d.safetensors"%(mode,seed));save_file(m.state_dict(),str(path));params=sum(z.numel() for z in m.parameters());ck(params<=250000,"PARAMETER_BUDGET")
            runs.append({"mode":mode,"seed":seed,"sha":h29.sha(path),"bytes":path.stat().st_size,"parameters":params,"train_cpu":cpu,"dev_cpu":pc,"policy":rule,"dev":metric})
    selected=max(runs,key=lambda z:(z["dev"]["accepted_correct"],z["dev"]["raw_exact"],-z["parameters"],-z["seed"]))
    rec={"schema":"r84h32.train.v1","train_n":len(tr),"dev_n":len(dv),"train_sha":digest(tr),"dev_sha":digest(dv),"states":STATES,"lexical_policy":lr,"lexical_dev":evalrows(dv,lp,lr),"lexical_dev_cpu":lc,"runs":runs,"selected":{"mode":selected["mode"],"seed":selected["seed"]},"exam_read":False,"production":False,"motto":MOTTO}
    write(out/"train.json",tr);write(out/"dev.json",dv);write(out/"receipt.json",rec);log("R84H32_TRAIN",rec)
def evaluate(demi,source,h31trained,trained,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);ck(r.blob((HERE/"verified_capabilities.py").read_bytes())==SOLVER,"SOLVER_PIN")
    rec=read(trained/"receipt.json");tr=read(trained/"train.json");dv=read(trained/"dev.json")
    ck(rec["train_sha"]==digest(tr) and rec["dev_sha"]==digest(dv) and not rec["exam_read"] and not rec["production"],"TRAIN_PIN")
    ex=h19.load_exam("R84H32_FROZEN_EXAM.json",EXAM);subjects=ex["subjects"];known=ex["statements"];unknown=ex["unknown"];pairs=ex["pairs"]
    texts=list(dict.fromkeys([z["text"] for z in known]+unknown+[t for z in pairs for t in (z["premise"],z["hypothesis"])]));norms={t:normalize(t,subjects) for t in texts};valid=[t for t in texts if norms[t] is not None]
    ref={key(z["text"]) for z in tr+dv};ck(not {key(norms[t][1]) for t in valid}&ref,"NORMALIZED_EXAM_OVERLAP")
    ck(all(exact.relation({"schema":"janus.relation.v1","facts":z["facts"],"claim":z["claim"]})==z["label"] for z in pairs),"PAIR_REFERENCE")
    def measure(probs,policy,cpu):
        lookup=dict(zip(valid,probs))
        def compile(t,guarded):
            if t not in lookup:return None
            j,c,g=candidate(lookup[t])
            if j==HOLD or (guarded and (policy is None or c<policy["confidence"] or g<policy["margin"])):return None
            a,v=STATES[j];return {"subject":norms[t][0],"attribute":a,"value":v}
        def assess(guarded):
            a=[compile(z["text"],guarded) for z in known];u=[compile(t,guarded) for t in unknown];answers=[]
            for z in pairs:
                p,q=compile(z["premise"],guarded),compile(z["hypothesis"],guarded)
                answers.append(exact.relation({"schema":"janus.relation.v1","facts":[p],"claim":q}) if p is not None and q is not None else None)
            return {"facts_exact":sum(x==z["fact"] for x,z in zip(a,known)),"facts_n":len(known),"facts_emitted":sum(x is not None for x in a),"fact_errors":sum(x is not None and x!=z["fact"] for x,z in zip(a,known)),"unknown_accepted":sum(x is not None for x in u),"unknown_n":len(unknown),"pairs_correct":sum(x==z["label"] for x,z in zip(answers,pairs)),"pairs_answered":sum(x is not None for x in answers),"pairs_wrong":sum(x is not None and x!=z["label"] for x,z in zip(answers,pairs)),"pairs_n":len(pairs)},{"facts":a,"unknown":u,"pairs":answers}
        raw,rd=assess(False);guarded,gd=assess(True)
        return {"raw":raw,"guarded":guarded,"cpu_seconds":cpu,"policy":policy},{"raw":rd,"guarded":gd}
    normalized=[norms[t][1] for t in valid];lp,lc=lexical(tr,normalized);lex,ld=measure(lp,rec["lexical_policy"],lc);runs=[];details=[{"mode":"lexical",**ld}]
    for z in rec["runs"]:
        mode=z["mode"];p=trained/("%s-%d.safetensors"%(mode,z["seed"]));ck(h29.sha(p)==z["sha"],"WEIGHT_PIN");m=Compiler(mode);m.load_state_dict(load_file(str(p)));m.eval()
        probs,cpu=predict(m,mode,normalized);metric,d=measure(probs,z["policy"],cpu);runs.append({"mode":mode,"seed":z["seed"],"parameters":z["parameters"],"bytes":z["bytes"],**metric});details.append({"mode":mode,"seed":z["seed"],**d})
    best=next(z for z in runs if z["mode"]==rec["selected"]["mode"] and z["seed"]==rec["selected"]["seed"]);ps=[(z["premise"],z["hypothesis"]) for z in pairs];ys=[z["label"] for z in pairs]
    parent=read(h31trained/"receipt.json");seed=parent["selected"]["distill"];old=next(z for z in parent["runs"] if z["mode"]=="distill" and z["seed"]==seed);p=h31trained/("distill-%d.safetensors"%seed);ck(h29.sha(p)==old["sha"],"H31_STUDENT_PIN");m=h31.Attention();m.load_state_dict(load_file(str(p)));m.eval();op,oc=h31.predict(m,"distill",ps)
    sem=h29.Semantic(demi,source);ck(sem.hashes==parent["teacher_hashes"],"TEACHER_PIN");start=time.process_time();tp=sem.pairs(ps);tc=time.process_time()-start
    g=best["guarded"];safe=g["fact_errors"]==0 and g["unknown_accepted"]==0 and g["pairs_wrong"]==0
    result={"schema":"r84h32.result.v1","motto":MOTTO,"selected":best,"all_runs":runs,"lexical":lex,"direct_h31":{"metrics":h30.metrics(op,ys),"cpu_seconds":oc},"teacher":{"metrics":h30.metrics(tp,ys),"cpu_seconds":tc},"normalized_overlap":0,"distinct_input_texts":len(texts),"subject_invalid":len(texts)-len(valid),"oracle_structured_pairs":len(pairs),"safety":"PASS_SHADOW" if safe and g["pairs_answered"]>=12 else "HOLD","improvement":"PASS_SHADOW" if safe and g["pairs_correct"]>h30.metrics(op,ys)["correct"] else "HOLD_NO_PROVEN_GAIN","production":False,"scope":"bounded compiler for six authored binary attributes with supplied subject inventory and exact unique literal subject matching. Not arbitrary NER or truth verification. Deduplicated H31 structured frames and new same-author paraphrases, heldout templates, same-author frozen exam; pair statements reuse extraction rows; exact solver isolated from uncertain extraction. No live gateway changes or automatic admission."}
    write(out/"result.json",result);write(out/"cases.json",details);log("R84H32_RESULT",result)
if __name__=="__main__":{"train":train,"evaluate":evaluate,"history":h30.history}[sys.argv[1]](*map(Path,sys.argv[2:]))
