"""H37 standalone verifier; no MiniLM, Transformers, network, or teacher labels."""
import sys,re,json,hashlib,time,socket,os,subprocess
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
import verified_capabilities as exact
ATTR=tuple(sorted(("cached_available","persist_first","multi_holder","full_replica","newer_recovery","all_or_none")))
A=len(ATTR);STATES=[(a,v) for a in ATTR for v in (0,1)]+[("HOLD",0)];HOLD=len(STATES)-1
EXAM="5eb12d1ec264521418f5e59e663d3977e3b4706d"
def ck(v,m):
    if not v:raise RuntimeError(m)
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,sort_keys=True,ensure_ascii=True)+"\n")
def log(t,x):print(t,json.dumps(x,sort_keys=True),flush=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def blob(p):
    raw=p.read_bytes();return hashlib.sha1(("blob %d\0"%len(raw)).encode()+raw).hexdigest()
def tokens(t):return re.findall(r"[a-z0-9]+",t.lower())
def key(t):return " ".join(tokens(t))
def normalize(t,subjects):
    if not isinstance(t,str) or not 1<=len(t)<=600:return None
    hits=[(s,r"(?<!\w)"+re.escape(s)+r"(?!\w)") for s in subjects if re.search(r"(?<!\w)"+re.escape(s)+r"(?!\w)",t,re.I)]
    if len(hits)!=1:return None
    s,rx=hits[0];return s," ".join(re.sub(rx," ENTITY ",t,flags=re.I).lower().split())
def scope(t):
    return not bool(re.search(r"\b(?:unknown|unclear|unspecified|not (?:known|established|specified|determined|reported|stated|given)|might|whether|may or may not|simultaneously present and absent|both present and absent|both before and after|compare|hypothetically|could (?:either|store)|either synchronous or deferred)\b",t,re.I))
def offline():
    ck("transformers" not in sys.modules and "huggingface_hub" not in sys.modules,"TEACHER_IMPORT")
    # Witness the actual process namespace, not the line count of a procfs file.
    # Fixed read-only iproute2 commands run in the same namespace.
    host=os.environ.get("JANUS_PARENT_NETNS","")
    current=os.readlink("/proc/self/ns/net")
    interfaces=socket.if_nameindex()
    def ip(*args):
        p=subprocess.run(["ip","-j",*args],check=True,capture_output=True,text=True,timeout=5)
        return json.loads(p.stdout)
    v4=ip("-4","route","show","table","all")
    v6=ip("-6","route","show","table","all")
    witness={"parent_namespace":host,"process_namespace":current,"uid":os.geteuid(),"interfaces":[name for _,name in interfaces],"ipv4_routes":v4,"ipv6_routes":v6}
    log("R84H37B_NETWORK_WITNESS",witness)
    ck(bool(re.fullmatch(r"net:\[\d+\]",host)) and current!=host,"NETWORK_NAMESPACE_NOT_DISTINCT")
    ck(os.geteuid()!=0,"NETWORK_PRIVILEGE")
    ck([name for _,name in interfaces]==["lo"],"NETWORK_NONLOOPBACK_INTERFACE")
    ck(all(z.get("dev")=="lo" and z.get("dst") not in ("default","0.0.0.0/0","::/0") and "gateway" not in z for z in v4+v6),"NETWORK_EXTERNAL_ROUTE")
    def denied(*args,**kwargs):raise RuntimeError("NETWORK_DISABLED")
    socket.create_connection=denied;socket.socket.connect=denied;socket.socket.connect_ex=denied
    torch.set_num_threads(2)
B=2048
def hid(s):return int.from_bytes(hashlib.blake2s(s.encode(),digest_size=4).digest(),"little")%B
def vector(t):
    w=tokens(t);features=["w:"+x for x in w]+["b:"+a+"_"+b for a,b in zip(w,w[1:])]+["t:"+a+"_"+b+"_"+c for a,b,c in zip(w,w[1:],w[2:])]
    features += ["c:"+x[j:j+3] for x in w for j in range(max(0,len(x)-2))]
    features += ["pos:"+str(min(3,4*i//max(1,len(w))))+":"+x for i,x in enumerate(w)]
    v=torch.zeros(B)
    for s in features:v[hid(s)]+=1
    v=torch.log1p(v);return v/v.norm().clamp_min(1e-8)
def inputs(ts):return torch.stack([vector(t) for t in ts])
class Verifier(nn.Module):
    def __init__(self):
        super().__init__();self.trunk=nn.Sequential(nn.Linear(B,64),nn.LayerNorm(64),nn.GELU());self.attr=nn.Linear(64,A);self.val=nn.Linear(64,A*2)
    def forward(self,x):
        v=self.trunk(x);return self.attr(v),self.val(v).reshape(-1,A,2)
def fit(tr,seed):
    torch.manual_seed(seed);m=Verifier();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.002);gen=torch.Generator().manual_seed(seed+1);x=inputs([z["text"] for z in tr]);y=torch.tensor([z["label"] for z in tr]);groups=[torch.where(y==i)[0] for i in range(len(STATES))];start=time.process_time()
    for i in range(420):
        ix=torch.cat([z[torch.randint(len(z),(4,),generator=gen)] for z in groups]);yy=y[ix];a,v=m(x[ix]);known=yy!=HOLD;ai=yy[known]//2;vv=yy[known]%2
        # All other properties and unknown texts are explicit negatives.
        at=torch.zeros_like(a);at[known,ai]=1;target=torch.zeros((len(yy),A*2));target[known,yy[known]]=1
        joint=(a.sigmoid().unsqueeze(-1)*v.softmax(-1)).flatten(1)
        loss=F.binary_cross_entropy_with_logits(a,at)+F.cross_entropy(v[known,ai],vv)+F.binary_cross_entropy(joint.clamp(1e-7,1-1e-7),target)
        ck(bool(torch.isfinite(loss)),"LOSS");opt.zero_grad();loss.backward();opt.step()
    return m.eval(),time.process_time()-start
def probabilities(models,ts):
    start=time.process_time();x=inputs(ts);allp=[]
    with torch.inference_mode():
        for m in models:
            a,v=m(x);allp.append((a.sigmoid(),v.softmax(-1)))
    # conservative min support across two separately seeded students
    ap=torch.stack([a for a,v in allp]).min(0).values;vp=torch.stack([v for a,v in allp]).min(0).values
    return [{"attribute":a,"value":v} for a,v in zip(ap.tolist(),vp.tolist())],time.process_time()-start
def supports(p,j,rule):
    if rule is None or j==HOLD:return False
    a,v=divmod(j,2);return p["attribute"][a]>=rule["attribute"] and p["value"][a][v]>=rule["value"] and p["value"][a][v]-p["value"][a][1-v]>=rule["margin"]

# New authored minimal contrasts; labels are authored state facts, not NLI.
# Separate unknown (third value) for every property. Each paired rendering
# differs in polarity/order, with no exam text admitted.
BANK={
"cached_available":[["The cache does not contain entity, requiring a remote read.","The cache contains entity, avoiding a remote read."],["The local result for entity is a miss.","The local result for entity is a hit."],["The backing service supplies entity after a cache miss.","The cache supplies entity after a successful lookup."],["The cached entry for entity is unavailable.","The cached entry for entity is available."],["A request for entity needs a source fetch because the cache is empty.","A request for entity needs no source fetch because the cache has it."]],
"persist_first":[["Acknowledgement of entity occurs before durability.","Durability of entity occurs before acknowledgement."],["Entity is confirmed first and persisted second.","Entity is persisted first and confirmed second."],["The update to entity becomes durable after success is reported.","The update to entity becomes durable before success is reported."],["Entity is still dirty in RAM at confirmation.","Entity is already on disk at confirmation."],["The acknowledgement precedes the disk write of entity.","The disk write of entity precedes acknowledgement."]],
"multi_holder":[["Entity admits no more than one holder.","Entity admits more than one holder."],["Entity allows only one worker to hold a permit.","Entity allows several workers to hold permits."],["Entity excludes a concurrent second owner.","Entity permits a concurrent second owner."],["One exclusive permit governs entity.","A pool of independent permits governs entity."],["Entity does not allow simultaneous callers.","Entity allows simultaneous callers."]],
"full_replica":[["Every node stores only a slice of entity.","Every node stores all of entity."],["Entity is divided into disjoint ranges across nodes.","Entity is duplicated in full across nodes."],["Each machine stores a different partition of entity.","Each machine stores an identical complete copy of entity."],["No host stores the complete dataset entity.","Each host stores the complete dataset entity."],["Entity has separate nonoverlapping pieces on distinct servers.","Entity has redundant complete copies on distinct servers."]],
"newer_recovery":[["Recovery of entity goes back to an earlier working version.","Recovery of entity goes forward to a later corrected version."],["The failed release of entity is rolled back.","The failed release of entity is repaired by a roll-forward."],["Entity restores its old build after failure.","Entity installs a newer patched build after failure."],["The remedy reinstates the previous version of entity.","The remedy deploys the next fixed version of entity."],["The recovery reuses the earlier release of entity.","The recovery uses a subsequent repaired release of entity."]],
"all_or_none":[["Entity permits a partial commit.","Entity prohibits a partial commit."],["Entity allows one linked edit to commit alone.","Entity forbids one linked edit to commit alone."],["The edits of entity can commit separately.","The edits of entity must commit together."],["Entity can retain only some linked changes after failure.","Entity cannot retain only some linked changes after failure."],["The transaction on entity may leave a committed subset.","The transaction on entity may not leave a committed subset."]]}
UNKNOWN=["The cache for entity has an eviction policy.","The cache state of entity is not specified.","Entity may or may not have a local copy.","The timing of confirmation for entity is unspecified.","Entity has a durable storage medium but no acknowledgement is described.","Entity may persist before or after confirmation.","Entity has a permit counter but the count is not given.","Entity has a token identifier.","The permitted number of owners of entity is unclear.","Entity has backup records on storage nodes.","The distribution of entity is unspecified.","Entity may be copied or partitioned.","Entity has a release number.","No recovery choice for entity is reported.","Entity may return to an old release or install a new release.","Entity has a transaction log.","The commit policy of entity is not established.","Entity may commit together or separately.","The description of entity is contradictory.","Entity is an object in the database.","Entity is cached and not cached at the same time.","Entity has a configuration file.","The data for entity is encrypted.","Entity has a unique identifier."]
DEV={"cached_available":["For entity the local lookup fails and the source responds.","For entity the local lookup succeeds and supplies the content."],"persist_first":["The client receives confirmation for entity prior to the permanent write.","The permanent write for entity occurs prior to client confirmation."],"multi_holder":["The resource entity admits a single owner at once.","The resource entity admits multiple owners at once."],"full_replica":["Distinct nodes carry separate fragments of entity.","Distinct nodes carry the whole collection entity."],"newer_recovery":["After failure, entity returns to the prior working package.","After failure, entity advances to a subsequently repaired package."],"all_or_none":["A linked modification of entity can remain committed without its companion.","A linked modification of entity cannot remain committed without its companion."]}
DEV_UNKNOWN=["The local lookup policy for entity is recorded but no result is provided.","The write for entity uses a disk but its completion order is not stated.","The permit for entity is represented by a number but no holder limit is given.","Entity has storage nodes but its data distribution is not stated.","Entity has a recovery procedure but its chosen revision is not reported.","Entity has linked edits but their commit relationship is unspecified."]
def row(t,y):
    a,v=STATES[y];return {"text":t.lower(),"label":y,"attribute":a,"value":v}
def unique(rows):
    d={}
    for z in rows:
        k=key(z["text"])
        if k in d:ck(d[k]["label"]==z["label"],"DUPLICATE_CONFLICT")
        else:d[k]=z
    return list(d.values())
def data(trained):
    rec=read(trained/"receipt.json");tr=read(trained/"train.json");dev=read(trained/"dev.json");records=read(trained/"correspondences.json")
    ck(rec["train_sha"]==digest(tr) and rec["dev_sha"]==digest(dev) and rec["dictionary_sha"]==digest(records) and len(tr)==len(records)==236 and not rec["exam_read"] and not rec["production"],"CORTEX_PIN")
    for z,r in zip(tr,records):
        ck(z["text"]==r["text"] and z["label"]==r["label"] and r["origin"]["training_allowed"] and r["origin"]["label_source"]=="AUTHORED_STRUCTURED_STATE" and hashlib.sha256(r["text"].encode()).hexdigest()==r["sha256"],"CORRESPONDENCE")
    base=len(tr)
    tr=unique(tr+[row(t,STATES.index((a,v))) for a,groups in BANK.items() for pair in groups for v,t in enumerate(pair)]+[row(t,HOLD) for t in UNKNOWN])
    dev=unique(dev+[row(t,STATES.index((a,v))) for a,p in DEV.items() for v,t in enumerate(p)]+[row(t,HOLD) for t in DEV_UNKNOWN]);k={key(z["text"]) for z in tr};dev=[z for z in dev if key(z["text"]) not in k]
    return rec,tr,dev,base
class V37(nn.Module):
    def __init__(self):
        super().__init__();self.trunk=nn.Sequential(nn.Linear(B,64),nn.LayerNorm(64),nn.GELU());self.attr=nn.Linear(64,A);self.val=nn.Linear(64,A*3)
    def forward(self,x):
        v=self.trunk(x);return self.attr(v),self.val(v).reshape(-1,A,3)
def fit37(tr,seed):
    torch.manual_seed(seed);m=V37();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.002);gen=torch.Generator().manual_seed(seed+1);x=inputs([z["text"] for z in tr]);y=torch.tensor([z["label"] for z in tr]);groups=[torch.where(y==i)[0] for i in range(len(STATES))];start=time.process_time()
    for _ in range(560):
        ix=torch.cat([g[torch.randint(len(g),(5,),generator=gen)] for g in groups]);yy=y[ix];a,v=m(x[ix]);ok=yy!=HOLD;ai=yy[ok]//2;vv=yy[ok]%2
        at=torch.zeros_like(a);at[ok,ai]=1;vt=torch.full((len(yy),A),2,dtype=torch.long);vt[ok,ai]=vv
        loss=F.binary_cross_entropy_with_logits(a,at)+F.cross_entropy(v.flatten(0,1),vt.flatten())+F.cross_entropy(v[ok,ai],vv)
        ck(bool(torch.isfinite(loss)),"LOSS");opt.zero_grad();loss.backward();opt.step()
    return m.eval(),time.process_time()-start
def prob37(models,ts):
    start=time.process_time();x=inputs(ts);out=[]
    with torch.inference_mode():
        for m in models:
            a,v=m(x);out.append((a.sigmoid(),v.softmax(-1)))
    ap=torch.stack([a for a,v in out]).min(0).values;vp=torch.stack([v for a,v in out]).min(0).values;up=torch.stack([v[:,:,2] for a,v in out]).max(0).values
    return [{"attribute":a,"value":v,"uncertainty":u} for a,v,u in zip(ap.tolist(),vp.tolist(),up.tolist())],time.process_time()-start
def accepted(p,rule,new,t):
    if rule is None or not scope(t):return []
    out=[]
    for j in range(HOLD):
        if supports(p,j,rule) and (not new or p["uncertainty"][j//2]<=rule["unknown_max"]):out.append(j)
    return out
def direct(ps,ys,ts,rule,new):
    m={"true_n":0,"true_support":0,"opposite_n":0,"opposite_false_support":0,"other_attribute_n":0,"other_attribute_false_support":0,"unknown_n":0,"unknown_false_support":0}
    for p,y,t in zip(ps,ys,ts):
        got=accepted(p,rule,new,t)
        for j in range(HOLD):
            k="unknown" if y==HOLD else "true" if j==y else "opposite" if j==(y^1) else "other_attribute"
            m[k+"_n"]+=1;m[k+("_support" if k=="true" else "_false_support")]+=j in got
    return m
def calibrate(ps,ys,ts):
    opts=[]
    for a in (.50,.60,.70,.80,.90,.95,.98):
        for v in (.60,.70,.80,.90,.95,.98):
            for g in (.15,.35,.50,.70):
                for u in (.10,.20,.30,.40):
                    r={"attribute":a,"value":v,"margin":g,"unknown_max":u};d=direct(ps,ys,ts,r,True)
                    if d["opposite_false_support"]==d["other_attribute_false_support"]==d["unknown_false_support"]==0 and d["true_support"]>0:opts.append((d["true_support"],a,v,g,-u,r))
    return max(opts,key=lambda x:x[:5])[-1] if opts else None
def train(trained,out):
    offline();out.mkdir(parents=True,exist_ok=False);rec,tr,dev,base=data(trained);models=[];runs=[]
    for seed in (371,372):
        m,cpu=fit37(tr,seed);models.append(m);p=out/("verify-%d.safetensors"%seed);save_file(m.state_dict(),str(p));n=sum(z.numel() for z in m.parameters());ck(n<=150000,"PARAMETERS")
        runs.append({"seed":seed,"parameters":n,"sha":sha(p),"bytes":p.stat().st_size,"train_cpu":cpu})
    ts=[z["text"] for z in dev];ys=[z["label"] for z in dev];ps,cpu=prob37(models,ts);rule=calibrate(ps,ys,ts)
    receipt={"schema":"r84h37.train.v1","base_n":base,"train_n":len(tr),"dev_n":len(dev),"train_sha":digest(tr),"dev_sha":digest(dev),"dictionary_sha":rec["dictionary_sha"],"runs":runs,"rule":rule,"dev_direct":direct(ps,ys,ts,rule,True),"teacher_labels_used":False,"teacher_modules_loaded":False,"network_namespace_no_routes":True,"exam_read":False,"production":False}
    write(out/"receipt.json",receipt);write(out/"train.json",tr);write(out/"dev.json",dev);log("R84H37_TRAIN",receipt)
def corpus(ex):
    ts=list(dict.fromkeys([z["text"] for z in ex["statements"]]+ex["unknown"]+[t for z in ex["pairs"] for t in (z["premise"],z["hypothesis"])]))
    return {t:normalize(t,ex["subjects"]) for t in ts}
def evaluate(own,old,out):
    offline();out.mkdir(parents=True,exist_ok=False);r=read(own/"receipt.json");tr=read(own/"train.json");dev=read(own/"dev.json");ck(r["train_sha"]==digest(tr) and r["dev_sha"]==digest(dev) and not r["exam_read"],"TRAIN_PIN")
    models=[];older=[];o=read(old/"receipt.json")
    for z in r["runs"]:
        p=own/("verify-%d.safetensors"%z["seed"]);ck(sha(p)==z["sha"],"H37_WEIGHTS");m=V37();m.load_state_dict(load_file(str(p)));models.append(m.eval())
    for z in o["runs"]:
        p=old/("verify-%d.safetensors"%z["seed"]);ck(sha(p)==z["sha"],"H36_WEIGHTS");m=Verifier();m.load_state_dict(load_file(str(p)));older.append(m.eval())
    here=Path(__file__).parent;exam=here/"R84H37_FROZEN_EXAM.json";ck(blob(exam)==EXAM and blob(here/"verified_capabilities.py")=="1b131dc27b0fb5524761473e3b27c6c4db51e759","EXAM_SOLVER_PIN")
    ex=read(exam);refs={key(z["text"]) for z in tr+dev}
    priors=[]
    for n,pin in [(33,"02e40d9360d717b0c8f3e5e8f420bdc5003e272b"),(34,"582ea0e9aa40c545af77964042121f7a95eb0b0a"),(36,"69823b8919ae31c23fae2f1f150feeaefdd28027")]:
        p=here/("R84H%d_FROZEN_EXAM.json"%n);ck(blob(p)==pin,"PRIOR_EXAM_PIN");prior=read(p);priors.append((n,prior));refs|={key(x[1]) for x in corpus(prior).values() if x}
    def assess(data,exclude):
        norms=corpus(data);blocked={t for t,x in norms.items() if x and key(x[1]) in exclude};known=[z for z in data["statements"] if z["text"] not in blocked];unknown=[t for t in data["unknown"] if t not in blocked];pairs=[z for z in data["pairs"] if z["premise"] not in blocked and z["hypothesis"] not in blocked]
        ck(len(known)>=20 and len(unknown)>=18 and len(pairs)>=18,"EXAM_SIZE")
        valid=[t for t,x in norms.items() if x and t not in blocked];ts=[norms[t][1] for t in valid];p,cpu=prob37(models,ts);q,oldcpu=probabilities(older,ts)
        look={t:{"h37":accepted(a,r["rule"],True,norms[t][1]),"h36":accepted(b,o["rule"],False,norms[t][1])} for t,a,b in zip(valid,p,q)}
        metrics={}
        for mode in ("h37","h36"):
            def compile(t):
                z=look.get(t,{}).get(mode,[])
                if len(z)!=1:return None
                a,v=STATES[z[0]];return {"subject":norms[t][0],"attribute":a,"value":v}
            a=[compile(z["text"]) for z in known];u=[compile(t) for t in unknown];answers=[]
            for z in pairs:
                ck(exact.relation({"schema":"janus.relation.v1","facts":z["facts"],"claim":z["claim"]})==z["label"],"PAIR_ORACLE")
                f,g=compile(z["premise"]),compile(z["hypothesis"]);answers.append(exact.relation({"schema":"janus.relation.v1","facts":[f],"claim":g}) if f and g else None)
            m={"facts_n":len(known),"facts_correct":sum(f==z["fact"] for f,z in zip(a,known)),"fact_errors":sum(f is not None and f!=z["fact"] for f,z in zip(a,known)),"unknown_n":len(u),"unknown_accepted":sum(f is not None for f in u),"pairs_n":len(pairs),"pairs_correct":sum(x==z["label"] for x,z in zip(answers,pairs)),"pairs_wrong":sum(x is not None and x!=z["label"] for x,z in zip(answers,pairs))}
            targets=[(z["text"],STATES.index((z["fact"]["attribute"],z["fact"]["value"]))) for z in known]+[(t,HOLD) for t in unknown];targets=[(t,y) for t,y in targets if t in look];lookup=dict(zip(valid,p if mode=="h37" else q))
            m["direct"]=direct([lookup[t] for t,y in targets],[y for t,y in targets],[norms[t][1] for t,y in targets],r["rule"] if mode=="h37" else o["rule"],mode=="h37");metrics[mode]=m
        return metrics,{"accepts":look,"excluded":sorted(blocked),"cpu_h37":cpu,"cpu_h36":oldcpu}
    fresh,detail=assess(ex,refs);replay={};details={"fresh":detail}
    for n,prior in priors:replay[str(n)],details[str(n)]=assess(prior,set())
    ck("transformers" not in sys.modules and "huggingface_hub" not in sys.modules,"TEACHER_IMPORT")
    result={"schema":"r84h37.offline.v1","teacher_modules_loaded":False,"teacher_weights_loaded":False,"network_namespace_no_routes":True,"train_n":r["train_n"],"rule":r["rule"],"parameters":sum(z["parameters"] for z in r["runs"]),"fresh":fresh,"regression":replay,"excluded_exact_overlap":len(detail["excluded"]),"standalone_exact_solver":True,"production":False,"scope":"Own-only verifier and exact JSON solver, explicit subject inventory, synthetic six-property English domain; no teacher or network inside train/evaluation process; separate comparison job, no deployment."}
    write(out/"offline.json",result);write(out/"accepts.json",details);log("R84H37_OFFLINE",result)
if __name__=="__main__":{"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
