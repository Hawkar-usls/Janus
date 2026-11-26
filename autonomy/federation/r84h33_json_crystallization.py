"""H33: bounded property/value compiler with native Cortex and selective verifier."""
import sys,re,json,time,random,math,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
import r84h32_json_semantic_compiler as h32
import r84h31_semantic_evolution as h31
import r84h30_semantic_inheritance as h30
import r84h29_semantic_resonance as h29
import r84h28_evidence_mastery as h28
import r84h27_trinity_critic as h27
import r84h19_exploration as h19
import r8_learning as r
import verified_capabilities as exact
HERE=Path(__file__).parent
EXAM="02e40d9360d717b0c8f3e5e8f420bdc5003e272b"
ATTR=h32.ATTR
STATES=h32.STATES
HOLD=h32.HOLD
ck=h32.ck;read=h32.read;write=h32.write;log=h32.log;digest=h32.digest
# Authored contrast variants. One explicit subject. Labels derive from a finite
# state table, not from the teacher. No automatic truth or source admission.
MORE={
"cached_available":{0:["The nearby cache cannot provide {s}; the authoritative store is asked for it.","An attempt to read {s} locally fails and the value is fetched from its backing service.","The local entry for {s} was evicted, requiring a fresh fetch from the database.","The fast tier reports a miss for {s} and the request continues to its origin.","The requested {s} is not cached, so its source must be consulted.","The buffer lacks {s}; the backing storage returns the missing value.","The application must retrieve {s} remotely because no local entry remains.","For {s}, local lookup is unsuccessful and the authoritative copy supplies the read."],1:["A successful local lookup returns {s} directly from the cache.","The fast-access copy of {s} is ready and the read is answered locally.","The application retrieves {s} from a previously cached value rather than the original database.","The saved entry for {s} satisfies the read without contacting the backing service.","A hit in local storage provides {s} immediately.","The value of {s} is already in the cache, so no upstream retrieval is necessary.","The buffer contains the needed {s} and supplies it directly.","For {s}, the read uses the existing local copy, avoiding a source request."]},
"persist_first":{0:["The client of {s} is told the write succeeded before the modified data reaches disk.","A write-back buffer acknowledges {s} while its change is only in RAM.","The dirty value of {s} is flushed to disk after the success reply.","The request modifying {s} completes for its caller prior to durability.","The acknowledgement of {s} is issued before a deferred persistent write.","An asynchronous writer persists {s} after the original edit has completed for the caller.","The change to {s} is buffered, acknowledged, and only later written to durable media.","The application can proceed after updating {s}, although the permanent copy is not yet updated."],1:["The response for {s} is issued after the changed data has been written to persistent storage.","A write-through update of {s} persists the new value before completion is acknowledged.","The backing disk receives the edit to {s} as part of the original operation.","The write request for {s} waits for durable storage before reporting success.","The persistent write of {s} completes before the success reply is sent.","The modification to {s} is stored synchronously in both layers before the caller continues.","No success acknowledgement for {s} is sent until its changed bytes are durable.","The service finishes the backing-store write of {s} and then acknowledges completion."]},
"multi_holder":{0:["The access lock for {s} admits exactly one thread at a time.","Two distinct tasks cannot possess the same exclusive permit for {s} concurrently.","The mutex protecting {s} excludes every worker except its single owner.","While a worker owns {s}, all other potential holders must wait.","The sole admission token for {s} is exclusively held by one caller.","Access to {s} allows one active holder and no second holder.","A single-holder gate restricts entry to {s} until its owner leaves.","The controller for {s} does not permit two simultaneous owners."],1:["The semaphore for {s} has a count of four, allowing four concurrent holders.","A finite token pool controls {s}; several callers may acquire separate tokens simultaneously.","The admission counter for {s} supports multiple active users within its capacity.","Each caller of {s} takes one of five permits and returns it after use.","Workers may share access to {s} concurrently if they hold different available permits.","There are three available admission slots for {s}, not a single exclusive lock.","More than one task can occupy {s} at the same time using separate tickets.","The resource {s} uses counting permits to bound the number of simultaneous holders."]},
"full_replica":{0:["The database {s} is partitioned into distinct subsets across servers.","Each node holds a different range of records from {s} rather than its full contents.","The records of {s} are split among machines with no overlap between their assigned portions.","A server stores only its own shard of {s}, never the entire collection.","Separate storage nodes retain different pieces of {s}.","The dataset {s} uses disjoint partitions, not complete replicated copies.","One machine keeps one part of {s} and another keeps a different part.","No machine contains the entire collection {s}; the pieces are distributed."],1:["Multiple servers retain matching complete copies of {s}.","The entire dataset {s} is maintained redundantly on every participating machine.","Each storage node has all records of {s} in an equivalent logical copy.","The full contents of {s} are replicated rather than divided into distinct shards.","A spare machine retains the complete collection {s}, identical to the primary copy.","The redundant database nodes each keep the whole dataset {s}.","The collection {s} is duplicated in its entirety on separate servers.","All participating servers store the same full logical record set {s}."]},
"newer_recovery":{0:["The failing release of {s} is replaced by its previous working build.","The service rolls {s} back to the stable version used before the upgrade.","Recovery reverses the deployment of {s} by reinstalling an older release.","The remedy for {s} restores a prior software revision, not a new patch.","After the update fails, the previous version of {s} is installed again.","The service repairs the failed deployment by going back to the older build of {s}.","The deployment system reverts {s} to its last known good predecessor.","The fix for the failed release of {s} is restoration of an earlier version."],1:["The failing release of {s} is replaced by a subsequent corrected build.","The service rolls {s} forward to a newer patched version.","Recovery deploys a new repaired revision of {s}, without restoring the previous release.","The remedy for {s} advances to a corrective update rather than reverting.","After the update fails, a later fixed version of {s} is installed.","The service repairs the failure by moving ahead to a new build of {s}.","The deployment system applies a fresh corrective patch to {s}.","The fix for the failed release of {s} is installation of a newer revision."]},
"all_or_none":{0:["The grouped edits of {s} can leave a partial committed result.","A transaction involving {s} permits one of its paired changes to survive alone.","The related modifications of {s} can be committed separately.","A failure while updating {s} may leave some changes applied and others unapplied.","The operation on {s} does not require all linked edits to commit as one unit.","One half of the paired update to {s} may commit while the other is rejected.","The changes to {s} need not succeed together; partial persistence is permitted.","The linked operation on {s} allows a committed subset after interruption."],1:["The grouped edits of {s} cannot leave a partial committed result.","The transaction involving {s} requires its paired changes to survive together or not at all.","The related modifications of {s} must be committed as one unit.","A failure while updating {s} leaves neither linked change committed.","The operation on {s} requires all linked edits to commit together or none to commit.","Neither half of the paired update to {s} may commit alone.","The changes to {s} succeed together or are entirely discarded.","The linked operation on {s} forbids a committed subset after interruption."]}}
MORENEG=["The cache index for {s} is sorted by insertion time.","The file containing {s} has a checksum.","The backing store of {s} supports encryption.","The memory allocator handling {s} merges adjacent free blocks.","The software release for {s} has a semantic version number.","The database containing {s} exposes a query planner.","The resource {s} is identified by a hexadecimal handle.","The transaction log for {s} contains a timestamp.","The records of {s} are encoded in JSON.","The disk storing {s} uses parity.","The service processing {s} applies a compression algorithm.","The cache for {s} uses a random eviction strategy.","The network path to {s} is routed using a prefix table.","The semaphore flag for {s} is red.","The state of {s} has not been specified.","The local availability of {s} is unknown.","No information establishes the order of confirmation and persistence for {s}.","The controller of {s} may grant exclusive or shared access; the description does not specify which.","The distribution strategy of {s} is not known.","The recovery plan of {s} does not specify which release will be installed.","The operation of {s} has an unspecified commit policy.","It is possible that {s} is cached, but no read result is given.","Whether {s} is durable when acknowledgement arrives is not established.","The description says nothing about how many holders {s} permits."]
def scope(t):
    # Hold explicit uncertainty and clearly mixed assertions; these controls
    # are deliberately conservative, not an arbitrary semantic proof.
    return not bool(re.search(r"\b(?:unknown|unclear|unspecified|not (?:known|established|specified|determined)|might|whether|may or may not|simultaneously present and absent|both present and absent|both before and after|compare|hypothetically|could (?:either|store)|either synchronous or deferred)\b",t,re.I))
def dataset(old):
    p=read(old/"receipt.json");tr=read(old/"train.json");dev=read(old/"dev.json")
    ck(p["train_sha"]==digest(tr) and p["dev_sha"]==digest(dev) and p["train_n"]==116 and not p["exam_read"],"H32_PIN")
    for a,vals in MORE.items():
        for v,forms in vals.items():
            tr += [h32.row(t.format(s="trainingobject"),"trainingobject",a,v) for t in forms]
    tr += [h32.row(t.format(s="trainingobject"),"trainingobject","HOLD",0) for t in MORENEG]
    tr=h32.unique(tr);keys={h32.key(z["text"]) for z in tr}
    dev=[z for z in dev if h32.key(z["text"]) not in keys]
    ck(len(tr)>=215 and len(dev)>=28 and set(z["label"] for z in dev)==set(range(len(STATES))),"CORPUS")
    return tr,dev
class Split(nn.Module):
    def __init__(self):
        super().__init__();self.word=nn.Embedding(h32.BUCKETS,40,padding_idx=0)
        self.trunk=nn.Sequential(nn.Linear(40,96),nn.GELU(),nn.LayerNorm(96))
        self.attr=nn.Linear(96,len(ATTR)+1);self.values=nn.Linear(96,len(ATTR)*2)
    def forward(self,x):
        m=x.ne(0).unsqueeze(-1);v=(self.word(x)*m).sum(1)/m.sum(1).clamp_min(1);z=self.trunk(v)
        return self.attr(z),self.values(z).reshape(-1,len(ATTR),2)
def probabilities(m,texts):
    start=time.process_time();x=h32.inputs(texts,"bag");out=[]
    with torch.inference_mode():
        for i in range(0,len(x),32):
            a,v=m(x[i:i+32]);a=a.softmax(-1);v=v.softmax(-1);joint=(a[:,:-1,None]*v).flatten(1);out+=torch.cat((joint,a[:,-1:]),1).tolist()
    return out,time.process_time()-start
def fit(tr,seed):
    torch.manual_seed(seed);m=Split();opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001);gen=torch.Generator().manual_seed(seed+1)
    x=h32.inputs([z["text"] for z in tr],"bag");y=torch.tensor([z["label"] for z in tr]);by=[torch.where(y==i)[0] for i in range(len(STATES))]
    start=time.process_time()
    for i in range(480):
        ix=torch.cat([k[torch.randint(len(k),(4,),generator=gen)] for k in by]);yy=y[ix];a,v=m(x[ix]);ay=yy//2;ay=torch.where(yy==HOLD,len(ATTR),ay)
        loss=F.cross_entropy(a,ay);ok=yy!=HOLD
        loss=loss+F.cross_entropy(v[ok,ay[ok]],yy[ok]%2)
        ck(bool(torch.isfinite(loss)),"LOSS");opt.zero_grad();loss.backward();opt.step()
    return m.eval(),time.process_time()-start
def crystal(demi,root,tr):
    m=h28.cortex(demi);root.mkdir();state=m.new_state();rows=[]
    for z in tr:
        text=z["text"];raw=text.encode();a,v=STATES[z["label"]]
        origin={"kind":"SYNTHETIC_CORRESPONDENCE","attribute":a,"value":v,"label":z["label"],"label_source":"AUTHORED_STRUCTURED_STATE","training_allowed":True,"external_truth_verified":False}
        ck(m.add_text(root,state,raw,origin) and not m.add_text(root,state,raw,origin),"DEDUP")
        rows.append({"text":text,"attribute":a,"value":v,"label":z["label"],"sha256":hashlib.sha256(raw).hexdigest(),"origin":origin})
    manifest=m.publish(root,state,"H33_DISPOSABLE_CORRESPONDENCE_DICTIONARY");m.write_json(root/"STATE.json",m.sealed(state));ck(len(m.load_state(root)["active"])==len(rows),"REOPEN")
    return rows,manifest
class Dictionary:
    def __init__(self,demi,root,records):
        self.m=h28.cortex(demi);self.root=root;self.state=self.m.load_state(root);self.records={z["sha256"]:z for z in records}
        ck(set(self.state["active"])==set(self.records),"DICTIONARY_PIN")
        self.cache={};self.calls=0;self.cpu=0.
    def search(self,t):
        if t in self.cache:return self.cache[t]
        start=time.process_time();words=[w for w in h32.tokens(t) if w not in h28.STOP and w!="entity"][:12];scores=[0.]*len(STATES);pointers=[None]*len(STATES)
        if words:
            q=self.m.query(self.root," ".join(words)[:120],limit=8);self.m.checked(q);self.calls+=1
            f=h32.lexical_features(t);den=math.sqrt(sum(x*x for x in f.values()))
            for hit in q["citations"]:
                z=self.records[hit["sha256"]];raw=self.m.read_object(self.root,hit["sha256"],hit["verified_bytes"])
                ck(raw.decode()==z["text"] and hit["origin"]==z["origin"] and hit["content_is_untrusted_data"],"POINTER")
                a=h32.lexical_features(z["text"]);s=sum(x*a.get(k,0) for k,x in f.items())/max(1e-12,den*math.sqrt(sum(x*x for x in a.values())))
                if s>scores[z["label"]]:scores[z["label"]]=s;pointers[z["label"]]=hit["sha256"]
        result=(scores,pointers);self.cache[t]=result;self.cpu+=time.process_time()-start;return result
def evidence(sem,texts,net,lex):
    # Only send local model a bounded pair for consensus candidates. This is
    # independent model evidence, not a definitive semantic oracle.
    indexes=[];pairs=[]
    for i,(t,n,l) in enumerate(zip(texts,net,lex)):
        j,_,_=h32.candidate(n);k,_,_=h32.candidate(l[0])
        if j==HOLD or j!=k or not scope(t) or l[1][j] is None:continue
        a,v=STATES[j];indexes.append(i);pairs.extend([(t,h31.HYP[a][v][0].format(o="entity")),(t,h31.HYP[a][1-v][0].format(o="entity"))])
    vals=sem.pairs(pairs) if pairs else [];out={}
    for k,i in enumerate(indexes):out[i]={"support":vals[2*k][1],"opposite_contradiction":vals[2*k+1][0],"opposite_entailment":vals[2*k+1][1]}
    return out
def select(t,n,l,e,policy):
    j,c,g=h32.candidate(n);k,s,m=h32.candidate(l[0]);proof={"neural":j,"lexical":k,"dictionary_sha256":l[1][k],"neural_score":c,"neural_margin":g,"lexical_score":s,"lexical_margin":m,"semantic":e,"scope":scope(t)}
    if policy is None or j==HOLD or j!=k or l[1][k] is None or not proof["scope"] or e is None:return None,proof
    if c<policy["confidence"] or g<policy["margin"] or s<policy["lexical"] or e["support"]<policy["semantic"] or e["opposite_contradiction"]<policy["opposite"] or e["opposite_entailment"]>.20:return None,proof
    return j,proof
def validate(rows,n,l,e,rule):
    result={"correct":0,"wrong":0,"unknown_accepted":0,"known_n":0,"unknown_n":0,"raw_exact":0,"raw_attribute":0}
    for i,z in enumerate(rows):
        truth=z["label"];j,_,_=h32.candidate(n[i]);v,_=select(z["text"],n[i],l[i],e.get(i),rule)
        if truth==HOLD:result["unknown_n"]+=1;result["unknown_accepted"]+=v is not None
        else:result["known_n"]+=1;result["raw_exact"]+=j==truth;result["raw_attribute"]+=STATES[j][0]==STATES[truth][0];result["correct"]+=v==truth;result["wrong"]+=v is not None and v!=truth
    return result
def calibrate(rows,n,l,e):
    opts=[]
    for c in (.35,.5,.7,.85,.95):
        for g in (.05,.15,.3):
            for s in (.2,.35,.5):
                for v in (.55,.75,.9):
                    p={"confidence":c,"margin":g,"lexical":s,"semantic":v,"opposite":v};m=validate(rows,n,l,e,p)
                    if m["wrong"]==0 and m["unknown_accepted"]==0 and m["correct"]>=3:opts.append((m["correct"],c,g,s,v,p))
    return max(opts,key=lambda z:z[:5])[-1] if opts else None
def train(demi,source,old,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);ck(r.blob((HERE/"verified_capabilities.py").read_bytes())==h32.SOLVER,"SOLVER")
    tr,dev=dataset(old);records,manifest=crystal(demi,out/"cortex",tr);dictionary=Dictionary(demi,out/"cortex",records)
    texts=[z["text"] for z in dev];lex=[dictionary.search(t) for t in texts];sem=h29.Semantic(demi,source);controls=h27.trinity(demi)
    runs=[]
    for seed in (331,332):
        m,cpu=fit(tr,seed);p,pcpu=probabilities(m,texts);ev=evidence(sem,texts,p,lex);rule=calibrate(dev,p,lex,ev);metric=validate(dev,p,lex,ev,rule);path=out/("split-%d.safetensors"%seed);save_file(m.state_dict(),str(path));params=sum(z.numel() for z in m.parameters());ck(params<=250000,"PARAMETERS")
        runs.append({"seed":seed,"sha":h29.sha(path),"parameters":params,"bytes":path.stat().st_size,"train_cpu":cpu,"dev_cpu":pcpu,"dev":metric,"policy":rule})
    best=max(runs,key=lambda z:(z["dev"]["correct"],z["dev"]["raw_exact"],-z["seed"]))
    receipt={"schema":"r84h33.train.v1","train_n":len(tr),"dev_n":len(dev),"train_sha":digest(tr),"dev_sha":digest(dev),"dictionary_sha":digest(records),"native_cortex_documents":manifest["indexed_documents"],"native_cortex_queries":dictionary.calls,"native_cortex_cpu":dictionary.cpu,"teacher_hashes":sem.hashes,"native_trinity_controls":controls,"runs":runs,"selected":best["seed"],"exam_read":False,"production":False}
    write(out/"train.json",tr);write(out/"dev.json",dev);write(out/"correspondences.json",records);write(out/"receipt.json",receipt);log("R84H33_TRAIN",receipt)
def evaluate(demi,source,old,trained,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);rec=read(trained/"receipt.json");tr=read(trained/"train.json");dev=read(trained/"dev.json");records=read(trained/"correspondences.json")
    ck(rec["train_sha"]==digest(tr) and rec["dev_sha"]==digest(dev) and rec["dictionary_sha"]==digest(records) and not rec["exam_read"] and not rec["production"],"TRAIN_PIN")
    ex=h19.load_exam("R84H33_FROZEN_EXAM.json",EXAM);previous=h19.load_exam("R84H32_FROZEN_EXAM.json",h32.EXAM)
    dictionary=Dictionary(demi,trained/"cortex",records);sem=h29.Semantic(demi,source);ck(sem.hashes==rec["teacher_hashes"],"TEACHER_PIN")
    def corpus(ex):
        texts=list(dict.fromkeys([z["text"] for z in ex["statements"]]+ex["unknown"]+[t for z in ex["pairs"] for t in (z["premise"],z["hypothesis"])]))
        norms={t:h32.normalize(t,ex["subjects"]) for t in texts};valid=[t for t in texts if norms[t] is not None]
        return texts,norms,valid
    _,oldnorm,oldvalid=corpus(previous);texts,norms,valid=corpus(ex);ref={h32.key(z["text"]) for z in tr+dev}|{h32.key(oldnorm[t][1]) for t in oldvalid}
    ck(not {h32.key(norms[t][1]) for t in valid}&ref,"EXAM_OVERLAP")
    ck(all(exact.relation({"schema":"janus.relation.v1","facts":z["facts"],"claim":z["claim"]})==z["label"] for z in ex["pairs"]),"PAIR_ORACLE")
    z=next(x for x in rec["runs"] if x["seed"]==rec["selected"]);path=trained/("split-%d.safetensors"%z["seed"]);ck(h29.sha(path)==z["sha"],"WEIGHT")
    m=Split();m.load_state_dict(load_file(str(path)));m.eval()
    def assess(ex):
        texts,norms,valid=corpus(ex);ts=[norms[t][1] for t in valid];net,cpu=probabilities(m,ts);lex=[dictionary.search(t) for t in ts];ev=evidence(sem,ts,net,lex);ix={t:i for i,t in enumerate(valid)}
        def compile(t,mode):
            if t not in ix:return None,{"status":"HOLD_SUBJECT"}
            i=ix[t];j,proof=select(ts[i],net[i],lex[i],ev.get(i),z["policy"])
            if mode=="neural":j=h32.candidate(net[i])[0]
            if mode=="lexical":j=h32.candidate(lex[i][0])[0]
            if mode=="agreement":
                j=h32.candidate(net[i])[0];k=h32.candidate(lex[i][0])[0]
                if j!=k or lex[i][1][k] is None or not scope(ts[i]):j=None
            if j is None or j==HOLD:return None,proof
            a,v=STATES[j];return {"subject":norms[t][0],"attribute":a,"value":v},proof
        metrics={};details={};confirmed=[]
        for mode in ("neural","lexical","agreement","guarded"):
            a=[compile(z0["text"],mode) for z0 in ex["statements"]];u=[compile(t,mode) for t in ex["unknown"]];answers=[]
            for q in ex["pairs"]:
                p,_=compile(q["premise"],mode);s,_=compile(q["hypothesis"],mode);answers.append(exact.relation({"schema":"janus.relation.v1","facts":[p],"claim":s}) if p and s else None)
            known=ex["statements"];metrics[mode]={"facts_n":len(known),"facts_exact":sum(f==z0["fact"] for (f,_),z0 in zip(a,known)),"facts_emitted":sum(f is not None for f,_ in a),"fact_errors":sum(f is not None and f!=z0["fact"] for (f,_),z0 in zip(a,known)),"unknown_n":len(u),"unknown_accepted":sum(f is not None for f,_ in u),"pairs_n":len(answers),"pairs_answered":sum(x is not None for x in answers),"pairs_correct":sum(x==q["label"] for x,q in zip(answers,ex["pairs"])),"pairs_wrong":sum(x is not None and x!=q["label"] for x,q in zip(answers,ex["pairs"]))}
            details[mode]={"facts":a,"unknown":u,"answers":answers}
            if mode=="guarded":
                for (f,proof),truth in zip(a,known):
                    if f==truth["fact"]:confirmed.append({"text":truth["text"],"fact":f,"proof":proof,"status":"CONFIRMED_ON_SYNTHETIC_EXAM","automatic_training":False,"production":False})
        return metrics,details,confirmed,cpu
    prior,_,_,_=assess(previous);fresh,details,confirmed,cpu=assess(ex);g=fresh["guarded"];safe=g["fact_errors"]==0 and g["unknown_accepted"]==0 and g["pairs_wrong"]==0
    report={"schema":"r84h33.result.v1","fresh":fresh,"disclosed_h32":prior,"selected_seed":z["seed"],"parameters":z["parameters"],"train_n":rec["train_n"],"dictionary_documents":len(records),"cortex_queries":dictionary.calls,"cortex_cpu":dictionary.cpu,"neural_cpu":cpu,"native_trinity_controls":h27.trinity(demi),"new_confirmed_correspondences":len(confirmed),"normalized_overlap":0,"structured_oracle":len(ex["pairs"]),"safety":"PASS_SHADOW" if safe and g["pairs_answered"]>=12 else "HOLD","production":False,"scope":"six attributes, explicit subjects, same-author synthetic facts and holdout, native Cortex disposable dictionary with hash-checked retrieval, two-head neural compiler, local MiniLM support/opposite checks; native TRINITY arithmetic controls only, no general semantic proof, no live corpus or gateway writes. Confirmed exam correspondences quarantined from training."}
    write(out/"result.json",report);write(out/"cases.json",details);write(out/"confirmed_correspondences.json",confirmed);log("R84H33_RESULT",report)
if __name__=="__main__":{"train":train,"evaluate":evaluate,"history":h30.history}[sys.argv[1]](*map(Path,sys.argv[2:]))
