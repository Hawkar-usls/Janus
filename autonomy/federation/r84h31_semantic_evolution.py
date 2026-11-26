"""H31: ordered attention, contrast groups and structured JSON state truth."""
import sys,time,re,random,hashlib,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
import r84h30_semantic_inheritance as h30
import r84h29_semantic_resonance as h29
import r84h19_exploration as h19
import r84h18_fog as h18
HERE=Path(__file__).parent
EXAM="d0c73404d76567fe872a2e6144c7e6c0dd43b06f"
LABELS=h30.LABELS
ck=h30.ck;read=h30.read;write=h30.write;log=h30.log;digest=h30.digest
MOTTO="Мы должны узнать"
# Each fact describes one explicit binary attribute for a fixed subject.
# Absence of the attribute is unknown, not false. No prose-to-frame parser.
def relation(q):
    if type(q) is not dict or set(q)!={"schema","facts","claim"} or q["schema"]!="janus.relation.v1":return "HOLD_INVALID"
    facts=q["facts"];claim=q["claim"]
    def valid(x):
        return type(x) is dict and set(x)=={"subject","attribute","value"} and all(type(x[k]) is str and 0<len(x[k])<=80 for k in ("subject","attribute")) and type(x["value"]) is int and x["value"] in (0,1)
    if type(facts) is not list or not 1<=len(facts)<=32 or not valid(claim) or not all(valid(x) for x in facts):return "HOLD_INVALID"
    seen={}
    for x in facts:
        k=(x["subject"],x["attribute"])
        if k in seen and seen[k]!=x["value"]:return "HOLD_CONFLICT"
        seen[k]=x["value"]
    k=(claim["subject"],claim["attribute"])
    return "neutral" if k not in seen else "entailment" if seen[k]==claim["value"] else "contradiction"
# Independent binary state table. Rendering is authored and still fallible.
# First 3 families are core, next 3 extend the earlier H30 curriculum.
BANK={
"cached_available":{
0:["The fast store has no {o}; a remote fetch supplies the requested value.","The lookup for {o} fails locally, so its source is contacted.","The desired {o} is missing from the local tier and obtained from the origin.","The local request cannot be satisfied because {o} was evicted; its origin supplies it."],
1:["The temporary store already contains {o}, so it serves the request locally.","The requested {o} resides in the fast tier; the lookup returns that copy.","No origin fetch is needed: the cache supplies {o}.","The nearby buffer supplies the resident {o} without a lower-level read."]},
"persist_first":{
0:["The service confirms the edit to {o} while only volatile memory holds the modification; persistence happens later.","A later flush makes {o} durable after the success response.","The writer acknowledges {o} before committing its changes to stable media.","The caller gets success for {o} and proceeds; a later background operation persists the change."],
1:["The service delays confirmation until the updated {o} has reached stable media.","Before acknowledging the change to {o}, the writer persists it.","The backing device receives the modified {o} as part of the original operation before success.","The write of {o} reaches durable storage and then the caller receives its acknowledgement."]},
"multi_holder":{
0:["The resource {o} gives a single exclusive access token to one worker at a time.","Only one task can hold the permit for {o}; other callers are excluded.","A second worker cannot enter {o} while one worker holds its unique permit.","The access controller for {o} grants ownership to one holder, not several."],
1:["The resource {o} permits three simultaneous holders, each with a separate token.","Four slots for {o} may be occupied together; each caller returns a permit on departure.","Several workers can enter {o} at the same time, provided a permit remains.","A pool of five permits limits concurrent holders of {o} to five, not one."]},
"full_replica":{
0:["The servers divide {o} into nonoverlapping subsets; no server stores the whole dataset.","Separate nodes each store a different disjoint partition of {o}.","One half of {o} is on the first node and the remaining half is on the other.","The dataset {o} is partitioned by key range across distinct machines, with no full copy."],
1:["Each of two machines holds a complete logical copy of {o}.","The full dataset {o} exists identically at multiple storage nodes.","Every server in the pair has all records of {o}.","Redundant storage maintains equivalent complete copies of {o} on distinct machines."]},
"newer_recovery":{
0:["After the failure of {o}, the service reinstalls the earlier working release.","Recovery of {o} returns to its previous stable software build.","The deployment of {o} is reversed to the version used before the update.","The broken update to {o} is undone by restoring its predecessor."],
1:["After the failure of {o}, recovery installs a newer corrected software release.","The service repairs {o} by advancing to a patched build instead of restoring the old one.","The next deployment of {o} introduces a fix rather than returning to an earlier version.","Recovery moves {o} forward to a newly repaired revision."]},
"all_or_none":{
0:["The operation on {o} permits one change to commit even if the other fails.","The two edits to {o} can take effect separately, leaving a partial result.","A failure can leave one of the paired modifications of {o} committed alone.","The application allows a subset of the linked changes to {o} to persist."],
1:["The paired modifications of {o} must commit together or neither may commit.","A failure leaves no partial committed result of the linked changes to {o}.","The operation on {o} is indivisible: either all changes take effect or none do.","The two edits of {o} are committed as a single unit or both discarded."]}}
HYP={
"cached_available":{0:["The local copy of {o} is missing.","The origin must supply {o}."],1:["The desired {o} is available in the cache.","The cache serves {o} without an origin fetch."]},
"persist_first":{0:["The acknowledgement of {o} precedes persistence.","The change to {o} is made durable after confirmation."],1:["Durability of {o} precedes acknowledgement.","The update to {o} reaches stable storage before success."]},
"multi_holder":{0:["At most one holder may access {o} simultaneously.","Multiple simultaneous holders of {o} are prohibited."],1:["More than one holder may access {o} simultaneously.","Multiple callers may hold separate permits for {o} at the same time."]},
"full_replica":{0:["The nodes hold disjoint subsets of {o}.","No node holds the complete dataset {o}."],1:["Each node holds a full copy of {o}.","The nodes maintain identical complete datasets {o}."]},
"newer_recovery":{0:["Recovery restores the earlier release of {o}.","The failed deployment of {o} is reversed."],1:["Recovery installs a newer corrected version of {o}.","The recovery of {o} advances to a repaired build."]},
"all_or_none":{0:["A partial commit of the linked changes to {o} is possible.","One linked modification of {o} can commit alone."],1:["The linked changes to {o} must commit as one unit.","A partial commit of the linked changes to {o} is prohibited."]}}
NEUTRAL=["The {o} has an identifier written in hexadecimal.","The service handling {o} runs on a machine in Helsinki.","The documentation of {o} is printed in blue.","The {o} is handled by a program written in Python."]
def generate(dev=False):
    rng=random.Random(3102 if dev else 3101);rows=[]
    objects=("item 211","block 223","resource 237") if dev else ("object 501","entry 512","service 523","record 534","asset 545","unit 556","page 567","buffer 578")
    for attr,states in BANK.items():
        for value,forms in states.items():
            for obj in objects:
                for form in (forms[3:] if dev else forms[:3]):
                    premise=form.format(o=obj);group=digest([attr,value,obj,premise])
                    for label,claimvalue in (("entailment",value),("contradiction",1-value),("neutral",1)):
                        targetattr=attr if label!="neutral" else "unreported_metadata"
                        claim={"subject":obj,"attribute":targetattr,"value":claimvalue}
                        frame={"schema":"janus.relation.v1","facts":[{"subject":obj,"attribute":attr,"value":value}],"claim":claim}
                        hyp=(rng.choice(HYP[attr][claimvalue]) if label!="neutral" else rng.choice(NEUTRAL)).format(o=obj)
                        ck(relation(frame)==label,"FRAME_LABEL")
                        rows.append({"premise":premise,"hypothesis":hyp,"label":label,"frame":frame,"group":group,"id":digest([premise,hyp]),"family":attr,"label_source":"FINITE_JSON_STATE_TABLE"})
    ck(len({z["id"] for z in rows})==len(rows),"DUPLICATE")
    return rows
BUCKETS=4096;DIM=40;LIMIT=96
def ids(a,b):
    def words(t):return [h30.hash_id("w:"+("<num>" if x.isdigit() else x)) for x in re.findall(r"[a-z]+|[0-9]+",t.lower())]
    p=words(a)[:46];q=words(b)[:46];z=[1]+p+[2]+q+[3];s=[0]*(len(p)+2)+[1]*(len(q)+1)
    return [z+[0]*(LIMIT-len(z)),s+[0]*(LIMIT-len(s))]
def encoded(ps):return torch.tensor([ids(a,b) for a,b in ps],dtype=torch.long)
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.word=nn.Embedding(BUCKETS,DIM,padding_idx=0);self.pos=nn.Embedding(LIMIT,DIM);self.segment=nn.Embedding(2,DIM)
        layer=nn.TransformerEncoderLayer(d_model=DIM,nhead=4,dim_feedforward=96,dropout=0.,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.Linear(DIM*4,96),nn.GELU(),nn.LayerNorm(96),nn.Linear(96,3))
    def forward(self,x):
        tok,seg=x[:,0],x[:,1];mask=tok.eq(0);z=self.word(tok)+self.pos(torch.arange(tok.shape[1],device=tok.device))[None]+self.segment(seg);z=self.encoder(z,src_key_padding_mask=mask)
        a=(~mask&seg.eq(0)).unsqueeze(-1);b=(~mask&seg.eq(1)).unsqueeze(-1);u=(z*a).sum(1)/a.sum(1).clamp_min(1);v=(z*b).sum(1)/b.sum(1).clamp_min(1)
        return self.head(torch.cat((u,v,(u-v).abs(),u*v),-1))
def model(mode):return h30.Student() if mode=="bag" else Attention()
def encode(mode,ps):return h30.inputs(ps) if mode=="bag" else encoded(ps)
def predict(m,mode,ps):
    start=time.process_time();x=encode(mode,ps);out=[]
    with torch.inference_mode():
        for i in range(0,len(x),32):out+=m(x[i:i+32]).softmax(-1).tolist()
    return out,time.process_time()-start
def learn(rows,dev,soft,mode,seed):
    torch.manual_seed(seed);m=model(mode);params=sum(v.numel() for v in m.parameters());ck(params<=250000,"PARAMETER_BUDGET")
    x=encode(mode,[h30.pair(z) for z in rows]);y=h30.labels(rows);probs=soft;pred=probs.argmax(1);mask=pred.eq(y)&probs.max(1).values.ge(.70);wrong=pred.ne(y)
    # Every generated group includes entailment/contradiction/neutral. Draw
    # intact groups; H30 replay included as individual cases.
    by={}
    for i,z in enumerate(rows):by.setdefault(z.get("group",z["id"]),[]).append(i)
    gs=list(by.values());rng=random.Random(seed);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.001);start=time.process_time()
    for step in range(420):
        idx=torch.tensor([i for g in rng.sample(gs,16) for i in g]);o=m(x[idx]);loss=F.cross_entropy(o,y[idx])
        if mode=="distill":
            good=mask[idx];bad=wrong[idx]
            if bool(good.any()):loss=loss+.5*F.kl_div(F.log_softmax(o[good],dim=-1),probs[idx][good],reduction="batchmean")
            if bool(bad.any()):
                # Explicit rejected teacher class; never use its wrong label
                # as positive target. Ground-truth CE remains on every row.
                wrongp=o[bad].softmax(-1).gather(1,pred[idx][bad,None]).squeeze(1);loss=loss+.15*(-torch.log1p(-wrongp.clamp(max=.999999))).mean()
        ck(bool(torch.isfinite(loss)),"LOSS");opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    cpu=time.process_time()-start;m.eval();p,dcpu=predict(m,mode,[h30.pair(z) for z in dev]);return m,{"parameters":params,"cpu":cpu,"dev_cpu":dcpu,"dev":h30.metrics(p,[z["label"] for z in dev])}
def train(demi,source,h30trained,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);prior=read(h30trained/"receipt.json");old=read(h30trained/"train.json");fb=read(h30trained/"teacher_feedback.json")
    ck(prior["train_sha"]==digest(old) and len(old)==864 and not prior["exam_read"] and not prior["production"],"PARENT")
    ck(len(fb)==len(old) and all(z["id"]==v["id"] for z,v in zip(old,fb)),"TEACHER_RECEIPT")
    new=generate();dev=generate(True);rows=old+new
    ck(not {z["id"] for z in rows}&{z["id"] for z in dev} and len({z["id"] for z in rows})==len(rows),"OVERLAP")
    sem=h29.Semantic(demi,source);ck(sem.hashes==prior["teacher_hashes"],"TEACHER_PIN")
    soft=torch.tensor([z["probabilities"] for z in fb]+sem.pairs([h30.pair(z) for z in new]),dtype=torch.float32)
    y=h30.labels(rows);pred=soft.argmax(1);accepted=pred.eq(y)&soft.max(1).values.ge(.70)
    runs=[]
    for mode in ("bag","attention","distill"):
        for seed in (311,312):
            m,rec=learn(rows,dev,soft,mode,seed);path=out/("%s-%d.safetensors"%(mode,seed));save_file(m.state_dict(),str(path));runs.append({"mode":mode,"seed":seed,"sha":h29.sha(path),"bytes":path.stat().st_size,**rec})
    best={mode:max((z for z in runs if z["mode"]==mode),key=lambda z:(z["dev"]["correct"],-z["seed"]))["seed"] for mode in ("bag","attention","distill")}
    rec={"schema":"r84h31.train.v1","motto":MOTTO,"train_n":len(rows),"new_json_n":len(new),"dev_n":len(dev),"train_sha":digest(rows),"dev_sha":digest(dev),"teacher_hashes":sem.hashes,"teacher_wrong_rejected":int(pred.ne(y).sum()),"teacher_accepted":int(accepted.sum()),"teacher_new_cpu":sem.cpu,"runs":runs,"selected":best,"exam_read":False,"production":False}
    write(out/"train.json",rows);write(out/"dev.json",dev);write(out/"receipt.json",rec);log("R84H31_TRAIN",rec)
def json_exam():
    # Exhaustive equality/absence reference over independent novel subjects
    # and attributes. Input already structured; no text parsing claimed.
    rows=[]
    for i in range(64):
        for attr in ("presence","sequence","access"):
            for a in (0,1):
                for b in (0,1):
                    for same in (False,True):
                        subject="holdout-%d"%i;fact={"subject":subject,"attribute":attr,"value":a};claim={"subject":subject if same else "other-"+subject,"attribute":attr,"value":b}
                        rows.append(({"schema":"janus.relation.v1","facts":[fact],"claim":claim},"neutral" if not same else "entailment" if a==b else "contradiction"))
    start=time.process_time();correct=sum(relation(json.loads(json.dumps(q)))==y for q,y in rows);cpu=time.process_time()-start
    f={"subject":"s","attribute":"a","value":1};q={"schema":"janus.relation.v1","facts":[f,{**f,"value":0}],"claim":f}
    ck(relation(q)=="HOLD_CONFLICT" and relation({**q,"facts":[{**f,"value":True}]})=="HOLD_INVALID","JSON_CONTROLS")
    return {"n":len(rows),"correct":correct,"cpu_seconds_including_json_roundtrip":cpu,"conflict_and_malformed_controls":2}
def evaluate(demi,source,h30trained,trained,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);r=read(trained/"receipt.json");rows=read(trained/"train.json");dev=read(trained/"dev.json");ck(r["train_sha"]==digest(rows) and r["dev_sha"]==digest(dev) and not r["exam_read"] and not r["production"],"TRAIN_PIN")
    ex=h19.load_exam("R84H31_FROZEN_EXAM.json",EXAM);old=h19.load_exam("R84H30_FROZEN_EXAM.json",h30.EXAM)
    ps=[(a,b) for a,b,_ in ex["pairs"]];ys=[y for _,_,y in ex["pairs"]];op=[(a,b) for a,b,_ in old["pairs"]];oy=[y for _,_,y in old["pairs"]]
    ref={(h18.clean(z["premise"]),h18.clean(z["hypothesis"])) for z in rows+dev}|{(h18.clean(a),h18.clean(b)) for a,b in op};ck(len(ps)==48 and not {(h18.clean(a),h18.clean(b)) for a,b in ps}&ref,"EXAM_OVERLAP")
    sem=h29.Semantic(demi,source);ck(sem.hashes==r["teacher_hashes"],"TEACHER_PIN");start=time.process_time();tp=sem.pairs(ps);tcpu=time.process_time()-start;top=sem.pairs(op)
    texts=[z[0] for z in old["scenarios"]]+old["unknown"];sp=[(t,sem.hyp[l]) for t in texts for l in sem.labels];teacher_s=sem.pairs(sp)
    def scenario(p):
        good=0;unknown=0
        for i,t in enumerate(texts):
            z=p[i*5:(i+1)*5];order=sorted(range(5),key=lambda j:(-z[j][1],j));j,k=order[:2]
            if i<len(old["scenarios"]):good+=sem.labels[j]==old["scenarios"][i][1]
            else:unknown+=z[j][1]>=.90 and z[j][0]<=.10 and z[j][1]-z[k][1]>=.20
        return {"top1":good,"target_n":10,"unknown_supported":unknown,"unknown_n":10}
    def report(p,q,s,cpu):
        lost=sum(max(range(3),key=lambda j:v[j])==LABELS.index(y) and max(range(3),key=lambda j:w[j])!=LABELS.index(y) for v,w,y in zip(top,q,oy))
        return {"fresh":h30.metrics(p,ys),"core":h30.metrics(p[:18],ys[:18]),"added":h30.metrics(p[18:36],ys[18:36]),"heldout_families":h30.metrics(p[36:],ys[36:]),"prior_h30":h30.metrics(q,oy),"prior_teacher_correct_lost":lost,"prior_scenario":scenario(s),"fresh_cpu":cpu}
    teacher=report(tp,top,teacher_s,tcpu)
    receipt=read(h30trained/"receipt.json");seed=receipt["selected_seeds"]["distill"];z=next(z for z in receipt["student_runs"] if z["mode"]=="distill" and z["seed"]==seed);path=h30trained/("distill-%d.safetensors"%seed);ck(h29.sha(path)==z["sha256"],"OLD_STUDENT")
    m=h30.Student();m.load_state_dict(load_file(str(path)));m.eval();p,cpu=h30.predict(m,ps);q,_=h30.predict(m,op);s,_=h30.predict(m,sp);baseline=report(p,q,s,cpu)
    res=[]
    for z in r["runs"]:
        mode=z["mode"];path=trained/("%s-%d.safetensors"%(mode,z["seed"]));ck(h29.sha(path)==z["sha"],"WEIGHTS");m=model(mode);m.load_state_dict(load_file(str(path)));m.eval();p,cpu=predict(m,mode,ps);q,_=predict(m,mode,op);s,_=predict(m,mode,sp)
        res.append({"mode":mode,"seed":z["seed"],"parameters":z["parameters"],"bytes":z["bytes"],**report(p,q,s,cpu)})
    chosen={mode:next(z for z in res if z["mode"]==mode and z["seed"]==r["selected"][mode]) for mode in r["selected"]};d=chosen["distill"];js=json_exam();safe=d["prior_teacher_correct_lost"]==0 and d["prior_scenario"]["unknown_supported"]==0
    result={"schema":"r84h31.result.v1","motto":MOTTO,"teacher":teacher,"h30_student":baseline,"selected":chosen,"all_seeds":res,"json_exact":js,"fresh_n":48,"exact_overlap":0,"teacher_parameters":82120707,"student_parameter_cap":250000,"attention_gain":chosen["attention"]["fresh"]["correct"]>chosen["bag"]["fresh"]["correct"],"distillation_gain":d["fresh"]["correct"]>chosen["attention"]["fresh"]["correct"],"safety":"PASS_SHADOW" if safe else "HOLD","replacement":"PASS_SHADOW" if safe and d["fresh"]["correct"]>=teacher["fresh"]["correct"] else "HOLD","production":False,"scope":"finite JSON relation solver only on supplied structured facts, not automatic natural-language conversion; same-author synthetic training and frozen exam; template-separated development for added cases, old H30 replay, attention over ordered token hashes; teacher errors penalized against independently assigned labels. No deployment or core weight changes."}
    write(out/"result.json",result);log("R84H31_RESULT",result)
if __name__=="__main__":{"train":train,"evaluate":evaluate,"history":h30.history}[sys.argv[1]](*map(Path,sys.argv[2:]))
