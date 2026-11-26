"""R84H30: train a compact student on structured labels with verified parent advice."""
import sys,json,time,re,hashlib,random,resource,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
import r84h29_semantic_resonance as h29
import r84h28_evidence_mastery as h28
import r84h25_metacognition as h25
import r84h19_exploration as h19
import r84h18_fog as h18
HERE=Path(__file__).parent
EXAM="5a0e789d6d65f5594045a6582aa4c7bec9faa73f"
LABELS=("contradiction","entailment","neutral")
MOTTO="Мы должны узнать"
ck=h25.ck;read=h25.read;write=h25.write;log=h25.log;digest=h25.digest
# Same author as exam, but labels are constructed from a finite state table
# independently of MiniLM. Natural language rendering remains an assumption.
BANK={
"cache":{
0:["The {obj} is absent from the local cache and must be retrieved from the origin.","The local lookup misses {obj}; the backing store provides it.","No cached copy of {obj} exists; fetch it from its source.","The request cannot find {obj} in the fast store and fetches it remotely."],
1:["The {obj} is already present in the cache and returned without an origin request.","The cached copy of {obj} satisfies the lookup.","The fast store contains {obj}; the backing store is not accessed.","The local lookup finds the resident {obj} and returns it."]},
"write":{
0:["The update to {obj} is acknowledged before persistence; a background flush happens later.","The changed {obj} remains dirty in memory when success is reported.","The buffered edit of {obj} reaches durable storage after the caller receives success.","The application changes {obj} in RAM, confirms the write, and persists it afterwards."],
1:["The update to {obj} reaches durable storage before acknowledgement.","The cache writes {obj} synchronously to backing storage before reporting success.","The caller waits for {obj} to be persisted before receiving confirmation.","The changed {obj} reaches both temporary and permanent storage in the original write."]},
"permits":{
0:["Only one worker can hold the permit for {obj} at once.","The resource {obj} grants exclusive access to a single holder.","No two callers may access {obj} simultaneously.","The resource {obj} has one permit, so only one task may enter at a time."],
1:["The resource {obj} has three permits and allows three simultaneous holders.","Several workers may hold permits for {obj} concurrently, within a finite limit.","A counting semaphore for {obj} grants one permit to each of several concurrent workers.","The pool for {obj} allows more than one caller to enter while free permits remain."]}}
HYP={
"cache":{0:["The {obj} is missing from the cache.","The requested {obj} must be fetched from its source."],1:["The {obj} is present in local cached storage.","The read of {obj} is supplied by an existing local copy."]},
"write":{0:["The write of {obj} is acknowledged before durable persistence.","The changed {obj} is persisted after acknowledgement."],1:["The {obj} is persisted before acknowledgement.","The backing store receives {obj} before success is reported."]},
"permits":{0:["Only one holder can access {obj} at a time.","Multiple simultaneous holders of {obj} are not allowed."],1:["More than one holder can access {obj} simultaneously.","Several permits for {obj} may be held at once."]}}
NEUTRAL=["The {obj} is stored in a data center in Oslo.","The {obj} is identified by a numeric code.","The {obj} is accessed using a wireless connection.","The {obj} is managed by software written in Java.","The {obj} is documented in a technical manual."]
TRAIN_OBJECTS=("record 7","page 11","block 23","entry 31","object 42","file 58","item 63","resource 74","buffer 85","service 96","document 107","packet 118")
DEV_OBJECTS=("unit 204","asset 219","table 235","slot 247")
def curriculum(objects,seed):
    rng=random.Random(seed);rows=[]
    for family,states in BANK.items():
        for state,forms in states.items():
            for obj in objects:
                for i,form in enumerate(forms):
                    premise=form.format(obj=obj)
                    for y,opts in (("entailment",HYP[family][state]),("contradiction",HYP[family][1-state]),("neutral",NEUTRAL)):
                        hyp=opts[rng.randrange(len(opts))].format(obj=obj)
                        row={"premise":premise,"hypothesis":hyp,"label":y,"family":family,"state":state,"subject":obj,"label_source":"STRUCTURED_STATE_TABLE_NOT_TEACHER"}
                        row["id"]=digest([premise,hyp]);rows.append(row)
    ck(len({z["id"] for z in rows})==len(rows),"CURRICULUM_DUPLICATE");rng.shuffle(rows);return rows
def pair(z):return z["premise"],z["hypothesis"]
def labels(rows):return torch.tensor([LABELS.index(z["label"]) for z in rows],dtype=torch.long)
BUCKETS=4096;DIM=40;WIDTH=96;LIMIT=112
def hash_id(s):return 1+int.from_bytes(hashlib.blake2s(s.encode(),digest_size=4).digest(),"little")%(BUCKETS-1)
def encode(t):
    w=re.findall(r"[a-z]+|[0-9]+",t.lower());w=["<num>" if x.isdigit() else x for x in w]
    # Ordered word and adjacent-word features, plus character prefixes.
    f=["w:"+x for x in w]+["b:"+a+"_"+b for a,b in zip(w,w[1:])]+["p:"+x[:4] for x in w if len(x)>4]
    ids=[hash_id(x) for x in f[:LIMIT]];return ids+[0]*(LIMIT-len(ids))
def inputs(pairs):
    return torch.tensor([[encode(a),encode(b)] for a,b in pairs],dtype=torch.long)
class Student(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(BUCKETS,DIM,padding_idx=0)
        self.project=nn.Sequential(nn.Linear(DIM*4,WIDTH),nn.LayerNorm(WIDTH),nn.GELU(),nn.Linear(WIDTH,WIDTH),nn.GELU(),nn.Linear(WIDTH,3))
    def forward(self,x):
        m=x.ne(0).unsqueeze(-1);e=self.embedding(x);v=(e*m).sum(2)/m.sum(2).clamp_min(1)
        a,b=v[:,0],v[:,1];return self.project(torch.cat((a,b,(a-b).abs(),a*b),dim=-1))
def predict(model,ps):
    start=time.process_time();x=inputs(ps);out=[]
    with torch.inference_mode():
        for i in range(0,len(x),64):out.extend(model(x[i:i+64]).softmax(-1).tolist())
    return out,time.process_time()-start
def metrics(ps,truth):
    counts={k:{"n":0,"correct":0} for k in LABELS}
    for v,y in zip(ps,truth):
        pred=LABELS[max(range(3),key=lambda i:v[i])];counts[y]["n"]+=1;counts[y]["correct"]+=pred==y
    return {"n":len(truth),"correct":sum(z["correct"] for z in counts.values()),"classes":counts}
def teacher(sem,rows):
    p=sem.pairs([pair(z) for z in rows]);y=labels(rows);v=torch.tensor(p,dtype=torch.float32)
    mask=v.argmax(1).eq(y)&v.max(1).values.ge(.70)
    return v,mask
def train_model(x,y,soft,mask,distill,seed):
    torch.manual_seed(seed);m=Student();opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001)
    gen=torch.Generator().manual_seed(seed+1);start=time.process_time()
    for i in range(450):
        idx=torch.randint(len(x),(64,),generator=gen);out=m(x[idx]);loss=F.cross_entropy(out,y[idx])
        if distill:
            use=mask[idx]
            if bool(use.any()):
                # Only correct and sufficiently confident parent soft targets
                # can modify loss, while structured labels always retained.
                v=soft[idx][use];loss=loss+.5*F.kl_div(F.log_softmax(out[use],dim=-1),v,reduction="batchmean")
        ck(bool(torch.isfinite(loss)),"TRAIN_LOSS");opt.zero_grad();loss.backward();opt.step()
    return m.eval(),time.process_time()-start
def train(demi,source,h29results,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);p=read(h29results/"result.json")
    ck(p["pair_n"]==30 and p["model_parameters"]==82120707 and not p["production"],"H29_PARENT")
    sem=h29.Semantic(demi,source);ck(sem.hashes["model.safetensors"]=="ec9c5b6a08b60d7a01b465c7ed44b496244dfd1bed3274c95875a74cea89ab35","TEACHER_HASH")
    tr=curriculum(TRAIN_OBJECTS,301);dev=curriculum(DEV_OBJECTS,302)
    ck(not {z["id"] for z in tr}&{z["id"] for z in dev},"DEV_OVERLAP")
    soft,mask=teacher(sem,tr);devsoft,devmask=teacher(sem,dev)
    x=inputs([pair(z) for z in tr]);y=labels(tr)
    seeds=(301,302,303);runs=[]
    for seed in seeds:
        for mode in ("hard","distill"):
            m,cpu=train_model(x,y,soft,mask,mode=="distill",seed)
            path=out/("%s-%d.safetensors"%(mode,seed));save_file(m.state_dict(),str(path))
            prob,pcpu=predict(m,[pair(z) for z in dev])
            runs.append({"mode":mode,"seed":seed,"sha256":h29.sha(path),"bytes":path.stat().st_size,"parameters":sum(v.numel() for v in m.parameters()),"train_cpu_seconds":cpu,"dev_cpu_seconds":pcpu,"dev":metrics(prob,[z["label"] for z in dev])})
    # Preselect per mode from dev only; all three seeds also tested later.
    best={mode:max((z for z in runs if z["mode"]==mode),key=lambda z:(z["dev"]["correct"],-z["seed"]))["seed"] for mode in ("hard","distill")}
    receipt={"schema":"r84h30.train.v1","motto":MOTTO,"train_n":len(tr),"dev_n":len(dev),"train_sha":digest(tr),"dev_sha":digest(dev),"teacher_hashes":sem.hashes,"teacher_parameters":sum(v.numel() for v in sem.model.parameters()),"teacher_train_correct":int(soft.argmax(1).eq(y).sum()),"teacher_admitted":int(mask.sum()),"teacher_wrong_rejected":int(soft.argmax(1).ne(y).sum()),"teacher_dev":metrics(devsoft.tolist(),[z["label"] for z in dev]),"teacher_cpu_seconds":sem.cpu,"student_runs":runs,"selected_seeds":best,"steps_per_model":450,"presentations_per_model":28800,"exam_read":False,"production":False}
    write(out/"train.json",tr);write(out/"dev.json",dev);write(out/"teacher_feedback.json",[{"id":z["id"],"probabilities":v,"accepted":bool(a)} for z,v,a in zip(tr,soft.tolist(),mask.tolist())]);write(out/"receipt.json",receipt);log("R84H30_TRAIN",receipt)
def evaluate(demi,source,trained,h29results,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);r=read(trained/"receipt.json");tr=read(trained/"train.json");dev=read(trained/"dev.json")
    ck(r["train_sha"]==digest(tr) and r["dev_sha"]==digest(dev) and not r["exam_read"] and not r["production"],"TRAIN_PIN")
    ex=h19.load_exam("R84H30_FROZEN_EXAM.json",EXAM);old=h19.load_exam("R84H29_FROZEN_EXAM.json",h29.EXAM);prior=read(h29results/"result.json");ck(prior["pair_n"]==30 and not prior["production"],"PARENT")
    ps=[(a,b) for a,b,_ in ex["pairs"]];truth=[y for _,_,y in ex["pairs"]]
    ref={(h18.clean(z["premise"]),h18.clean(z["hypothesis"])) for z in tr+dev}|{(h18.clean(a),h18.clean(b)) for a,b,_ in old["pairs"]}
    ck(len(ps)==60 and len(set(ps))==60 and not {(h18.clean(a),h18.clean(b)) for a,b in ps}&ref,"PAIR_OVERLAP")
    op=[(a,b) for a,b,_ in old["pairs"]];oy=[y for _,_,y in old["pairs"]]
    sem=h29.Semantic(demi,source);ck(sem.hashes==r["teacher_hashes"],"TEACHER_REPLAY")
    start=time.process_time();tp=sem.pairs(ps);teacher_cpu=time.process_time()-start;top=sem.pairs(op)
    texts=[x[0] for x in ex["scenarios"]]+ex["unknown"];sp=[(t,sem.hyp[l]) for t in texts for l in sem.labels];tsp=sem.pairs(sp)
    def scenario(v):
        ranked=[];correct=0;unknown_support=0
        for i,t in enumerate(texts):
            z=v[i*5:(i+1)*5];order=sorted(range(5),key=lambda j:(-z[j][1],j));j=order[0];k=order[1]
            # Fixed, not tuned on new exam. Ranking is separate from admission.
            support=z[j][1]>=.90 and z[j][0]<=.10 and z[j][1]-z[k][1]>=.20
            pred=sem.labels[j]
            if i<len(ex["scenarios"]):correct+=pred==ex["scenarios"][i][1]
            else:unknown_support+=support
            ranked.append({"text":t,"candidate":pred,"supported":support,"probabilities":z[j]})
        return {"target_n":len(ex["scenarios"]),"top1_correct":correct,"unknown_n":len(ex["unknown"]),"unknown_supported":unknown_support},ranked
    tsc,_=scenario(tsp);teacher_result={"fresh":metrics(tp,truth),"in_domain":metrics(tp[:36],truth[:36]),"transfer":metrics(tp[36:],truth[36:]),"prior_h29":metrics(top,oy),"scenario":tsc,"fresh_cpu_seconds":teacher_cpu}
    runs=[];detail=[]
    for z in r["student_runs"]:
        path=trained/("%s-%d.safetensors"%(z["mode"],z["seed"]));ck(h29.sha(path)==z["sha256"],"STUDENT_HASH");m=Student();m.load_state_dict(load_file(str(path)));m.eval()
        prob,cpu=predict(m,ps);oldprob,_=predict(m,op);svp,_=predict(m,sp);sc,rank=scenario(svp)
        a=metrics(prob,truth);b=metrics(oldprob,oy);lost=sum(LABELS[max(range(3),key=lambda j:v[j])]==y and LABELS[max(range(3),key=lambda j:q[j])]!=y for v,q,y in zip(top,oldprob,oy))
        runs.append({"mode":z["mode"],"seed":z["seed"],"parameters":z["parameters"],"bytes":z["bytes"],"fresh":a,"in_domain":metrics(prob[:36],truth[:36]),"transfer":metrics(prob[36:],truth[36:]),"prior_h29":b,"prior_teacher_correct_lost":lost,"scenario":sc,"fresh_cpu_seconds":cpu})
        detail.append({"mode":z["mode"],"seed":z["seed"],"pair_probs":prob,"scenario_rank":rank})
    selected={mode:next(z for z in runs if z["mode"]==mode and z["seed"]==r["selected_seeds"][mode]) for mode in ("hard","distill")}
    ds=selected["distill"];hs=selected["hard"]
    safe=ds["prior_teacher_correct_lost"]==0 and ds["scenario"]["unknown_supported"]==0
    gain=ds["fresh"]["correct"]>hs["fresh"]["correct"]
    report={"schema":"r84h30.result.v1","motto":MOTTO,"teacher":teacher_result,"selected":selected,"all_seeds":runs,"teacher_parameters":r["teacher_parameters"],"student_parameters":ds["parameters"],"compression_ratio":r["teacher_parameters"]/ds["parameters"],"teacher_admitted":r["teacher_admitted"],"teacher_wrong_rejected":r["teacher_wrong_rejected"],"fresh_pairs":len(ps),"prior_pairs":len(op),"exact_pair_overlap":0,"peak_process_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"distillation_gain":"PASS_SHADOW" if gain else "HOLD_NO_PROVEN_GAIN","student_safety":"PASS_SHADOW" if safe else "HOLD","replacement_gate":"PASS_SHADOW" if safe and ds["fresh"]["correct"]>=teacher_result["fresh"]["correct"] and gain else "HOLD","production":False,"scope":"same-author synthetic state table, independently assigned labels but not independent external examiner; compact trainable hash embeddings + MLP, not general language understanding. Parent soft labels only on verified matching classes. Dev templates shared with training but disjoint entities. Fresh transfer concepts not trained. Prior H29 replay not a fresh exam. Historical core untouched, separate job tests exact replay. No deployment."}
    write(out/"result.json",report);write(out/"cases.json",detail);log("R84H30_RESULT",report)
def history(*args):
    *paths,out=args;torch.set_num_threads(2);ctx=h25.Context(*paths);out.mkdir(parents=True,exist_ok=False)
    pred,routes,cpu=ctx.neural(ctx.hist);n=sum(ctx.prev[i]["answer"]==int(pred[i]) and ctx.prev[i]["route"]==int(routes[i]) for i in range(671));ck(n==671,"HISTORY")
    report={"schema":"r84h30.history.v1","historical_parity":n,"n":671,"correct":sum(ctx.prev[i]["correct"] for i in range(671)),"cpu_seconds":cpu,"production":False,"scope":"legacy route unchanged; not evidence that deploying the student preserves all skills"}
    write(out/"result.json",report);log("R84H30_HISTORY",report)
if __name__=="__main__":{"train":train,"evaluate":evaluate,"history":history}[sys.argv[1]](*map(Path,sys.argv[2:]))
