"""H36 own bounded property/value verifier, no teacher labels or live writes."""
import sys,math,time,hashlib,re
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
import r84h34_semantic_crystal_expansion as h34
import r84h33_json_crystallization as h33
import r84h32_json_semantic_compiler as h32
import r84h31_semantic_evolution as h31
import r84h29_semantic_resonance as h29
import r84h27_trinity_critic as h27
import r84h30_semantic_inheritance as h30
import r84h19_exploration as h19
import verified_capabilities as exact
EXAM="69823b8919ae31c23fae2f1f150feeaefdd28027"
ATTR=h32.ATTR;STATES=h32.STATES;HOLD=h32.HOLD;A=len(ATTR)
ck=h32.ck;read=h32.read;write=h32.write;log=h32.log;digest=h32.digest
B=2048
def hid(s):return int.from_bytes(hashlib.blake2s(s.encode(),digest_size=4).digest(),"little")%B
def vector(t):
    w=h32.tokens(t);features=["w:"+x for x in w]+["b:"+a+"_"+b for a,b in zip(w,w[1:])]+["t:"+a+"_"+b+"_"+c for a,b,c in zip(w,w[1:],w[2:])]
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
def observe(c,models,ts):
    net,ncpu=h33.probabilities(c.model,ts);lex=[c.d.search(t) for t in ts];ev=h33.evidence(c.sem,ts,net,lex);own,cpu=probabilities(models,ts);rows=[]
    for i,t in enumerate(ts):
        j,conf,gap=h32.candidate(net[i]);k,s,_=h32.candidate(lex[i][0]);base,proof=h33.select(t,net[i],lex[i],ev.get(i),c.policy)
        eligible=j!=HOLD and j==k and lex[i][1][k] is not None and h33.scope(t) and conf>=c.policy["confidence"] and gap>=c.policy["margin"] and s>=c.policy["lexical"]
        rows.append({"text":t,"candidate":j if eligible else None,"baseline":base,"own":own[i],"proof":proof,"scope":h33.scope(t)})
    return rows,cpu
def select(z,rule,mode):
    j=z["candidate"];own=j if j is not None and supports(z["own"],j,rule) else None
    if mode=="parent":return z["baseline"]
    if mode=="own":return own
    return z["baseline"] if z["baseline"] is not None else own
def metrics(rows,ys,rule,mode):
    m={"known_n":0,"unknown_n":0,"correct":0,"wrong":0,"unknown_accepted":0,"baseline_correct_lost":0,"additional_correct":0}
    for z,y in zip(rows,ys):
        j=select(z,rule,mode)
        if y==HOLD:m["unknown_n"]+=1;m["unknown_accepted"]+=j is not None
        else:
            m["known_n"]+=1;m["correct"]+=j==y;m["wrong"]+=j is not None and j!=y;m["baseline_correct_lost"]+=z["baseline"]==y and j!=y;m["additional_correct"]+=j==y and z["baseline"]!=y
    return m
def direct_own(rows,ys,rule):
    m={"true_support":0,"true_n":0,"opposite_false_support":0,"opposite_n":0,"other_attribute_false_support":0,"other_attribute_n":0,"unknown_false_support":0,"unknown_n":0}
    for z,y in zip(rows,ys):
        for j in range(HOLD):
            yes=z["scope"] and supports(z["own"],j,rule)
            k="unknown" if y==HOLD else "true" if j==y else "opposite" if j==y^1 else "other_attribute"
            m[k+"_n"]+=1;m[k+("_support" if k=="true" else "_false_support")]+=yes
    return m
def calibrate(rows,ys):
    opts=[]
    for a in (.50,.60,.70,.80,.90,.95,.98):
        for v in (.55,.65,.75,.85,.90,.95,.98):
            for g in (.10,.20,.35,.50,.70):
                rule={"attribute":a,"value":v,"margin":g};d=direct_own(rows,ys,rule);m=metrics(rows,ys,rule,"hybrid")
                if d["opposite_false_support"]==d["other_attribute_false_support"]==d["unknown_false_support"]==m["wrong"]==m["unknown_accepted"]==0 and d["true_support"]>0:
                    opts.append((m["additional_correct"],d["true_support"],a,v,g,rule))
    return max(opts,key=lambda x:x[:5])[-1] if opts else None
def train(demi,source,trained,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);c=h34.Context(demi,source,trained)
    # Validate every training record from native Cortex, not exam-confirmed rows.
    records=read(trained/"correspondences.json");tr=c.tr;dev=c.dev
    ck(len(records)==len(tr)==236,"TRAIN_COUNT")
    for z,rec in zip(tr,records):
        ck(z["text"]==rec["text"] and z["label"]==rec["label"] and rec["origin"]["training_allowed"] and rec["origin"]["label_source"]=="AUTHORED_STRUCTURED_STATE","CORRESPONDENCE")
        raw=c.d.m.read_object(c.d.root,rec["sha256"],c.d.state["records"][rec["sha256"]]["size"]);ck(raw.decode()==z["text"],"CORTEX_RECORD")
    runs=[];models=[]
    for seed in (361,362):
        m,cpu=fit(tr,seed);models.append(m);path=out/("verify-%d.safetensors"%seed);save_file(m.state_dict(),str(path));n=sum(p.numel() for p in m.parameters());ck(n<=150000,"PARAM_BUDGET")
        runs.append({"seed":seed,"parameters":n,"bytes":path.stat().st_size,"sha":h29.sha(path),"train_cpu":cpu})
    rows,cpu=observe(c,models,[z["text"] for z in dev]);ys=[z["label"] for z in dev];rule=calibrate(rows,ys)
    parent=metrics(rows,ys,None,"parent");own=metrics(rows,ys,rule,"own");hybrid=metrics(rows,ys,rule,"hybrid");ck(parent["correct"]==4 and parent["wrong"]==parent["unknown_accepted"]==0,"H33_DEV")
    rec={"schema":"r84h36.train.v1","train_sha":c.rec["train_sha"],"dictionary_sha":c.rec["dictionary_sha"],"source_weight_sha":c.z["sha"],"train_n":len(tr),"dev_n":len(dev),"runs":runs,"rule":rule,"dev_parent":parent,"dev_own":own,"dev_hybrid":hybrid,"dev_direct_own":direct_own(rows,ys,rule),"enabled":rule is not None and hybrid["additional_correct"]>0 and hybrid["wrong"]==hybrid["unknown_accepted"]==0,"teacher_hashes":c.sem.hashes,"own_dev_cpu":cpu,"trinity_arithmetic_controls":h27.trinity(demi),"exam_read":False,"teacher_labels_used":False,"production":False}
    write(out/"receipt.json",rec);write(out/"dev_evidence.json",rows);log("R84H36_TRAIN",rec)
def direct_parent(c,ts,ys):
    # Same 12 candidate claims for each normalized input, same H33 first-template thresholds.
    ps=[]
    for t in ts:
        for a,v in STATES[:HOLD]:ps.extend([(t,h31.HYP[a][v][0].format(o="entity")),(t,h31.HYP[a][1-v][0].format(o="entity"))])
    start=time.process_time();p=c.sem.pairs(ps);cpu=time.process_time()-start
    m={"true_support":0,"true_n":0,"opposite_false_support":0,"opposite_n":0,"other_attribute_false_support":0,"other_attribute_n":0,"unknown_false_support":0,"unknown_n":0}
    for i,(t,y) in enumerate(zip(ts,ys)):
        for j in range(HOLD):
            s,o=p[2*(i*HOLD+j):2*(i*HOLD+j)+2]
            yes=h33.scope(t) and s[1]>=c.policy["semantic"] and o[0]>=c.policy["opposite"] and o[1]<=.20
            k="unknown" if y==HOLD else "true" if j==y else "opposite" if j==y^1 else "other_attribute"
            m[k+"_n"]+=1;m[k+("_support" if k=="true" else "_false_support")]+=yes
    return m,cpu,len(ps)
def evaluate(demi,source,trained,own,out):
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);c=h34.Context(demi,source,trained);rec=read(own/"receipt.json")
    ck(rec["train_sha"]==c.rec["train_sha"] and rec["dictionary_sha"]==c.rec["dictionary_sha"] and rec["source_weight_sha"]==c.z["sha"] and rec["teacher_hashes"]==c.sem.hashes and not rec["exam_read"] and not rec["production"],"TRAIN_PIN")
    models=[]
    for r in rec["runs"]:
        p=own/("verify-%d.safetensors"%r["seed"]);ck(h29.sha(p)==r["sha"],"WEIGHT");m=Verifier();m.load_state_dict(load_file(str(p)));models.append(m.eval())
    old=h19.load_exam("R84H33_FROZEN_EXAM.json",h33.EXAM);prev=h19.load_exam("R84H34_FROZEN_EXAM.json",h34.EXAM);earlier=h19.load_exam("R84H32_FROZEN_EXAM.json",h32.EXAM);ex=h19.load_exam("R84H36_FROZEN_EXAM.json",EXAM)
    ref={h32.key(z["text"]) for z in c.tr+c.dev}
    for prior in (old,prev,earlier):ref|={h32.key(n[1]) for n in h34.corpus(prior).values() if n}
    def assess(data,exclude,compare_direct):
        norm=h34.corpus(data);blocked={t for t,n in norm.items() if n and h32.key(n[1]) in exclude}
        known=[z for z in data["statements"] if z["text"] not in blocked];unknown=[t for t in data["unknown"] if t not in blocked];pairs=[z for z in data["pairs"] if z["premise"] not in blocked and z["hypothesis"] not in blocked]
        ck(len(known)>=30 and len(unknown)>=18 and len(pairs)>=24,"SIZE")
        ts=[t for t,n in norm.items() if n and t not in blocked];obs,cpu=observe(c,models,[norm[t][1] for t in ts]);lookup=dict(zip(ts,obs));rule=rec["rule"];res={};details={}
        def compile(t,mode):
            if t not in lookup:return None
            j=select(lookup[t],rule if mode!="guarded" or rec["enabled"] else None,"hybrid" if mode=="guarded" else mode)
            if j is None:return None
            a,v=STATES[j];return {"subject":norm[t][0],"attribute":a,"value":v}
        for mode in ("parent","own","hybrid","guarded"):
            a=[compile(z["text"],mode) for z in known];u=[compile(t,mode) for t in unknown];ans=[]
            for q in pairs:
                ck(exact.relation({"schema":"janus.relation.v1","facts":q["facts"],"claim":q["claim"]})==q["label"],"ORACLE")
                p,s=compile(q["premise"],mode),compile(q["hypothesis"],mode);ans.append(exact.relation({"schema":"janus.relation.v1","facts":[p],"claim":s}) if p and s else None)
            res[mode]={"facts_n":len(known),"facts_correct":sum(f==z["fact"] for f,z in zip(a,known)),"fact_errors":sum(f is not None and f!=z["fact"] for f,z in zip(a,known)),"unknown_n":len(u),"unknown_accepted":sum(f is not None for f in u),"pairs_n":len(ans),"pairs_answered":sum(x is not None for x in ans),"pairs_correct":sum(x==q["label"] for x,q in zip(ans,pairs)),"pairs_wrong":sum(x is not None and x!=q["label"] for x,q in zip(ans,pairs))}
            details[mode]={"facts":a,"unknown":u,"answers":ans}
        res["own"]["baseline_correct_lost"]=sum(a==z["fact"] and b!=z["fact"] for a,b,z in zip(details["parent"]["facts"],details["own"]["facts"],known))
        if compare_direct:
            targets=[(z["text"],STATES.index((z["fact"]["attribute"],z["fact"]["value"]))) for z in known]+[(t,HOLD) for t in unknown];targets=[(t,y) for t,y in targets if t in lookup];ys=[y for t,y in targets];r=[lookup[t] for t,y in targets];p,pc,n=direct_parent(c,[norm[t][1] for t,y in targets],ys)
            res["direct_comparison"]={"parent":p,"own":direct_own(r,ys,rule),"teacher_cpu":pc,"teacher_pairs":n,"normalized_inputs":len(targets)}
        confirmed=[{"text":z["text"],"fact":f,"evidence":lookup[z["text"]],"status":"CONFIRMED_ON_SYNTHETIC_EXAM","training_allowed":False} for z,f in zip(known,details["guarded"]["facts"]) if f==z["fact"]]
        return res,{"details":details,"observations":lookup,"excluded":sorted(blocked)},confirmed,cpu
    oldr,_,_,_=assess(old,set(),False);prevr,_,_,_=assess(prev,set(),False)
    ck(oldr["parent"]["facts_correct"]==3 and prevr["parent"]["facts_correct"]==4 and prevr["parent"]["fact_errors"]==prevr["parent"]["unknown_accepted"]==0,"PARENT_REPLAY")
    fresh,details,confirmed,cpu=assess(ex,ref,True)
    def safe(m):return m["fact_errors"]==m["unknown_accepted"]==m["pairs_wrong"]==0
    z=fresh["guarded"];d=fresh["direct_comparison"]["own"];dp=fresh["direct_comparison"]["parent"]
    direct_safe=d["opposite_false_support"]==d["other_attribute_false_support"]==d["unknown_false_support"]==0
    hybrid_ok=rec["enabled"] and all(safe(x["guarded"]) for x in (oldr,prevr,fresh)) and direct_safe and z["facts_correct"]>fresh["parent"]["facts_correct"] and z["pairs_answered"]>=12
    own_ok=all(safe(x["own"]) and x["own"]["baseline_correct_lost"]==0 for x in (oldr,prevr,fresh)) and direct_safe and fresh["own"]["facts_correct"]>fresh["parent"]["facts_correct"] and fresh["own"]["pairs_answered"]>=12 and d["true_support"]>dp["true_support"]
    result={"schema":"r84h36.result.v1","fresh":fresh,"h33_regression":oldr,"h34_regression":prevr,"gate_enabled":rec["enabled"],"rule":rec["rule"],"train_n":rec["train_n"],"verifier_parameters_total":sum(z["parameters"] for z in rec["runs"]),"own_inference_cpu_seconds":cpu,"new_confirmed":len(confirmed),"excluded_exact_overlap":len(details["excluded"]),"hybrid_gate":"PASS_SHADOW" if hybrid_ok else "HOLD","replacement_gate":"PASS_SHADOW" if own_ok else "HOLD","production":False,"scope":"Frozen same-author English six-property exam; native Cortex training-only authored correspondences, two separately seeded own models with shared training data; no teacher targets, teacher used for baseline and calibration comparison only; all candidate claims tested on valid subject texts; pair statements reuse extraction rows; no general semantics or external truth guarantee."}
    write(out/"result.json",result);write(out/"cases.json",details);write(out/"confirmed_correspondences.json",confirmed);log("R84H36_RESULT",result)
if __name__=="__main__":{"train":train,"evaluate":evaluate,"history":h30.history}[sys.argv[1]](*map(Path,sys.argv[2:]))
