"""R84E: matched-budget DOVOD shadow neural continual-learning ablation."""
import hashlib,importlib.util,json,math,random,re,sys,time,urllib.request
from pathlib import Path
import jfp_gateway as g
import r8_learning as r
import r84b_intake as b
HERE=Path(__file__).parent
EXAM="d6e328ef2ae8983cfc03a4504ac245b097b65501"
DOVOD="5f7de85ea89ae3ffcee7c9cf13d6eb653e43ccd0"
SOURCE_PACKET="f0df278a6037105271097f8a9da9e2558062582474a7a7e482e9f81b1e793567"
ARMS=("baseline","replay","activity","joint")
SEEDS=(111,112,113)
PARENT_STEPS=180;NEW_STEPS=180;BATCH=12
TEMPLATES={"primary":["give primary category for {id}","report {id} primary subject category","identify primary subject of {id}"],"author_count":["give author count for {id}","report {id} author count","count authors listed on {id}"],"license":["give license for {id}","report {id} license","identify publication license of {id}"],"claim":["give reported topic for {id}","report {id} thematic research topic","identify author reported topic of {id}"]}
def ck(v,why):g.require(v,why)
def log(tag,o):print(tag,json.dumps(o,sort_keys=True),flush=True)
def sha(x):return hashlib.sha256(x).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n")
def tokenize(s):return re.findall(r"2611\.\d{5}v1|[a-z]+|\d+",s.lower())
def source_label(k,v):
    if k=="primary":return {"cs.LG":"cat_cs_lg","cs.CV":"cat_cs_cv","cs.RO":"cat_cs_ro"}[v]
    if k=="author_count":return "count_"+str(v)
    if k=="license":return {"CC-BY-4.0":"lic_cc_by","OTHER_OR_RESTRICTED":"lic_restricted"}[v]
    if k=="claim":return {"RL replay comparison":"topic_replay","firing-rate threshold control":"topic_firing"}[v]
    raise ValueError(k)
def exam():
    raw=(HERE/"R84E_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");e=json.loads(raw)
    old=[x for group in b.exams() for x in group];new=[[q.format(id=i),y] for i,k,y in e["facts"] for q in e["templates"][k]]
    return old,new,e["facts"]
def prepare(source,out):
    p=read(source/"packet.json");ck(p["sha256"]==SOURCE_PACKET and g.digest({k:v for k,v in p.items() if k!="sha256"})==SOURCE_PACKET and p["production"] is False and len(p["records"])==5 and not p["holds"],"SOURCE_PACKET")
    old=r.OLD+r.fixture()["examples"]+[x for p in b.fixture() for x in p["examples"]]
    facts=[[x["id"],k,source_label(k,v)] for x in p["records"] for k,v in x["fields"].items()]
    new=[[q.format(id=i),y] for i,k,y in facts for q in TEMPLATES[k]]
    eo,en,ef=exam();ck(sorted(facts)==sorted(ef),"FROZEN_FACT_IDENTITY")
    ck(len(old)==46 and len(new)==51 and len(eo)==38 and len(en)==34 and not {x[0] for x in old+new}&{x[0] for x in eo+en},"SPLIT")
    vocab={w:i+2 for i,w in enumerate(sorted({w for t,y in old+new for w in tokenize(t)}))}
    labels={y:i+200 for i,y in enumerate(sorted({y for t,y in old+new}))}
    ck(max(vocab.values())<200 and max(labels.values())<258,"VOCAB")
    with urllib.request.urlopen("https://raw.githubusercontent.com/Hawkar-usls/janus-meta-registry/main/DovoD.py",timeout=25) as f:raw=f.read(262145)
    ck(r.blob(raw)==DOVOD,"DOVOD_PIN")
    out.mkdir(parents=True,exist_ok=False);(out/"dovod.py").write_bytes(raw)
    d={"old":old,"new":new,"vocab":vocab,"labels":labels,"exam_blob":EXAM,"packet_sha256":SOURCE_PACKET,"production":False};write(out/"data.json",d)
    log("R84E_PREPARED",{"old_train":len(old),"new_train":len(new),"old_exam":len(eo),"new_exam":len(en),"source_facts":len(facts),"data_sha256":g.digest(d),"train_exam_prompt_overlap":0})
def lib(p):
    import torch
    torch.set_num_threads(2)
    path=p/"dovod.py";ck(r.blob(path.read_bytes())==DOVOD,"SOURCE_PIN")
    s=importlib.util.spec_from_file_location("dovod_original",path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return torch,m.HierarchicalTransformer
def model(L):
    torch,D=L;m=D(vocab_size=258,n_embd=124,n_head=4,n_layer=2,block_size=128)
    ck(sum(p.numel() for p in m.parameters())==452352,"PARAMS");return m
def encode(d,s):
    # Input-only word-token mapping learned from training text; unseen tokens map to UNK.
    ids=[0]+[d["vocab"].get(w,1) for w in tokenize(s)]+[0]
    ck(len(ids)<=24,"LENGTH");return ids
def batch(L,d,rows):
    torch,_=L;xs=[encode(d,x[0]) for x in rows];lens=torch.tensor([len(x)-1 for x in xs]);n=max(map(len,xs))
    x=torch.tensor([z+[0]*(n-len(z)) for z in xs]);y=torch.tensor([d["labels"][z[1]] for z in rows]);return x,lens,y
def forward(L,m,x,lens):
    # Exact batched form of the original 1D DOVOD forward; causal padding lies after the scored token.
    torch,_=L;n=x.shape[1];z=m.wte(x)+m.wpe(torch.arange(n));mask=torch.triu(torch.ones(n,n,dtype=torch.bool),diagonal=1)
    z=m.transformer_encoder(z,mask=mask);z=m.ln_f(z);return m.lm_head(z[torch.arange(x.shape[0]),lens])
def state_hash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def match_original(L,m,d):
    torch,_=L;m.eval();x,l,y=batch(L,d,d["old"][:3])
    with torch.no_grad():
        z=forward(L,m,x,l)
        for i in range(len(l)):ck(torch.allclose(z[i],m(x[i,:l[i]+1])[-1],atol=2e-5,rtol=2e-5),"BATCH_EQUIVALENCE")
def optimize(L,m,d,seed,phase,arm):
    torch,_=L;rng=random.Random(seed+phase*10000);torch.manual_seed(seed+phase*10000)
    opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=.01);activity=arm in ("activity","joint") and phase==2;replay=arm in ("replay","joint") and phase==2
    capture=[];hooks=[layer.linear1.register_forward_hook(lambda mod,inp,out:capture.append(out)) for layer in m.transformer_encoder.layers]
    steps=PARENT_STEPS if phase==1 else NEW_STEPS;examples=0;old_examples=0;new_examples=0;start=time.monotonic();last=None
    try:
        for j in range(steps):
            if phase==1:rows=rng.choices(d["old"],k=BATCH);old_examples+=BATCH
            else:
                # Same candidate stream for all arms. Replay swaps exactly 3 of 12 slots.
                nr=rng.choices(d["new"],k=BATCH);orr=rng.choices(d["old"],k=3)
                rows=nr[:9]+orr if replay else nr
                old_examples+=3 if replay else 0;new_examples+=9 if replay else 12
            examples+=len(rows);m.train();capture.clear();opt.zero_grad(set_to_none=True)
            x,l,y=batch(L,d,rows);z=forward(L,m,x,l);ce=torch.nn.functional.cross_entropy(z,y)
            ck(len(capture)==2,"ACTIVITY_HOOK")
            mask=(torch.arange(x.shape[1])[None,:]<=l[:,None]).unsqueeze(-1)
            rates=[(torch.sigmoid(v/.1)*mask).sum((0,1))/mask.sum() for v in capture]
            reg=sum(((rate-.30)**2).mean() for rate in rates)/len(rates)
            loss=ce+(.2*reg if activity else 0.);ck(math.isfinite(float(loss.detach())),"LOSS");loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            if j==steps-1:last={"ce":float(ce.detach()),"regularizer":float(reg.detach())}
    finally:
        for h in hooks:h.remove()
    return {"steps":steps,"examples":examples,"old_examples":old_examples,"new_examples":new_examples,"seconds":time.monotonic()-start,**last}
def save(L,m,d,out,record):
    torch,_=L;out.mkdir(parents=True,exist_ok=False);p=out/"shadow.pt";torch.save(m.state_dict(),p)
    record.update({"model_sha256":state_hash(m),"file_sha256":sha(p.read_bytes()),"data_sha256":g.digest(d),"production":False});write(out/"receipt.json",record)
def load(L,d,out):
    torch,_=L;rec=read(out/"receipt.json");p=out/"shadow.pt";ck(rec["file_sha256"]==sha(p.read_bytes()) and rec["data_sha256"]==g.digest(d) and rec["production"] is False,"CHECKPOINT")
    m=model(L);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True)
    ck(state_hash(m)==rec["model_sha256"] and all(torch.isfinite(x).all() for x in m.state_dict().values()),"MODEL_HASH")
    return m,rec
def parents(prep,out):
    L=lib(prep);torch,_=L;d=read(prep/"data.json")
    out.mkdir(parents=True,exist_ok=False)
    for seed in SEEDS:
        torch.manual_seed(seed);m=model(L);match_original(L,m,d);meta=optimize(L,m,d,seed,1,"baseline")
        rec={"seed":seed,"phase":1,**meta};save(L,m,d,out/str(seed),rec);log("R84E_PARENT",rec)
def train(prep,parent,out,arm):
    ck(arm in ARMS,"ARM");L=lib(prep);d=read(prep/"data.json");out.mkdir(parents=True,exist_ok=False)
    for seed in SEEDS:
        m,p=load(L,d,parent/str(seed));ck(p["seed"]==seed and p["phase"]==1,"PARENT");meta=optimize(L,m,d,seed,2,arm)
        rec={"seed":seed,"phase":2,"arm":arm,"parent_sha256":p["model_sha256"],**meta}
        save(L,m,d,out/str(seed),rec);log("R84E_TRAINED",rec)
def score(L,m,d,rows):
    torch,_=L;m.eval();correct=0;preds=[];rates=[];handles=[layer.linear1.register_forward_hook(lambda mod,inp,out:rates.append(out.detach())) for layer in m.transformer_encoder.layers]
    try:
        with torch.no_grad():
            x,l,y=batch(L,d,rows);z=forward(L,m,x,l);p=z.argmax(-1);correct=int((p==y).sum());preds=p.tolist()
            mask=(torch.arange(x.shape[1])[None,:]<=l[:,None]).unsqueeze(-1)
            fire=[((a>0)*mask).sum((0,1)).float()/mask.sum() for a in rates]
            inactive=sum(int((a<.01).sum()) for a in fire)/sum(a.numel() for a in fire)
    finally:
        for h in handles:h.remove()
    return {"exact":correct,"n":len(rows),"inactive_fraction":inactive,"preds":preds}
def evaluate(prep,parent,out,*folders):
    L=lib(prep);d=read(prep/"data.json");eo,en,_=exam();results=[]
    ck(not {x[0] for x in d["old"]+d["new"]}&{x[0] for x in eo+en},"EXAM_OVERLAP")
    for seed in SEEDS:
        m,p=load(L,d,parent/str(seed));old=score(L,m,d,eo);new=score(L,m,d,en)
        log("R84E_BASE",{"seed":seed,"old":old["exact"],"new":new["exact"],"old_n":len(eo),"new_n":len(en)})
        for arm,folder in zip(ARMS,folders):
            q,rec=load(L,d,folder/str(seed));ck(rec["arm"]==arm and rec["seed"]==seed and rec["parent_sha256"]==p["model_sha256"] and rec["steps"]==NEW_STEPS and rec["examples"]==NEW_STEPS*BATCH,"ARM_MATCH")
            a=score(L,q,d,eo);b=score(L,q,d,en)
            retained=sum(x==d["labels"][row[1]] and y==d["labels"][row[1]] for row,x,y in zip(eo,old["preds"],a["preds"]))
            z={"seed":seed,"arm":arm,"old_before":old["exact"],"old_after":a["exact"],"old_retained":retained,"old_n":len(eo),"new_before":new["exact"],"new_after":b["exact"],"new_n":len(en),"inactive_fraction_old":a["inactive_fraction"],"inactive_fraction_new":b["inactive_fraction"],"seconds":rec["seconds"],"steps":rec["steps"],"old_examples":rec["old_examples"],"new_examples":rec["new_examples"],"parent_sha256":p["model_sha256"],"model_sha256":rec["model_sha256"]}
            results.append(z);log("R84E_RESULT",z)
    out.mkdir(parents=True,exist_ok=False);write(out/"results.json",{"schema":"r84e.result.v1","results":results,"scope":"internal paraphrase generalization for known source identities; old regression reused","source_inspired_not_reproduction":True,"production":False,"external_validation":False})
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x) for x in sys.argv[2:]]
    if mode=="prepare":prepare(*args)
    elif mode=="parents":parents(*args)
    elif mode=="train":train(*args[:3],sys.argv[5])
    elif mode=="evaluate":evaluate(*args)
    else:raise ValueError("MODE")
