"""R7C8 native byte versus compact representation and matched DOVOD architecture."""
import hashlib, importlib.util, json, math, random, sys, urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXAM="2020ee05a064e1ca1ff460034ec8e402ae579d44"
DOVOD="5f7de85ea89ae3ffcee7c9cf13d6eb653e43ccd0"
BASE="06d7e6228ca9aef2dc58eb55fe1c805aa3b66e4f8fb90b0e946ce6af47b1a662"
SEEDS=(81,82,83)
ARMS=("promoted_text","promoted_compact","scratch_text","scratch_compact","dovod_compact")
STEPS=1200
def check(ok,why):
    if not ok:raise RuntimeError(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def blob(raw):return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,o):Path(p).write_text(json.dumps(o,sort_keys=True,separators=(",",":"))+"\n")
def digest(o):return sha(json.dumps(o,sort_keys=True,separators=(",",":")).encode())
def pair(a,b):return tuple(sorted((a,b)))
def exam():
    raw=(HERE/"R7C8_FROZEN_COGNITIVE_EXAM.json").read_bytes();check(blob(raw)==EXAM,"EXAM_PIN")
    rows=json.loads(raw)["rows"]
    check(len(rows)==96 and len({pair(a,b) for k,a,b,y in rows})==96,"EXAM_GROUPS")
    check(all(y==(int(a>b) if k=="compare" else a+b) for k,a,b,y in rows),"EXAM_LABELS")
    return rows
def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    with urllib.request.urlopen("https://raw.githubusercontent.com/Hawkar-usls/janus-meta-registry/main/DovoD.py",timeout=30) as f:raw=f.read(262145)
    check(blob(raw)==DOVOD,"DOVOD_PIN");(out/"dovod.py").write_bytes(raw)
    ex=exam();excluded={pair(a,b) for k,a,b,y in ex}
    for filename,pin in (("R7C5_FROZEN_INTERNAL_EXAM.json","84c033c7bf38824249255cdc90f44db2a475f1af"),("R7C6_FROZEN_FOUNDATION_EXAM.json","393a6490bba4f0e4f41cf8484729407ceaeb91d7"),("R7C7_FROZEN_ANTI_OVERFIT_EXAM.json","bec690ffa67c085fa997a15ed6558e847852efd9")):
        raw=(HERE/filename).read_bytes();check(blob(raw)==pin,"OLD_EXAM_PIN");x=json.loads(raw)
        if "R7C5" in filename:old=[(a,b) for a,b,y,c in x["cases"]]
        elif "R7C6" in filename:old=[(a,b) for s,k,a,b,y in x["cases"] if k=="add"]
        else:old=[(a,b) for rows in x["cohorts"].values() for a,b,y in rows]
        excluded.update(pair(a,b) for a,b in old)
    excluded.update(pair(11+7*i%38,13+11*i%39) for i in range(24))
    rng=random.Random(8100);train=[];dev=[]
    # stratified by operand magnitude and carry; group split across operations/orientations
    for small in (True,False):
        for carry in (False,True):
            pool=[(a,b) for a in range(100) for b in range(a+1,100) if ((b<10)==small) and ((a%10+b%10>=10)==carry) and (a,b) not in excluded]
            rng.shuffle(pool)
            ndev=min(12,len(pool)//3);ntrain=min(200,len(pool)-ndev)
            check(ndev>=2 and ntrain>=2,"POOL")
            for dest,p in ((dev,pool[:ndev]),(train,pool[ndev:ndev+ntrain])):
                for a,b in p:
                    dest.extend([["compare",a,b,0],["compare",b,a,1],["addition",a,b,a+b]])
    tr={pair(a,b) for k,a,b,y in train};dv={pair(a,b) for k,a,b,y in dev}
    check(not tr&dv and not (tr|dv)&excluded,"LEAK")
    for k,a,b,y in train+dev:
        if k=="compare":check(y==int(b<a),"LABEL_COMPARE")
        else:
            z=a
            for _ in range(b):z+=1
            check(z==y,"LABEL_ADD")
    data={"train":train,"dev":dev,"exam_blob":EXAM,"dovod_blob":DOVOD,"promotion":False}
    write(out/"data.json",data)
    print("R7C8_PREPARED",json.dumps({"train_rows":len(train),"train_groups":len(tr),"dev_rows":len(dev),"dev_groups":len(dv),"exam":96,"overlap":0,"data_sha256":digest(data)}),flush=True)
def imports(demi,prepared):
    import torch
    sys.path.insert(0,str(demi))
    from janus_model.model import ByteTokenizer,JanusModelConfig,JanusTinyTransformer
    from janus_model.train_registry import load_checkpoint
    source=prepared/"dovod.py";check(blob(source.read_bytes())==DOVOD,"SOURCE_PIN")
    spec=importlib.util.spec_from_file_location("dovod_original",source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return torch,ByteTokenizer,JanusModelConfig,JanusTinyTransformer,load_checkpoint,mod.HierarchicalTransformer
def prompt(row,compact):
    k,a,b,y=row
    if compact:return f"{a}>{b}?" if k=="compare" else f"{a}+{b}="
    return f"Is {a} greater than {b}? Reply 1 for yes or 0 for no." if k=="compare" else f"Compute {a} + {b}. Reply with the integer only."
def make(arm,lib,demi):
    torch,T,C,N,load,D=lib
    if arm.startswith("promoted"):return load(demi/"janus_model/checkpoints/promoted.pt",expected_sha256=BASE)[0]
    if arm.startswith("scratch"):return N(C())
    if arm=="dovod_compact":return D(vocab_size=258,n_embd=124,n_head=4,n_layer=2,block_size=128)
    raise ValueError(arm)
def ops(lib,model,arm):
    torch,T,*_=lib;compact=arm.endswith("compact");is_dovod=arm=="dovod_compact"
    def prefix(r):return T.encode("Q:"+prompt(r,compact)+"\nJANUS:",bos=True)
    def logits(ids):
        x=torch.tensor(ids,dtype=torch.long)
        return model(x) if is_dovod else model(x.unsqueeze(0))[0][0]
    def loss(r):
        pre=prefix(r);ans=T.encode(str(r[3]),eos=True);seq=pre+ans
        check(len(seq)-1<=128,"CONTEXT")
        pred=logits(seq[:-1]);target=torch.tensor([-100]*(len(pre)-1)+ans,dtype=torch.long)
        return torch.nn.functional.cross_entropy(pred,target)
    @torch.no_grad()
    def answer(r):
        ids=prefix(r);out=[]
        for _ in range(8):
            n=int(logits(ids)[-1].argmax().item())
            if n==T.EOS:return T.decode(out) if all(48<=i<=57 for i in out) else None
            out.append(n);ids.append(n)
        return None
    @torch.no_grad()
    def score(rows):
        model.eval();answers=[answer(r) for r in rows];losses=[float(loss(r)) for r in rows]
        return {"exact":sum(a==str(r[3]) for a,r in zip(answers,rows)),"n":len(rows),"loss":sum(losses)/len(losses),"answers":answers}
    return loss,score,prefix
def count(m):return sum(p.numel() for p in m.parameters())
def train(demi,prepared,out,arm):
    check(arm in ARMS,"ARM");lib=imports(demi,prepared);torch,T,C,N,load,D=lib
    torch.set_num_threads(4);data=read(prepared/"data.json");check(data["exam_blob"]==EXAM and not data["promotion"],"DATA")
    check(not {pair(a,b) for k,a,b,y in data["train"]}&{pair(a,b) for k,a,b,y in data["dev"]},"DEV_LEAK")
    out.mkdir(parents=True,exist_ok=False);native=count(N(C()))
    for seed in SEEDS:
        torch.manual_seed(seed);rng=random.Random(seed);m=make(arm,lib,demi);n=count(m);check(abs(n-native)/native<=.05,"PARAMETER_MATCH")
        loss,score,prefix=ops(lib,m,arm);opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=.01)
        folder=out/str(seed);folder.mkdir();history=[]
        def snapshot():return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        def rank(s):return (s["exact"],-s["loss"])
        best=score(data["dev"]);best_step=0;best_state=snapshot()
        for step in range(1,STEPS+1):
            r=rng.choice(data["train"]);m.train();opt.zero_grad(set_to_none=True);l=loss(r);check(math.isfinite(l.item()),"FINITE");l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
            if step%200==0:
                s=score(data["dev"])
                if rank(s)>rank(best):best=s;best_step=step;best_state=snapshot()
                e={"arm":arm,"seed":seed,"steps":step,"dev_exact":s["exact"],"dev_n":s["n"],"dev_loss":s["loss"],"best_step":best_step}
                history.append(e);print("R7C8_BLOCK",json.dumps(e),flush=True)
        m.load_state_dict(best_state);target=folder/"shadow.pt";torch.save(best_state,target)
        r={"arm":arm,"seed":seed,"steps":STEPS,"selected_step":best_step,"dev_exact":best["exact"],"params":n,"native_params":native,
           "mean_train_prefix_bytes":sum(len(prefix(r)) for r in data["train"])/len(data["train"]),
           "shadow_sha256":sha(target.read_bytes()),"data_sha256":digest(data),"history":history,"promotion":False}
        write(folder/"receipt.json",r);print("R7C8_TRAINED",json.dumps({k:v for k,v in r.items() if k!="history"}),flush=True)
    check(sha((demi/"janus_model/checkpoints/promoted.pt").read_bytes())==BASE,"BASE_CHANGED")
def evaluate(demi,prepared,out,*folders):
    lib=imports(demi,prepared);torch,T,C,N,load,D=lib;torch.set_num_threads(4)
    data=read(prepared/"data.json");rows=exam();check(not {pair(a,b) for k,a,b,y in rows}&{pair(a,b) for k,a,b,y in data["train"]+data["dev"]},"LEAK")
    # Causality: changing future input must not change earlier logits.
    for arm in ("scratch_compact","dovod_compact"):
        m=make(arm,lib,demi);m.eval()
        with torch.no_grad():
            x=torch.tensor([256,49,43,50]);y=torch.tensor([256,49,43,57])
            a=m(x) if arm.startswith("dovod") else m(x.unsqueeze(0))[0][0]
            b=m(y) if arm.startswith("dovod") else m(y.unsqueeze(0))[0][0]
            check(torch.allclose(a[:-1],b[:-1],atol=1e-5,rtol=1e-5),"CAUSAL")
    print("R7C8_TOKENIZER",json.dumps({str(n):T.encode(str(n)) for n in (0,7,42,123,999)}),flush=True)
    results=[]
    def measure(m,arm,seed,selected,label):
        loss,score,prefix=ops(lib,m,arm);metrics={}
        for k in ("compare","addition","transfer"):
            subset=[r for r in rows if r[0]==k];s=score(subset)
            metrics[k]={"exact":s["exact"],"n":s["n"],"answers":s["answers"]}
        r={"arm":label,"seed":seed,"selected_step":selected,"params":count(m),"metrics":metrics}
        results.append(r);print("R7C8_MODEL",json.dumps({**r,"metrics":{k:{x:y for x,y in v.items() if x!="answers"} for k,v in metrics.items()}}),flush=True)
    for arm in ("promoted_text","promoted_compact"):measure(make(arm,lib,demi),arm,0,0,"original_"+arm)
    for arm,folder in zip(ARMS,folders):
        for seed in SEEDS:
            r=read(folder/str(seed)/"receipt.json");p=folder/str(seed)/"shadow.pt"
            check(r["arm"]==arm and r["seed"]==seed and r["data_sha256"]==digest(data) and r["steps"]==STEPS and not r["promotion"],"RECEIPT")
            check(sha(p.read_bytes())==r["shadow_sha256"],"SHA")
            m=make(arm,lib,demi);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True)
            measure(m,arm,seed,r["selected_step"],arm)
    out.mkdir(parents=True,exist_ok=False)
    write(out/"result.json",{"status":"INTERNAL_MEASUREMENT_ONLY","exam_blob":EXAM,"results":results,"external_validation":False,"broad_regression":"NOT_RUN","promotion":False})
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x).resolve() for x in sys.argv[2:]]
    if mode=="prepare":prepare(*args)
    elif mode=="train":train(*args[:3],sys.argv[5])
    elif mode=="evaluate":evaluate(*args)
    else:raise ValueError("MODE")
