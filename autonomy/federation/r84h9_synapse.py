"""R84H9 bounded frozen-expert routing ablation with train-only calibration."""
import sys,random,math,time,json
from pathlib import Path
import r84h8_neurogenesis as h8
import r84h7_immortal as h7
import r84h6_heritage as h6
import r84h5_control as h5
import r84h4_equilibrium as h4
import r84h3_stability as h3
import r84h2_growth as h2
import r84g_adaptive as a
import r84e_neural as e
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="312b57c0532a0fb8f34699c785ce19263825cf4a"
ROOTS=h8.ROOTS
BUNDLES=("e928f79b0e23530b43c9da9d2e4ebd7c72112fa4ac67b266ccfcb24aff08bfab","fb3e6271c0e8de03643cfe65d421c0d53ec2377caac4a670d6fb56c2b6e30b9a","7faa36bdcf3774a73fcb8d36bcd7bccd41b928bbc0ecb6a12dd4b595caadc3c7")
MODES=("baseline","natural","guarded","binary","consensus")
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def source(L,root,d,seed,i,root_path,bundle_path):
    core,cr=h7.load(L,root,root_path/"dual50"/str(seed),seed)
    b,br=h8.load_bundle(L,root,d,bundle_path/"modular"/str(seed),3,seed)
    ck(cr["model_sha256"]==ROOTS[i] and br["model_sha256"]==BUNDLES[i],"PIN")
    core.requires_grad_(False);b.requires_grad_(False);core.eval();b.eval();return core,b
def inputs(L,core,b,root,d,rows):
    torch,_=L;x,base=h8.features(L,core,root,d,rows)
    with torch.no_grad():
        orig=b["router"](x).softmax(-1)
        outs=[m(x) for m in b["modules_list"]]
        cand=[base]
        for i,o in enumerate(outs):
            ids=torch.tensor([d["labels"][y] for y in sorted(h8.TRAIN[i])])
            cand.append(ids[o.argmax(-1)])
        # All expert logits, probabilities and old core confidence, no answer labels.
        corelog=x[:,-root["output_size"]:]*10.
        cp=corelog.softmax(-1);top=cp.topk(2,dim=1).values
        extra=torch.cat([orig,top]+[o.softmax(-1) for o in outs],dim=1)
    return x,torch.cat([x,extra],dim=1),orig,torch.stack(cand,dim=1)
def split(rows,mapping,seed):
    by={}
    for i,(_,y) in enumerate(rows):by.setdefault(y,[]).append(i)
    rng=random.Random(seed+84901);fit=[];dev=[]
    for y,ids in sorted(by.items()):
        rng.shuffle(ids);n=min(len(ids)-1,max(1,len(ids)//5)) if mapping[y]==0 else min(len(ids)-1,2)
        dev+=ids[:n];fit+=ids[n:]
    ck(not set(fit)&set(dev) and len(fit)+len(dev)==len(rows),"SPLIT")
    return sorted(fit),sorted(dev)
def model(L,dim,seed):
    torch,_=L;torch.manual_seed(seed+84900)
    return torch.nn.ModuleDict({"natural":torch.nn.Sequential(torch.nn.Linear(dim,48),torch.nn.Tanh(),torch.nn.Linear(48,4)),"binary":torch.nn.Sequential(torch.nn.Linear(dim,48),torch.nn.Tanh(),torch.nn.Linear(48,3)),"consensus":torch.nn.Sequential(torch.nn.Linear(dim+12,48),torch.nn.Tanh(),torch.nn.Linear(48,3))})
def fit_one(L,m,x,orig,y,fit,seed,mode):
    torch,_=L;rng=random.Random(seed+84902);order=[];n=len(fit)
    while len(order)<240*32:
        q=list(fit);rng.shuffle(q);order+=q
    order=order[:240*32];ck(len(set(order))==n,"FIT_COVERAGE")
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    for j in range(240):
        ids=order[j*32:(j+1)*32];z=m(x[ids]);v=y[ids]
        if mode=="natural":loss=torch.nn.functional.cross_entropy(z,v)
        else:
            targets=(v[:,None]==torch.arange(1,4)[None,:]).float()
            hard=(1.+2.*orig[ids,1:])* (v[:,None]==0).float()
            weight=torch.where(targets>0,3.,1.+hard)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,targets,reduction="none")*weight).mean()
        ck(math.isfinite(float(loss.detach())),"LOSS");opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return len(order)
def probabilities(L,m,x,xx,orig):
    torch,_=L;m.eval()
    with torch.no_grad():
        return {"baseline":orig[:,1:],"natural":m["natural"](x).softmax(-1)[:,1:],"binary":m["binary"](x).sigmoid(),"consensus":m["consensus"](xx).sigmoid()}
def route(L,p,threshold):
    torch,_=L;v,idx=p.max(-1);return torch.where(v>=threshold,idx+1,torch.zeros_like(idx))
def calibrate(L,p,y,dev):
    torch,_=L;q=p[dev];yy=y[dev];nold=int((yy==0).sum());nnew=int((yy>0).sum())
    ck(nold>0 and nnew>0,"DEV")
    best=None
    for th in sorted(set([0.,1.01]+[float(v) for v in q.max(-1).values])):
        z=route(L,q,th);false=int(((z!=0)&(yy==0)).sum());hit=int(((z==yy)&(yy>0)).sum())
        if false/nold>.02:continue
        key=(hit,-false,th)
        if best is None or key>best[0]:best=(key,{"threshold":th,"old_n":nold,"new_n":nnew,"old_misroute":false,"new_route_correct":hit})
    ck(best is not None,"CALIBRATION");return best[1]
def train(prep,root_path,bundle_path,out):
    L=e.lib(prep);torch,_=L;*_,ds=h8.data(prep);root,d=ds[0],ds[-1];rows=d["old"]+d["new"];mapping=h8.routes(root,ds,3)
    out.mkdir(parents=True,exist_ok=False)
    for i,seed in enumerate(e.SEEDS):
        core,b=source(L,root,d,seed,i,root_path,bundle_path);start=time.process_time()
        x,xx,orig,_=inputs(L,core,b,root,d,rows);y=torch.tensor([mapping[v] for _,v in rows]);fit,dev=split(rows,mapping,seed)
        m=model(L,x.shape[1],seed);cpus={};steps={}
        for mode,feat in [("natural",x),("binary",x),("consensus",xx)]:
            t=time.process_time();steps[mode]=fit_one(L,m[mode],feat,orig,y,fit,seed,mode);cpus[mode]=time.process_time()-t
        probs=probabilities(L,m,x,xx,orig);thresholds={mode:calibrate(L,probs[mode],y,dev) for mode in ("natural","binary","consensus")}
        ck(e.state_hash(core)==ROOTS[i] and e.state_hash(b)==BUNDLES[i],"FROZEN")
        rec={"seed":seed,"core_sha256":ROOTS[i],"bundle_sha256":BUNDLES[i],"fit_n":len(fit),"dev_n":len(dev),"fit_indices":fit,"dev_indices":dev,"thresholds":thresholds,"steps_each":240,"presentations":steps,"router_cpu_seconds":cpus,"total_cpu_seconds":time.process_time()-start}
        e.save(L,m,d,out/str(seed),rec);log("R84H9_TRAIN",{k:v for k,v in rec.items() if k not in ("fit_indices","dev_indices")})
def exam_groups(prep,all_data):
    d0,old,hist,prior,pre,h6ds,h7ds,ds=all_data;groups=h2.exams(d0,old)
    for mod,dd,name in [(h3,hist,"H3"),(h4,prior,"H4"),(h5,pre,"H5"),(h6,h6ds,"H6"),(h7,h7ds,"H7"),(h8,ds,"H8")]:
        groups+=h5.clean_exam("R84"+name+"_FROZEN_EXAM.json",mod.EXAM,dd,groups)
    ck(sum(map(len,groups))==419,"PRIOR")
    d=ds[-1];raw=(HERE/"R84H9_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN")
    used=d["old"]+d["new"]+sum(d["monitor"],[])+sum(groups,[])
    norm={a.norm(t) for t,_ in used};enc={tuple(e.encode(d,t)) for t,_ in used};fresh=[]
    for grp in json.loads(raw)["cohorts"]:
        clean=[]
        for t,y in grp:
            n=a.norm(t);k=tuple(e.encode(d,t))
            if n in norm or k in enc:continue
            clean.append([t,y]);norm.add(n);enc.add(k)
        ck(len(clean)>=3,"FRESH");fresh.append(clean)
    return groups,fresh
def evaluate(prep,root_path,bundle_path,trained,out):
    L=e.lib(prep);torch,_=L;dd=h8.data(prep);ds=dd[-1];root,d=ds[0],ds[-1];groups,fresh=exam_groups(prep,dd);groups+=fresh
    rows=sum(groups,[]);labels=d["labels"];mapping=h8.routes(root,ds,3)
    y=torch.tensor([mapping[v] for _,v in rows]);actual=torch.tensor([labels[v] for _,v in rows]);old=y==0;new=y>0
    log("R84H9_EXAM",{"prior_n":419,"fresh_n":list(map(len,fresh)),"old_n":int(old.sum()),"new_n":int(new.sum())})
    results=[]
    for i,seed in enumerate(e.SEEDS):
        core,b=source(L,root,d,seed,i,root_path,bundle_path);x,xx,orig,cand=inputs(L,core,b,root,d,rows)
        rec=e.read(trained/str(seed)/"receipt.json");p=trained/str(seed)/"shadow.pt"
        ck(rec["file_sha256"]==e.sha(p.read_bytes()) and rec["data_sha256"]==g.digest(d) and rec["core_sha256"]==ROOTS[i] and rec["bundle_sha256"]==BUNDLES[i] and rec["production"] is False,"RECEIPT")
        m=model(L,x.shape[1],seed);m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True);ck(e.state_hash(m)==rec["model_sha256"],"HASH")
        probs=probabilities(L,m,x,xx,orig);base=cand[:,0];oracle=cand[torch.arange(len(rows)),y]
        for mode in MODES:
            if mode=="baseline":z=orig.argmax(-1)
            elif mode=="natural":
                with torch.no_grad():z=m["natural"](x).argmax(-1)
            else:
                key="natural" if mode=="guarded" else mode
                z=route(L,probs[key],rec["thresholds"][key]["threshold"])
            pred=cand[torch.arange(len(rows)),z];correct=pred==actual;bc=base==actual
            by=[{"route":k,"n":int((y==k).sum()),"correct":int(((z==y)&(y==k)).sum())} for k in range(4)]
            old_loss=int((bc&~correct&old).sum())
            row={"seed":seed,"mode":mode,"dep":int(correct.sum()),"n":len(rows),"old_correct":int((correct&old).sum()),"new_correct":int((correct&new).sum()),"old_n":int(old.sum()),"new_n":int(new.sum()),"old_base_correct":int((bc&old).sum()),"root_lost":old_loss,"old_misroute":int(((z!=0)&old).sum()),"new_route_correct":int(((z==y)&new).sum()),"route_correct":int((z==y).sum()),"route_by_class":by,"oracle_dep":int((oracle==actual).sum()),"fresh_dep":int(correct[-sum(map(len,fresh)):].sum()),"fresh_n":sum(map(len,fresh)),"fresh_route_correct":int((z[-sum(map(len,fresh)):] == y[-sum(map(len,fresh)):]).sum()),"threshold":rec["thresholds"]["natural" if mode=="guarded" else mode]["threshold"] if mode in ("guarded","binary","consensus") else None,"core_unchanged":e.state_hash(core)==ROOTS[i],"modules_unchanged":e.state_hash(b)==BUNDLES[i]}
            results.append(row);log("R84H9_RESULT",row)
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h9.result.v1","results":results,"production":False,"external_validation":False,"scope":"frozen experts; router fit/dev partition of prior expert training rows, dev independent of router fitting but not expert training; fixed <=2% dev old misroute threshold; fresh and historic exam never consumed by learner; aggregate route accuracy is imbalanced, report new-route recall and macro per-route"})
if __name__=="__main__":
    if sys.argv[1]=="train":train(*[Path(x) for x in sys.argv[2:]])
    elif sys.argv[1]=="evaluate":evaluate(*[Path(x) for x in sys.argv[2:]])
    else:raise ValueError("MODE")
