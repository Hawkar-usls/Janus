"""R84H22 finite, shared-weight graph neural benchmark; no automatic admission."""
import sys,json,hashlib,time,random
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).parent
EXAM="79cd2734668552d2647950bd878ad691a2dc012f"
PARENT="4db450886556a413688a29038860f86188bb905564aa99ceee3995ccac6f9d0c"
TRAIN="f17dc866e96e65ab74951bcad78c87a319d7eedcd79d62fa875a55681c41a624"
MOTTO="Мы должны узнать"
def ck(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(("blob %d\0"%len(b)).encode()+b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def key(z):return sha(canonical([z["edges"],z["roots"],z["target"]]).encode())
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(canonical(x)+"\n")
def log(t,x):print(t,canonical(x),flush=True)
class GraphNet(nn.Module):
    """Size-independent shared message/update functions; max aggregation.
    No deterministic reachability label or closure supplied to forward.
    Root indicators and directed edges are the only graph inputs.
    """
    def __init__(self):
        super().__init__()
        self.start=nn.Linear(1,16)
        self.message=nn.Linear(16,16)
        self.update=nn.Sequential(nn.Linear(33,32),nn.ReLU(),nn.Linear(32,16))
        self.readout=nn.Linear(16,2)
    def forward(self,edges,roots,target):
        n=roots.shape[1];root=roots.unsqueeze(-1);h=torch.tanh(self.start(root))
        # incoming edge u->v, independent of ordering; n bounded rounds.
        mask=edges.transpose(1,2).unsqueeze(-1).bool()
        for _ in range(n):
            msg=torch.tanh(self.message(h)).unsqueeze(1)
            incoming=torch.where(mask,msg,-2.).amax(dim=2)
            h=torch.tanh(self.update(torch.cat((h,incoming,root),dim=-1)))
        return self.readout(h[torch.arange(len(h)),target])
class PropagationNet(nn.Module):
    """Explicit monotone propagation inductive bias, trainable scalar transfer.
    Structural reachability knowledge is built into this architecture, not learned
    from scratch. Compare initialization and exact oracle to isolate learning gain.
    """
    def __init__(self):
        super().__init__()
        self.gain=nn.Parameter(torch.tensor(2.0))
        self.bias=nn.Parameter(torch.tensor(-1.0))
        self.scale=nn.Parameter(torch.tensor(1.0))
        self.cut=nn.Parameter(torch.tensor(0.5))
    def forward(self,edges,roots,target):
        h=roots
        for _ in range(roots.shape[1]):
            incoming=(edges.transpose(1,2)*h.unsqueeze(1)).amax(dim=2)
            present=edges.transpose(1,2).amax(dim=2)
            msg=torch.sigmoid(torch.nn.functional.softplus(self.gain)*incoming+self.bias)*present
            h=torch.maximum(h,msg)
        y=torch.nn.functional.softplus(self.scale)*(h[torch.arange(len(h)),target]-self.cut)
        return torch.stack((-y,y),dim=-1)
def tensors(rows):
    n=len(rows[0]["roots"]);return (torch.tensor([z["edges"] for z in rows],dtype=torch.float32).reshape(-1,n,n),torch.tensor([z["roots"] for z in rows],dtype=torch.float32),torch.tensor([z["target"] for z in rows]))
def bfs(z):
    n=len(z["roots"]);visited={i for i,v in enumerate(z["roots"]) if v};pending=list(visited)
    while pending:
        u=pending.pop()
        for v in range(n):
            if z["edges"][u*n+v] and v not in visited:visited.add(v);pending.append(v)
    return int(z["target"] in visited)
def parent(folder):
    ck(sha((folder/"shadow.pt").read_bytes())==PARENT and sha((folder/"train.json").read_bytes())==TRAIN,"H21_PIN")
    rec=read(folder/"train_receipt.json");ck(rec["production"] is False and rec["paper_text_used"] is False and rec["examples"]==3000,"H21_RECEIPT")
    rows=read(folder/"train.json");ck(len(rows)==3000 and len({key(z) for z in rows})==3000 and all(bfs(z)==z["label"] for z in rows),"TRAIN_DATA")
    return rows
def train(h21,out):
    torch.set_num_threads(1);rows=parent(h21);edges,roots,target=tensors(rows);y=torch.tensor([z["label"] for z in rows]);out.mkdir(parents=True,exist_ok=False);records=[]
    for name,cls in (("gnn",GraphNet),("propagation",PropagationNet)):
        torch.manual_seed(22072);m=cls();torch.save(m.state_dict(),out/(name+"-initial.pt"))
        opt=torch.optim.AdamW(m.parameters(),lr=.003 if name=="gnn" else .02,weight_decay=0)
        start=time.process_time()
        for step in range(600):
            ix=torch.randperm(len(rows))[:64];loss=nn.functional.cross_entropy(m(edges[ix],roots[ix],target[ix]),y[ix]);ck(bool(torch.isfinite(loss)),"LOSS");opt.zero_grad();loss.backward();opt.step()
        torch.save(m.state_dict(),out/(name+".pt"));records.append({"mode":name,"parameters":sum(p.numel() for p in m.parameters()),"steps":600,"presentations":38400,"cpu_seconds":time.process_time()-start,"initial_sha":sha((out/(name+"-initial.pt")).read_bytes()),"trained_sha":sha((out/(name+".pt")).read_bytes())})
    rec={"schema":"r84h22.train.v1","parent_train_sha":TRAIN,"train_n":3000,"train_graph_size":6,"exam_read":False,"production":False,"models":records,"motto":MOTTO}
    write(out/"receipt.json",rec);log("R84H22_TRAIN",rec)
def measure(pred,truth):
    counts=[truth.count(k) for k in (0,1)];correct=[sum(a==b==k for a,b in zip(pred,truth)) for k in (0,1)]
    ck(min(counts)>0,"CLASS_COUNTS")
    return {"n":len(truth),"correct":sum(correct),"counts":counts,"class_correct":correct,"balanced_accuracy":sum(c/n for c,n in zip(correct,counts))/2}
def evaluate(h21,trained,out):
    torch.set_num_threads(1);trainrows=parent(h21);keys={key(z) for z in trainrows};raw=(HERE/"R84H22_FROZEN_GRAPH_EXAM.json").read_bytes();ck(blob(raw)==EXAM,"EXAM_PIN")
    cases=json.loads(raw)["cases"];ck(len(cases)==840 and len({key(z) for z in cases})==840 and all(bfs(z)==z["expected"] for z in cases),"REFERENCE")
    rec=read(trained/"receipt.json");ck(rec["parent_train_sha"]==TRAIN and not rec["exam_read"] and not rec["production"],"RECEIPT")
    models={};params={}
    for row,cls in zip(rec["models"],(GraphNet,PropagationNet)):
        name=row["mode"];params[name]=row["parameters"]
        for tag,suffix,pin in (("initial","-initial.pt",row["initial_sha"]),("trained",".pt",row["trained_sha"])):
            p=trained/(name+suffix);ck(sha(p.read_bytes())==pin,"WEIGHT_PIN");m=cls();m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True));m.eval();models[name+"_"+tag]=m
    mlp=nn.Sequential(nn.Linear(48,64),nn.ReLU(),nn.Linear(64,2));mlp.load_state_dict(torch.load(h21/"shadow.pt",map_location="cpu",weights_only=True));mlp.eval()
    results=[];overlap=0
    for n in (6,12,24):
        for kind in ("random","long_chain"):
            orig=[z for z in cases if z["n"]==n and z["kind"]==kind];rows=[z for z in orig if key(z) not in keys];overlap+=len(orig)-len(rows);truth=[z["expected"] for z in rows];edges,roots,target=tensors(rows)
            modes={}
            for name,m in models.items():
                start=time.process_time()
                with torch.no_grad():
                    pred=[]
                    for i in range(0,len(rows),32):pred+=m(edges[i:i+32],roots[i:i+32],target[i:i+32]).argmax(-1).tolist()
                modes[name]={**measure(pred,truth),"cpu_seconds":time.process_time()-start}
            if n==6:
                x=torch.cat((edges.flatten(1),roots,nn.functional.one_hot(target,6).float()),dim=1);start=time.process_time()
                with torch.no_grad():pred=mlp(x).argmax(-1).tolist()
                modes["h21_mlp"]={**measure(pred,truth),"cpu_seconds":time.process_time()-start}
            start=time.process_time();pred=[bfs(z) for z in rows];modes["bfs"]={**measure(pred,truth),"cpu_seconds":time.process_time()-start}
            result={"size":n,"kind":kind,"frozen_n":len(orig),"overlap_excluded":len(orig)-len(rows),"modes":modes};results.append(result);log("R84H22_COHORT",result)
    gates={name:all(z["modes"][name]["balanced_accuracy"]>=.95 for z in results) for name in ("gnn_trained","propagation_trained")}
    report={"schema":"r84h22.exam.v1","exam_n":840,"overlap_excluded":overlap,"cohorts":results,"gates":gates,"parameters":params,"production":False,"motto":MOTTO,"scope":"finite graph reachability, synthetic same-owner exam; propagation baseline encodes algorithmic prior; no general natural-language competence, no production or promotion"}
    out.mkdir(parents=True,exist_ok=False);write(out/"result.json",report);log("R84H22_GRAPH_GATE",{"gates":gates,"overlap_excluded":overlap,"production":False})
def historical(prep,rootpath,bundlepath,h10path,h11path,h12path,h17results,snapshot,out):
    import r84h15_world as h15
    import r84h14_partition as h14
    import r84h13_streaming as h13
    import r84h16_chunk as h16
    L,dd,root,d,mapping,labels=h15.base_data(prep);groups,_=h14.exams(prep,dd,d);hist=sum(groups,[]);ck(len(hist)==671,"HIST")
    cells,_,_,_,_=h13.catalog(L,d,mapping,labels,h15.SEED,0,h11path,h12path)
    base,br,_=h13.persistent(L,root,d,h15.SEED,0,rootpath,bundlepath,h10path,mapping,hist);pred,routes,_=h13.eager(L,cells,hist,base,br)
    prev={x["i"]:x for x in read(h17results/"cases.json") if x["cohort"]=="historical"}
    parity=sum(prev[i]["pred"]==[pred[i]] and prev[i]["routes"]==[routes[i]] for i in range(671));dep=sum(p==labels[y] for p,(_,y) in zip(pred,hist))
    receipt=read(snapshot/"snapshot.json");ck(receipt["size"]==100000 and h16.sha((snapshot/"world.sqlite").read_bytes())==receipt["db_sha256"],"WORLD")
    ck(parity==671 and dep==386,"HIST_PARITY")
    out.mkdir(parents=True,exist_ok=False);z={"parity":parity,"dep":dep,"world_sha256":receipt["db_sha256"],"production":False,"scope":"old model replay only, new graph model not routed into old text tasks"};write(out/"history.json",z);log("R84H22_HISTORY",z)
def summarize(exam,history,trained,out):
    a=read(exam/"result.json");b=read(history/"history.json");t=read(trained/"receipt.json");ck(b["parity"]==671 and not a["production"] and not b["production"],"REGRESSION")
    out.mkdir(parents=True,exist_ok=False);candidates=[]
    for row in t["models"]:
        if a["gates"][row["mode"]+"_trained"]:
            candidates.append({"schema":"r84h22.chunk_proposal.v1","namespace":"candidate/tracing-reachability/"+row["mode"],"weight_sha256":row["trained_sha"],"parameters":row["parameters"],"input":"directed adjacency matrix, roots, target; bounded n<=24","historical_replay":671,"review_required":True,"production":False,"auto_promote":False})
    z={"schema":"r84h22.summary.v1","graph_gates":a["gates"],"historical_parity":b["parity"],"historical_dep":b["dep"],"candidates":candidates,"motto":MOTTO,"production":False,"native_world_modified":False}
    write(out/"summary.json",z);log("R84H22_SUMMARY",z)
if __name__=="__main__":
    {"train":train,"evaluate":evaluate,"historical":historical,"summarize":summarize}[sys.argv[1]](*map(Path,sys.argv[2:]))
