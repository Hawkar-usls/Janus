"""R84H15 finite scale audit. 15 original experts + up to 105 synthetic load fixtures."""
import sys,json,time,math,random,gc,resource
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.sparse import csr_matrix,vstack
import r84h14_partition as h14
import r84h13_streaming as h13
import r84h12_evolution as h12
import r84h11b_repair as repair
import r84h11_cortex as h11
import r84h10_resonance as h10
import r84e_neural as e
import r84g_adaptive as a
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="9882c3e97d1849e00b1b3028d273184ba96daba4"
SEED=111
SIZES=(15,30,60,120)
MODES=("exact","sprites16","tree16","tree16_prefetch")
DOMAINS=[c+" "+n for c in ("amber","cobalt","jade","silver","ivory","coral","violet") for n in ("ledger","relay","archive","queue","catalog","buffer","vault","index","journal","registry","mirror","channel","broker","cache","terminal")]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def rss():
    return int(next(x.split()[1] for x in Path("/proc/self/status").read_text().splitlines() if x.startswith("VmRSS:")))*1024
def peak():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
def fixture_rows(i):
    n=DOMAINS[i];ys=["h15_%03d_%d"%(i,j) for j in (0,1)]
    return [["gate practice "+n+" "+t,ys[j]] for j,ts in enumerate((["ingest incoming item","append new data","persist submitted entry","write received object"],["retrieve saved item","read existing data","fetch persisted entry","return archived object"])) for t in ts]
def base_data(prep):
    L=e.lib(prep);dd=h12.h8.data(prep);ds=dd[-1];root,d=ds[0],ds[-1];mapping,labels=h12.maps(root,ds,d);return L,dd,root,d,mapping,labels
def banks(d,mapping,i):
    old,_=repair.refs(d,SEED,mapping)
    old=old+[x for grp in h12.FIT for x in h10.extra(grp)]
    # Shared real negative anchors plus two examples from each other fixture.
    # Fixed full 105-fixture fit bank, independent of tested scale.
    rng=random.Random(15000);old=[old[j] for j in sorted(rng.sample(range(len(old)),96))]
    old += [fixture_rows(j)[k] for j in range(105) if j!=i for k in (0,4)]
    return old,fixture_rows(i)
def fixture_train(L,x,seed):
    torch,_=L;m=h11.model(L,x.shape[1],seed);opt=torch.optim.AdamW(m.parameters(),lr=.005,weight_decay=.01);y=torch.tensor([0]*4+[1]*4)
    for _ in range(60):
        loss=torch.nn.functional.cross_entropy(m(x),y);ck(math.isfinite(float(loss.detach())),"LOSS");opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return m
def prepare(prep,out):
    L,dd,root,d,mapping,labels=base_data(prep);out.mkdir(parents=True,exist_ok=False);start=time.process_time();pins=[]
    for i in range(105):
        fo,fn=banks(d,mapping,i);lex=h10.lexical_fit(fo+fn);x=h10.lexical(L,lex,fn);m=fixture_train(L,x,15000+i)
        rec={"seed":SEED,"fixture":i,"domain":DOMAINS[i],"lex":lex,"ref_sha256":g.digest(fo+fn),"parameters":sum(p.numel() for p in m.parameters()),"steps":60,"presentations":480,"fixture_only":True}
        e.save(L,m,d,out/str(i),rec);pins.append({"i":i,"sha":rec["model_sha256"],"file":rec["file_sha256"]})
    rec={"schema":"r84h15.fixture.v1","seed":SEED,"pins":pins,"count":105,"cpu_seconds":time.process_time()-start,"production":False}
    e.write(out/"manifest.json",rec);log("R84H15_FIXTURES",rec)
def fixture_cells(L,d,mapping,labels,folder,size):
    manifest=e.read(folder/"manifest.json");ck(manifest["count"]==105 and manifest["seed"]==SEED and manifest["production"] is False,"MANIFEST")
    start=max(labels.values())+1;cells=[]
    for i in range(size-15):
        path=folder/str(i);rec=e.read(path/"receipt.json");pin=manifest["pins"][i];fo,fn=banks(d,mapping,i)
        ck(rec["fixture"]==i and rec["fixture_only"] is True and rec["production"] is False and rec["domain"]==DOMAINS[i] and rec["model_sha256"]==pin["sha"] and rec["file_sha256"]==pin["file"] and rec["data_sha256"]==g.digest(d) and rec["ref_sha256"]==g.digest(fo+fn) and rec["lex"]==h10.lexical_fit(fo+fn),"FIXTURE_PIN")
        ids=[start+2*i,start+2*i+1]
        cells.append(h13.Cell(L,19+i,path,rec,fo,fn,ids,pin["sha"],15000+i))
    return cells
class Bounds:
    """Unsigned 16 bit directed rounding: positive bounds round upward,
    negative anchors downward. Sparse zeros remain zero. No precision-loss
    rejection unless an additional floating-point guard is satisfied."""
    def __init__(self,m,npositive):
        m=m.tocsr();m.eliminate_zeros();ck(np.all(m.data>=0),"NONNEGATIVE")
        self.scale=max(float(m.data.max(initial=0)),1e-12)/65534
        q=m.copy();rows=np.repeat(np.arange(m.shape[0]),np.diff(m.indptr))
        q.data=np.where(rows<npositive,np.ceil(m.data/self.scale),np.floor(m.data/self.scale)).clip(0,65535).astype(np.uint16);q.eliminate_zeros()
        self.mat=q;self.npositive=npositive
    def rejects(self,q,eps):
        v=(self.mat@q)*self.scale
        return float(v[:self.npositive].max())+eps<float(v[self.npositive:].max())
    def bytes(self):return h13.sparse_bytes(self.mat)
class Node:
    def __init__(self,begin,end,bounds,children):
        self.begin=begin;self.end=end;self.bounds=bounds;self.children=children
class Atlas:
    def __init__(self,cells,tree):
        self.keys=sorted({k for c in cells for k in c.lex.lookup});self.lookup={k:i for i,k in enumerate(self.keys)};nkeys=len(self.keys)
        self.scale=max(w for c in cells for _,w in c.lex.lookup.values());self.sprites=[];self.nodes=[];self.tree=tree
        self.sprite_bytes=0;self.uncompressed_sprite_bytes=0;self.tree_bytes=0;leafbounds=[]
        for c in cells:
            keys=c.rec["lex"]["keys"];scales=np.array([c.lex.lookup[k][1] for k in keys],dtype=np.float64);idx=np.array([self.lookup[k] for k in keys])
            co=c.ref.astype(np.float64).multiply(scales).tocoo();ref=csr_matrix((co.data,(co.row,idx[co.col])),shape=(c.ref.shape[0],nkeys))
            neg=ref[:c.nold];pos=ref[c.nold:];ck(neg.shape[0]>=16 and pos.shape[0]>0,"ANCHORS")
            near=(neg@pos.T).toarray();inds=[]
            for j in range(pos.shape[0]):
                for k in np.argsort(-near[:,j],kind="stable")[:4]:
                    if int(k) not in inds:inds.append(int(k))
            for k in np.argsort(-near.max(axis=1),kind="stable"):
                if int(k) not in inds:inds.append(int(k))
                if len(inds)>=48:break
            sprite=vstack([pos,neg[inds[:48]]],format="csr");b=Bounds(sprite,pos.shape[0]);self.sprites.append(b);self.sprite_bytes+=b.bytes();self.uncompressed_sprite_bytes+=h13.sparse_bytes(sprite)
            if tree:
                # Common lower envelopes: min of valid child negative vectors
                # is a valid lower bound for every child, regardless of anchor identity.
                upper=pos.max(axis=0).toarray().ravel()
                chosen=np.argsort(-near.max(axis=1),kind="stable")[:16];lower=neg[chosen].toarray()
                leafbounds.append((upper,lower))
        self.depth=0
        if tree:
            layer=[(Node(i,i+1,None,[]),u,l) for i,(u,l) in enumerate(leafbounds)]
            while len(layer)>1:
                nextlayer=[];self.depth+=1
                for i in range(0,len(layer),3):
                    grp=layer[i:i+3];up=np.maximum.reduce([x[1] for x in grp]);lo=np.minimum.reduce([x[2] for x in grp]);m=csr_matrix(np.vstack([up,lo]));b=Bounds(m,1)
                    children=[x[0] for x in grp];node=Node(children[0].begin,children[-1].end,b,children);self.nodes.append(node);self.tree_bytes+=b.bytes();nextlayer.append((node,up,lo))
                layer=nextlayer
            self.root=layer[0][0]
        self.vocab_bytes=len(json.dumps(self.keys).encode())
    def query(self,terms):
        q=np.zeros(len(self.keys),dtype=np.float64)
        for k,v in terms.items():
            if k in self.lookup:q[self.lookup[k]]=1+math.log(v)
        return q,2e-5*(1+self.scale*np.linalg.norm(q))
    def candidates(self,terms,stats):
        q,eps=self.query(terms)
        def leaf(j):
            stats["sprite_checks"]+=1
            if self.sprites[j].rejects(q,eps):stats["sprite_rejected"]+=1;return
            yield j
        def visit(n):
            if n.bounds is None:
                yield from leaf(n.begin);return
            stats["node_checks"]+=1
            if n.bounds.rejects(q,eps):
                stats["nodes_rejected"]+=1;stats["tree_cells_rejected"]+=n.end-n.begin;return
            for child in reversed(n.children):yield from visit(child)
        if self.tree:yield from visit(self.root)
        else:
            for j in range(len(self.sprites)-1,-1,-1):yield from leaf(j)
class Predictive(h13.Cache):
    """Bounded synchronous prefetch from observed transitions only.
    Use a spare slot; never evict an occupied cell speculatively."""
    def __init__(self):
        super().__init__(3);self.transitions=defaultdict(Counter);self.prev=None;self.prefetched=set();self.prefetch_loads=0;self.prefetch_used=0;self.demand_loads=0;self.demand_hits=0
    def demand(self,c,cells):
        if c.route in self.models:self.demand_hits+=1
        else:self.demand_loads+=1
        if c.route in self.prefetched:self.prefetched.remove(c.route);self.prefetch_used+=1
        m=super().get(c);self.prefetched.intersection_update(self.models)
        if self.prev is not None:self.transitions[self.prev][c.route]+=1
        self.prev=c.route
        return m
    def prefetch(self,cells):
        row=self.transitions[self.prev]
        if len(self.models)>=self.cap or sum(row.values())<3:return
        target,n=row.most_common(1)[0]
        if n/sum(row.values())<.7 or target in self.models:return
        c=cells[target-4];super().get(c);self.prefetched.add(target);self.prefetch_loads+=1
    def stats(self):return {**super().stats(),"prefetch_loads":self.prefetch_loads,"prefetch_used":self.prefetch_used,"demand_loads":self.demand_loads,"demand_hits":self.demand_hits}
def run(prep,rootpath,bundlepath,h10path,h11path,h12path,fixtures,size,mode,out):
    size=int(size);ck(size in SIZES and mode in MODES,"MODE")
    L,dd,root,d,mapping,labels=base_data(prep);rss_start=rss();t0=time.process_time()
    cells,coarse,mat,offsets,indexbytes=h13.catalog(L,d,mapping,labels,SEED,0,h11path,h12path);cells+=fixture_cells(L,d,mapping,labels,fixtures,size)
    full_index=sum(h13.sparse_bytes(c.ref)+len(json.dumps(c.rec["lex"]).encode()) for c in cells)
    del coarse,mat,offsets;gc.collect();rss_catalog=rss()
    atlas=None if mode=="exact" else Atlas(cells,mode.startswith("tree"));build_cpu=time.process_time()-t0;gc.collect();rss_index=rss()
    groups,_=h14.exams(prep,dd,d);hist=sum(groups,[]);ck(len(hist)==671,"HISTORICAL")
    raw=(HERE/"R84H15_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");ex=json.loads(raw);ck(ex["domains"]==DOMAINS and len(ex["rows"])==210,"PROBES")
    startid=max(labels.values())+1
    fixture_queries=[(t,i,k) for t,i,k in ex["rows"]];fit_norm={a.norm(t) for i in range(105) for t,_ in fixture_rows(i)}
    ck(not fit_norm&{a.norm(t) for t,_,_ in fixture_queries},"FIXTURE_COLLISION")
    rows=hist+[[t,"fixture"] for t,_,_ in fixture_queries];actual=[labels[y] for _,y in hist]+[startid+2*i+k for _,i,k in fixture_queries]
    # Base persistent inference requires a known dummy label only for encoding;
    # label is never used by gate or prediction. Replace fixture placeholder.
    dummy=next(iter(d["labels"]));base_rows=hist+[[t,dummy] for t,_,_ in fixture_queries]
    base,br,persistent=h13.persistent(L,root,d,SEED,0,rootpath,bundlepath,h10path,mapping,base_rows)
    order=list(range(len(rows)));random.Random(15001).shuffle(order)
    pred=list(base);routes=list(br);cache=Predictive() if mode.endswith("prefetch") else h13.Cache(3)
    stats={k:0 for k in ("gate_checks","boundary_checks","sprite_checks","sprite_rejected","node_checks","nodes_rejected","tree_cells_rejected","neural_forwards_rows")}
    rss_before_stream=rss();cpu=time.process_time();wall=time.perf_counter()
    for ix in order:
        terms=h10.terms(rows[ix][0]);visible=range(len(cells)-1,-1,-1) if atlas is None else atlas.candidates(terms,stats)
        for j in visible:
            c=cells[j]
            if not c.qualifies(terms,stats):continue
            m=cache.demand(c,cells) if isinstance(cache,Predictive) else cache.get(c);pred[ix]=c.predict(m,terms);del m;routes[ix]=c.route;stats["neural_forwards_rows"]+=1
            if isinstance(cache,Predictive):cache.prefetch(cells)
            break
    elapsed=time.process_time()-cpu;wall=time.perf_counter()-wall;stats.update(cache.stats());peak_rss=peak()
    hist_dep=sum(x==y for x,y in zip(pred[:len(hist)],actual[:len(hist)]))
    active=2*(size-15);fixture_correct=sum(x==y for x,y in zip(pred[len(hist):len(hist)+active],actual[len(hist):len(hist)+active]))
    row={"schema":"r84h15.run.v1","seed":SEED,"size":size,"real_cells":15,"fixture_cells":size-15,"mode":mode,"n":len(rows),"historical_n":len(hist),"historical_dep":hist_dep,"active_fixture_n":active,"active_fixture_correct":fixture_correct,"inactive_fixture_routes":sum(z>=19 for z in routes[len(hist)+active:]),"historical_fixture_intercepts":sum(z>=19 for z in routes[:len(hist)]),"persistent_weight_bytes":persistent,"full_gate_index_estimate_bytes":full_index,"compressed_sprite_bytes":atlas.sprite_bytes if atlas else 0,"uncompressed_sprite_bytes":atlas.uncompressed_sprite_bytes if atlas else 0,"hlod_bytes":atlas.tree_bytes if atlas else 0,"vocab_serialized_bytes":atlas.vocab_bytes if atlas else 0,"tree_depth":atlas.depth if atlas else 0,"tree_nodes":len(atlas.nodes) if atlas else 0,"rss_start_bytes":rss_start,"rss_catalog_bytes":rss_catalog,"rss_index_bytes":rss_index,"rss_before_stream_bytes":rss_before_stream,"rss_peak_bytes":peak_rss,"build_cpu_seconds":build_cpu,"stream_cpu_seconds":elapsed,"stream_wall_seconds":wall,**stats}
    log("R84H15_RESULT",row);ck(stats["max_resident_cells"]<=3,"CACHE_CAP");out.mkdir(parents=True,exist_ok=True);e.write(out/(mode+".json"),{"metrics":row,"pred":pred,"routes":routes,"actual":actual,"production":False});cache.close()
def compare(folder,size):
    size=int(size);records={m:e.read(folder/(m+".json")) for m in MODES};base=records["exact"];n=base["metrics"]["n"];result=[]
    for m,x in records.items():
        ck(x["actual"]==base["actual"] and x["metrics"]["size"]==size and x["production"] is False,"WORKLOAD")
        p=sum(u==v for u,v in zip(x["pred"],base["pred"]));z=sum(u==v for u,v in zip(x["routes"],base["routes"]))
        lost=sum(u==a and v!=a for u,v,a in zip(base["pred"],x["pred"],base["actual"]))
        metrics={**x["metrics"],"prediction_parity":p,"route_parity":z,"lost":lost};result.append(metrics);log("R84H15_COMPARE",metrics)
        ck(p==n and z==n and lost==0,"PARITY")
    e.write(folder/"comparison.json",{"schema":"r84h15.comparison.v1","size":size,"results":result,"production":False})
    log("R84H15_GATE",{"size":size,"parity":True,"production":False})
def summarize(folder,out):
    allrows=[]
    for size in SIZES:
        x=e.read(folder/("r84h15-scale-"+str(size))/"comparison.json");ck(x["size"]==size and x["production"] is False,"SIZE");allrows+=x["results"]
    # Original 671-prompt regression for seed111 is checked against the H14
    # exact reference separately; no claim that synthetic fixture interference is absent.
    out.mkdir(parents=True,exist_ok=True);e.write(out/"result.json",{"schema":"r84h15.result.v1","results":allrows,"production":False,"external_validation":False,"scope":"single seed111 load test; 15 original trained experts plus up to 105 independently trained synthetic two-class domain fixtures; identical 881 inputs across scales; exact routing parity within each scale, historical interference measured separately; directed uint16 bounds, multi-level HLOD, synchronous spare-slot prefetch based on past transitions only; isolated Linux process RSS includes runtime, catalogs, model reads and construction; not service RSS or network streaming; no promotion"})
    log("R84H15_SUMMARY",{"sizes":list(SIZES),"all_parity":True,"production":False})
if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="prepare":prepare(*map(Path,sys.argv[2:]))
    elif mode=="run":run(*[Path(x) for x in sys.argv[2:9]],sys.argv[9],sys.argv[10],Path(sys.argv[11]))
    elif mode=="compare":compare(Path(sys.argv[2]),sys.argv[3])
    elif mode=="summarize":summarize(*map(Path,sys.argv[2:]))
    else:raise ValueError("MODE")
