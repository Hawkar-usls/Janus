"""R84H14: bounded HLOD regions, competence sprites and cache locality audit."""
import sys,json,time,math,random,gc
from collections import deque
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,vstack
import r84h13_streaming as h13
import r84h12_evolution as h12
import r84h11b_repair as repair
import r84h11_cortex as h11
import r84h10_resonance as h10
import r84e_neural as e
import r84g_adaptive as a
import r8_learning as r
HERE=Path(__file__).parent
EXAM="864a0a544e7fd1f1bbcd82ff2f7b2d320122ab52"
MODES=("exact","sprite","hlod","hlod_adaptive")
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def sb(m):return h13.sparse_bytes(m)
def exams(prep,dd,d):
    groups,_=h13.exams(prep,dd,d);ck(sum(map(len,groups))==641,"PRIOR")
    raw=(HERE/"R84H14_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN")
    used=d["old"]+d["new"]+h10.extra(h10.FIT)+h10.extra(h10.DEV)+[row for grp in h11.FIT+h11.DEV+h12.FIT+h12.DEV for row in h10.extra(grp)]+h10.extra(repair.FIT)+h10.extra(repair.DEV)+sum(d["monitor"],[])+sum(groups,[])
    seen={a.norm(t) for t,_ in used};fresh=[]
    for grp in json.loads(raw)["cohorts"]:
        clean=[]
        for t,y in grp:
            s=a.norm(t)
            if s in seen:continue
            seen.add(s);clean.append([t,y])
        ck(len(clean)>=6,"FRESH");fresh.append(clean)
    log("R84H14_EXAM",{"prior_n":641,"fresh_n":list(map(len,fresh))})
    return groups+fresh,sum(map(len,fresh))
class Atlas:
    """Positive sprite scores are exact in raw coordinates; negative anchors are lower bounds.
    All term weights nonnegative. A common query norm cancels within each cell.
    A conservative margin protects floating point and H13's near-boundary fallback.
    HLOD upper envelope covers all children, common lower envelopes bound each child.
    Region bounds and sprite bounds never select an answer, only reject candidates.
    """
    def __init__(self,cells):
        self.keys=sorted({k for c in cells for k in c.lex.lookup});self.lookup={k:i for i,k in enumerate(self.keys)}
        self.scale=max(w for c in cells for _,w in c.lex.lookup.values());self.sprites=[];self.regions=[];self.n=len(cells);nkeys=len(self.keys)
        temp=[]
        for c in cells:
            keys=c.rec["lex"]["keys"];scales=np.array([c.lex.lookup[k][1] for k in keys],dtype=np.float64);idx=np.array([self.lookup[k] for k in keys])
            co=c.ref.astype(np.float64).multiply(scales).tocoo()
            ref=csr_matrix((co.data,(co.row,idx[co.col])),shape=(c.ref.shape[0],nkeys))
            neg=ref[:c.nold];pos=ref[c.nold:];ck(pos.shape[0]>0 and neg.shape[0]>0,"EMPTY")
            # At most 48 negative anchor rows. Deterministic and fit-only: favor
            # hard negatives near each positive plus high aggregate similarity.
            near=(neg@pos.T).toarray();anchors=[]
            for j in range(pos.shape[0]):
                for ix in np.argsort(-near[:,j],kind="stable")[:4]:
                    if int(ix) not in anchors:anchors.append(int(ix))
            for ix in np.argsort(-near.max(axis=1),kind="stable"):
                if int(ix) not in anchors:anchors.append(int(ix))
                if len(anchors)>=48:break
            anchors=anchors[:48];sprite=vstack([pos,neg[anchors]],format="csr")
            self.sprites.append((sprite,pos.shape[0]));temp.append((neg,pos))
        starts=[];matrix=[];nrow=0
        for sprite,np_ in self.sprites:
            starts.extend([nrow,nrow+np_]);matrix.append(sprite);nrow+=sprite.shape[0]
        self.matrix=vstack(matrix,format="csr");self.starts=np.array(starts,dtype=np.int64)
        # Contiguous regions preserve the original newest-first route precedence.
        for begin in range(0,len(cells),3):
            end=min(begin+3,len(cells));items=temp[begin:end];upper=np.maximum.reduce([np.asarray(p.max(axis=0).toarray()).ravel() for _,p in items])
            # Each common lower envelope is componentwise <= one negative
            # reference in every child, hence it is a lower bound for ALL.
            common_n=min(n.shape[0] for n,_ in items)
            score=sum(np.asarray(n[:common_n]@upper).ravel() for n,_ in items)
            inds=np.argsort(-score,kind="stable")[:16]
            lower=[np.minimum.reduce([n[int(j)].toarray().ravel() for n,_ in items]) for j in inds]
            bounds=csr_matrix(np.vstack([upper]+lower))
            self.regions.append((begin,end,bounds))
        self.sprite_bytes=sum(sb(x) for x,_ in self.sprites)+sb(self.matrix)
        self.region_bytes=sum(sb(m) for _,_,m in self.regions)
        self.vocab_bytes=len(json.dumps(self.keys).encode())
        self.anchor_rows=sum(x.shape[0]-n for x,n in self.sprites)
        self.positive_rows=sum(n for _,n in self.sprites)
    def query(self,terms):
        q=np.zeros(len(self.keys),dtype=np.float64)
        for k,v in terms.items():
            if k in self.lookup:q[self.lookup[k]]=1+math.log(v)
        margin=2e-5*(1+self.scale*np.linalg.norm(q))
        return q,margin
    def visible(self,terms,mode,stats):
        q,eps=self.query(terms)
        if mode=="sprite":
            score=self.matrix@q;vals=np.maximum.reduceat(score,self.starts).reshape(-1,2)
            ok=vals[:,0]+eps>=vals[:,1];stats["sprite_checks"]+=self.n;stats["sprite_rejected"]+=int((~ok).sum())
            return [i for i in range(self.n-1,-1,-1) if ok[i]]
        out=[]
        for begin,end,bounds in reversed(self.regions):
            v=bounds@q;stats["region_checks"]+=1
            if v[0]+eps<v[1:].max():
                stats["regions_rejected"]+=1;stats["region_cells_rejected"]+=end-begin;continue
            for j in range(end-1,begin-1,-1):
                m,n=self.sprites[j];v=m@q;stats["sprite_checks"]+=1
                if v[:n].max()+eps<v[n:].max():stats["sprite_rejected"]+=1;continue
                out.append(j)
        return out
class Adaptive(h13.Cache):
    """3..6 resident cells. Expand on recent misses, shrink after sustained locality.
    Only past and current accesses are used; no exam order lookahead.
    """
    def __init__(self):
        super().__init__(3);self.recent=deque(maxlen=12);self.changes=0;self.high=3
    def get(self,c):
        self.recent.append(int(c.route not in self.models));target=self.cap
        if len(self.recent)>=8 and sum(self.recent)>=math.ceil(.75*len(self.recent)):target=min(6,self.cap+1)
        elif len(self.recent)==12 and sum(self.recent)<=2:target=max(3,self.cap-1)
        if target!=self.cap:self.cap=target;self.changes+=1
        self.high=max(self.high,self.cap)
        while len(self.models)>self.cap:
            _,old=self.models.popitem(last=False);self.resident-=h13.bytes_model(old);del old;self.evictions+=1
        return super().get(c)
    def stats(self):return {**super().stats(),"cache_capacity_changes":self.changes,"max_capacity":self.high}
def stream(cells,atlas,rows,base,br,mode,order):
    cache=Adaptive() if mode=="hlod_adaptive" else h13.Cache(3)
    pred=list(base);routes=list(br);stats={k:0 for k in ("gate_checks","boundary_checks","sprite_checks","sprite_rejected","region_checks","regions_rejected","region_cells_rejected","neural_forwards_rows")}
    start=time.process_time();wall=time.perf_counter()
    for ix in order:
        terms=h10.terms(rows[ix][0]);visible=range(len(cells)-1,-1,-1) if mode=="exact" else atlas.visible(terms,"sprite" if mode=="sprite" else "hlod",stats)
        for j in visible:
            c=cells[j]
            if c.qualifies(terms,stats):
                pred[ix]=c.predict(cache.get(c),terms);routes[ix]=c.route;stats["neural_forwards_rows"]+=1;break
    stats.update(cache.stats());stats.update({"cpu_seconds":time.process_time()-start,"wall_seconds":time.perf_counter()-wall})
    cache.close();gc.collect();return pred,routes,stats
def evaluate(prep,rootpath,bundlepath,h10path,h11path,h12path,out):
    L=e.lib(prep);dd=h12.h8.data(prep);ds=dd[-1];root,d=ds[0],ds[-1];mapping,labels=h12.maps(root,ds,d);groups,nf=exams(prep,dd,d);rows=sum(groups,[])
    actual=[labels[y] for _,y in rows];results=[];baseline_hist=0
    for i,seed in enumerate(e.SEEDS):
        cells,coarse,mat,offsets,indexbytes=h13.catalog(L,d,mapping,labels,seed,i,h11path,h12path)
        start=time.process_time();atlas=Atlas(cells);build=time.process_time()-start
        base,br,persistent=h13.persistent(L,root,d,seed,i,rootpath,bundlepath,h10path,mapping,rows)
        ep,er,es=h13.eager(L,cells,rows,base,br);baseline_hist+=sum(x==v for x,v in zip(ep[:-nf],actual[:-nf]))
        shuffled=list(range(len(rows)));random.Random(seed+14000).shuffle(shuffled)
        # A separate thematic order uses only input text similarity, never labels.
        # Here use coarse highest-similarity cell and stable input order; offline
        # replay scenario, not a claim about real online request scheduling.
        def cluster(ix):
            sim=mat@coarse.vector(h10.terms(rows[ix][0]))
            vals=[float(sim[offsets[j]:offsets[j+1]].max()) for j in range(len(cells))]
            return max(range(len(vals)),key=lambda j:(vals[j],j))
        topical=sorted(shuffled,key=cluster)
        for workload,order in (("shuffled",shuffled),("topical",topical)):
            for mode in MODES:
                p,z,s=stream(cells,atlas,rows,base,br,mode,order)
                same=sum(x==y for x,y in zip(p,ep));rsame=sum(x==y for x,y in zip(z,er));lost=sum(x==v and y!=v for x,y,v in zip(ep,p,actual))
                row={"seed":seed,"mode":mode,"workload":workload,"n":len(rows),"dep":sum(x==v for x,v in zip(p,actual)),"historical_dep":sum(x==v for x,v in zip(p[:-nf],actual[:-nf])),"fresh_dep":sum(x==v for x,v in zip(p[-nf:],actual[-nf:])),"fresh_n":nf,"prediction_parity":same,"route_parity":rsame,"lost":lost,"eager_dep":sum(x==v for x,v in zip(ep,actual)),"persistent_weight_bytes":persistent,"base_index_serialized_estimate":indexbytes,"sprite_sparse_bytes":atlas.sprite_bytes,"hlod_sparse_bytes":atlas.region_bytes,"atlas_vocab_serialized_bytes":atlas.vocab_bytes,"sprite_positive_rows":atlas.positive_rows,"sprite_anchor_rows":atlas.anchor_rows,"atlas_build_cpu_seconds":build,"regions":len(atlas.regions),**s}
                log("R84H14_RESULT",row);results.append(row)
                ck(same==len(rows) and rsame==len(rows),"PARITY");ck(s["max_resident_cells"]<=(6 if mode=="hlod_adaptive" else 3),"CACHE_CAP")
    ck(baseline_hist==1058,"H13_REGRESSION")
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h14.result.v1","results":results,"production":False,"external_validation":False,"scope":"finite HLOD region bounds, sparse positive sprites and negative anchor lower bounds; conservative reject only; exact routing and weights unchanged; same online replay order within workload; topical ordering is offline simulation based on input similarity; adaptive 3..6 cell cache, no prefetch or future lookahead; CPU includes lazy weight loads, excludes common catalog/base/atlas build; sparse bytes and weight bytes not RSS; no network streaming or promotion"})
    log("R84H14_GATE",{"baseline_historical":baseline_hist,"all_parity":True,"production":False})
if __name__=="__main__":evaluate(*[Path(x) for x in sys.argv[1:]])
