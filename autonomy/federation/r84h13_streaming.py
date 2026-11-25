"""R84H13 bounded world streaming: immutable catalog, sparse gates, lazy LRU expert cells."""
import sys,json,time,math,random,gc
from collections import OrderedDict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import r84h12_evolution as h12
import r84h11b_repair as repair
import r84h11_cortex as h11
import r84h10_resonance as h10
import r84h9_synapse as h9
import r84h8_neurogenesis as h8
import r84e_neural as e
import r84g_adaptive as a
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="8200691e4764de499eef34189763191f40876988"
PINS=[
["4de93080aa250e5bce2fce4438701fd2e0dde43ab8d3bcc1601cb54916e84bde","012eb72db49f6b8d1bdfd72b326bbb614e7dcab0ebbc796e9acc54a3142274e5","f35e7937998f988de6dfb68505db1ff0ecd042acc7f3fbdb6fddc21b689ca6","db228ce960bd4a47be02b9c4661f1302f243e2f3d95f177d72d8623812e40e76","698ef5516f8790bcb981f1ede417c65bee6a7dead4c0a17cc385effd2ed43159","ab82f1287b6c7f46caf619bf0ba67e9c8c2704662e69dbc4dde78e1de32aa530","76f03231863c8c2c54d8c9d01077957a4b4682f50fbb25135fead29cbdabc2c8","bc41824abdef4d93fd6814e88074117bdc7d25d99e0483cbbfb05dcb17386a44","5cdd7ae38f61f1b64ecbb18330b3eb994ab4a3103d34dcab0209de15d55caa06","622df6d388dbdf4d78bfe10022f62f27df41f3a473dc3b8f94f84b17a61cc059"],
["24614f84c1e9ffd5f910ad6b767eaa26708a856bb2621aea398252701aa4852d","963744e7701f7905b3c350d001a9a72dd9e26e5e39b523de7a636549325a6891","51be12795cd98485dec66017e52abf9be631a0e4f0570649621706d085ad139f","344747d2574d4bdf33d5716b9b0fccea7a83c60fa2a46c3e0ceb1ef590d6135f","54c370d224ba83eebf8c60bfc1363591570e794962b5ab906886df7e949eabc1","35b0d003ab99e7b0ad2c5c96da6a56455b603b01c5eead0c3c6167b5a3add465","729758e2bb55ffe66f0bd19f195873b079663513b00b58d4ac12e5eb294c521d","6fd6c8768f030ef9997a62d64388849e0127f09b8004d8887fead53bc7a9bea2","963332c2e10e7e3d3cc78cb3876b64bb05494e4fe19d5c1492a8c6bb1c7eb0ab","f01c4fd2bbb75762036fc299152040ed22848f0df2718ba0096457a8d5c7f262"],
["1677efcf4e93d9d0f478926e2ce974aa6ed94b9282bab8dfcf6465f6dd49bbaf","f06c977a313d63ba7a823eedab9c59217ab3bb114843599bdbcd321d254fdfc7","afc024b25b37434ff21594408fd897c815551d20ab77e3764d6d13695e3832f1","c04daf68430cff103e83ae6a706cd76245af0bb4e6b17846565d26d728921466","1f656c1dfecfcfa23d7926484745c2fa6bd25306ac46b0a5fc9293463625bccc","a6e72e4419f640a2f0d073387c94df3c449942905a22bebc3a827a32268ca42c","20e266c11b4cc9f6fe19962ed06e639452c89a3362fa4e0f1e21afd111fd8903","fe37043ebf40d6a943cf6b917cf2b41d6ecfb63dd7cc3ac3a13c71c089aede36","7c0fb0b1b1d54cb297ee7ba05b1f378db3a93696863a8bfd5f5cb1e70d018bca","26ce458f8d59921506bae2688dc72c79661c9914d2f0cf619a1ff1ab036a5ae2"]]
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def bytes_model(m):return sum(p.numel()*p.element_size() for p in m.parameters())+sum(p.numel()*p.element_size() for p in m.buffers())
def sparse_bytes(m):return m.data.nbytes+m.indices.nbytes+m.indptr.nbytes
class Lex:
    def __init__(self,lex):
        self.lex=lex;self.lookup={k:(i,w*(.6 if k.startswith("c:") else 1.)) for i,(k,w) in enumerate(zip(lex["keys"],lex["idf"]))}
    def vector(self,terms):
        x=np.zeros(len(self.lookup),dtype=np.float32)
        for k,v in terms.items():
            if k in self.lookup:
                i,w=self.lookup[k];x[i]=(1+math.log(v))*w
        n=np.linalg.norm(x)
        if n>0:x/=n
        return x
class Cell:
    def __init__(self,L,route,folder,rec,fo,fn,ids,pin,seed):
        self.L=L;self.route=route;self.folder=folder;self.rec=rec;self.ids=ids;self.pin=pin;self.seed=seed
        self.lex=Lex(rec["lex"]);self.nold=len(fo);self.ref=csr_matrix(h10.lexical(L,rec["lex"],fo+fn).numpy());self.positive=fn
        self.expected_bytes=rec["parameters"]*4
    def qualifies(self,terms,stats):
        x=self.lex.vector(terms);sim=self.ref@x;diff=float(sim[self.nold:].max()-sim[:self.nold].max());stats["gate_checks"]+=1
        # Match the historic float32 >= 0.5 rule near the decision boundary.
        if abs(diff)<=2e-6:
            if not np.any(x):return True
            torch,_=self.L;stats["boundary_checks"]+=1
            q=torch.from_numpy(x).reshape(1,-1);ref=torch.from_numpy(self.ref.toarray())
            return bool(h11.score_gate(self.L,q,ref,self.nold)[0]>=.5)
        return diff>0
    def load(self):
        torch,_=self.L;p=self.folder/"shadow.pt";raw=p.read_bytes()
        ck(e.sha(raw)==self.rec["file_sha256"],"FILE_SHA")
        import io
        m=h11.model(self.L,len(self.lex.lookup),self.seed);m.load_state_dict(torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True),strict=True);m.eval();m.requires_grad_(False)
        ck(e.state_hash(m)==self.pin and bytes_model(m)==self.expected_bytes,"WEIGHT_SHA")
        return m,len(raw)
    def predict(self,m,terms):
        torch,_=self.L
        with torch.no_grad():return self.ids[int(m(torch.from_numpy(self.lex.vector(terms)).reshape(1,-1)).argmax(-1)[0])]
class Cache:
    def __init__(self,cap):
        self.cap=cap;self.models=OrderedDict();self.loads=0;self.hits=0;self.evictions=0;self.read_bytes=0;self.resident=0;self.peak=0;self.max_cells=0
    def get(self,c):
        if c.route in self.models:
            self.hits+=1;self.models.move_to_end(c.route);return self.models[c.route]
        if len(self.models)>=self.cap:
            _,old=self.models.popitem(last=False);self.resident-=bytes_model(old);del old;self.evictions+=1
        m,n=c.load();self.loads+=1;self.read_bytes+=n;self.models[c.route]=m;self.resident+=bytes_model(m);self.peak=max(self.peak,self.resident);self.max_cells=max(self.max_cells,len(self.models))
        ck(len(self.models)<=self.cap,"CACHE_CAP");return m
    def stats(self):
        return {"loads":self.loads,"hits":self.hits,"evictions":self.evictions,"read_bytes":self.read_bytes,"peak_resident_weight_bytes":self.peak,"max_resident_cells":self.max_cells,"resident_cells":len(self.models)}
    def close(self):self.models.clear();self.resident=0
def catalog(L,d,mapping,labels,seed,i,h11path,h12path):
    cells=[];parent=h11.PINS[i]
    for cy in range(1,16):
        old=cy<=5;k=cy if old else cy-5;folder=(h11path if old else h12path)/str(k)/str(seed)
        rec=e.read(folder/"receipt.json");fo,fn=h11.banks(d,seed,k,"fit") if old else h12.banks(d,seed,k,mapping,"fit")
        pin=repair.PINS[k-1][i] if old else PINS[i][k-1];grp=h11.FIT[k-1] if old else h12.FIT[k-1]
        ck(rec["production"] is False and rec["seed"]==seed and rec["cycle"]==k and rec["parent_sha256"]==parent and rec["model_sha256"]==pin and rec["data_sha256"]==g.digest(d) and rec["ref_sha256"]==g.digest(fo+fn) and rec["lex"]==h10.lexical_fit(fo+fn),"CATALOG_PIN")
        if old:ck(rec["base_sha256"]==h11.PINS[i],"BASE_PIN")
        c=Cell(L,cy+3,folder,rec,fo,fn,[labels[y] for y in sorted(grp)],pin,seed+k);cells.append(c);parent=pin
    # Coarse positive-only level-of-detail directory. No neural weights are loaded here.
    positives=sum([c.positive for c in cells],[]);lex=h10.lexical_fit(positives);coarse=Lex(lex);mat=csr_matrix(h10.lexical(L,lex,positives).numpy())
    offsets=np.cumsum([0]+[len(c.positive) for c in cells])
    size=sum(sparse_bytes(c.ref)+len(json.dumps(c.rec["lex"]).encode()) for c in cells)+sparse_bytes(mat)+len(json.dumps(lex).encode())
    return cells,coarse,mat,offsets,size
def exams(prep,dd,d):
    groups,_=h12.exams(prep,dd,d);ck(sum(map(len,groups))==611,"PRIOR")
    raw=(HERE/"R84H13_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN")
    used=d["old"]+d["new"]+h10.extra(h10.FIT)+h10.extra(h10.DEV)+[row for grp in h11.FIT+h11.DEV+h12.FIT+h12.DEV for row in h10.extra(grp)]+h10.extra(repair.FIT)+h10.extra(repair.DEV)+sum(d["monitor"],[])+sum(groups,[])
    norm={a.norm(t) for t,_ in used};fresh=[]
    for grp in json.loads(raw)["cohorts"]:
        clean=[]
        for t,y in grp:
            s=a.norm(t)
            if s in norm:continue
            norm.add(s);clean.append([t,y])
        ck(len(clean)>=6,"FRESH");fresh.append(clean)
    log("R84H13_EXAM",{"prior_n":611,"fresh_n":list(map(len,fresh))});return groups+fresh,sum(map(len,fresh))
def persistent(L,root,d,seed,i,rootpath,bundlepath,h10path,mapping,rows):
    torch,_=L;core,b=h9.source(L,root,d,seed,i,rootpath,bundlepath);hr,f=h11.load_h10(L,d,h10path/str(seed),seed,i)
    _,_,_,cand=h9.inputs(L,core,b,root,d,rows);lex=hr["lex"];p=h10.prototype(L,h10.lexical(L,lex,rows),h10.lexical(L,lex,f),torch.tensor([mapping[y] for _,y in f]))
    z=h9.route(L,p,hr["thresholds"]["prototype"]["threshold"]);return cand[torch.arange(len(rows)),z].tolist(),z.tolist(),bytes_model(core)+bytes_model(b)
def eager(L,cells,rows,base,br):
    torch,_=L;cache=Cache(len(cells));pred=torch.tensor(base);routes=torch.tensor(br);start=time.process_time();wall=time.perf_counter()
    for c in cells:cache.get(c)
    load_cpu=time.process_time()-start;run=time.process_time();runwall=time.perf_counter()
    for c in cells:
        m=cache.get(c);x=h10.lexical(L,c.rec["lex"],rows);ref=torch.from_numpy(c.ref.toarray());take=h11.score_gate(L,x,ref,c.nold)>=.5
        with torch.no_grad():ans=torch.tensor(c.ids)[m(x).argmax(-1)]
        pred=torch.where(take,ans,pred);routes=torch.where(take,torch.full_like(routes,c.route),routes)
    s=cache.stats();s.update({"cpu_seconds":time.process_time()-start,"wall_seconds":time.perf_counter()-wall,"load_cpu_seconds":load_cpu,"inference_cpu_seconds":time.process_time()-run,"inference_wall_seconds":time.perf_counter()-runwall,"neural_forwards_rows":len(rows)*len(cells),"gate_checks":len(rows)*len(cells)})
    del m;cache.close();gc.collect();return pred.tolist(),routes.tolist(),s
def streaming(cells,coarse,mat,offsets,rows,base,br,mode,order):
    cache=Cache(3);pred=list(base);routes=list(br);stats={"gate_checks":0,"boundary_checks":0,"fallbacks":0,"neural_forwards_rows":0};start=time.process_time();wall=time.perf_counter()
    for ix in order:
        terms=h10.terms(rows[ix][0]);chosen=None
        if mode=="exact":visible=list(range(len(cells)-1,-1,-1))
        else:
            k=2 if mode=="camera2" else 4
            sim=mat@coarse.vector(terms);ranks=[float(sim[offsets[j]:offsets[j+1]].max()) for j in range(len(cells))]
            visible=sorted(sorted(range(len(cells)),key=lambda j:(ranks[j],j),reverse=True)[:k],reverse=True)
        for j in visible:
            if cells[j].qualifies(terms,stats):chosen=j;break
        if chosen is None and mode!="exact":
            stats["fallbacks"]+=1
            for j in range(len(cells)-1,-1,-1):
                if j not in visible and cells[j].qualifies(terms,stats):chosen=j;break
        if chosen is not None:
            c=cells[chosen];pred[ix]=c.predict(cache.get(c),terms);routes[ix]=c.route;stats["neural_forwards_rows"]+=1
    stats.update(cache.stats());stats.update({"cpu_seconds":time.process_time()-start,"wall_seconds":time.perf_counter()-wall})
    cache.close();gc.collect();return pred,routes,stats
def evaluate(prep,rootpath,bundlepath,h10path,h11path,h12path,out):
    L=e.lib(prep);torch,_=L;dd=h8.data(prep);ds=dd[-1];root,d=ds[0],ds[-1];mapping,labels=h12.maps(root,ds,d);groups,nf=exams(prep,dd,d);rows=sum(groups,[])
    actual=[labels[y] for _,y in rows];truth=[mapping[y] for _,y in rows];results=[]
    for i,seed in enumerate(e.SEEDS):
        start=time.process_time();cells,coarse,mat,offsets,indexbytes=catalog(L,d,mapping,labels,seed,i,h11path,h12path);indexcpu=time.process_time()-start
        start=time.process_time();base,br,persistentbytes=persistent(L,root,d,seed,i,rootpath,bundlepath,h10path,mapping,rows);basecpu=time.process_time()-start
        # Exact baseline is the original latest-qualified fixed gate with all expert weights resident.
        ep,er,es=eager(L,cells,rows,base,br);hist=sum(p==v for p,v in zip(ep[:-nf],actual[:-nf]))
        order=list(range(len(rows)));random.Random(seed+13000).shuffle(order)
        for mode in ("eager","exact","camera2","camera4"):
            p,z,s=(ep,er,es) if mode=="eager" else streaming(cells,coarse,mat,offsets,rows,base,br,mode,order)
            same=sum(x==y for x,y in zip(p,ep));routesame=sum(x==y for x,y in zip(z,er));lost=sum(x==v and y!=v for x,y,v in zip(ep,p,actual));gain=sum(x!=v and y==v for x,y,v in zip(ep,p,actual))
            row={"seed":seed,"mode":mode,"n":len(rows),"dep":sum(x==v for x,v in zip(p,actual)),"baseline_dep":sum(x==v for x,v in zip(ep,actual)),"historical_dep":sum(x==v for x,v in zip(p[:-nf],actual[:-nf])),"fresh_dep":sum(x==v for x,v in zip(p[-nf:],actual[-nf:])),"fresh_n":nf,"prediction_parity":same,"route_parity":routesame,"lost":lost,"gained":gain,"route_correct":sum(x==v for x,v in zip(z,truth)),"persistent_weight_bytes":persistentbytes,"serialized_index_bytes_estimate":indexbytes,"index_build_cpu_seconds":indexcpu,"persistent_batch_cpu_seconds":basecpu,**s}
            results.append(row);log("R84H13_RESULT",row)
            if mode=="exact":ck(s["max_resident_cells"]<=3,"STREAM_CAP")
    baseline=sum(x["historical_dep"] for x in results if x["mode"]=="eager");ck(baseline==997,"H12_BASELINE_REGRESSION")
    parity=all(x["prediction_parity"]==x["n"] and x["route_parity"]==x["n"] for x in results if x["mode"]=="exact")
    out.mkdir(parents=True,exist_ok=False);e.write(out/"result.json",{"schema":"r84h13.result.v1","results":results,"exact_parity":parity,"production":False,"external_validation":False,"scope":"finite 15 lazy expert cells plus persistent H8 core/bundle; same fixed gate precedence; exact reverse sparse gate scan vs approximate top2/top4 camera with fallback only if none accepts; LRU capacity 3; eager vectorized batch vs shuffled single-query streaming; weight bytes not process RSS; index bytes compressed arrays and serialized metadata estimate; local artifact file reads, no remote network streaming; CPU includes lazy load/hash; no weight changes/promotion"})
    log("R84H13_GATE",{"exact_parity":parity,"baseline_historical":baseline,"production":False})
if __name__=="__main__":evaluate(*[Path(x) for x in sys.argv[1:]])
