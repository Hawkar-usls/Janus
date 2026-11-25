"""R84H16 finite, addressed chunk store benchmark. No autonomous training or production writes."""
import sys,json,time,hashlib,sqlite3,random,gc
from pathlib import Path
from collections import OrderedDict
import r84h15_world as h15
import r84h14_partition as h14
import r84h13_streaming as h13
import r84h10_resonance as h10
import r84h11b_repair as repair
import r84e_neural as e
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
PROTOCOL="d78fdd5a95a954c4853a44be246718cb8c36eca1"
WORLD="JANUS-WORLD-v1"
SIZES=(120,10000,100000)
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def address(name):
    ck(isinstance(name,str) and 0<len(name)<=120 and name==name.strip().lower() and name.isascii(),"CANONICAL_NAMESPACE")
    return sha((WORLD+"\0"+name).encode())
def protocol():
    raw=(HERE/"R84H16_FROZEN_PROTOCOL.json").read_bytes();ck(r.blob(raw)==PROTOCOL,"PROTOCOL_PIN")
    p=json.loads(raw);ck(p["seed"]==WORLD and p["scales"]==list(SIZES),"PROTOCOL");return p
def pack(name,kind,payload):
    return {"address":address(name),"name":name,"world":WORLD,"schema":1,"kind":kind,"payload":payload}
class Store:
    def __init__(self,path,readonly=False):
        self.path=path
        self.db=sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True) if readonly else sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size=-2048");self.db.execute("PRAGMA temp_store=MEMORY")
        if not readonly:
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("CREATE TABLE IF NOT EXISTS chunks(address TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE,body TEXT NOT NULL,digest TEXT NOT NULL) WITHOUT ROWID")
    def put(self,record):
        name=record["name"];key=address(name);ck(record["address"]==key and record["world"]==WORLD,"ADDRESS")
        body=canonical(record);digest=sha(body.encode())
        old=self.db.execute("SELECT body,digest FROM chunks WHERE address=?",(key,)).fetchone()
        if old:
            ck(old==(body,digest),"IMMUTABLE_CONFLICT");return False
        self.db.execute("INSERT INTO chunks VALUES(?,?,?,?)",(key,name,body,digest));return True
    def get(self,name):
        key=address(name);row=self.db.execute("SELECT name,body,digest FROM chunks WHERE address=?",(key,)).fetchone()
        if row is None:return None
        n,body,digest=row;ck(n==name and sha(body.encode())==digest,"ROW_INTEGRITY")
        obj=json.loads(body);ck(obj["address"]==key and obj["name"]==name and obj["world"]==WORLD and obj["schema"]==1,"ROW_IDENTITY");return obj
    def count(self):return self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    def close(self):self.db.close()
def build(fixtures,size,out):
    size=int(size);p=protocol();ck(size in SIZES,"SIZE")
    out.mkdir(parents=True,exist_ok=False);path=out/"world.sqlite";s=Store(path);start=time.process_time()
    manifest=e.read(fixtures/"manifest.json");ck(manifest["count"]==105 and manifest["production"] is False and manifest["seed"]==h15.SEED,"FIXTURE_MANIFEST")
    with s.db:
        for i in range(15):
            pin=repair.PINS[i][0] if i<5 else h13.PINS[0][i-5]
            s.put(pack("legacy/cell-%02d"%i,"legacy",{"route":i+4,"model_sha256":pin}))
        for i,n in enumerate(h15.DOMAINS):
            pin=manifest["pins"][i];rec=e.read(fixtures/str(i)/"receipt.json")
            ck(pin["i"]==i and rec["fixture"]==i and rec["domain"]==n and rec["model_sha256"]==pin["sha"] and rec["file_sha256"]==pin["file"] and rec["fixture_only"] is True and rec["production"] is False,"RECEIPT")
            s.put(pack("fixture/"+n,"fixture_ready",{"i":i,"route":19+i,"model_sha256":pin["sha"],"file_sha256":pin["file"],"fixture_only":True}))
        for i in range(size-120):
            n=p["unknown"][i] if i<len(p["unknown"]) else "unexplored/chunk-%09d"%i
            s.put(pack(n,"empty",{"knowledge":False}))
    ck(s.count()==size,"COUNT")
    # Check idempotency and protection without weakening the immutable catalog.
    probe=s.get("fixture/"+h15.DOMAINS[0]);ck(not s.put(probe),"IDEMPOTENT")
    refused=False
    try:s.put(pack(probe["name"],"empty",{"knowledge":False}))
    except Exception:refused=True
    ck(refused,"IMMUTABLE_GUARD")
    bad=False
    try:address(" FIXTURE/INVALID ")
    except Exception:bad=True
    ck(bad,"CANONICAL_GUARD")
    ck(s.db.execute("PRAGMA quick_check").fetchone()[0]=="ok","SQLITE_CHECK")
    s.close();receipt={"schema":"r84h16.snapshot.v1","world":WORLD,"size":size,"ready_fixture":105,"legacy":15,"empty":size-120,"db_sha256":sha(path.read_bytes()),"db_bytes":path.stat().st_size,"build_cpu_seconds":time.process_time()-start,"production":False}
    e.write(out/"snapshot.json",receipt);log("R84H16_BUILD",receipt)
class Lazy:
    """A bounded metadata cache in addition to the independent bounded weight cache."""
    def __init__(self,L,d,mapping,labels,fixtures):
        self.L=L;self.d=d;self.mapping=mapping;self.labels=labels;self.fixtures=fixtures;self.cache=OrderedDict();self.loads=0;self.max_cells=0
    def get(self,record):
        p=record["payload"];i=p["i"];ck(0<=i<105 and record["name"]=="fixture/"+h15.DOMAINS[i] and p["route"]==19+i and p["fixture_only"] is True,"FIXTURE_ADDRESS")
        if i in self.cache:self.cache.move_to_end(i);return self.cache[i]
        folder=self.fixtures/str(i);rec=e.read(folder/"receipt.json");fo,fn=h15.banks(self.d,self.mapping,i)
        ck(rec["fixture"]==i and rec["domain"]==h15.DOMAINS[i] and rec["fixture_only"] is True and rec["production"] is False and rec["model_sha256"]==p["model_sha256"] and rec["file_sha256"]==p["file_sha256"] and rec["data_sha256"]==g.digest(self.d) and rec["ref_sha256"]==g.digest(fo+fn) and rec["lex"]==h10.lexical_fit(fo+fn),"LAZY_PIN")
        first=max(self.labels.values())+1;cell=h13.Cell(self.L,19+i,folder,rec,fo,fn,[first+2*i,first+2*i+1],p["model_sha256"],15000+i)
        if len(self.cache)>=3:self.cache.popitem(last=False)
        self.cache[i]=cell;self.loads+=1;self.max_cells=max(self.max_cells,len(self.cache));return cell
def evaluate(prep,rootpath,bundlepath,h10path,h11path,h12path,fixtures,snapshot,out):
    p=protocol();receipt=e.read(snapshot/"snapshot.json");path=snapshot/"world.sqlite";ck(receipt["world"]==WORLD and receipt["size"] in SIZES and receipt["production"] is False and sha(path.read_bytes())==receipt["db_sha256"],"SNAPSHOT_SHA")
    size=receipt["size"];s=Store(path,True);ck(s.count()==size and s.db.execute("PRAGMA quick_check").fetchone()[0]=="ok","REOPEN")
    # Repeated point queries, independently timed; no table scan in the lookup plan.
    plan=s.db.execute("EXPLAIN QUERY PLAN SELECT name,body,digest FROM chunks WHERE address=?",(address("fixture/"+h15.DOMAINS[0]),)).fetchall()
    ck(any("SEARCH" in x[3] for x in plan),"INDEXED_LOOKUP")
    names=["fixture/"+n for n in h15.DOMAINS]+p["unknown"];rng=random.Random(16001);probes=[rng.choice(names) for _ in range(12000)]
    rss0=h15.rss();t=time.process_time();hits=0
    for name in probes:
        q=s.get(name);hits+=int(q is not None and q["kind"]=="fixture_ready")
    lookup_cpu=time.process_time()-t
    L,dd,root,d,mapping,labels=h15.base_data(prep);cells,coarse,mat,offsets,indexbytes=h13.catalog(L,d,mapping,labels,h15.SEED,0,h11path,h12path)
    for i,c in enumerate(cells):
        rec=s.get("legacy/cell-%02d"%i);ck(rec["kind"]=="legacy" and rec["payload"]=={"route":c.route,"model_sha256":c.pin},"LEGACY_PIN")
    atlas=h14.Atlas(cells);groups,_=h14.exams(prep,dd,d);hist=sum(groups,[]);ck(len(hist)==671,"HIST")
    actual=[labels[y] for _,y in hist];base,br,persistent=h13.persistent(L,root,d,h15.SEED,0,rootpath,bundlepath,h10path,mapping,hist)
    ep,er,_=h13.eager(L,cells,hist,base,br);ck(sum(x==y for x,y in zip(ep,actual))==386,"H14_BASELINE")
    raw=(HERE/"R84H15_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==h15.EXAM,"FIXTURE_EXAM")
    ex=json.loads(raw);ck(ex["domains"]==h15.DOMAINS and len(ex["rows"])==210,"FIXTURE_ROWS")
    # Explicit namespace metadata is supplied to fixture requests. Historical
    # inputs are unaddressed, and therefore cannot be overridden by new chunks.
    # This intentionally does not claim a solved natural-language address router.
    requests=[(t,None,k) for k,(t,_) in enumerate(hist)]+[(t,"fixture/"+h15.DOMAINS[i],len(hist)+j) for j,(t,i,k) in enumerate(ex["rows"])]+[("unknown addressed request",n,len(hist)+210+j) for j,n in enumerate(p["unknown"])]
    rng=random.Random(16002);rng.shuffle(requests);pred=list(base)+[None]*(210+len(p["unknown"]));routes=list(br)+[None]*(210+len(p["unknown"]))
    weight_cache=h13.Cache(3);lazy=Lazy(L,d,mapping,labels,fixtures);stats={k:0 for k in ("gate_checks","boundary_checks","sprite_checks","sprite_rejected","region_checks","regions_rejected","region_cells_rejected")};abstained=0;address_lookups=0;fixture_gate_rejects=0
    gc.collect();rss_before=h15.rss();t=time.process_time();wall=time.perf_counter()
    for text,name,ix in requests:
        terms=h10.terms(text)
        if name is None:
            for j in atlas.visible(terms,"sprite",stats):
                c=cells[j]
                if c.qualifies(terms,stats):
                    pred[ix]=c.predict(weight_cache.get(c),terms);routes[ix]=c.route;break
        else:
            address_lookups+=1;rec=s.get(name)
            if rec is None or rec["kind"]=="empty":abstained+=1;continue
            ck(rec["kind"]=="fixture_ready","NAMESPACE_KIND");c=lazy.get(rec)
            if not c.qualifies(terms,stats):fixture_gate_rejects+=1;continue
            pred[ix]=c.predict(weight_cache.get(c),terms);routes[ix]=c.route
    stream_cpu=time.process_time()-t;stream_wall=time.perf_counter()-wall
    first=max(labels.values())+1;fixture_actual=[first+2*i+k for _,i,k in ex["rows"]]
    same=sum(x==y for x,y in zip(pred[:671],ep));rsame=sum(x==y for x,y in zip(routes[:671],er));hist_dep=sum(x==y for x,y in zip(pred[:671],actual));fixture_correct=sum(x==y for x,y in zip(pred[671:881],fixture_actual))
    ck(same==671 and rsame==671 and hist_dep==386,"LEGACY_PARITY");ck(abstained==len(p["unknown"]) and all(x is None for x in pred[881:]),"UNKNOWN_ABSTAIN")
    ck(fixture_correct==210 and fixture_gate_rejects==0,"FIXTURE_ACCURACY");ck(weight_cache.max_cells<=3 and lazy.max_cells<=3,"RESIDENCY")
    ck(s.count()==size,"READONLY_COUNT");s.close();ck(sha(path.read_bytes())==receipt["db_sha256"],"SNAPSHOT_UNCHANGED")
    row={"size":size,"actual_neural_cells":120,"empty_chunks":size-120,"historical_n":671,"historical_dep":hist_dep,"historical_prediction_parity":same,"historical_route_parity":rsame,"fixture_n":210,"fixture_correct":fixture_correct,"unknown_abstained":abstained,"address_lookup_n":len(probes),"address_lookup_ready_hits":hits,"address_lookup_cpu_seconds":lookup_cpu,"index_plan":plan,"stream_address_lookups":address_lookups,"stream_cpu_seconds":stream_cpu,"stream_wall_seconds":stream_wall,"rss_start_bytes":rss0,"rss_before_stream_bytes":rss_before,"peak_rss_bytes":h15.peak(),"db_bytes":receipt["db_bytes"],"db_sha256":receipt["db_sha256"],"lazy_metadata_loads":lazy.loads,"max_lazy_metadata_cells":lazy.max_cells,"persistent_weight_bytes":persistent,"fixture_gate_rejects":fixture_gate_rejects,**weight_cache.stats(),**stats,"production":False}
    weight_cache.close();out.mkdir(parents=True,exist_ok=True);e.write(out/"result.json",row);log("R84H16_RESULT",row)
    log("R84H16_GATE",{"size":size,"restart_integrity":True,"historical_parity":True,"fixture_accuracy":True,"unknown_abstention":True,"production":False})
if __name__=="__main__":
    if sys.argv[1]=="build":build(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]))
    elif sys.argv[1]=="evaluate":evaluate(*map(Path,sys.argv[2:]))
    else:raise ValueError("MODE")
