"""R84H23 bounded mixed typed routing: exact graph answer, neural shadow, no deployment."""
import sys,json,time,random,shutil
from pathlib import Path
import torch
import r84h22_mastery as h22
import r84h17_biome as h17
import r84h16_chunk as h16
import r84h15_world as h15
import r84h14_partition as h14
import r84h13_streaming as h13
import r84h10_resonance as h10
import r8_learning as r
import r84e_neural as e
HERE=Path(__file__).parent
EXAM="09e9a46ce46661e8d11fb6d8dc023eb030700f66"
PIN="0e5cd399d7105028e38855ca4730e32eedd7d2a66b85a9e8556cc5280c6c7e14"
NAME="candidate/tracing-reachability/propagation"
def ck(v,m):e.ck(v,m)
def valid(q):
    if type(q) is not dict or set(q)!={"schema","edges","roots","target"} or q["schema"]!="janus.graph.reachability.v1":return False
    roots=q["roots"];edges=q["edges"];target=q["target"]
    if type(roots) is not list or not 1<=len(roots)<=24 or type(edges) is not list or len(edges)!=len(roots)**2:return False
    return type(target) is int and 0<=target<len(roots) and all(type(x) is int and x in (0,1) for x in roots+edges)
def request(z):return {"schema":"janus.graph.reachability.v1",**{k:z[k] for k in ("edges","roots","target")}}
def permute(z,rng):
    n=len(z["roots"]);p=list(range(n));rng.shuffle(p)
    return {"schema":z["schema"],"roots":[z["roots"][i] for i in p],"edges":[z["edges"][i*n+j] for i in p for j in p],"target":p.index(z["target"])}
class Graph:
    def __init__(self,store,trained):
        self.store=store;self.trained=trained;self.model=None;self.loads=0;self.calls=0;self.exact_cpu=0.;self.shadow_cpu=0.;self.load_cpu=0.
    def solve(self,q):
        if not valid(q):return {"status":"REJECT","answer":None}
        record=self.store.get(NAME)
        ck(record["kind"]=="graph_candidate" and record["payload"]["weight_sha256"]==PIN,"GRAPH_CATALOG_PIN")
        if self.model is None:
            t=time.process_time();p=self.trained/"propagation.pt";ck(h22.sha(p.read_bytes())==PIN,"WEIGHT_PIN")
            self.model=h22.PropagationNet();self.model.load_state_dict(torch.load(p,map_location="cpu",weights_only=True));self.model.eval()
            self.loads+=1;self.load_cpu+=time.process_time()-t
        t=time.process_time();answer=h22.bfs(q);self.exact_cpu+=time.process_time()-t
        t=time.process_time()
        with torch.no_grad():shadow=int(self.model(*h22.tensors([q])).argmax(-1).item())
        self.shadow_cpu+=time.process_time()-t;self.calls+=1
        return {"status":"GRAPH","answer":answer,"shadow":shadow,"route":"exact_bfs","candidate":NAME}
def evaluate(prep,rootpath,bundlepath,h10path,h11path,h12path,fixtures,snapshot,h17results,h21,h22trained,h22summary,out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    summary=e.read(h22summary/"summary.json");ck(summary["production"] is False and summary["historical_parity"]==671 and summary["graph_gates"]["propagation_trained"] and len(summary["candidates"])==1,"H22_GATE")
    candidate=summary["candidates"][0];ck(candidate["namespace"]==NAME and candidate["weight_sha256"]==PIN and candidate["auto_promote"] is False,"CANDIDATE")
    receipt=e.read(snapshot/"snapshot.json");source=snapshot/"world.sqlite";ck(receipt["size"]==100000 and h16.sha(source.read_bytes())==receipt["db_sha256"],"WORLD_PIN")
    path=out/"shadow-world.sqlite";shutil.copyfile(source,path);w=h16.Store(path)
    with w.db:
        rec=h16.pack(NAME,"graph_candidate",{"weight_sha256":PIN,"parameters":4,"production":False,"answer_authority":"exact_bfs","neural_shadow":True})
        ck(w.put(rec) and not w.put(rec),"CANDIDATE_INSERT")
    ck(w.count()==100001,"CATALOG_COUNT");w.close();overlay_sha=h16.sha(path.read_bytes());s=h16.Store(path,True)
    graph=Graph(s,h22trained);router=h17.Router();L,dd,root,d,mapping,labels=h15.base_data(prep)
    cells,_,_,_,_=h13.catalog(L,d,mapping,labels,h15.SEED,0,h11path,h12path)
    for i,c in enumerate(cells):ck(s.get("legacy/cell-%02d"%i)["payload"]=={"route":c.route,"model_sha256":c.pin},"LEGACY_PIN")
    atlas=h14.Atlas(cells);groups,_=h14.exams(prep,dd,d);hist=sum(groups,[]);ck(len(hist)==671,"HIST")
    base,br,_=h13.persistent(L,root,d,h15.SEED,0,rootpath,bundlepath,h10path,mapping,hist)
    prior={(x["cohort"],x["i"]):x for x in e.read(h17results/"cases.json")};ck(len(prior)==959,"H17_CASES")
    raw=(HERE/"R84H17_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==h17.EXAM,"H17_PIN");old=json.loads(raw)
    raw=(HERE/"R84H15_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==h15.EXAM,"H15_PIN");orig=json.loads(raw)
    textcases=[("historical",i,t) for i,(t,_) in enumerate(hist)]+[("original",i,t) for i,(t,_,_) in enumerate(orig["rows"])]
    for k in ("fresh","cross","ambiguous","unknown","mixed","openUnknown"):textcases.extend((k,i,x["text"]) for i,x in enumerate(old[k]))
    ck(len(textcases)==959,"TEXT_SIZE")
    cache=h13.Cache(3);lazy=h16.Lazy(L,d,mapping,labels,fixtures);stats={k:0 for k in ("gate_checks","boundary_checks","sprite_checks","sprite_rejected","region_checks","regions_rejected","region_cells_rejected")}
    def dispatch(q,legacy=None):
        # The same entry point handles text and typed graph inputs; graph shape
        # validation is never applied to plain text, and malformed objects do not fall back.
        if type(q) is not str:return graph.solve(q)
        status,plan=router.plan(q);domains=[x["domain"] for x in plan];pred=[];routes=[]
        if status=="LEGACY" and legacy is not None:
            y,z=legacy;terms=h10.terms(q)
            for j in atlas.visible(terms,"sprite",stats):
                c=cells[j]
                if c.qualifies(terms,stats):y=c.predict(cache.get(c),terms);z=c.route;break
            pred=[y];routes=[z]
        elif status in ("SINGLE","CROSS"):
            records=[]
            for p in plan:
                rec=s.get(p["address"])
                if rec is None or rec["kind"]!="fixture_ready":status="FOG";break
                records.append(rec)
            if status!="FOG":
                for p,rec in zip(plan,records):
                    c=lazy.get(rec);terms=h10.terms(p["text"])
                    if c.qualifies(terms,stats):pred.append(c.predict(cache.get(c),terms));routes.append(c.route)
                    else:pred.append(None);routes.append(None)
        return {"status":status,"domains":domains,"pred":pred,"routes":routes}
    raw=(HERE/"R84H23_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");exam=json.loads(raw)["rows"];ck(len(exam)==1836 and all(h22.bfs(x)==x["expected"] for x in exam),"EXAM_REFERENCE")
    raw=(HERE/"R84H22_FROZEN_GRAPH_EXAM.json").read_bytes();ck(r.blob(raw)==h22.EXAM,"H22_EXAM_PIN")
    exclude={h22.key(z) for z in h22.parent(h21)}|{h22.key(z) for z in json.loads(raw)["cases"]}
    fresh=[z for z in exam if h22.key(z) not in exclude];ck(len(fresh)>=1700,"EXAM_OVERLAP")
    malformed=[None,[],{},{"schema":"other"},request({"edges":[0],"roots":[0],"target":True}),request({"edges":[2],"roots":[0],"target":0}),request({"edges":[0],"roots":[True],"target":0}),request({"edges":[],"roots":[0],"target":0}),request({"edges":[0],"roots":[0],"target":1}),request({"edges":[],"roots":[],"target":0}),request({"edges":[0]*625,"roots":[0]*25,"target":0}),{**request({"edges":[0],"roots":[0],"target":0}),"expected":0}]
    rng=random.Random(23073);stream=[("text",x) for x in textcases]+[("graph",z) for z in fresh]+[("bad",q) for q in malformed];rng.shuffle(stream)
    parity={};by={};permutation_n=0;permutation_exact=0;permutation_shadow=0;bad=0;before=time.process_time()
    for kind,obj in stream:
        if kind=="text":
            cohort,i,t=obj;old=prior[(cohort,i)];calls=graph.calls
            q=dispatch(t,(base[i],br[i]) if cohort=="historical" else None);ck(graph.calls==calls,"TEXT_INTERCEPT")
            p=parity.setdefault(cohort,{"n":0,"exact":0});p["n"]+=1;p["exact"]+=int(all(q[k]==old[k] for k in ("status","domains","pred","routes")))
        elif kind=="bad":
            calls=graph.calls;ans=dispatch(obj);ck(ans=={"status":"REJECT","answer":None} and graph.calls==calls,"MALFORMED");bad+=1
        else:
            z=obj;q=request(z);ans=dispatch(q);ck(ans["answer"]==z["expected"],"EXACT_ORACLE")
            k=str(z["n"])+"/"+z["kind"];p=by.setdefault(k,{"n":0,"exact":0,"shadow":0,"class_n":[0,0],"class_shadow":[0,0]});p["n"]+=1;p["exact"]+=1;y=z["expected"];p["shadow"]+=int(ans["shadow"]==y);p["class_n"][y]+=1;p["class_shadow"][y]+=int(ans["shadow"]==y)
            # One bounded deterministic permutation per graph.
            perm=dispatch(permute(q,rng));permutation_n+=1;permutation_exact+=int(perm["answer"]==ans["answer"]);permutation_shadow+=int(perm["shadow"]==ans["shadow"])
    elapsed=time.process_time()-before;ck(sum(x["exact"] for x in parity.values())==959 and permutation_exact==permutation_n and bad==len(malformed),"MIXED_GATE")
    ck(cache.max_cells<=3 and lazy.max_cells<=3 and graph.loads==1,"CACHE")
    s.close();ck(h16.sha(source.read_bytes())==receipt["db_sha256"] and h16.sha(path.read_bytes())==overlay_sha,"IMMUTABLE")
    # Do not choose neural output over exact BFS; no advantage established.
    report={"schema":"r84h23.result.v1","motto":"Мы должны узнать","text_parity":parity,"historical_parity":parity["historical"]["exact"],"graph_frozen_n":len(exam),"graph_overlap_excluded":len(exam)-len(fresh),"graph_cohorts":by,"graph_exact":len(fresh),"graph_shadow":sum(x["shadow"] for x in by.values()),"permutation_n":permutation_n,"permutation_exact":permutation_exact,"permutation_shadow":permutation_shadow,"malformed_rejected":bad,"graph_loads":graph.loads,"graph_calls":graph.calls,"exact_cpu_seconds":graph.exact_cpu,"neural_cpu_seconds":graph.shadow_cpu,"neural_load_cpu_seconds":graph.load_cpu,"stream_cpu_seconds":elapsed,"max_text_weight_cells":cache.max_cells,"max_text_metadata_cells":lazy.max_cells,"graph_scalar_parameters":4,"overlay_catalog_size":100001,"overlay_sha256":overlay_sha,"original_catalog_unchanged":True,"answer_authority":"exact_bfs","neural_promotion":False,"production":False,"scope":"same-owner frozen typed graph benchmark and mixed H17 replay; historical base prepared in batch, no general text runtime or natural-language graph extraction; candidate shadow only, disposable overlay; no training"}
    e.write(out/"result.json",report);e.log("R84H23_RESULT",report);cache.close()
if __name__=="__main__":evaluate(*map(Path,sys.argv[1:]))
