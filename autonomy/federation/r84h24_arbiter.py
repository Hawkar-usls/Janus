"""R84H24 finite cost-aware arbiter: exact graph, legacy neural, native retrieval, research quarantine."""
import sys,json,time,random,shutil,importlib
import r84h19_exploration as h19
import r84h18_fog as h18
import r84h12_evolution as h12
import r84h23_integration as h23
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
NEW_EXAM="fbda534b26226e3a4e0e16f897eddda1fe6003fa"
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
    def __init__(self,store):
        self.store=store;self.calls=0;self.exact_cpu=0.
    def solve(self,q):
        if not valid(q):return {"status":"REJECT","answer":None}
        t=time.process_time();record=self.store.get(NAME)
        ck(record["kind"]=="graph_candidate" and record["payload"]["weight_sha256"]==PIN,"GRAPH_CATALOG_PIN")
        answer=h22.bfs(q);self.exact_cpu+=time.process_time()-t;self.calls+=1
        return {"status":"GRAPH","answer":answer,"route":"exact_bfs"}
def evaluate(prep,rootpath,bundlepath,h10path,h11path,h12path,fixtures,snapshot,h17results,h21,h22trained,h22summary,h19results,h21sources,demi,out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    summary=e.read(h22summary/"summary.json");ck(summary["production"] is False and summary["historical_parity"]==671 and summary["graph_gates"]["propagation_trained"] and len(summary["candidates"])==1,"H22_GATE")
    candidate=summary["candidates"][0];ck(candidate["namespace"]==NAME and candidate["weight_sha256"]==PIN and candidate["auto_promote"] is False,"CANDIDATE")
    receipt=e.read(snapshot/"snapshot.json");source=snapshot/"world.sqlite";ck(receipt["size"]==100000 and h16.sha(source.read_bytes())==receipt["db_sha256"],"WORLD_PIN")
    path=out/"shadow-world.sqlite";shutil.copyfile(source,path);w=h16.Store(path)
    with w.db:
        rec=h16.pack(NAME,"graph_candidate",{"weight_sha256":PIN,"parameters":4,"production":False,"answer_authority":"exact_bfs","neural_shadow":True})
        ck(w.put(rec) and not w.put(rec),"CANDIDATE_INSERT")
    ck(w.count()==100001,"CATALOG_COUNT");w.close();overlay_sha=h16.sha(path.read_bytes());s=h16.Store(path,True)
    graph=Graph(s);router=h17.Router();L,dd,root,d,mapping,labels=h15.base_data(prep)
    raw=(HERE/"R84H24_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==NEW_EXAM,"NEW_EXAM_PIN");new=json.loads(raw)
    f,n=h12.banks(d,h15.SEED,10,mapping,"fit");v,w=h12.banks(d,h15.SEED,10,mapping,"dev");fit=h18.uniq(f+n);dev=h18.uniq(v+w)
    ck(len(fit)==553 and len(dev)==200,"SPLIT");det=h19.Evidence(L,fit)
    h19rec=e.read(h19results/"result.json");policy=h19rec["policy"];ck(h19rec["scientific"]=="PASS_SHADOW" and h19rec["production"] is False and policy["mode"]=="support" and abs(policy["unknown_below"]-.02964504361152649)<1e-12 and abs(policy["known_at_least"]-.07922915220260619)<1e-12,"POLICY_PIN")
    knownnorm={h18.clean(t) for t,_ in fit+dev};ck(not knownnorm&{h18.clean(t) for t in new["unknown"]},"NEW_COLLISION")
    for name,pin in (("memory_cortex_growth.py","626b74379e195d2e72d7c5b727198f6a3e297f04"),("memory_512_audit_index_r3.py","a01fe718018cd10bf4b0e5c1ae3b83dd90364ff3")):
        ck(r.blob((demi/"janus_model"/name).read_bytes())==pin,"CORTEX_PIN")
    sys.path.insert(0,str(demi.resolve()));m=importlib.import_module("janus_model.memory_cortex_growth")
    sourceaudit=e.read(h21sources/"source_result.json");memroot=h21sources/"cortex-shadow";memstate=m.load_state(memroot)
    ck(sourceaudit["native_index_documents"]==1 and len(memstate["active"])==1 and sourceaudit["external_metadata_admitted"]==0,"MEMORY_PARENT")
    memsha=memstate["active"][0];ck(memstate["records"][memsha]["origin"]["kind"]=="OWN_AUTHORED_SYNTHETIC_CONTROL","MEMORY_SCOPE")
    mempins={str(p.relative_to(memroot)):h16.sha(p.read_bytes()) for p in memroot.rglob("*") if p.is_file()}
    plans={};memory_cpu=0.;score_cpu=0.;memory_calls=0;flags={"historical":0,"fresh_unknown":0};route_counts={}
    def research(t,state,score):
        p=h19.study(t,state,score,policy);plans[p["id"]]=p;return p["id"]
    def familiarity(t):
        nonlocal score_cpu
        ts=time.process_time();score=det.score(t);state=h19.classify(score,policy);score_cpu+=time.process_time()-ts;return state,score
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
        nonlocal memory_cpu,memory_calls
        if type(q) is dict and q.get("schema")=="janus.memory.query.v1":
            if set(q)!={"schema","query"} or type(q["query"]) is not str or not 1<=len(q["query"])<=120 or not q["query"].strip():return {"status":"REJECT","answer":None}
            ts=time.process_time();z=m.query(memroot,q["query"],limit=5);m.checked(z);memory_cpu+=time.process_time()-ts;memory_calls+=1
            if z["citations"]:
                ck(all(x["sha256"]==memsha and x["content_is_untrusted_data"] for x in z["citations"]),"MEMORY_POINTER")
                return {"status":"MEMORY_HIT","citations":[x["sha256"] for x in z["citations"]]}
            state,score=familiarity(q["query"]);research(q["query"],"FRONTIER",score);return {"status":"RESEARCH","citations":[]}
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
        if status=="LEGACY":
            state,score=familiarity(q)
            if state!="KNOWN":
                research(q,state,score)
            # Compatibility outputs retained as classification candidates; do
            # not turn low familiarity into a verified factual answer.
            return {"status":status,"domains":domains,"pred":pred,"routes":routes,"research_flag":state!="KNOWN","familiarity":state}
        return {"status":status,"domains":domains,"pred":pred,"routes":routes,"research_flag":False}
    raw=(HERE/"R84H23_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");exam=json.loads(raw)["rows"];ck(len(exam)==1836 and all(h22.bfs(x)==x["expected"] for x in exam),"EXAM_REFERENCE")
    raw=(HERE/"R84H22_FROZEN_GRAPH_EXAM.json").read_bytes();ck(r.blob(raw)==h22.EXAM,"H22_EXAM_PIN")
    exclude={h22.key(z) for z in h22.parent(h21)}|{h22.key(z) for z in json.loads(raw)["cases"]}
    fresh=[z for z in exam if h22.key(z) not in exclude];ck(len(fresh)>=1700,"EXAM_OVERLAP")
    malformed=[None,[],{},{"schema":"other"},request({"edges":[0],"roots":[0],"target":True}),request({"edges":[2],"roots":[0],"target":0}),request({"edges":[0],"roots":[True],"target":0}),request({"edges":[],"roots":[0],"target":0}),request({"edges":[0],"roots":[0],"target":1}),request({"edges":[],"roots":[],"target":0}),request({"edges":[0]*625,"roots":[0]*25,"target":0}),{**request({"edges":[0],"roots":[0],"target":0}),"expected":0}]
    rng=random.Random(23073);stream=[("text",x) for x in textcases]+[("graph",z) for z in fresh]+[("bad",q) for q in malformed]+[("memory",q) for q in new["memory"]]+[("unknown",t) for t in new["unknown"]];rng.shuffle(stream)
    parity={};by={};permutation_n=0;permutation_exact=0;bad=0;memory_correct=0;unknown_n=0;bench=[];before=time.process_time()
    for kind,obj in stream:
        if kind=="text":
            cohort,i,t=obj;old=prior[(cohort,i)];calls=graph.calls
            q=dispatch(t,(base[i],br[i]) if cohort=="historical" else None);ck(graph.calls==calls,"TEXT_INTERCEPT")
            p=parity.setdefault(cohort,{"n":0,"exact":0});p["n"]+=1;p["exact"]+=int(all(q[k]==old[k] for k in ("status","domains","pred","routes")))
            if cohort=="historical":flags["historical"]+=int(q["research_flag"])
        elif kind=="memory":
            ans=dispatch({"schema":"janus.memory.query.v1","query":obj["query"]});memory_correct+=int(ans["status"]==("MEMORY_HIT" if obj["expected"]=="HIT" else "RESEARCH"))
            route_counts[ans["status"]]=route_counts.get(ans["status"],0)+1
        elif kind=="unknown":
            ans=dispatch(obj);unknown_n+=1;flags["fresh_unknown"]+=int(ans["research_flag"]);ck(not ans["pred"],"UNKNOWN_UNVERIFIED_ANSWER")
        elif kind=="bad":
            calls=graph.calls;ans=dispatch(obj);ck(ans=={"status":"REJECT","answer":None} and graph.calls==calls,"MALFORMED");bad+=1
        else:
            z=obj;q=request(z);ans=dispatch(q);ck(ans["answer"]==z["expected"],"EXACT_ORACLE")
            k=str(z["n"])+"/"+z["kind"];p=by.setdefault(k,{"n":0,"exact":0,"shadow":0,"class_n":[0,0],"class_shadow":[0,0]});p["n"]+=1;p["exact"]+=1;y=z["expected"];p["class_n"][y]+=1;bench.append(q)
            # One bounded deterministic permutation per graph.
            pq=permute(q,rng);perm=dispatch(pq);permutation_n+=1;permutation_exact+=int(perm["answer"]==ans["answer"]);bench.append(pq)
    elapsed=time.process_time()-before;ck(sum(x["exact"] for x in parity.values())==959 and permutation_exact==permutation_n and bad==len(malformed),"MIXED_GATE")
    ck(cache.max_cells<=3 and lazy.max_cells<=3 and memory_correct==len(new["memory"]),"CACHE_MEMORY")
    # Same runner, same typed inputs; compare exact-only path with H23 always
    # shadow route, measured separately. Includes native catalog lookup in both.
    baseline=h23.Graph(s,h22trained);t=time.process_time()
    for q in bench:
        ans=baseline.solve(q);ck(ans["answer"]==h22.bfs(q),"BASELINE")
    baseline_cpu=time.process_time()-t
    queue=sorted(plans.values(),key=lambda p:(p["score"],p["id"]))[:12]
    ck(h19.save_queue(out/"exploration.sqlite",queue)==len(queue) and h19.save_queue(out/"exploration.sqlite",queue)==0,"QUEUE")
    e.write(out/"plans.json",queue)
    ck(mempins=={str(p.relative_to(memroot)):h16.sha(p.read_bytes()) for p in memroot.rglob("*") if p.is_file()},"MEMORY_UNCHANGED")
    s.close();ck(h16.sha(source.read_bytes())==receipt["db_sha256"] and h16.sha(path.read_bytes())==overlay_sha,"IMMUTABLE")
    # Do not choose neural output over exact BFS; no advantage established.
    report={"schema":"r84h24.result.v1","motto":"Мы должны узнать","text_parity":parity,"historical_parity":parity["historical"]["exact"],"graph_exact":len(fresh),"graph_overlap_excluded":len(exam)-len(fresh),"graph_permutation_exact":permutation_exact,"malformed_rejected":bad,"memory_correct":memory_correct,"memory_n":len(new["memory"]),"memory_calls":memory_calls,"memory_routes":route_counts,"memory_cpu_seconds":memory_cpu,"familiarity_cpu_seconds":score_cpu,"flags":flags,"fresh_unknown_n":unknown_n,"research_candidates":len(plans),"quarantine_queue":len(queue),"exact_only_cpu_seconds":graph.exact_cpu,"always_shadow_cpu_seconds":baseline_cpu,"always_shadow_neural_cpu_seconds":baseline.shadow_cpu,"always_shadow_loads":baseline.loads,"primary_graph_neural_loads":0,"graph_calls":graph.calls,"max_text_weight_cells":cache.max_cells,"max_text_metadata_cells":lazy.max_cells,"peak_rss_bytes":h15.peak(),"mixed_stream_cpu_seconds":elapsed,"original_catalog_unchanged":True,"memory_unchanged":True,"scientific":"PASS_SHADOW" if flags["fresh_unknown"]>=new["controls"]["unknown_min_flagged"] else "HOLD","production":False,"scope":"bounded typed graph and memory dispatch; plain text familiarity side-channel, preserves historical candidates including false flags; fresh unknown probes have no verified answer; native Cortex read of single isolated original control, not live corpus; research proposals only, no BEETLE/TRINITY execution here; same-owner synthetic exam, no learned universal arbiter or promotion"}
    e.write(out/"result.json",report);e.log("R84H24_RESULT",report);cache.close()
if __name__=="__main__":evaluate(*map(Path,sys.argv[1:]))
