"""R7C.4: observed native failures -> real Beetle director -> EvoAgentX curriculum.
Research only. No live dispatch, exam reads, neural training, or admission.
"""
import base64, importlib.util, json, os, random, sys, tempfile, urllib.request
from pathlib import Path
from dataclasses import dataclass
import jdg_r7c3 as previous
import jfp_gateway as g

DEMI="Hawkar-usls/Janus-Demiurge"
SOURCES={
 "receipt":(DEMI,"e7d0f016cfb59c253ab2363cbeecf0facf06cc17","research/essence/trinity_parental_school/R1_REAL_CPU_DIAGNOSTIC_RESULT_20261123.json","0d5847fe97de0570a6aa4abd670ec2c1038083de"),
 "contract":(DEMI,"e7d0f016cfb59c253ab2363cbeecf0facf06cc17","tools/trinity_parental_contract.py","4636318842c64196c916c75a7b59fac0c876e2fa"),
 "director":(DEMI,"48aa4210f68fc59d00da5e4bc0a45ca4b16a1fe1","tools/demiurge_beetle_director.py","79f221c8ec6ab4e7cc82d0f4df9b35ca9409674d"),
 "bridge":(DEMI,"48aa4210f68fc59d00da5e4bc0a45ca4b16a1fe1","tools/essence_beetle_core_bridge.py","3b0b7ad61eb433a4568f7b4257833f2b859978a9"),
 "directives":(DEMI,"48aa4210f68fc59d00da5e4bc0a45ca4b16a1fe1","beetle/DEMIURGE_BEETLE_DIRECTIVES.json","8d29dcb8ff7a5e9c9c8725d095ae39f30140fee4")}
SEED=74
BUDGET=60
SPACE={"carry":[20,25,30,35,40,45,50,55,60,65,70,75,80]}
CKPT="06d7e6228ca9aef2dc58eb55fe1c805aa3b66e4f8fb90b0e946ce6af47b1a662"
check=previous.check
sha=previous.sha
def get_sources(root):
    result={}
    for key,(repo,rev,path,pin) in SOURCES.items():
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{rev}/{path}",timeout=30) as f:
            raw=f.read(262145)
        check(0<len(raw)<=262144 and previous.blob(raw)==pin,"SOURCE_PIN:"+key)
        result[key]=raw
        (root/(key+".txt")).write_bytes(raw)
    return result
def profile(src):
    r=json.loads(src["receipt"])
    check(r["context"]["original_promoted_checkpoint_sha256"]==CKPT,"CHECKPOINT")
    check(r["source"]["workflow_run"].endswith("/45191245034") and r["source"]["github_action_conclusion"]=="success","RUN")
    check(r["context"]["measurements"][0]["context"]==128,"CONTEXT")
    m=r["context"]["measurements"][0]
    check((m["all_cases"],m["fit"],m["exact"])==(36,24,0),"OBSERVED_OUTCOME")
    check(r["context"]["probe_scope"].startswith("24 training-only short tasks"),"PARTITION")
    check(r["known_exam_policy"]["exam_data_used_by_critic_or_context_probe"] is False,"EXAM_SCOPE")
    # Independently reconstruct the pinned training-only generator, never import its exam readers.
    c=src["contract"].decode()
    check(all(x in c for x in ('for i in range(24):','a = 11 + (7*i) % 38','b = 13 + (11*i) % 39','"partition": "TRAIN_ONLY"')),"TRAIN_GENERATOR")
    rows=[(11+(7*i)%38,13+(11*i)%39) for i in range(24)]
    n=sum(a%10+b%10>=10 for a,b in rows)
    check(0<n<24 and len(set(rows))==24,"STRATA")
    return {"checkpoint_sha256":CKPT,"observed_fit":24,"observed_exact":0,
      "observed_failures":24,"carry_failures":n,"no_carry_failures":24-n,
      "scope":"INFERRED_FROM_ALL_WRONG_TRAIN_ONLY_AGGREGATE","per_answer_text_available":False,
      "unmeasured_operations":["subtract","multiply"],"source_run":45191245034}
def desire(p):
    return [{"id":"adaptive_school_native_addition",
      "title":"Find source-verifiable two-digit addition teaching material, carry and no carry",
      "keywords":["two digit addition","place value","carry regrouping","worked examples","arithmetic learning","counterexample","heldout contamination"],
      "target_repositories":["Hawkar-usls/janus-meta-registry"],"priority":75}]
def load_director(root,src):
    path=root/"director.py";path.write_bytes(src["director"])
    spec=importlib.util.spec_from_file_location("pinned_beetle_director",path)
    (root/"essence_beetle_core_bridge.py").write_bytes(src["bridge"])
    sys.path.insert(0,str(root))
    try:
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    finally:
        sys.path.pop(0)
    return m
def route(root,src,p):
    m=load_director(root,src)
    allowed=set(json.loads(src["directives"])["allowed_target_repositories"])
    items=m.clean_adhoc(desire(p),allowed,"R7C4_RESEARCH_ONLY",max_items=1)
    check(len(items)==1 and items[0]["target_repositories"]==["Hawkar-usls/janus-meta-registry"],"BEETLE_ROUTE")
    bad=[dict(desire(p)[0],target_repositories=["unauthorized/target"])]
    check(m.clean_adhoc(bad,allowed,"R7C4_NEGATIVE",max_items=1)==[],"BEETLE_ALLOWLIST")
    return items
def metrics(p,cfg):
    check(set(cfg)=={"carry"} and type(cfg["carry"]) is int and cfg["carry"] in SPACE["carry"],"CFG")
    n=cfg["carry"]
    # Allocation proxy only: match the observed distribution of failed training examples.
    return {"score":float(2400-abs(24*n-100*p["carry_failures"])),"carry":float(n)}
@dataclass
class Program:
    carry:int=50
    def run(self):return {"carry":self.carry}
def replay(p,hist):
    check(len(hist)==BUDGET,"BUDGET")
    best={}
    for i,h in enumerate(hist):
        m=metrics(p,h["cfg"]);cell=min(4,int((h["cfg"]["carry"]-20)/60*5))
        check(h["step"]==i and h["source"] in ("mutate","random") and h["metrics"]==m and h["fitness"]==m["score"] and h["cell"]==[cell],"REPLAY")
        accept=cell not in best or m["score"]>best[cell]["fitness"]
        check(h["accepted"] is accept,"ARCHIVE")
        if accept:best[cell]={"cfg":h["cfg"],"fitness":m["score"]}
    check(bool(best),"EMPTY")
    return best
def scores(p,hist):
    b=replay(p,hist)
    rng=random.Random(SEED);rb={}
    for _ in range(BUDGET):
        n=rng.choice(SPACE["carry"]);cell=min(4,int((n-20)/60*5));s=metrics(p,{"carry":n})["score"]
        rb[cell]=max(rb.get(cell,0),s)
    cfg=max(b.values(),key=lambda x:x["fitness"])["cfg"]
    return cfg,{"map_elites_best":metrics(p,cfg)["score"],"random_best":max(rb.values()),
      "uniform_50_50":metrics(p,{"carry":50})["score"],"cells":len(b),"random_cells":len(rb),
      "metric":"allocation_alignment_proxy_NOT_learning"}
def tasks(cfg):
    pools={False:[],True:[]}
    # New operand domain, disjoint from the 24 diagnostic pairs; not certified against sealed exams.
    for a in range(51,100):
        for b in range(a,100):
            pools[a%10+b%10>=10].append((a,b))
    rows=[];rng=random.Random(SEED)
    for carry,n in ((True,cfg["carry"]),(False,100-cfg["carry"])):
        for a,b in rng.sample(pools[carry],n):
            rows.append({"id":f"adaptive-{len(rows):03}","a":a,"b":b,"op":"+","answer":a+b,
              "carry":carry,"hint":"Add ones, regroup a ten when needed, then add tens.",
              "partition":"TRAIN_ONLY_PROPOSAL"})
    return rows
def request():
    return {"schema":"janus.jdg.adaptive_school.v1","sources":{k:list(v) for k,v in SOURCES.items()},
       "upstream_revision":previous.REV,"upstream_pins":previous.PINS,"source_sha256":sha(Path(__file__).read_bytes()),
       "adapter_sha256":sha(Path(previous.__file__).read_bytes()),"run":os.environ["GITHUB_RUN_ID"],
       "attempt":os.environ["GITHUB_RUN_ATTEMPT"],"head":os.environ["GITHUB_SHA"],
       "training_authority":False,"action_authority":False}
def worker(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);src=get_sources(root);p=profile(src);routes=route(root,src,p)
        upstream=root/"upstream";upstream.mkdir();previous.obtain(upstream);sys.path.insert(0,str(upstream))
        from evoagentx.optimizers.map_elites_optimizer import MapElitesOptimizer
        from evoagentx.optimizers.engine.registry import ParamRegistry
        prog=Program();reg=ParamRegistry();reg.track(prog,"carry",name="carry")
        opt=MapElitesOptimizer(registry=reg,evaluator=lambda c:metrics(p,c),program=prog.run,
          search_space=SPACE,feature_dimensions=["carry"],feature_ranges={"carry":(20.,80.)},
          feature_bins=5,n_iterations=BUDGET,exploration_ratio=.6,random_seed=SEED)
        cfg,res=opt.optimize();hist=json.loads(g.canonical(res["history"]));best=replay(p,hist)
        check(best=={k[0]:{"cfg":v.cfg,"fitness":v.fitness} for k,v in res["archive"].items()},"UPSTREAM_ARCHIVE")
        chosen,summary=scores(p,hist);check(metrics(p,cfg)==metrics(p,chosen),"BEST_SCORE")
        data={"profile":p,"beetle_route_proposal":routes,"history":hist,"summary":summary,"allocation":chosen,
          "tasks":tasks(chosen),"beetle_live_dispatch":False,"external_materials_retrieved":False,
          "exam_overlap_audit":"PENDING","training_authority":False}
        raw=g.canonical(data);check(len(raw)<=previous.LIMIT,"LIMIT");(out/"ADAPTIVE.json").write_bytes(raw)
        (out/"UPSTREAM_LICENSE.txt").write_bytes((upstream/"LICENSE").read_bytes())
        k,pub=g.keys(root,"adaptive")
        report={"request":request(),"payload_sha256":sha(raw),"profile":p,"summary":summary,
          "task_count":100,"beetle_live_dispatch":False,"model_training":False,"training_authority":False}
        g.write_json(out/"RESULT.json",{"report":report,"signature":base64.b64encode(g.crypto(k,g.canonical(report))).decode()})
        (out/"PUBLIC.pem").write_bytes(pub.read_bytes())
        with open(os.environ["GITHUB_OUTPUT"],"a") as f:f.write("key_sha="+sha(pub.read_bytes())+"\n")
        print("JDG_R7C4_WORKER",json.dumps(report,sort_keys=True))
def verify(out,key_sha):
    out=Path(out);pub=out/"PUBLIC.pem";check(sha(pub.read_bytes())==key_sha,"KEY")
    e=g.read_json(out/"RESULT.json");check(set(e)=={"report","signature"},"ENVELOPE")
    r=e["report"];g.crypto(pub,g.canonical(r),base64.b64decode(e["signature"],validate=True))
    check(r["request"]==request(),"REQUEST")
    data=previous.read_data(out/"ADAPTIVE.json")
    check(sha(g.canonical(data))==r["payload_sha256"],"PAYLOAD")
    check(previous.blob((out/"UPSTREAM_LICENSE.txt").read_bytes())==previous.PINS["LICENSE"],"LICENSE")
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);src=get_sources(root);p=profile(src);routes=route(root,src,p)
        check(p==data["profile"]==r["profile"] and routes==data["beetle_route_proposal"],"SOURCE_REPLAY")
    cfg,summary=scores(p,data["history"]);check(cfg==data["allocation"] and summary==data["summary"]==r["summary"],"OPTIMIZER")
    rows=data["tasks"];check(rows==tasks(cfg) and len(rows)==r["task_count"]==100,"TASKS")
    check(len({(t["a"],t["b"]) for t in rows})==100,"DEDUP")
    for t in rows:
        check(t["answer"]-t["a"]==t["b"] and t["carry"]==(t["a"]%10+t["b"]%10>=10)
          and t["partition"]=="TRAIN_ONLY_PROPOSAL" and min(t["a"],t["b"])>=51,"ARITHMETIC")
    check(sum(t["carry"] for t in rows)==cfg["carry"],"ALLOCATION")
    check(not any((data["beetle_live_dispatch"],data["external_materials_retrieved"],data["training_authority"],r["model_training"],r["training_authority"])) and data["exam_overlap_audit"]=="PENDING","AUTHORITY")
    bad=json.loads(g.canonical(data));bad["tasks"][0]["answer"]+=1
    check(sha(g.canonical(bad))!=r["payload_sha256"] and bad["tasks"]!=tasks(cfg),"TAMPER")
    print("JDG_R7C4_VERIFIED",json.dumps({"status":"PASS_ADAPTIVE_PROPOSAL_ONLY","profile":p,"summary":summary,
      "allocation":cfg,"tasks":100,"beetle_director_executed":True,"beetle_live_dispatch":False,
      "model_training":False,"exam_overlap_audit":"PENDING"},sort_keys=True))
if __name__=="__main__":
    if sys.argv[1]=="worker":worker(sys.argv[2])
    elif sys.argv[1]=="verify":verify(sys.argv[2],sys.argv[3])
    else:raise ValueError("MODE")
