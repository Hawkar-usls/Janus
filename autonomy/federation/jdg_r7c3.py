"""JDG R7C.3: actual MIT EvoAgentX MAP-Elites, CPU curriculum proposal only."""
import base64, hashlib, json, os, random, sys, tempfile, urllib.request
from pathlib import Path
from dataclasses import dataclass
import jfp_gateway as g

UPSTREAM="ANative-Lab/EvoAgentX"
REV="e3df900662443e94dd92141be5633b040a69a70a"
PINS={
 "evoagentx/optimizers/map_elites_optimizer.py":"f135733bcd8b0605a71231308d79a67a2f82e13b",
 "evoagentx/optimizers/engine/base.py":"c21210f8730308fdea422d6b2ffe722b5ec114fa",
 "evoagentx/optimizers/engine/decorators.py":"376bc5d5d496bbad2a433842e8a9b781fbe9a491",
 "evoagentx/optimizers/engine/registry.py":"bac52998bad5fdc0402d7b46ba28e72c0a2ad5bc",
 "LICENSE":"c5b7391d0674144008cabf42f8860ea0ba0234d4"}
SPACE={"op":[0,1,2],"difficulty":[1,2,3],"variant":[0,1,2,3]}
# Synthetic prior: successful answers out of 12, by operation and difficulty.
# Not measured JANUS performance; not an independent examination.
COUNTS=[[11,7,3],[10,6,2],[8,4,1]]
SEED=71
BUDGET=120
LIMIT=262144
def check(ok,why):
    if not ok: raise ValueError(why)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def blob(raw): return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def request():
    return {"schema":"janus.jdg.curriculum.request.v1","capability":"curriculum.map_elites.evoagentx.v1",
      "provider":UPSTREAM,"revision":REV,"pins":PINS,"seed":SEED,"budget":BUDGET,
      "fixture":"synthetic-train-only-v1","source_sha256":sha(Path(__file__).read_bytes()),
      "run":os.environ["GITHUB_RUN_ID"],"attempt":os.environ["GITHUB_RUN_ATTEMPT"],
      "head":os.environ["GITHUB_SHA"],"action_authority":False,"training_authority":False}
def obtain(root):
    for path,pin in PINS.items():
        url=f"https://raw.githubusercontent.com/{UPSTREAM}/{REV}/{path}"
        with urllib.request.urlopen(url,timeout=30) as r: data=r.read(131073)
        check(0<len(data)<=131072 and blob(data)==pin,"UPSTREAM_BLOB_PIN:"+path)
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    # Explicit dependency adapter: the generic upstream registry imports PromptTemplate
    # even for integer attributes. No prompt APIs supported or simulated.
    p=root/"evoagentx/prompts/template.py";p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("class PromptTemplate:\n    def __init__(self,*a,**kw):\n        raise RuntimeError('prompt API not supported')\n")
    for name in ["evoagentx","evoagentx/optimizers","evoagentx/optimizers/engine","evoagentx/prompts"]:
        (root/name/"__init__.py").write_text("")
def metric(cfg):
    check(set(cfg)==set(SPACE) and all(type(cfg[k]) is int and cfg[k] in SPACE[k] for k in SPACE),"CONFIG")
    o,d,v=(cfg[k] for k in ("op","difficulty","variant"))
    n=max(0,min(12,COUNTS[o][d-1]+v-1))
    # Prefer frontier tasks (half solved), with a tie preference for weaker operation.
    return {"score":float(12-abs(2*n-12))+(2-o)*0.0,"op":float(o),"difficulty":float(d)}
@dataclass
class Program:
    op:int=0
    difficulty:int=1
    variant:int=0
    def run(self):return vars(self).copy()
def reduce_history(history):
    check(type(history) is list and len(history)==BUDGET,"BUDGET")
    best={}
    for i,h in enumerate(history):
        check(h["step"]==i and h["source"] in ("random","mutate"),"HISTORY")
        cfg=h["cfg"];m=metric(cfg);cell=(cfg["op"],cfg["difficulty"]-1)
        check(h["metrics"]==m and h["fitness"]==m["score"] and list(cell)==h["cell"],"FITNESS")
        accepted=cell not in best or m["score"]>best[cell]["fitness"]
        check(h["accepted"] is accepted,"ARCHIVE_RULE")
        if accepted:best[cell]={"cfg":cfg,"fitness":m["score"]}
    return best
def baseline():
    rng=random.Random(SEED)
    best={}
    for _ in range(BUDGET):
        cfg={k:rng.choice(v) for k,v in SPACE.items()}
        cell=(cfg["op"],cfg["difficulty"]-1);s=metric(cfg)["score"]
        if cell not in best or s>best[cell]:best[cell]=s
    return {"cells":len(best),"qd_score":sum(best.values())}
def curriculum(best):
    check(len(best)==9,"COVERAGE")
    tasks=[]
    for i in range(100):
        cell=sorted(best)[i%9];cfg=best[cell]["cfg"]
        o,d,v=(cfg[k] for k in ("op","difficulty","variant"))
        # Each record has a unique (operation,a,b), finite integer answer, no executable text.
        # Difficulty denotes magnitude band of first operand; all variants bounded.
        low=10**(d-1); span=9*low
        a=low+(i+v*17)%span
        b=1+i+v*101
        symbol=("+","-","*")[o]
        answer=(a+b,a-b,a*b)[o]
        tasks.append({"id":f"train-{i:03d}","op":symbol,"a":a,"b":b,"answer":answer,
           "cell":list(cell),"partition":"TRAIN_ONLY"})
    check(len({(t["op"],t["a"],t["b"]) for t in tasks})==100,"DUPLICATE_TASK")
    return tasks
def read_data(path):
    raw=Path(path).read_bytes();check(0<len(raw)<=LIMIT,"DATA_SIZE")
    obj=json.loads(raw);check(g.canonical(obj)==raw,"NONCANONICAL")
    return obj
def make_data(hist,archive):
    best=reduce_history(hist)
    observed={k:{"cfg":v.cfg,"fitness":v.fitness} for k,v in archive.items()}
    check(best==observed,"UPSTREAM_ARCHIVE_REPLAY")
    return {"history":hist,"tasks":curriculum(best),"map_elites":{"cells":len(best),
      "qd_score":sum(v["fitness"] for v in best.values())},"random_baseline":baseline()}
def worker(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);obtain(root)
        sys.path.insert(0,str(root))
        from evoagentx.optimizers.map_elites_optimizer import MapElitesOptimizer
        from evoagentx.optimizers.engine.registry import ParamRegistry
        p=Program();reg=ParamRegistry()
        for k in SPACE:reg.track(p,k,name=k)
        opt=MapElitesOptimizer(registry=reg,evaluator=metric,program=p.run,search_space=SPACE,
            feature_dimensions=["op","difficulty"],feature_ranges={"op":(0.,2.),"difficulty":(1.,3.)},
            feature_bins=3,n_iterations=BUDGET,exploration_ratio=.6,random_seed=SEED)
        cfg,result=opt.optimize()
        hist=json.loads(g.canonical(result["history"]))
        data=make_data(hist,result["archive"])
        check(cfg==result["best"].cfg and metric(cfg)["score"]==max(h["fitness"] for h in hist),"BEST")
        raw=g.canonical(data);check(len(raw)<=LIMIT,"DATA_LIMIT")
        (out/"CURRICULUM.json").write_bytes(raw)
        (out/"UPSTREAM_LICENSE.txt").write_bytes((root/"LICENSE").read_bytes())
        key,pub=g.keys(root,"jdg-curriculum")
        report={"request":request(),"data_sha256":sha(raw),"task_count":len(data["tasks"]),
            "map_elites":data["map_elites"],"random_baseline":data["random_baseline"],
            "scope":"SYNTHETIC_TRAIN_ONLY","model_training":False,"external_peer":False,
            "action_authority":False,"training_authority":False}
        sig=base64.b64encode(g.crypto(key,g.canonical(report))).decode()
        g.write_json(out/"RESULT.json",{"report":report,"signature":sig})
        (out/"PUBLIC.pem").write_bytes(pub.read_bytes())
        with open(os.environ["GITHUB_OUTPUT"],"a") as f:f.write("key_sha="+sha(pub.read_bytes())+"\n")
        print("JDG_R7C3_WORKER",json.dumps(report,sort_keys=True))
def verify(out,key_sha):
    out=Path(out);pub=out/"PUBLIC.pem";check(sha(pub.read_bytes())==key_sha,"KEY_PIN")
    envelope=g.read_json(out/"RESULT.json");check(set(envelope)=={"report","signature"},"ENVELOPE")
    r=envelope["report"];g.crypto(pub,g.canonical(r),base64.b64decode(envelope["signature"],validate=True))
    check(r["request"]==request(),"REQUEST_BINDING")
    check(all(r[k] is False for k in ("model_training","external_peer","action_authority","training_authority")),"AUTHORITY")
    check(r["scope"]=="SYNTHETIC_TRAIN_ONLY","SCOPE")
    data=read_data(out/"CURRICULUM.json")
    check(sha(g.canonical(data))==r["data_sha256"],"PAYLOAD_HASH")
    check(blob((out/"UPSTREAM_LICENSE.txt").read_bytes())==PINS["LICENSE"],"LICENSE_PIN")
    best=reduce_history(data["history"])
    check(data["tasks"]==curriculum(best) and r["task_count"]==100,"TASK_REPLAY")
    # Independent arithmetic checks, no eval and no imported upstream code.
    for t in data["tasks"]:
        a,b=t["a"],t["b"]
        if t["op"]=="+":ok=t["answer"]==sum((a,b))
        elif t["op"]=="-":ok=t["answer"]+b==a
        else:ok=t["op"]=="*" and t["answer"]%b==0 and t["answer"]//b==a
        check(ok and t["partition"]=="TRAIN_ONLY","ANSWER_OR_PARTITION")
    summary={"cells":len(best),"qd_score":sum(v["fitness"] for v in best.values())}
    check(data["map_elites"]==r["map_elites"]==summary and data["random_baseline"]==r["random_baseline"]==baseline(),"SUMMARY")
    # Negative controls: wrong answer and altered score must not be admitted.
    tampered=json.loads(g.canonical(data));tampered["tasks"][0]["answer"]+=1
    check(sha(g.canonical(tampered))!=r["data_sha256"] and tampered["tasks"]!=curriculum(best),"TAMPER_CONTROL")
    bad=json.loads(g.canonical(data["history"]));bad[0]["fitness"]+=1
    try:reduce_history(bad)
    except ValueError:pass
    else:raise ValueError("BAD_SCORE_ACCEPTED")
    print("JDG_R7C3_VERIFIED",json.dumps({"status":"PASS_REAL_UPSTREAM_CURRICULUM_OPTIMIZER",
      "tasks":100,"map_elites":summary,"random_baseline":baseline(),"tamper_controls":2,
      "model_learning_gain":"NOT_MEASURED","training_authority":False,"external_peers":0},sort_keys=True))
if __name__=="__main__":
    if sys.argv[1]=="worker":worker(sys.argv[2])
    elif sys.argv[1]=="verify":verify(sys.argv[2],sys.argv[3])
    else:raise ValueError("MODE")
