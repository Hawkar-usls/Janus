"""R84H19 finite exploration shadow audit; no autonomous dispatch or promotion."""
import sys,json,time,sqlite3,math
from pathlib import Path
import numpy as np
import r84h18_fog as h18
import r84h17_biome as h17
import r84h16_chunk as h16
import r84h15_world as h15
import r84h14_partition as h14
import r84h13_streaming as h13
import r84h12_evolution as h12
import r84e_neural as e
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="9f181faf481fe70c3f04c3ddce901690d44375a8"
MOTTO="Мы должны узнать"
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def load_exam(name,pin):
    raw=(HERE/name).read_bytes();ck(r.blob(raw)==pin,"EXAM_PIN");return json.loads(raw)
class Evidence:
    def __init__(self,L,fit):
        self.base=h18.Familiarity(L,fit);self.y=np.array([y for _,y in fit]);self.fit=fit
    def score(self,t):
        s=self.base.score(t);v=self.base.lex.vector(h18.terms(t));sim=self.base.ref@v
        order=np.argsort(-sim,kind="stable")[:5];top=self.y[order[0]]
        same=float(np.mean(self.y[order]==top));s["support"]=s["joint"]*(.5+.5*same)
        s["consensus"]=same
        return s
def calibrate(det,dev,cal):
    k=[det.score(t) for t,_ in dev];u=[det.score(t) for t in cal];opts=[]
    for mode in ("cosine","coverage","joint","support"):
        kv=[x[mode] for x in k];uv=[x[mode] for x in u]
        lo=h18.threshold(kv,.98);hi=h18.threshold(kv,.90)
        opts.append({"mode":mode,"unknown_below":lo,"known_at_least":hi,"dev_unknown":sum(x<lo for x in kv),"dev_frontier":sum(lo<=x<hi for x in kv),"dev_n":len(kv),"cal_unknown":sum(x<lo for x in uv),"cal_flagged":sum(x<hi for x in uv),"cal_n":len(uv),"auc":h18.auc(kv,uv)})
    return max(opts,key=lambda p:(p["cal_unknown"],p["cal_flagged"],p["auc"],-p["dev_unknown"]))
def classify(score,policy):
    v=score[policy["mode"]]
    return "UNKNOWN" if v<policy["unknown_below"] else "FRONTIER" if v<policy["known_at_least"] else "KNOWN"
def study(text,state,score,policy):
    p=h18.proposal(text,score,{"mode":policy["mode"],"threshold":policy["unknown_below"]})
    p.update({"schema":"r84h19.exploration_plan.v1","source_state":state,"state":"EXPLORING","meaning":"local research plan prepared; no external dispatch","motto":MOTTO,"attribution":"user-supplied wording, attribution unverified","review_required":True})
    return p
def save_queue(path,plans):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS plans(id TEXT PRIMARY KEY, body TEXT NOT NULL, digest TEXT NOT NULL)")
        n=0
        for p in plans:
            body=h16.canonical(p);digest=h16.sha(body.encode());old=db.execute("SELECT body,digest FROM plans WHERE id=?",(p["id"],)).fetchone()
            if old:ck(old==(body,digest),"IMMUTABLE_PLAN")
            else:db.execute("INSERT INTO plans VALUES(?,?,?)",(p["id"],body,digest));n+=1
        return n
def restore(folder):
    rec=e.read(folder/"result.json");plans=e.read(folder/"plans.json");ck(rec["production"] is False and len(plans)<=12,"RESTORE")
    with sqlite3.connect(folder/"exploration.sqlite") as db:
        rows=db.execute("SELECT id,body,digest FROM plans ORDER BY id").fetchall()
        ck(len(rows)==len(plans) and all(h16.sha(body.encode())==digest and json.loads(body)["id"]==i for i,body,digest in rows),"PLAN_HASH")
        ck({i:json.loads(body) for i,body,_ in rows}=={p["id"]:p for p in plans},"PLAN_CONTENT")
    ck(save_queue(folder/"exploration.sqlite",plans)==0,"REPLAY_IDEMPOTENCY")
    log("R84H19_RESTORE",{"plans":len(plans),"idempotent":True,"production":False,"dispatch":False,"motto":MOTTO})
def evaluate(prep,rootpath,bundlepath,h10path,h11path,h12path,fixtures,snapshot,h17results,h18results,out):
    L,dd,root,d,mapping,labels=h15.base_data(prep)
    ex=load_exam("R84H19_FROZEN_EXAM.json",EXAM);old=load_exam("R84H18_FROZEN_EXAM.json",h18.EXAM);old17=load_exam("R84H17_FROZEN_EXAM.json",h17.EXAM)
    f,n=h12.banks(d,h15.SEED,10,mapping,"fit");v,w=h12.banks(d,h15.SEED,10,mapping,"dev");fit=h18.uniq(f+n);dev=h18.uniq(v+w)
    ck(len(fit)==553 and len(dev)==200,"H18_SPLIT")
    fitnorm={h18.clean(t) for t,_ in fit};devnorm={h18.clean(t) for t,_ in dev}
    ck(not fitnorm&devnorm,"FIT_DEV")
    h18rec=e.read(h18results/"result.json");ck(h18rec["production"] is False and h18rec["historical_live_parity"]==671,"H18_RECEIPT")
    cal=old["calibrationUnknown"]+ex["calibrationNear"];det=Evidence(L,fit);policy=calibrate(det,dev,cal);log("R84H19_CALIBRATION",policy)
    groups,_=h14.exams(prep,dd,d);hist=sum(groups,[]);fresh=ex["freshKnown"];ck(len(hist)==671 and len(fresh)==20,"KNOWN_SIZE")
    occupied=fitnorm|devnorm|{h18.clean(t) for t,_ in hist}|{h18.clean(t) for t in cal}
    ck(not occupied&{h18.clean(t) for t,_ in fresh},"FRESH_KNOWN_COLLISION")
    oldunknown=[x["text"] for x in old17["openUnknown"]]+[t for _,t in old["freshUnknown"]+old["nearUnknown"]]
    newunknown=ex["freshUnknown"];ck(len(newunknown)==20 and not {h18.clean(t) for t in newunknown}&(occupied|{h18.clean(t) for t in oldunknown}),"FRESH_UNKNOWN_COLLISION")
    world=e.read(snapshot/"snapshot.json");path=snapshot/"world.sqlite";ck(world["size"]==100000 and h16.sha(path.read_bytes())==world["db_sha256"],"WORLD")
    cells,_,_,_,_=h13.catalog(L,d,mapping,labels,h15.SEED,0,h11path,h12path)
    known=hist+fresh;actual=[labels[y] for _,y in known];base,br,_=h13.persistent(L,root,d,h15.SEED,0,rootpath,bundlepath,h10path,mapping,known);pred,routes,_=h13.eager(L,cells,known,base,br)
    prev={x["i"]:x for x in e.read(h17results/"cases.json") if x["cohort"]=="historical"}
    ck(len(prev)==671 and all(prev[i]["pred"]==[pred[i]] and prev[i]["routes"]==[routes[i]] for i in range(671)),"HIST_PARITY")
    router=h17.Router();rows=[];start=time.process_time()
    for cohort,items in (("historical",[t for t,_ in hist]),("fresh_known",[t for t,_ in fresh]),("prior_unknown",oldunknown),("fresh_unknown",newunknown)):
        for i,t in enumerate(items):
            status,_=router.plan(t);score=det.score(t);state=classify(score,policy) if status=="LEGACY" else "KNOWN" if status in ("SINGLE","CROSS") else "FRONTIER"
            row={"cohort":cohort,"i":i,"text":t,"state":state,"scores":score,"prior_route":status}
            if cohort in ("historical","fresh_known"):
                ix=i if cohort=="historical" else 671+i;row.update({"answer":pred[ix],"route":routes[ix],"correct":pred[ix]==actual[ix],"answer_preserved":True})
            rows.append(row)
    metrics={}
    for cohort in ("historical","fresh_known","prior_unknown","fresh_unknown"):
        q=[x for x in rows if x["cohort"]==cohort];metrics[cohort]={"n":len(q),"known":sum(x["state"]=="KNOWN" for x in q),"frontier":sum(x["state"]=="FRONTIER" for x in q),"unknown":sum(x["state"]=="UNKNOWN" for x in q),"flagged":sum(x["state"]!="KNOWN" for x in q),"correct":sum(x.get("correct",False) for x in q),"correct_flagged":sum(x.get("correct",False) and x["state"]!="KNOWN" for x in q)}
    # Rank by score only; neither hidden labels nor cohort control the queue.
    candidates={}
    for x in rows:
        if x["state"] in ("UNKNOWN","FRONTIER"):
            p=study(x["text"],x["state"],x["scores"],policy);candidates[p["id"]]=p
    plans=sorted(candidates.values(),key=lambda p:(p["source_state"]!="UNKNOWN",p["score"],p["id"]))[:12]
    ck(all(not p["auto_dispatch"] and not p["auto_train"] and not p["auto_promote"] and p["review_required"] for p in plans),"QUEUE_POLICY")
    out.mkdir(parents=True,exist_ok=False);db=out/"exploration.sqlite";ck(save_queue(db,plans)==len(plans),"QUEUE_WRITE");ck(save_queue(db,plans)==0,"QUEUE_DEDUP")
    ck(h16.sha(path.read_bytes())==world["db_sha256"],"WORLD_UNCHANGED")
    m={"schema":"r84h19.result.v1","motto":MOTTO,"motto_attribution":"user supplied; unverified","policy":policy,"metrics":metrics,"historical_prediction_parity":671,"historical_dep":metrics["historical"]["correct"],"candidate_n":len(candidates),"plans_n":len(plans),"score_cpu_seconds":time.process_time()-start,"production":False,"external_validation":False,"scientific":"PASS_SHADOW" if metrics["fresh_unknown"]["flagged"]>=16 and metrics["fresh_known"]["flagged"]<=2 else "HOLD","scope":"seed111 finite side-channel familiarity states; fresh synthetic probes, prior regression; known answers remain unchanged even when flagged; unknown probes have no answer validation; EXPLORING means local quarantined plan only, not execution; no external discovery or training, no organ dispatch, no promotion"}
    e.write(out/"result.json",m);e.write(out/"cases.json",rows);e.write(out/"plans.json",plans);log("R84H19_RESULT",m)
    log("R84H19_GATE",{"engineering":"PASS","scientific":m["scientific"],"historical_parity":671,"dispatch":False,"production":False})
if __name__=="__main__":
    if sys.argv[1]=="evaluate":evaluate(*map(Path,sys.argv[2:]))
    elif sys.argv[1]=="restore":restore(Path(sys.argv[2]))
    else:raise ValueError("MODE")
