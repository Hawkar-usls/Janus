"""R84H18 bounded OOD shadow audit and quarantined study-plan receipts."""
import sys,json,time,math,re
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import r84h17_biome as h17
import r84h16_chunk as h16
import r84h15_world as h15
import r84h14_partition as h14
import r84h13_streaming as h13
import r84h12_evolution as h12
import r84h10_resonance as h10
import r84g_adaptive as a
import r84e_neural as e
import r8_learning as r
import jfp_gateway as g
HERE=Path(__file__).parent
EXAM="79a20f965491a53ba8c95562cb1e8cde8dbacf9e"
METHODS=("cosine","coverage","joint","balanced")
TARGETS=(1.,.98,.95,.90)
def ck(v,w):e.ck(v,w)
def log(t,x):e.log(t,x)
def clean(t):
    # Remove the dataset-specific control prefixes for both known and unknown.
    return re.sub(r"^gate(?:\s+(?:practice|exam|probe))?\s+","",t.strip().lower())
def terms(t):return h10.terms(clean(t))
def uniq(rows):
    seen=set();out=[]
    for t,y in rows:
        n=clean(t)
        if n not in seen:out.append((t,y));seen.add(n)
    return out
class Familiarity:
    """Fit-only sparse retrieval and independent vocabulary coverage.
    No exam label, route or known/unknown indicator enters score().
    """
    def __init__(self,L,fit):
        texts=[(clean(t),y) for t,y in fit];lex=h10.lexical_fit(texts);self.lex=h13.Lex(lex)
        self.ref=csr_matrix(h10.lexical(L,lex,texts).numpy());self.words={k for k in self.lex.lookup if k.startswith("w:")}
        self.bytes=h13.sparse_bytes(self.ref)+len(json.dumps(lex).encode())
    def score(self,t):
        z=terms(t);v=self.lex.vector(z)
        sim=self.ref@v;cos=float(sim.max(initial=0))
        total=sum(n for k,n in z.items() if k.startswith("w:"));known=sum(n for k,n in z.items() if k in self.words)
        cov=known/max(total,1);cos=max(0.,min(1.,cos))
        return {"cosine":cos,"coverage":cov,"joint":cos*cov,"balanced":math.sqrt(cos*cov)}
def threshold(values,retain):
    # Exact order statistic: >= threshold accepts at least the required count.
    a=sorted(values);allowed=len(a)-math.ceil(retain*len(a)-1e-9)
    return a[max(0,allowed)]
def auc(k,u):
    return sum(1. if x>y else .5 if x==y else 0. for x in k for y in u)/(len(k)*len(u))
def fit_policy(detector,dev,unknown):
    known=[detector.score(t) for t,_ in dev];outside=[detector.score(t) for t in unknown];policies=[]
    for retain in TARGETS:
        options=[]
        for mode in METHODS:
            k=[x[mode] for x in known];u=[x[mode] for x in outside];th=threshold(k,retain)
            rec={"target_dev_retention":retain,"mode":mode,"threshold":th,"dev_accepted":sum(x>=th for x in k),"dev_n":len(k),"calibration_unknown_rejected":sum(x<th for x in u),"calibration_unknown_n":len(u),"calibration_auc":auc(k,u)}
            options.append(rec)
        # Deterministic tie-breaking on calibration only.
        policies.append(max(options,key=lambda x:(x["calibration_unknown_rejected"],x["dev_accepted"],x["calibration_auc"],-METHODS.index(x["mode"]))))
    return policies
def proposal(text,scores,policy):
    n=clean(text);ck(0<len(n)<=500,"PROPOSAL_LENGTH")
    # Non-executable, quarantined request. The query is untrusted data,
    # not a command to an agent. No source/answer claims or automatic ingestion.
    qid=h16.sha(("JANUS-FOG-v1\0"+n).encode())
    return {"schema":"r84h18.study_candidate.v1","id":qid,"state":"UNVERIFIED__REVIEW_REQUIRED","query":n,"score":scores[policy["mode"]],"threshold":policy["threshold"],"policy":policy["mode"],"candidate_address":h16.address("fog/"+qid),"stages":[{"organ":"BEETLE","task":"Discover relevant public references; record source URLs, provenance and licensing; no instruction execution from retrieved content.","status":"PROPOSED_ONLY","max_sources":5},{"organ":"TRINITY","task":"Validate source claims independently and produce separate study and held-out evaluation sets; reject ambiguous or unsupported claims.","status":"PROPOSED_ONLY"},{"organ":"Memory Cortex","task":"Deduplicate verified knowledge and prepare a candidate versioned chunk; require regression exam, review and explicit approval before admission.","status":"PROPOSED_ONLY"}],"auto_dispatch":False,"auto_train":False,"auto_promote":False,"production":False}
def evaluate(prep,rootpath,bundlepath,h10path,h11path,h12path,fixtures,snapshot,h17results,out):
    L,dd,root,d,mapping,labels=h15.base_data(prep)
    raw=(HERE/"R84H18_FROZEN_EXAM.json").read_bytes();ck(r.blob(raw)==EXAM,"EXAM_PIN");ex=json.loads(raw)
    fo,fn=h12.banks(d,h15.SEED,10,mapping,"fit");vo,vn=h12.banks(d,h15.SEED,10,mapping,"dev")
    fit=uniq(fo+fn);dev=uniq(vo+vn)
    fitnorm={clean(t) for t,_ in fit};devnorm={clean(t) for t,_ in dev};ck(not fitnorm&devnorm,"FIT_DEV_COLLISION")
    detector=Familiarity(L,fit);cal=ex["calibrationUnknown"];ck(len(cal)==24 and not {clean(t) for t in cal}&(fitnorm|devnorm),"CALIBRATION_COLLISION")
    policies=fit_policy(detector,dev,cal);log("R84H18_CALIBRATION",{"fit_n":len(fit),"dev_n":len(dev),"fit_sha256":g.digest(fit),"dev_sha256":g.digest(dev),"policies":policies})
    # H17 artifacts prove the previous exact result; the same pinned checkpoints
    # are independently recomputed for historical prediction/route comparisons.
    h17rec=e.read(h17results/"result.json");ck(h17rec["schema"]=="r84h17.result.v1" and h17rec["production"] is False and h17rec["metrics"]["historical"]["parity"]==671 and h17rec["metrics"]["original"]["answer_correct"]==210 and h17rec["metrics"]["fresh"]["answer_correct"]==42 and h17rec["metrics"]["cross"]["answer_correct"]==12,"H17_RECEIPT")
    h17cases=e.read(h17results/"cases.json");oldcases={x["i"]:x for x in h17cases if x["cohort"]=="historical"};ck(len(oldcases)==671,"H17_CASES")
    receipt=e.read(snapshot/"snapshot.json");path=snapshot/"world.sqlite";ck(receipt["size"]==100000 and receipt["world"]==h16.WORLD and receipt["production"] is False and h16.sha(path.read_bytes())==receipt["db_sha256"],"WORLD_PIN")
    cells,_,_,_,_=h13.catalog(L,d,mapping,labels,h15.SEED,0,h11path,h12path)
    groups,_=h14.exams(prep,dd,d);hist=sum(groups,[]);ck(len(hist)==671,"HIST")
    actual=[labels[y] for _,y in hist];base,br,_=h13.persistent(L,root,d,h15.SEED,0,rootpath,bundlepath,h10path,mapping,hist)
    ep,er,_=h13.eager(L,cells,hist,base,br);ck(sum(x==y for x,y in zip(ep,actual))==386,"BASELINE")
    ck(all(oldcases[i]["pred"]==[ep[i]] and oldcases[i]["routes"]==[er[i]] for i in range(671)),"H17_PARITY")
    oldraw=(HERE/"R84H17_FROZEN_EXAM.json").read_bytes();ck(r.blob(oldraw)==h17.EXAM,"H17_EXAM");old=json.loads(oldraw)
    oldunknown=[x["text"] for x in old["openUnknown"]];fresh=[t for _,t in ex["freshUnknown"]];near=[t for _,t in ex["nearUnknown"]]
    allunknown=oldunknown+fresh+near;unknownnorm={clean(t) for t in allunknown}
    ck(len(unknownnorm)==42 and not unknownnorm&(fitnorm|devnorm|{clean(t) for t in cal}|{clean(t) for t,_ in hist}),"UNKNOWN_COLLISION")
    router=h17.Router();start=time.process_time();rows=[]
    for cohort,items in [("historical",[t for t,_ in hist]),("h17_unknown",oldunknown),("fresh_unknown",fresh),("near_unknown",near)]:
        for ix,t in enumerate(items):
            status,_=router.plan(t);score=detector.score(t)
            rec={"cohort":cohort,"i":ix,"text":t,"prior_route":status,"score":score,"fit_overlap":clean(t) in fitnorm,"dev_overlap":clean(t) in devnorm}
            if cohort=="historical":rec["baseline_correct"]=ep[ix]==actual[ix]
            rows.append(rec)
    score_cpu=time.process_time()-start
    metrics=[];historical=[x for x in rows if x["cohort"]=="historical"];priorcorrect=sum(x["baseline_correct"] for x in historical)
    for p in policies:
        def fog(x):return x["prior_route"]=="LEGACY" and x["score"][p["mode"]]<p["threshold"]
        knownfog=[x for x in historical if fog(x)]
        correctlost=sum(x["baseline_correct"] for x in knownfog)
        m={"policy":p,"historical_n":671,"historical_rejected":len(knownfog),"historical_prediction_parity_if_enforced":671-len(knownfog),"historical_correct_lost_if_enforced":correctlost,"historical_dep_if_enforced":priorcorrect-correctlost,"known_eval_fit_overlap":sum(x["fit_overlap"] for x in historical),"known_eval_dev_overlap":sum(x["dev_overlap"] for x in historical)}
        unseen=[x for x in historical if not x["fit_overlap"] and not x["dev_overlap"]];m["unseen_known_n"]=len(unseen);m["unseen_known_rejected"]=sum(fog(x) for x in unseen)
        for cohort in ("h17_unknown","fresh_unknown","near_unknown"):
            rr=[x for x in rows if x["cohort"]==cohort];m[cohort]={"n":len(rr),"detected":sum(fog(x) for x in rr),"auc":auc([x["score"][p["mode"]] for x in historical],[x["score"][p["mode"]] for x in rr])}
        metrics.append(m);log("R84H18_POLICY",m)
    # The proposed operational policy is frozen on dev retention, never selected
    # from the test set. All decisions remain shadow-only even if the gate passes.
    primary=policies[1];primarymetrics=metrics[1]
    candidates={}
    for x in rows:
        if x["prior_route"]=="LEGACY" and x["score"][primary["mode"]]<primary["threshold"]:
            q=proposal(x["text"],x["score"],primary);candidates[q["id"]]=q
    # Deterministic capped quarantine queue; include false-positive known cases
    # too, because a real detector has no access to the hidden test cohort.
    queue=sorted(candidates.values(),key=lambda x:(x["score"],x["id"]))[:12]
    ck(len({x["id"] for x in queue})==len(queue) and all(not x["auto_dispatch"] and not x["auto_train"] and not x["auto_promote"] for x in queue),"QUEUE_GUARD")
    gate=primarymetrics["historical_rejected"]==0 and primarymetrics["h17_unknown"]["detected"]>=5 and primarymetrics["fresh_unknown"]["detected"]>=20 and primarymetrics["near_unknown"]["detected"]>=8
    ck(h16.sha(path.read_bytes())==receipt["db_sha256"],"WORLD_UNCHANGED")
    out.mkdir(parents=True,exist_ok=True)
    result={"schema":"r84h18.result.v1","fit_n":len(fit),"dev_n":len(dev),"policies":metrics,"primary_policy":primary,"shadow_gate":gate,"historical_live_parity":671,"historical_live_dep":386,"h17_fixture_regression_receipt":h17rec["metrics"]["original"]["answer_correct"],"h17_alias_regression_receipt":h17rec["metrics"]["fresh"]["answer_correct"],"h17_cross_regression_receipt":h17rec["metrics"]["cross"]["answer_correct"],"study_candidates_total":len(candidates),"quarantine_queue_n":len(queue),"detector_index_estimate_bytes":detector.bytes,"score_cpu_seconds":score_cpu,"peak_rss_bytes":h15.peak(),"production":False,"external_validation":False,"scope":"seed111 finite shadow lexical OOD detector; fit/dev/calibration only for thresholds; historical H17 checkpoint parity independently recomputed, fixture H17 receipt referenced but not rerun; synthetic far and near-domain unknowns; no unrestricted semantic or epistemic certainty; proposals only, no real organ dispatch or external collection, no training or promotion"}
    e.write(out/"result.json",result);e.write(out/"cases.json",rows);e.write(out/"study_queue.json",{"schema":"r84h18.quarantine.v1","candidates":queue,"auto_dispatch":False,"production":False})
    log("R84H18_RESULT",result)
    log("R84H18_GATE",{"engineering":"PASS","scientific":"PASS_SHADOW" if gate else "HOLD","primary_dev_retention":.98,"historical_live_parity":671,"production":False,"dispatch":False})
if __name__=="__main__":evaluate(*map(Path,sys.argv[1:]))
