"""R84C: bounded discovery, fail-closed oracle and cross-repo shadow receipt."""
import base64,hashlib,importlib.util,json,os,re,sys,tempfile,time,urllib.parse
from pathlib import Path
import jfp_gateway as g
import r8_learning as r
import r84_loop as a
import r84b_intake as b
J=r.J; G="Hawkar-usls/Janus_Genesis"
PARENT="9282022f21ee976a33fb426a73cee295427c4d93"
STATE_BLOB="2855a5437cbfaedea33758a4893072574345bc26"
STATE_HASH="a562a9f909c0b249e382f7a91b5da26e6df5d7334697e080dcdf6f7382d7f737"
MODEL_HASH="7cfdd9f9faa9c564b26e298706c934b348af4eb791597d9784467c727563e87a"
BEETLE="cdeb4777fdd1df49090f60e36a995951fe214924"
TRINITY="7792bae07705e040504052bcf673342b661858f7"
ESSENCE="4401b6113f838695ed813612f24129b18274b964"
BRANCH="research/r84c-engine-20261125"
OUT=Path("data/r84c-engine/RECEIPT.json")
NODES=[J,r.M,G,"Hawkar-usls/Janus-Demiurge","Hawkar-usls/janus-lapis"]
def ck(v,s):g.require(v,s)
def log(tag,o):print(tag,json.dumps(o,sort_keys=True),flush=True)
def module(name,path,pin):
    path=Path(path);ck(r.blob(path.read_bytes())==pin,"SOURCE_PIN:"+name)
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def identity_gate(reg):
    ck(set(reg)==set(NODES),"NODE_SCOPE")
    for n,fp in reg.items():ck(type(fp) is str and re.fullmatch("[a-f0-9]{64}",fp),"UNPROVISIONED_KEY:"+n)
    ck(len(set(reg.values()))==len(NODES),"SHARED_KEY")
    return True
def policy_tests():
    res=[]
    for tag,reg in (("missing",{n:None for n in NODES}),("unknown",{**{n:"0"*64 for n in NODES},"external":None}),("shared",{n:"0"*64 for n in NODES})):
        try:identity_gate(reg)
        except g.Reject as e:res.append([tag,str(e)])
        else:raise g.Reject("IDENTITY_NEGATIVE_ACCEPTED")
    return res
def prior_state():
    s=a.get(J,PARENT,"data/r84b-learning/STATE.json",STATE_BLOB)
    q=a.get(J,PARENT,"data/r84b-learning/RESULT.json")
    p=r.verify(q,J,r.M,"SKILL_R84B",q["packet"]["created"])
    b.validate(s,at=p["created"])
    ck(g.digest(s)==STATE_HASH and g.digest(s["model"])==MODEL_HASH and p["payload"]["state_sha256"]==STATE_HASH and p["payload"]["model_sha256"]==MODEL_HASH,"PARENT")
    scores=[sum(b.exact(s["model"],x)) for x in b.exams()]
    ck(scores==[6,16,8,8],"PARENT_REGRESSION")
    return s,scores
def scout(lapis,out):
    import requests
    m=module("beetle_r84c",Path(lapis)/"tools/universal_beetle.py",BEETLE)
    maxbytes=256000
    class Bounded(requests.Session):
        calls=0
        def get(self,url,**kw):
            u=urllib.parse.urlparse(url);ck(u.scheme=="https" and u.netloc=="export.arxiv.org" and u.path=="/api/query" and self.calls==0,"SCOUT_SCOPE")
            self.calls+=1
            kw.update(stream=True,allow_redirects=False,timeout=(10,20))
            with super().get(url,**kw) as resp:
                ck(resp.status_code==200,"SOURCE_HTTP_"+str(resp.status_code))
                chunks=[];total=0
                for x in resp.iter_content(8192):
                    total+=len(x);ck(total<=maxbytes,"METADATA_CAP");chunks.append(x)
                resp._content=b"".join(chunks);resp._content_consumed=True
                return resp
    sess=Bounded();m.session=lambda:sess
    state=m.State(out/"beetle")
    cfg={"global_budget":{"max_requests_per_run":1,"max_new_candidates_per_run":5,"max_metadata_bytes_per_run":maxbytes},"directions":[{"id":"r84c","title":"Continual learning discovery only","keywords":["continual learning","catastrophic forgetting","lifelong learning"],"min_score":1,"sources":[{"kind":"arxiv","query":'all:"continual learning" OR all:"catastrophic forgetting"',"max_results":5}]}]}
    summary=m.run_scout(cfg,state,"r84c")
    ck(sess.calls<=1 and summary["requests"]<=1 and summary["metadata_bytes"]<=maxbytes and len(state.seen)<=5,"SCOUT_BUDGET")
    leads=[]
    for key,v in sorted(state.seen.items()):
        url=urllib.parse.urlparse(v["url"])
        ck(url.scheme=="https" and url.netloc in ("arxiv.org","www.arxiv.org") and url.path.startswith("/abs/"),"LEAD_URL")
        leads.append({"id":key,"url":v["url"],"disposition":"QUARANTINE_UNVERIFIED","training_allowed":False})
    report={"schema":"r84c.discovery.v1","source_blob":BEETLE,"attempts":sess.calls,"requests":summary["requests"],"metadata_bytes":summary["metadata_bytes"],"errors":len(summary["errors"]),"leads":leads,"production":False}
    g.write_json(out/"discovery.json",report);log("R84C_BEETLE",report)
def oracle(demi,discovery):
    m=module("trinity_r84c",Path(demi)/"tools/trinity_r14_verified_oracle.py",TRINITY)
    rows=[]
    for problem,answer,expected in [("Calculate 8 + 9. Answer with one integer.","17","APPROVE"),("Calculate 8 + 9. Answer with one integer.","18","VETO_WRONG"),("Calculate 8 + 9. Answer with one integer.","17 words","VETO_FORMAT"),("Verify license and provenance for a paper.","yes","HOLD_UNSUPPORTED")]:
        v=m.verify(problem,answer);ck(v["verdict"]==expected,"ORACLE");rows.append(v["verdict"])
    for lead in discovery["leads"]:
        v=m.verify("Review discovery candidate "+lead["url"],"approve")
        ck(v["verdict"]=="HOLD_UNSUPPORTED" and not v["eligible_for_train"],"LEAD_APPROVAL")
    return {"source_blob":TRINITY,"controls":rows,"lead_holds":len(discovery["leads"]),"new_training_labels":0,"parent_models_executed":False}
def seal(genesis,obj):
    m=module("essence_r84c",Path(genesis)/"tools/essence_v54_neural_witness.py",ESSENCE)
    v=m.seal(obj);m.check(v)
    try:m.check({**v,"production_promotion":True})
    except ValueError:pass
    else:raise g.Reject("SEAL_NEGATIVE")
    return v
def build(demi,genesis,inp,out):
    ck(os.environ["GITHUB_REPOSITORY"]==J,"REPO")
    d=g.read_json(inp/"discovery.json");ck(d["source_blob"]==BEETLE and d["production"] is False and d["attempts"]<=1 and d["metadata_bytes"]<=256000 and len(d["leads"])<=5 and all(x["training_allowed"] is False for x in d["leads"]),"DISCOVERY")
    s,scores=prior_state()
    report=seal(genesis,{"schema":"janus.r84c.shadow_engine_receipt.v1","parent_commit":PARENT,"parent_state_sha256":STATE_HASH,"parent_model_sha256":MODEL_HASH,"scores":scores,"discovery":d,"trinity":oracle(demi,d),"identity_controls":policy_tests(),"persistent_keys":False,"schedule":False,"new_training":0,"production_promotion":False,"sentinel_dispatch":False,"essence_scope":"V54_SEAL_CHECK_ONLY_NOT_MODEL_ADMISSION"})
    out.mkdir(parents=True,exist_ok=False);g.write_json(out/"receipt.json",report)
    log("R84C_VERIFIED",{"sha256":report["sha256"],"scores":scores,"discovery":len(d["leads"]),"trinity":report["trinity"],"identity_controls":report["identity_controls"],"new_training":0})
def publish(inp):
    ck(os.environ.get("GITHUB_REPOSITORY")==J and os.environ.get("R84C_HEAD")==BRANCH and os.environ.get("GITHUB_RUN_ATTEMPT")=="1","WRITE_SCOPE")
    ck(a.cmd(["git","remote","get-url","origin"]) in ("https://github.com/"+J,"https://github.com/"+J+".git"),"ORIGIN")
    z=g.read_json(inp/"receipt.json");ck(z["sha256"]==g.digest({k:v for k,v in z.items() if k!="sha256"}) and z["parent_state_sha256"]==STATE_HASH and z["production_promotion"] is False,"RECEIPT")
    OUT.parent.mkdir(parents=True,exist_ok=True);g.write_json(OUT,z);a.cmd(["git","add","--",str(OUT)])
    ck(a.cmd(["git","diff","--cached","--name-only"])==str(OUT),"STAGED_SCOPE")
    a.cmd(["git","-c","user.name=JANUS R84C research","-c","user.email=41898282+github-actions[bot]@users.noreply.github.com","commit","-m","research(R84C): publish bounded engine receipt"])
    a.cmd(["git","push","origin","HEAD:refs/heads/"+BRANCH],45);log("R84C_PUBLISHED",{"commit":a.cmd(["git","rev-parse","HEAD"]),"sha256":z["sha256"]})
def witness(genesis,out):
    ck(os.environ.get("GITHUB_REPOSITORY")==G,"REPO")
    import urllib.request,urllib.error
    end=time.monotonic()+180
    while time.monotonic()<end:
        try:
            refs=a.cmd(["git","ls-remote","--heads","https://github.com/"+J+".git","refs/heads/"+BRANCH],10).split()
            if len(refs)==2:
                z=a.get(J,refs[0],str(OUT));break
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError):pass
        time.sleep(3)
    else:raise g.Reject("WITNESS_TIMEOUT")
    m=module("essence_r84c_witness",Path(genesis)/"tools/essence_v54_neural_witness.py",ESSENCE);m.check(z)
    s,scores=prior_state();ck(z["scores"]==scores and z["parent_state_sha256"]==g.digest(s) and z["parent_model_sha256"]==g.digest(s["model"]) and z["production_promotion"] is False and z["new_training"]==0,"WITNESS")
    report=m.seal({"schema":"janus.r84c.genesis_witness.v1","janus_commit":refs[0],"receipt_sha256":z["sha256"],"scores":scores,"quarantined":len(z["discovery"]["leads"]),"production_promotion":False,"full_essence_model_admission":False})
    m.write_once(out/"witness.json",report);log("R84C_GENESIS_WITNESS",report)
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x) for x in sys.argv[2:]]
    if mode=="scout":
        args[1].mkdir(parents=True,exist_ok=False);scout(*args)
    elif mode=="verify":build(*args)
    elif mode=="publish":publish(*args)
    elif mode=="witness":witness(*args)
    else:raise ValueError("MODE")
