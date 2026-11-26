"""R84H20 one-shot actual BEETLE discovery, TRINITY oracle, isolated memory intake."""
import sys,json,hashlib,importlib.util,re,sqlite3,urllib.parse
from pathlib import Path
HERE=Path(__file__).parent
PROTOCOL="a71ab9fd79218c8f22f17596cb42c4bb2b0c1f40"
BEETLE="cdeb4777fdd1df49090f60e36a995951fe214924"
TRINITY="7792bae07705e040504052bcf673342b661858f7"
MOTTO="Мы должны узнать"
def ck(v,m):
    if not v:raise ValueError(m)
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def blob(raw):return hashlib.sha1(("blob %d\0"%len(raw)).encode()+raw).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(canonical(x)+"\n")
def log(tag,x):print(tag,canonical(x),flush=True)
def pinned(name,p,pin):
    raw=p.read_bytes();ck(blob(raw)==pin,"SOURCE_PIN_"+name)
    sys.path.insert(0,str(p.parent));spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def policy():
    raw=(HERE/"R84H20_FROZEN_PROTOCOL.json").read_bytes();ck(blob(raw)==PROTOCOL,"PROTOCOL_PIN");return json.loads(raw)
def allowed():
    # Public-facing search text restricted to synthetic unknown science probes;
    # never transmit a historical JANUS query or arbitrary incoming text.
    ex=read(HERE/"R84H19_FROZEN_EXAM.json");old=read(HERE/"R84H18_FROZEN_EXAM.json")
    return {t.strip().lower() for t in ex["freshUnknown"]+[t for _,t in old["freshUnknown"]+old["nearUnknown"]]}
def scout(lapis,h19,out):
    import requests
    p=policy();out.mkdir(parents=True,exist_ok=False);rec=read(h19/"result.json");plans=read(h19/"plans.json")
    ck(rec["schema"]=="r84h19.result.v1" and rec["scientific"]=="PASS_SHADOW" and rec["historical_prediction_parity"]==671 and rec["production"] is False and len(plans)==12,"H19_GATE")
    with sqlite3.connect("file:"+str((h19/"exploration.sqlite").resolve())+"?mode=ro",uri=True) as db:
        records=db.execute("SELECT id,body,digest FROM plans").fetchall()
    ck(len(records)==len(plans) and all(hashlib.sha256(body.encode()).hexdigest()==sha for _,body,sha in records),"H19_QUEUE_HASH")
    ck({i:json.loads(body) for i,body,_ in records}=={x["id"]:x for x in plans},"H19_QUEUE_MATCH")
    selected=[x for x in plans if x["query"] in allowed()]
    ck(all(x["review_required"] and not x["auto_dispatch"] and not x["auto_train"] and not x["auto_promote"] and x["production"] is False for x in plans),"PLAN_GUARD")
    if not selected:
        report={"schema":"r84h20.discovery.v1","status":"HOLD_NO_ALLOWED_SYNTHETIC_PLAN","leads":[],"requests":0,"production":False,"motto":MOTTO};write(out/"discovery.json",report);log("R84H20_DISCOVERY",report);return
    plan=selected[0];stop=set("what how why does do describe explain the a an of to in on by for is are from its it their and with during between use role causes purpose".split())
    words=[w for w in re.findall(r"[a-z]{4,}",plan["query"]) if w not in stop];kw=list(dict.fromkeys(words))[:6];ck(len(kw)>=2,"QUERY_SCOPE")
    m=pinned("beetle_r84h20",lapis/"tools/universal_beetle.py",BEETLE)
    class Bounded(requests.Session):
        calls=0
        bytes_read=0
        def get(self,url,**kwargs):
            u=urllib.parse.urlparse(url);ck(u.scheme=="https" and u.netloc=="export.arxiv.org" and u.path=="/api/query" and self.calls==0,"SCOUT_SCOPE")
            self.calls+=1;kwargs.update(stream=True,allow_redirects=False,timeout=(10,20))
            with super().get(url,**kwargs) as resp:
                ck(resp.status_code==200,"SOURCE_HTTP_"+str(resp.status_code));chunks=[]
                for data in resp.iter_content(8192):
                    self.bytes_read+=len(data);ck(self.bytes_read<=p["max_bytes"],"BYTE_CAP");chunks.append(data)
                resp._content=b"".join(chunks);resp._content_consumed=True;return resp
    sess=Bounded();m.session=lambda:sess
    state=m.State(out/"beetle");cfg={"global_budget":{"max_requests_per_run":1,"max_new_candidates_per_run":5,"max_metadata_bytes_per_run":p["max_bytes"]},"directions":[{"id":"h20_"+plan["id"][:12],"title":"Synthetic research plan","keywords":kw,"min_score":1,"sources":[{"kind":"arxiv","query":" AND ".join('all:"'+w+'"' for w in kw[:2]),"max_results":5}]}]}
    summary=m.run_scout(cfg,state,cfg["directions"][0]["id"]);ck(sess.calls<=1 and summary["requests"]<=1 and len(state.seen)<=5 and summary["metadata_bytes"]<=p["max_bytes"],"SCOUT_BUDGET")
    leads=[]
    for i,v in sorted(state.seen.items()):
        u=urllib.parse.urlparse(v["url"]);ck(u.scheme=="https" and u.netloc in ("arxiv.org","www.arxiv.org") and u.path.startswith("/abs/"),"LEAD_URL")
        leads.append({"candidate_id":i,"url":v["url"],"status":"QUARANTINE_UNVERIFIED","training_allowed":False})
    report={"schema":"r84h20.discovery.v1","status":"DISCOVERY_CANDIDATES" if leads else "HOLD_NO_LEADS","plan_id":plan["id"],"query":plan["query"],"keywords":kw,"requests":sess.calls,"summary":summary,"leads":leads,"receipt_head":state.prev_receipt,"production":False,"motto":MOTTO}
    write(out/"discovery.json",report);log("R84H20_DISCOVERY",report)
def verify(demi,scoutdir,out):
    p=policy();out.mkdir(parents=True,exist_ok=False);d=read(scoutdir/"discovery.json");ck(d["schema"]=="r84h20.discovery.v1" and d["production"] is False and len(d["leads"])<=5,"DISCOVERY")
    m=pinned("trinity_r84h20",demi/"tools/trinity_r14_verified_oracle.py",TRINITY)
    controls=[]
    for prompt,answer,expected in p["controls"]:
        verdict=m.verify(prompt,answer);ck(verdict["verdict"]==expected,"ORACLE_CONTROL")
        controls.append({"expected":expected,"actual":verdict["verdict"]})
    claims=[]
    for lead in d["leads"]:
        ck(not lead["training_allowed"],"UNVERIFIED")
        result=m.verify("Assess the scientific claims from "+lead["url"],"unverified");ck(result["verdict"]=="HOLD_UNSUPPORTED" and not result["eligible_for_train"],"ORACLE_BOUNDARY")
        claims.append({**lead,"oracle_verdict":result["verdict"],"license_verified":False,"claim_verified":False})
    report={"schema":"r84h20.verification.v1","discovery_sha256":digest(d),"plan_id":d.get("plan_id"),"controls":controls,"claims":claims,"production":False,"training_allowed":False,"motto":MOTTO}
    write(out/"verification.json",report);log("R84H20_VERIFY",{"controls_pass":len(controls),"scientific_claims_held":len(claims),"production":False})
def intake(scoutdir,verified,out):
    d=read(scoutdir/"discovery.json");v=read(verified/"verification.json");ck(v["discovery_sha256"]==digest(d) and v["plan_id"]==d.get("plan_id") and len(v["controls"])==6 and not v["training_allowed"] and not v["production"],"HANDOFF")
    out.mkdir(parents=True,exist_ok=False);path=out/"candidate_intake.sqlite"
    def insert():
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY,body TEXT NOT NULL,sha256 TEXT NOT NULL)")
            n=0
            for x in v["claims"]:
                ck(x["oracle_verdict"]=="HOLD_UNSUPPORTED" and not x["claim_verified"] and not x["license_verified"] and not x["training_allowed"],"INTAKE_SCOPE")
                body=canonical(x);sha=digest(x);old=db.execute("SELECT body,sha256 FROM candidates WHERE id=?",(x["candidate_id"],)).fetchone()
                if old:ck(old==(body,sha),"IMMUTABLE_INTAKE")
                else:db.execute("INSERT INTO candidates VALUES(?,?,?)",(x["candidate_id"],body,sha));n+=1
            return n
    n=insert();ck(insert()==0,"DUPLICATE")
    with sqlite3.connect(path) as db:
        rows=db.execute("SELECT body,sha256 FROM candidates").fetchall();ck(len(rows)==n and all(digest(json.loads(body))==sha for body,sha in rows),"RESTORE")
    result={"schema":"r84h20.link_exam.v1","plan_id":d.get("plan_id"),"requests":d["requests"],"discovery_candidates":len(d["leads"]),"oracle_controls_pass":len(v["controls"]),"quarantined":n,"duplicate_writes":0,"native_memory_cortex_connected":False,"memory_intake_adapter":"ISOLATED_SQLITE","scientific":"HOLD_UNVERIFIED","engineering":"PASS" if n else "PASS_CONTROLS__DISCOVERY_HOLD","training_transitions":0,"production":False,"motto":MOTTO}
    write(out/"result.json",result);log("R84H20_GATE",result)
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x) for x in sys.argv[2:]]
    {"scout":scout,"verify":verify,"intake":intake}[mode](*args)
