"""H28 finite source-pointer retrieval pilot, no autonomous admission or external authority."""
import sys,json,re,time,math,tempfile,importlib,urllib.parse
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
import requests
import r84h20_link as h20
import r84h21_admission as h21
import r84h25_metacognition as h25
import r84h27_trinity_critic as h27
import r84h19_exploration as h19
import r84h18_fog as h18
import r8_learning as r
HERE=Path(__file__).parent
EXAM="dc15bebce75c06cb5756e4f95d9a9b5c05f31c05"
CARDS="4a3c8dfc078ddf883b505e6c6402831ce40e24cd"
MOTTO="Мы должны узнать"
def ck(v,m):h25.ck(v,m)
def read(p):return h25.read(p)
def write(p,x):h25.write(p,x)
def log(t,x):h25.log(t,x)
def digest(x):return h25.digest(x)
def cards():return h19.load_exam("R84H28_SOURCE_CARDS.json",CARDS)["cards"]
def cortex(demi):
    for name,pin in (("memory_cortex_growth.py",h21.CORTEX),("memory_512_audit_index_r3.py",h21.AUDIT)):
        ck(r.blob((demi/"janus_model"/name).read_bytes())==pin,"CORTEX_PIN")
    sys.path.insert(0,str(demi.resolve()));return importlib.import_module("janus_model.memory_cortex_growth")
class Visible(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.depth=0;self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in ("script","style"):self.depth+=1
    def handle_endtag(self,tag):
        if tag in ("script","style"):self.depth=max(0,self.depth-1)
    def handle_data(self,data):
        if not self.depth:self.parts.append(data)
def normalized(t):return " ".join(t.lower().replace("\u2019","'").split())
class Bounded(requests.Session):
    def __init__(self,urls,cap,maxcalls):
        super().__init__();self.trust_env=False;self.urls=set(urls);self.cap=cap;self.maxcalls=maxcalls;self.calls=0;self.nbytes=0
    def get(self,url,**kwargs):
        u=urllib.parse.urlsplit(url);base=urllib.parse.urlunsplit((u.scheme,u.netloc,u.path,"",""))
        ck(base in self.urls and u.scheme=="https" and self.calls<self.maxcalls,"NETWORK_SCOPE")
        self.calls+=1;kwargs.update(stream=True,allow_redirects=False,timeout=(8,20))
        with super().get(url,**kwargs) as resp:
            ck(resp.status_code==200,"HTTP_"+str(resp.status_code));chunks=[];n=0
            for b in resp.iter_content(8192):
                n+=len(b);self.nbytes+=len(b);ck(n<=self.cap,"SOURCE_SIZE");chunks.append(b)
            resp._content=b"".join(chunks);resp._content_consumed=True;return resp
def sources(demi,lapis,out):
    out.mkdir(parents=True,exist_ok=False);m=cortex(demi);cs=cards();root=out/"cortex";root.mkdir();state=m.new_state()
    urls=sorted({z["url"] for z in cs});sess=Bounded(urls,1500000,len(urls));pages={};audit=[]
    for url in urls:
        try:
            response=sess.get(url);raw=response.content;p=Visible();p.feed(raw.decode("utf-8",errors="replace"));text=normalized(" ".join(p.parts))
            pages[url]=(text,h21.sha(raw))
        except Exception as exc:audit.append({"url":url,"error":type(exc).__name__+":"+str(exc)[:120]})
    admitted=[]
    for z in cs:
        body,sha=pages.get(z["url"],("",None));ok=sha is not None and all(normalized(a) in body for a in z["anchors"])
        rec={"label":z["label"],"url":z["url"],"provider_sha256":sha,"anchors_present":bool(ok),"semantic_truth_verified":False,"training_allowed":False}
        if ok:
            text=z["text"].encode();origin={"kind":"OWN_AUTHORED_SOURCE_REFERENCED_CARD","url":z["url"],"label":z["label"],"provider_sha256":sha,"training_allowed":False}
            ck(m.add_text(root,state,text,origin) and not m.add_text(root,state,text,origin),"CORTEX_DEDUP")
            rec["card_sha256"]=h21.sha(text);admitted.append(rec)
        audit.append(rec)
    manifest=m.publish(root,state,"H28_ISOLATED_AUTHORED_CARDS") if admitted else None
    if admitted:m.write_json(root/"STATE.json",m.sealed(state));ck(len(m.load_state(root)["active"])==len(admitted),"CORTEX_REOPEN")
    # Native BEETLE executes a bounded, fixed public search, never sends user
    # or exam texts. Discovery leads are not evidence and are not admitted.
    scout=h20.pinned("beetle_h28",lapis/"tools/universal_beetle.py",h20.BEETLE)
    bs=Bounded(["https://export.arxiv.org/api/query"],256000,2);scout.session=lambda:bs
    dirs=[{"id":"cache","title":"Cache write policies","keywords":["cache","write-through","write-back"],"min_score":1,"sources":[{"kind":"arxiv","query":'all:"write-back" AND all:"cache"',"max_results":3}]},{"id":"semaphore","title":"Counting semaphore","keywords":["semaphore","concurrency"],"min_score":1,"sources":[{"kind":"arxiv","query":'all:"semaphore" AND all:"concurrency"',"max_results":3}]}]
    with tempfile.TemporaryDirectory() as td:
        st=scout.State(Path(td));summary=scout.run_scout({"global_budget":{"max_requests_per_run":2,"max_new_candidates_per_run":6,"max_metadata_bytes_per_run":512000},"directions":dirs},st,None)
        leads=[{"candidate_id":i,"url":v["url"],"status":"QUARANTINE_UNVERIFIED","training_allowed":False} for i,v in sorted(st.seen.items())]
        ck(len(leads)<=6 and bs.calls<=2,"SCOUT_BUDGET")
        head=st.prev_receipt
    report={"schema":"r84h28.sources.v1","card_blob":CARDS,"official_requests":sess.calls,"official_bytes":sess.nbytes,"audit":audit,"admitted":admitted,"indexed_documents":manifest["indexed_documents"] if manifest else 0,"beetle_requests":bs.calls,"beetle_summary":summary,"beetle_leads":leads,"beetle_receipt_head":head,"external_text_stored":False,"production":False}
    write(out/"sources.json",report);log("R84H28_SOURCES",report)
STOP=set("a an the is are was were be been being of to in on at by for with from as and or that this it its they their can may must will has have had does do which what how when after before".split())
def tokens(t):return [w for w in re.findall(r"[a-z0-9]+",t.lower()) if w not in STOP]
def features(t):
    words=tokens(t);f=Counter("w:"+w for w in words)
    for w in words:
        for i in range(max(0,len(w)-2)):f["c:"+w[i:i+3]]+=.25
    return f
def cosine(a,b):
    return sum(x*b.get(k,0) for k,x in a.items())/max(1e-12,math.sqrt(sum(x*x for x in a.values())*sum(x*x for x in b.values())))
def scope(t):
    # General conservative abstention for mixed or explicitly negated claims.
    return not re.search(r"\b(?:also|either|compare|hypothetic\w*|suppose|imagine|explicitly not|does not|do not|not asserted|never retries|does not say)\b",t.lower()) and "?" not in t
class Retrieval:
    def __init__(self,demi,source):
        self.m=cortex(demi);self.root=source/"cortex";self.rec=read(source/"sources.json")
        ck(self.rec["card_blob"]==CARDS and not self.rec["production"],"SOURCES_PIN")
        self.cards={z["label"]:z for z in cards()};self.active={z["card_sha256"]:z for z in self.rec["admitted"]}
        self.state=self.m.load_state(self.root);ck(set(self.state["active"])==set(self.active),"CORTEX_ACTIVE")
        self.cache={};self.cpu=0.;self.calls=0
    def inspect(self,t):
        if t in self.cache:return self.cache[t]
        start=time.process_time();out={"candidates":[],"scope":scope(t)}
        if out["scope"] and self.active:
            words=list(dict.fromkeys(tokens(t)));query=" ".join(words)[:120].rsplit(" ",1)[0] if len(" ".join(words))>120 else " ".join(words)
            if query:
                q=self.m.query(self.root,query,limit=5);self.m.checked(q);self.calls+=1;f=features(t)
                for z in q["citations"]:
                    a=self.active[z["sha256"]];rec=self.state["records"][z["sha256"]]
                    raw=self.m.read_object(self.root,z["sha256"],rec["size"]);card=self.cards[a["label"]]
                    ck(raw==card["text"].encode() and z["origin"]["provider_sha256"]==a["provider_sha256"] and z["content_is_untrusted_data"],"RETRIEVED_POINTER")
                    out["candidates"].append({"label":a["label"],"sha256":z["sha256"],"url":a["url"],"similarity":cosine(f,features(raw.decode()))})
                out["candidates"].sort(key=lambda x:(-x["similarity"],x["label"]))
        self.cache[t]=out;self.cpu+=time.process_time()-start;return out
def baseline(z,c,enabled):
    a,v=h27.verdict(z,c,enabled)
    return a, v["verdict"]=="SUPPORTED_CANDIDATE" and v["matches"][0]["label"] in enabled
def choose(z,c,enabled,ret,labels,policy):
    base,supported=baseline(z,c,enabled);q=ret.inspect(z["text"]);a=base;used=False
    # Keep accepted H27 evidence; retrieval only supplements unresolved cases.
    if not supported and q["scope"] and q["candidates"]:
        top=q["candidates"][0];margin=top["similarity"]-(q["candidates"][1]["similarity"] if len(q["candidates"])>1 else 0.)
        rule=policy.get(top["label"])
        if rule and top["similarity"]>=rule["min_similarity"] and margin>=rule["min_margin"]:
            a=int(labels[top["label"]]);used=True
    return a,base,used,q,supported
def measure(rows,c,enabled,ret,labels,policy):
    m={"known_n":0,"base_correct":0,"correct":0,"rescued":0,"lost":0,"retrieval_used":0,"retrieval_wrong":0,"unknown_n":0,"unknown_retrieval_supported":0,"joint_wrong":0,"joint_wrong_correct":0};detail=[]
    for z in rows:
        a,b,used,q,supported=choose(z,c,enabled,ret,labels,policy);d={"id":z["id"],"text":z["text"],"kind":z["kind"],"retrieval_used":used,"retrieval":q,"h27_supported":supported}
        if z["kind"]=="known":
            good=a==z["truth"];old=b==z["truth"];joint=z["neural"]==z["lexical"] and z["neural"]!=z["truth"]
            m["known_n"]+=1;m["base_correct"]+=old;m["correct"]+=good;m["rescued"]+=good and not old;m["lost"]+=old and not good;m["retrieval_used"]+=used;m["retrieval_wrong"]+=used and not good;m["joint_wrong"]+=joint;m["joint_wrong_correct"]+=joint and good;d.update({"truth":z["truth"],"answer":a,"h27_answer":b,"correct":good,"joint_wrong":joint})
        else:m["unknown_n"]+=1;m["unknown_retrieval_supported"]+=used
        detail.append(d)
    return m,detail
def past(ctx,h25folder,h26folder,h27gate,h27result):
    p=read(h27gate/"gate.json");a=read(h25folder/"feedback.json");b=read(h26folder/"feedback.json");z=read(h27result/"result.json")
    ck(p["h26_feedback_sha"]==digest(b) and p["h25_feedback_sha"]==digest(a) and z["improvement"]=="PASS_SHADOW" and z["fresh_guarded"]["correct"]==38 and not z["production"],"H27_PARENT")
    ex=h19.load_exam("R84H27_FROZEN_EXAM.json",h27.EXAM);refs={h18.clean(t) for t,_ in ctx.fit+ctx.dev+ctx.oldknown}|{h18.clean(t) for t in ctx.oldunknown}|{h18.clean(t["text"]) for t in a+b}
    k=[x for x in ex["known"] if h18.clean(x[0]) not in refs];u=[x for x in ex["unknown"] if h18.clean(x) not in refs]
    c,cost=ctx.observations(k,u,"H27_FRESH_EVAL")
    ck(len(c)==60,"H27_REPLAY_SIZE");return a,b,c,set(p["enabled"])
def gate(*args):
    *paths,h25folder,h26folder,h27gate,h27result,source,demi,out=args;ctx=h25.Context(*paths);out.mkdir(parents=True,exist_ok=False);a,b,c,enabled=past(ctx,h25folder,h26folder,h27gate,h27result)
    critic=h27.Critic(ctx.labels);ret=Retrieval(demi,source);base,_=measure(c,critic,enabled,ret,ctx.labels,{});ck(base["correct"]==38,"H27_REPLAY")
    # Select thresholds only on disclosed prior evidence. H28 exam not available.
    policy={};audit={};rows=a+b+c
    for card in cards():
        label=card["label"];options=[]
        for s in (.20,.30,.40,.50,.60):
            for gap in (.03,.08,.15):
                p={label:{"min_similarity":s,"min_margin":gap}};m,_=measure(rows,critic,enabled,ret,ctx.labels,p)
                if m["lost"]==0 and m["retrieval_wrong"]==0 and m["unknown_retrieval_supported"]==0 and m["rescued"]>=1:options.append((m["rescued"],s,gap,m))
        if options:
            _,s,gap,m=max(options,key=lambda x:(x[0],x[1],x[2]));policy[label]={"min_similarity":s,"min_margin":gap};audit[label]=m
    prev,detail=measure(b,critic,enabled,ret,ctx.labels,policy)
    ck(prev["joint_wrong"]==9 and prev["lost"]==0,"PRIOR_GATE")
    report={"schema":"r84h28.gate.v1","source_sha":digest(ret.rec),"enabled_h27":sorted(enabled),"retrieval_policy":policy,"validation":audit,"prior":prev,"prior_joint_failures":[z for z in detail if z.get("joint_wrong")],"trinity_controls":h27.trinity(demi),"retrieval_cpu_seconds":ret.cpu,"retrieval_queries":ret.calls,"exam_read":False,"production":False}
    write(out/"gate.json",report);log("R84H28_GATE",report)
def evaluate(*args):
    *paths,h25folder,h26folder,h27gate,h27result,source,demi,gateout,out=args;ctx=h25.Context(*paths);out.mkdir(parents=True,exist_ok=False)
    policy=read(gateout/"gate.json");ret=Retrieval(demi,source);ck(policy["source_sha"]==digest(ret.rec) and not policy["exam_read"] and not policy["production"],"GATE_PIN")
    a,b,c,enabled=past(ctx,h25folder,h26folder,h27gate,h27result);ck(set(policy["enabled_h27"])==enabled,"H27_PIN")
    ex=h19.load_exam("R84H28_FROZEN_EXAM.json",EXAM);refs={h18.clean(t) for t,_ in ctx.fit+ctx.dev+ctx.oldknown}|{h18.clean(t) for t in ctx.oldunknown}|{h18.clean(z["text"]) for z in a+b+c}
    known=[x for x in ex["known"] if h18.clean(x[0]) not in refs];unknown=[t for t in ex["unknown"] if h18.clean(t) not in refs];ck(len(known)>=35 and len(unknown)>=18,"EXAM_OVERLAP")
    rows,cost=ctx.observations(known,unknown,"H28_FRESH_EVAL");critic=h27.Critic(ctx.labels)
    base,_=measure(rows,critic,enabled,ret,ctx.labels,{});fresh,details=measure(rows,critic,enabled,ret,ctx.labels,policy["retrieval_policy"])
    conflicts=[]
    for t in ex["conflict"]:
        z={"kind":"unknown","text":t};ans,b0,used,q,supported=choose(z,critic,enabled,ret,ctx.labels,policy["retrieval_policy"])
        conflicts.append({"text":t,"retrieval_used":used,"h27_supported":supported,"retrieval":q})
    # H27 is retained as the comparator, including its possible false support;
    # report combined risk, not merely retrieval-added errors.
    unknown_combined=sum(z["kind"]=="unknown" and (z["retrieval_used"] or z["h27_supported"]) for z in details)
    conflict_held=sum(not z["retrieval_used"] and not z["h27_supported"] for z in conflicts)
    pred,routes,histcpu=ctx.neural(ctx.hist);parity=sum(ctx.prev[i]["answer"]==int(pred[i]) and ctx.prev[i]["route"]==int(routes[i]) for i in range(671));ck(parity==671,"HISTORY")
    safe=fresh["lost"]==0 and fresh["retrieval_wrong"]==0 and unknown_combined==0 and conflict_held==len(conflicts)
    report={"schema":"r84h28.result.v1","motto":MOTTO,"source_cards":len(ret.active),"beetle_leads_quarantined":len(ret.rec["beetle_leads"]),"beetle_requests":ret.rec["beetle_requests"],"retrieval_policy":policy["retrieval_policy"],"prior_joint_correct":policy["prior"]["joint_wrong_correct"],"prior_joint_total":9,"fresh_h27":base,"fresh_h28":fresh,"unknown_combined_supported":unknown_combined,"conflict_held":conflict_held,"conflict_n":len(conflicts),"overlap_excluded":60-len(known)-len(unknown),"historical_parity":parity,"retrieval_cpu_seconds":ret.cpu,"retrieval_queries":ret.calls,"cost":cost,"historical_cpu_seconds":histcpu,"trinity_controls":h27.trinity(demi),"safety":"PASS_SHADOW" if safe else "HOLD","improvement":"PASS_SHADOW" if safe and fresh["rescued"]>0 else "HOLD_NO_PROVEN_GAIN","production":False,"scope":"live bounded official source anchor checks, authored paraphrase cards only, native Cortex verified pointers and lexical similarity, not semantic truth verification; actual BEETLE discovery quarantined, native TRINITY arithmetic controls only; same-owner scenario exam; separate historical replay, no live production corpus or automatic training"}
    write(out/"result.json",report);write(out/"cases.json",details);write(out/"conflicts.json",conflicts);log("R84H28_RESULT",report)
if __name__=="__main__":{"sources":sources,"gate":gate,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
