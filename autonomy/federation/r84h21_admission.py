"""R84H21 finite source admission and independently authored GC graph shadow study."""
import hashlib,importlib,sys,json,re,time,urllib.request,urllib.error,urllib.parse,xml.etree.ElementTree as ET,random
from pathlib import Path
from html.parser import HTMLParser
HERE=Path(__file__).parent
PROTO="07938d1fc6e1e2284bff3a0581fdf0b751840fc2"
EXAM="b7f6ae2dc183e0d1c42d975c4ee72e51869bde1d"
CORTEX="626b74379e195d2e72d7c5b727198f6a3e297f04"
AUDIT="a01fe718018cd10bf4b0e5c1ae3b83dd90364ff3"
MOTTO="Мы должны узнать"
def ck(x,s):
    if not x:raise ValueError(s)
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(("blob %d\0"%len(b)).encode()+b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(canonical(x)+"\n")
def log(t,x):print(t,canonical(x),flush=True)
def protocol():
    raw=(HERE/"R84H21_PROTOCOL.json").read_bytes();ck(blob(raw)==PROTO,"PROTOCOL");return json.loads(raw)
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None
class LicenseParser(HTMLParser):
    # Only an explicit rel=license on the requested version page is evidence.
    def __init__(self):
        super().__init__();self.links=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag in ("a","link") and "license" in (a.get("rel") or "").lower().split() and a.get("href"):
            self.links.add(a["href"])
def fetch(url,cap):
    u=urllib.parse.urlsplit(url)
    ck(u.scheme=="https" and ((u.netloc=="export.arxiv.org" and u.path=="/api/query") or (u.netloc=="arxiv.org" and re.fullmatch(r"/abs/\d{4}\.\d{4,5}v\d+",u.path))),"FETCH_SCOPE")
    req=urllib.request.Request(url,headers={"User-Agent":"JANUS-Knowledge-Admission/1.0","Accept":"application/atom+xml,text/html"})
    with urllib.request.build_opener(NoRedirect).open(req,timeout=25) as r:
        b=r.read(cap+1);ck(len(b)<=cap,"RESPONSE_CAP");return b
def sources(demi,h20,out):
    p=protocol();out.mkdir(parents=True,exist_ok=False)
    for name,pin in (("memory_cortex_growth.py",CORTEX),("memory_512_audit_index_r3.py",AUDIT)):
        ck(blob((demi/"janus_model"/name).read_bytes())==pin,"CORTEX_PIN")
    sys.path.insert(0,str(demi.resolve()));m=importlib.import_module("janus_model.memory_cortex_growth")
    v=read(h20/"verification.json");ck(v["schema"]=="r84h20.verification.v1" and v["production"] is False and not v["training_allowed"] and len(v["claims"])==5,"PARENT")
    expected={"https://arxiv.org/abs/"+x for x in p["ids"]}
    ck({x["url"] for x in v["claims"]}==expected and all(x["oracle_verdict"]=="HOLD_UNSUPPORTED" for x in v["claims"]),"PARENT_IDS")
    state=m.new_state();root=out/"cortex-shadow";root.mkdir();rows=[];requests=0
    for item in sorted(v["claims"],key=lambda x:x["url"]):
        url=item["url"];ident,primary=m.canonical_lead(url);rec={"id":ident,"url":url,"primary_identity":False,"license_status":"HOLD_NO_EXPLICIT_VERSION_LICENSE","relevance":"UNASSESSED","claim_truth":"HOLD_UNVERIFIED","training_allowed":False}
        text=None
        try:
            requests+=1;raw=fetch(primary,p["max_response_bytes"]);text=m.primary_text(ident,raw)
            rec.update({"primary_identity":True,"provider_sha256":sha(raw),"normalized_text_sha256":sha(text),"title":text.decode().split("\n\n",1)[0]})
            low=text.decode().lower()
            rec["relevance"]="GC_KEYWORD_MATCH_ONLY" if re.search(r"\bgarbage.collect(?:ion|or|ing)\b",low) and re.search(r"\b(heap|reachab\w*|trac\w*|mark.and.sweep)\b",low) else "HOLD_RELEVANCE"
        except Exception as exc:rec["metadata_error"]=type(exc).__name__+":"+str(exc)[:120]
        time.sleep(3.1)
        try:
            requests+=1;page=fetch(url,p["max_response_bytes"]);parser=LicenseParser();parser.feed(page.decode("utf-8",errors="strict"))
            rec["license_page_sha256"]=sha(page);rec["license_links"]=sorted(parser.links)
            if len(parser.links)==1 and next(iter(parser.links)) in p["licenses"]:
                rec["license_status"]="EXPLICIT_ALLOWLIST_METADATA_STORAGE_ONLY";rec["license_url"]=next(iter(parser.links))
        except Exception as exc:rec["license_error"]=type(exc).__name__+":"+str(exc)[:120]
        # Only independently retrieved, relevant, explicitly licensed metadata may
        # enter the disposable native corpus. No claims or paper-derived labels.
        eligible=rec["primary_identity"] and rec["relevance"]=="GC_KEYWORD_MATCH_ONLY" and rec["license_status"]=="EXPLICIT_ALLOWLIST_METADATA_STORAGE_ONLY"
        rec["metadata_admitted"]=False
        if eligible:
            origin={"kind":"LICENSE_GATED_ABSTRACT_METADATA_NOT_PROOF","url":url,"primary_id":ident,"provider_sha256":rec["provider_sha256"],"license_url":rec["license_url"],"training_allowed":False}
            rec["metadata_admitted"]=m.add_text(root,state,text,origin)
        rows.append(rec);log("R84H21_SOURCE",rec)
    ck(requests<=p["max_requests"],"REQUEST_CAP")
    # Original authored graph principle is independently valid by graph closure,
    # not attributed to or derived from any of the five papers.
    own=b"JANUS synthetic directed graph reachability control. Starting at designated root vertices, repeatedly visit outgoing neighbors until no additional vertex is reachable. This models only abstract tracing reachability, not a complete garbage collector."
    ownorigin={"kind":"OWN_AUTHORED_SYNTHETIC_CONTROL","url":None,"training_allowed":False}
    ck(m.add_text(root,state,own,ownorigin) and not m.add_text(root,state,own,ownorigin),"NATIVE_DEDUP")
    manifest=m.publish(root,state,"R84H21_DISPOSABLE_LICENSE_GATED_METADATA_AND_CONTROL")
    m.write_json(root/"STATE.json",m.sealed(state));q=m.query(root,"reachability",limit=5);m.checked(q)
    ck(any(x["sha256"]==sha(own) and x["origin"]["kind"]=="OWN_AUTHORED_SYNTHETIC_CONTROL" for x in q["citations"]),"NATIVE_QUERY_CONTROL")
    restored=m.load_state(root);ck(len(restored["active"])==1+sum(x["metadata_admitted"] for x in rows),"NATIVE_REOPEN")
    report={"schema":"r84h21.source_audit.v1","requests":requests,"sources":rows,"identity_verified":sum(x["primary_identity"] for x in rows),"explicit_license":sum(x["license_status"]=="EXPLICIT_ALLOWLIST_METADATA_STORAGE_ONLY" for x in rows),"relevance_matches":sum(x["relevance"]=="GC_KEYWORD_MATCH_ONLY" for x in rows),"external_metadata_admitted":sum(x["metadata_admitted"] for x in rows),"scientific_claims_admitted":0,"paper_training_examples":0,"native_cortex_api":True,"native_cortex_durable_state_changed":False,"native_index_documents":manifest["indexed_documents"],"native_query_control":True,"native_dedup":True,"motto":MOTTO,"production":False}
    write(out/"source_result.json",report);log("R84H21_SOURCE_GATE",report)
def reachable(edges,roots,target):
    visited={i for i,v in enumerate(roots) if v};pending=list(visited)
    while pending:
        u=pending.pop()
        for v in range(6):
            if edges[u*6+v] and v not in visited:visited.add(v);pending.append(v)
    return int(target in visited)
def key(edges,roots,target):return sha(canonical([edges,roots,target]).encode())
def vector(edges,roots,target):return edges+roots+[int(i==target) for i in range(6)]
def train(out):
    import torch
    torch.set_num_threads(1);torch.manual_seed(21072);rng=random.Random(21073)
    out.mkdir(parents=True,exist_ok=False);rows=[];seen=set();counts=[0,0]
    for _ in range(30000):
        if min(counts)>=1500:break
        p=rng.choice((.20,.32,.44,.56));edges=[int(i%7!=0 and rng.random()<p) for i in range(36)];roots=[int(rng.random()<.2) for _ in range(6)]
        if not any(roots):roots[rng.randrange(6)]=1
        target=rng.randrange(6);k=key(edges,roots,target);y=reachable(edges,roots,target)
        if k in seen or counts[y]>=1500:continue
        rows.append({"edges":edges,"roots":roots,"target":target,"label":y,"key":k});seen.add(k);counts[y]+=1
    ck(counts==[1500,1500],"TRAIN_BALANCE")
    x=torch.tensor([vector(z["edges"],z["roots"],z["target"]) for z in rows],dtype=torch.float32);y=torch.tensor([z["label"] for z in rows])
    model=torch.nn.Sequential(torch.nn.Linear(48,64),torch.nn.ReLU(),torch.nn.Linear(64,2));opt=torch.optim.AdamW(model.parameters(),lr=.003)
    start=time.process_time()
    for step in range(600):
        ix=torch.randperm(len(rows))[:64];loss=torch.nn.functional.cross_entropy(model(x[ix]),y[ix]);ck(bool(torch.isfinite(loss)),"LOSS");opt.zero_grad();loss.backward();opt.step()
    torch.save(model.state_dict(),out/"shadow.pt");write(out/"train.json",rows)
    rec={"schema":"r84h21.synthetic_train.v1","examples":len(rows),"counts":counts,"steps":600,"presentations":38400,"parameters":sum(p.numel() for p in model.parameters()),"model_sha256":sha((out/"shadow.pt").read_bytes()),"train_sha256":sha((out/"train.json").read_bytes()),"cpu_seconds":time.process_time()-start,"paper_text_used":False,"exam_read":False,"production":False}
    write(out/"train_receipt.json",rec);log("R84H21_TRAIN",rec)
def evaluate(trained,out):
    import torch
    torch.set_num_threads(1)
    raw=(HERE/"R84H21_GRAPH_EXAM.json").read_bytes();ck(blob(raw)==EXAM,"EXAM_PIN");ex=json.loads(raw);rec=read(trained/"train_receipt.json");rows=read(trained/"train.json")
    ck(rec["production"] is False and not rec["paper_text_used"] and not rec["exam_read"] and sha((trained/"shadow.pt").read_bytes())==rec["model_sha256"] and sha((trained/"train.json").read_bytes())==rec["train_sha256"],"TRAIN_PIN")
    known={x["key"] for x in rows};model=torch.nn.Sequential(torch.nn.Linear(48,64),torch.nn.ReLU(),torch.nn.Linear(64,2));model.load_state_dict(torch.load(trained/"shadow.pt",map_location="cpu",weights_only=True));model.eval()
    cases=ex["cases"];ck(len(cases)==240 and len({key(x["edges"],x["roots"],x["target"]) for x in cases})==240,"EXAM_SIZE")
    overlap=sum(key(x["edges"],x["roots"],x["target"]) in known for x in cases);clean=[x for x in cases if key(x["edges"],x["roots"],x["target"]) not in known]
    x=torch.tensor([vector(z["edges"],z["roots"],z["target"]) for z in clean],dtype=torch.float32);truth=[z["expected"] for z in clean]
    with torch.no_grad():pred=model(x).argmax(-1).tolist()
    oracle=sum(reachable(z["edges"],z["roots"],z["target"])==z["expected"] for z in cases)
    counts=[truth.count(0),truth.count(1)];correct=[sum(p==y==k for p,y in zip(pred,truth)) for k in (0,1)]
    ck(oracle==240 and min(counts)>0 and len(clean)>=230,"ORACLE_REFERENCE")
    bal=sum(correct[k]/counts[k] for k in (0,1))/2
    result={"schema":"r84h21.graph_exam_result.v1","frozen_n":240,"exact_training_overlap_excluded":overlap,"held_out_n":len(clean),"neural_correct":sum(correct),"class_counts":counts,"class_correct":correct,"balanced_accuracy":bal,"majority_baseline":max(counts),"deterministic_oracle_correct":oracle,"oracle_n":240,"shadow_neural_gate":"PASS_SHADOW" if overlap==0 and bal>=.95 else "HOLD","paper_text_used":False,"production":False,"scope":"6-node synthetic tracing reachability only; not freeform garbage collection knowledge or external scientific validation"}
    out.mkdir(parents=True,exist_ok=False);write(out/"graph_result.json",result);log("R84H21_GRAPH_GATE",result)
if __name__=="__main__":
    {"sources":sources,"train":train,"evaluate":evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
