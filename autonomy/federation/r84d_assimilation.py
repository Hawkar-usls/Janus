"""R84D bounded source evidence and shadow associative factual memory; not neural weight training."""
import json,hashlib,re,sys,time,unicodedata,urllib.request,urllib.parse,urllib.error,xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
import jfp_gateway as g
import r8_learning as r
import r84_loop as a
import r84c_engine as c
PARENT="0aecd2eb4aa3731941495e3ebdade6dd63715960"
RECEIPT_BLOB="ff58b423d1d709ef204f3cee476dab89de745f1a"
RECEIPT_HASH="e55bcff19a43ae6e39608d326b20da13163ef2d718f3402027335228bf0fb83d"
EXAM="d51742fd14cfd7b10f962b802b8158517baf5620"
SCOPE={"2611.32496v1":{"terms":["PPO","SAC","replay"],"claim":"RL replay comparison"},"2611.32691v1":{"terms":["firing rate","threshold","plasticity"],"claim":"firing-rate threshold control"}}
CC={"https://creativecommons.org/licenses/by/4.0/":"CC-BY-4.0","https://creativecommons.org/publicdomain/zero/1.0/":"CC0-1.0"}
META_TERMS="https://info.arxiv.org/help/api/tou.html"
def ck(x,msg):g.require(x,msg)
def log(tag,obj):print(tag,json.dumps(obj,sort_keys=True),flush=True)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def prior():
    z=a.get(r.J,PARENT,"data/r84c-engine/RECEIPT.json",RECEIPT_BLOB)
    ck(z["sha256"]==RECEIPT_HASH and g.digest({k:v for k,v in z.items() if k!="sha256"})==RECEIPT_HASH,"DISCOVERY_PIN")
    ids=[x["url"].split("/")[-1] for x in z["discovery"]["leads"]]
    ck(len(ids)==5 and len(set(ids))==5 and all(re.fullmatch(r"2611\.\d{5}v1",x) for x in ids) and set(SCOPE)<=set(ids),"IDS")
    return ids
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None
last=0
def fetch(url):
    global last
    u=urllib.parse.urlparse(url)
    ck(u.scheme=="https" and u.netloc in ("export.arxiv.org","arxiv.org") and (u.path=="/api/query" or re.fullmatch(r"/abs/2611\.\d{5}v1",u.path)),"URL")
    time.sleep(max(0,3.2-(time.monotonic()-last)));last=time.monotonic()
    req=urllib.request.Request(url,headers={"User-Agent":"JANUS R84D bounded research metadata check","Accept":"application/atom+xml,text/html"})
    with urllib.request.build_opener(NoRedirect).open(req,timeout=20) as f:
        ck(f.status==200 and f.geturl()==url,"HTTP")
        data=f.read(512001)
    ck(0<len(data)<=512000,"BYTE_CAP");return data
class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.meta={};self.links=[];self.href=None;self.text=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=="meta" and d.get("name","").startswith("citation_"):self.meta.setdefault(d["name"],[]).append(d.get("content",""))
        if tag=="a":self.href=d.get("href");self.text=[]
    def handle_data(self,d):
        if self.href:self.text.append(d)
    def handle_endtag(self,tag):
        if tag=="a":
            if self.href:self.links.append((self.href," ".join("".join(self.text).split()).lower()))
            self.href=None;self.text=[]
def norm(x):return " ".join(x.split())
def extract(atom,page,id):
    ns={"a":"http://www.w3.org/2005/Atom","x":"http://arxiv.org/schemas/atom"}
    entries=[e for e in ET.fromstring(atom).findall("a:entry",ns) if e.findtext("a:id","",ns).split("/")[-1]==id]
    ck(len(entries)==1,"EXACT_VERSION");e=entries[0];p=Page();p.feed(page.decode("utf-8","strict"))
    title=norm(e.findtext("a:title","",ns));authors=[norm(x.findtext("a:name","",ns)) for x in e.findall("a:author",ns)]
    primary=e.find("x:primary_category",ns);ck(primary is not None,"PRIMARY")
    pt=norm(p.meta.get("citation_title",[""])[0]);pa=[norm(x) for x in p.meta.get("citation_author",[])]
    # arXiv Atom uses given-name first; citation_author can use family-name first.
    # Preserve author count and sequence, compare tokens of each name (not fuzzy matching).
    def namekey(s):
        return sorted(re.findall(r"[^\W_]+",unicodedata.normalize("NFKC",s).casefold(),re.UNICODE))
    def titlekey(s):return norm(unicodedata.normalize("NFKC",s)).casefold()
    same_title=bool(title) and titlekey(title)==titlekey(pt)
    same_authors=bool(authors) and [namekey(x) for x in authors]==[namekey(x) for x in pa]
    ck(same_title and same_authors,"CROSS_SOURCE_IDENTITY:title="+str(same_title)+",authors="+str(same_authors)+",counts="+str(len(authors))+"/"+str(len(pa)))

    links=[h.replace("http://","https://",1).rstrip("/")+"/" for h,t in p.links if t in ("view license","license","(license)") and ("creativecommons.org/" in h or "arxiv.org/licenses/" in h)]
    ck(len(set(links))==1,"LICENSE_LINK")
    lic=CC.get(links[0],"OTHER_OR_RESTRICTED")
    summary=norm(e.findtext("a:summary","",ns));fields={"primary":primary.attrib["term"],"author_count":len(authors),"license":lic}
    claim=None
    if id in SCOPE and lic in CC.values():
        spec=SCOPE[id];ck(all(x.casefold() in summary.casefold() for x in spec["terms"]),"CLAIM_EVIDENCE")
        fields["claim"]=spec["claim"];claim={"kind":"AUTHOR_REPORTED_TOPIC_NOT_INDEPENDENTLY_VERIFIED","label":spec["claim"],"evidence_terms":spec["terms"],"abstract_sha256":sha(summary.encode()),"transformation":"human-reviewed short topic label; no full text copied"}
    return {"id":id,"title":title,"authors":authors,"source":"https://arxiv.org/abs/"+id,"license_url":links[0],"metadata_terms":META_TERMS,"fields":fields,"claim":claim,"atom_sha256":sha(atom),"page_sha256":sha(page)}
def harvest(out):
    ids=prior();out.mkdir(parents=True,exist_ok=False)
    u="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"id_list":",".join(ids),"max_results":5})
    records=[];holds=[];requests=0
    try:requests+=1;atom=fetch(u)
    except Exception as e:
        atom=None;holds=[{"id":i,"reason":"ATOM_UNAVAILABLE:"+type(e).__name__} for i in ids]
    if atom:
        for i in ids:
            try:
                requests+=1;page=fetch("https://arxiv.org/abs/"+i);records.append(extract(atom,page,i))
            except Exception as e:holds.append({"id":i,"reason":str(e)[:150]})
    ck(requests<=6,"REQUEST_BUDGET")
    # Persist source-attributed metadata only. Raw documents are not shared.
    pack={"schema":"r84d.verified_source_packet.v1","parent_receipt":RECEIPT_HASH,"requests":requests,"records":records,"holds":holds,"full_text_ingested":False,"independent_scientific_validation":False,"production":False,"acknowledgement":"Thank you to arXiv for use of its open access interoperability."}
    pack["sha256"]=g.digest(pack);g.write_json(out/"packet.json",pack)
    log("R84D_HARVEST",{"records":[{"id":x["id"],"fields":x["fields"],"license_url":x["license_url"]} for x in records],"holds":holds,"requests":requests,"packet_sha256":pack["sha256"]})
def validate(p):
    ck(p["schema"]=="r84d.verified_source_packet.v1" and p["sha256"]==g.digest({k:v for k,v in p.items() if k!="sha256"}) and p["parent_receipt"]==RECEIPT_HASH and p["production"] is False and p["full_text_ingested"] is False,"PACKET")
    ids=prior();ck(sorted(x["id"] for x in p["records"]+p["holds"])==sorted(ids) and p["requests"]<=6,"COVERAGE")
    for x in p["records"]:
        ck(x["source"]=="https://arxiv.org/abs/"+x["id"] and x["metadata_terms"]==META_TERMS and x["fields"]["author_count"]==len(x["authors"]) and x["fields"]["license"]==CC.get(x["license_url"],"OTHER_OR_RESTRICTED"),"RECORD")
        if x["claim"]:ck(x["id"] in SCOPE and x["fields"]["license"] in CC.values() and x["fields"]["claim"]==SCOPE[x["id"]]["claim"] and x["claim"]["kind"]=="AUTHOR_REPORTED_TOPIC_NOT_INDEPENDENTLY_VERIFIED","CLAIM")
    return p
def assimilate(inp,out):
    p=validate(g.read_json(inp/"packet.json"));m={"schema":"r84d.associative_memory.v1","parent_model":c.MODEL_HASH,"packet_sha256":p["sha256"],"facts":{x["id"]:x["fields"] for x in p["records"]},"production":False}
    out.mkdir(parents=True,exist_ok=False);g.write_json(out/"memory.json",m);g.write_json(out/"packet.json",p)
    log("R84D_ASSIMILATE",{"records":len(m["facts"]),"fields":sum(len(x) for x in m["facts"].values()),"memory_sha256":g.digest(m),"neural_training":False})
def answer(m,id,field):
    if field not in ("primary","author_count","license","claim"):return "HOLD"
    return m.get("facts",{}).get(id,{}).get(field,"HOLD")
def audit(inp,out):
    m=g.read_json(inp/"memory.json");p=validate(g.read_json(inp/"packet.json"))
    ck(m["facts"]=={x["id"]:x["fields"] for x in p["records"]} and m["packet_sha256"]==p["sha256"],"MEMORY")
    ex=(Path(__file__).parent/"R84D_FROZEN_EXAM.json").read_bytes();ck(r.blob(ex)==EXAM,"EXAM")
    rows=json.loads(ex)["cases"];before=[answer({},x["id"],x["field"])==x["expected"] for x in rows];after=[answer(m,x["id"],x["field"])==x["expected"] for x in rows]
    known=[i for i,x in enumerate(rows) if x["expected"]!="HOLD"];unknown=[i for i,x in enumerate(rows) if x["expected"]=="HOLD"]
    _,scores=c.prior_state();ck(scores==[6,16,8,8],"REGRESSION")
    # Revoke the entire imported packet; original model hash and empty fact store are unchanged.
    revoked={**m,"facts":{}};ck(all(answer(revoked,x["id"],x["field"])=="HOLD" for x in rows),"ROLLBACK")
    report={"schema":"r84d.result.v1","packet_sha256":p["sha256"],"memory_sha256":g.digest(m),"source_records":len(p["records"]),"source_holds":p["holds"],"new_before":sum(before[i] for i in known),"new_after":sum(after[i] for i in known),"new_n":len(known),"abstention":sum(after[i] for i in unknown),"abstention_n":len(unknown),"old_scores":scores,"rollback_exact_parent":True,"exam_internal":True,"neural_training":False,"production":False,"status":"PASS_SOURCE_FACT_MEMORY" if all(after) else "HOLD_INCOMPLETE_SOURCE_OR_EXAM"}
    out.mkdir(parents=True,exist_ok=False);g.write_json(out/"report.json",report);log("R84D_AUDIT",report)
if __name__=="__main__":
    mode=sys.argv[1];args=[Path(x) for x in sys.argv[2:]]
    {"harvest":harvest,"assimilate":assimilate,"audit":audit}[mode](*args)
