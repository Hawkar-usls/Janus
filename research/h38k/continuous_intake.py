"""Bounded read-only source admission bridge, independent of model promotion.
Consumes real published Memory Cortex state; provenance is not scientific truth.
"""
import argparse,hashlib,json,re,subprocess,tempfile,unittest
from pathlib import Path
from collections import Counter
SRC_SCHEMA="janus.memory_cortex.growth.v1"
SCHEMA="janus.h38k.intake.v1"
PRIMARY="PRIMARY_ABSTRACT_METADATA_NOT_FULL_PAPER_OR_PROOF"
HEX=re.compile(r"[0-9a-f]{64}")
COMMIT=re.compile(r"[0-9a-f]{40}")
ID=re.compile(r"(?:arxiv:[a-zA-Z0-9./_-]{1,80}|zenodo:[0-9]{1,20})")
CAP=4096;BUDGET=8;MAX_DOC=1024*1024
def ck(ok,msg):
    if not ok:raise ValueError(msg)
def canonical(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def seal(o):
    b={k:v for k,v in o.items() if k!="receipt_sha256"};return {**b,"receipt_sha256":digest(canonical(b))}
def checked(o):
    ck(type(o) is dict and seal(o)==o,"RECEIPT_HASH");return o
def read(p,limit=32*MAX_DOC):
    ck(not p.is_symlink() and p.is_file() and p.stat().st_size<=limit,"FILE_LIMIT")
    return checked(json.loads(p.read_text(encoding="utf-8")))
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(".tmp");ck(not p.is_symlink() and not q.is_symlink(),"SYMLINK")
    q.write_text(json.dumps(seal(o),indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8");q.replace(p)
def source(root):
    s=read(root/"STATE.json");p=read(root/"CURRENT.json")
    ck(s["schema"]==p["schema"]==SRC_SCHEMA and type(s["generation"]) is int and s["generation"]>=0,"SOURCE_SCHEMA")
    gen=s["generation"];path=f"generations/{gen:06d}.json";ck(p["generation"]==gen and p["manifest_path"]==path,"POINTER")
    m=read(root/path);a=s["active"];pending=s["pending"];rs=s["records"]
    ck(m["schema"]==SRC_SCHEMA and m["generation"]==gen and m["receipt_sha256"]==p["manifest_sha256"],"MANIFEST")
    ck(len(a)==len(set(a)) and len(pending)==len(set(pending)) and not set(a)&set(pending) and set(a)|set(pending)==set(rs),"MEMBERSHIP")
    ck(m["documents"]==sorted(a) and m["verified_unique_source_bytes"]==sum(rs[x]["size"] for x in a),"MANIFEST_MEMBERSHIP")
    return s,m
def initial():return {"schema":SCHEMA,"records":{},"sources_seen":0,"last_generation":-1,"last_manifest":None}
def verify_doc(root,h,rec,manifest,commit):
    ck(HEX.fullmatch(h) and type(rec["size"]) is int and 0<rec["size"]<=MAX_DOC,"OBJECT_METADATA")
    o=rec["origin"];ck(o.get("kind")==PRIMARY and ID.fullmatch(o.get("primary_id","")) and HEX.fullmatch(o.get("provider_sha256","")) and COMMIT.fullmatch(o.get("beetle_commit","")),"ORIGIN")
    ck(o.get("beetle_repository")=="Hawkar-usls/janus-lapis","ORIGIN_REPOSITORY")
    p=root/"objects"/h[:2]/(h[2:]+".txt")
    ck(not p.is_symlink() and not p.parent.is_symlink() and p.is_file() and p.stat().st_size==rec["size"],"SOURCE_BYTES")
    raw=p.read_bytes();ck(digest(raw)==h,"SOURCE_HASH");raw.decode("utf-8","strict")
    # The facts are about this documented intake pipeline, NOT paper findings.
    # The upstream consumer fetched the primary record, then stored its text;
    # publish admitted the stored object. Those stages are ordered by construction.
    a="fetch_"+h[:16];b="store_"+h[:16];c="publish_"+h[:16]
    atoms=[{"left":a,"rel":"BEFORE","right":b,"positive":True},{"left":b,"rel":"BEFORE","right":c,"positive":True}]
    # A separate simple verifier reconstructs the expected stage path, rejects
    # reversed/negated labels and preserves the source metadata witness.
    witness={"source_sha256":h,"source_bytes":len(raw),"primary_id":o["primary_id"],"provider_sha256":o["provider_sha256"],"beetle_commit":o["beetle_commit"],"cortex_commit":commit,"manifest_sha256":manifest["receipt_sha256"],"generation":manifest["generation"]}
    q={"kind":"VERIFIED_OPERATIONAL_PROVENANCE_NOT_SCIENTIFIC_FACT","witness":witness,"atoms":atoms,"derived":{"left":a,"rel":"BEFORE","right":c,"positive":True},"training_status":"QUARANTINED_PENDING_SEPARATE_SEMANTIC_EXAM","scientific_truth":False,"model_training_allowed":False}
    ck(verify_packet(q),"PACKET_VERIFY")
    return seal(q)
def verify_packet(q):
    try:
        w=q["witness"];h=w["source_sha256"];a="fetch_"+h[:16];b="store_"+h[:16];c="publish_"+h[:16]
        expect=lambda x,y:{"left":x,"rel":"BEFORE","right":y,"positive":True}
        return bool(HEX.fullmatch(h) and q["atoms"]==[expect(a,b),expect(b,c)] and q["derived"]==expect(a,c) and q["scientific_truth"] is False and q["model_training_allowed"] is False and q["training_status"]=="QUARANTINED_PENDING_SEPARATE_SEMANTIC_EXAM")
    except (KeyError,TypeError):return False
def scan(root,statepath,commit,budget=BUDGET):
    ck(COMMIT.fullmatch(commit) and type(budget) is int and 1<=budget<=BUDGET,"BOUND")
    s,m=source(root);old=read(statepath) if statepath.exists() else initial()
    ck(old["schema"]==SCHEMA and len(old["records"])<=CAP,"QUEUE_SCHEMA")
    for k,v in old["records"].items():
        ck(checked(v)["witness"]["source_sha256"]==k and verify_packet(v),"QUEUE_INTEGRITY")
    ck(m["generation"]>=old["last_generation"],"SOURCE_ROLLBACK")
    if m["generation"]==old["last_generation"]:ck(m["receipt_sha256"]==old["last_manifest"],"GENERATION_CONFLICT")
    eligible=[x for x in s["active"] if s["records"][x]["origin"].get("kind")==PRIMARY]
    # deterministic content addressing; exact duplicate documents never restudied.
    unseen=sorted(set(eligible)-set(old["records"]))
    selected=unseen[:min(budget,CAP-len(old["records"]))]
    new={}
    for h in selected:new[h]=verify_doc(root,h,s["records"][h],m,commit)
    rows={**old["records"],**new};state={"schema":SCHEMA,"records":rows,"sources_seen":len(eligible),"last_generation":m["generation"],"last_manifest":m["receipt_sha256"]}
    if seal(state)!=seal(old):write(statepath,state)
    return seal({"schema":"janus.h38k.scan.v1","source_commit":commit,"cortex_generation":m["generation"],"manifest_sha256":m["receipt_sha256"],"active_docs":len(s["active"]),"active_source_bytes":m["verified_unique_source_bytes"],"primary_docs":len(eligible),"already_seen":len(set(eligible)&set(old["records"])),"new_verified_operational_packets":len(new),"total_packets":len(rows),"remaining_unseen":len(unseen)-len(new),"capacity_hold":len(rows)>=CAP,"new_source_sha256":sorted(new),"scientific_claims_admitted":0,"neural_training_admitted":0,"production":False})
class Tests(unittest.TestCase):
    def fixture(self,root):
        h=digest(b"Title\n\nPublic metadata description with sufficient content.\n");raw=b"Title\n\nPublic metadata description with sufficient content.\n";p=root/"objects"/h[:2]/(h[2:]+".txt");p.parent.mkdir(parents=True);p.write_bytes(raw)
        o={"kind":PRIMARY,"primary_id":"arxiv:2611.00001","provider_sha256":"1"*64,"beetle_repository":"Hawkar-usls/janus-lapis","beetle_commit":"2"*40}
        st={"schema":SRC_SCHEMA,"generation":0,"active":[h],"pending":[],"records":{h:{"size":len(raw),"origin":o}}}
        manifest=seal({"schema":SRC_SCHEMA,"generation":0,"documents":[h],"verified_unique_source_bytes":len(raw)})
        write(root/"STATE.json",st);write(root/"generations/000000.json",manifest);write(root/"CURRENT.json",{"schema":SRC_SCHEMA,"generation":0,"manifest_path":"generations/000000.json","manifest_sha256":manifest["receipt_sha256"]})
        return h,p
    def test_restart_and_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/"src";h,p=self.fixture(r);q=Path(td)/"queue.json";a=scan(r,q,"3"*40);raw=q.read_bytes();b=scan(r,q,"3"*40)
            self.assertEqual((a["new_verified_operational_packets"],b["new_verified_operational_packets"],b["already_seen"]),(1,0,1));self.assertEqual(raw,q.read_bytes())
    def test_tampered_source(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/"src";h,p=self.fixture(r);p.write_bytes(b"x"*p.stat().st_size)
            with self.assertRaisesRegex(ValueError,"SOURCE_HASH"):scan(r,Path(td)/"q.json","3"*40)
    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/"src";self.fixture(r);p=r/"CURRENT.json";a=json.loads(p.read_text());a["generation"]=1;p.write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError,"RECEIPT_HASH"):scan(r,Path(td)/"q.json","3"*40)
    def test_reversed_label(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/"src";h,p=self.fixture(r);q=Path(td)/"q.json";scan(r,q,"3"*40);v=read(q)["records"][h];v["derived"]["positive"]=False;self.assertFalse(verify_packet(v))
    def test_resealed_queue_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/"src";h,p=self.fixture(r);q=Path(td)/"q.json";scan(r,q,"3"*40);v=read(q);v["records"][h]["derived"]["positive"]=False;v["records"][h]=seal(v["records"][h]);write(q,v)
            with self.assertRaisesRegex(ValueError,"QUEUE_INTEGRITY"):scan(r,q,"3"*40)
def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path);p.add_argument("--state",type=Path);p.add_argument("--commit");p.add_argument("--report",type=Path);p.add_argument("--test",action="store_true");a=p.parse_args()
    if a.test:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    report=scan(a.source,a.state,a.commit);write(a.report,report);print("R84H38K_SCAN",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":main()
