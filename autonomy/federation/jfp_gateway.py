"""JFP 0.1 pilot: signed, bidirectional, file-transport gateway.
Only one read-only capability; no automatic network dispatch or code execution.
Keys are provisioned out of band by the operator, never taken from a message.
"""
import argparse
import base64
import hashlib
import json
import re
import sqlite3
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path

VERSION = "0.1-pilot"
CAP = "registry.read_fixture.v1"
MAX_PACKET = 16384
TTL = 300
FIELDS = {"version","type","id","sender","recipient","capability","created","expires","reply_to","payload","payload_sha256","signature"}
ID = re.compile(r"[a-zA-Z0-9:_-]{1,100}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
FIXTURE = {"goal":"compare documented capabilities","release_gate":{"status":"HOLD"},"next_allowed":"review source"}

class Reject(ValueError):
    pass

def require(condition, reason):
    if not condition:
        raise Reject(reason)

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False).encode("utf-8")

def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()

def read_json(path):
    def unique(pairs):
        out={}
        for k,v in pairs:
            require(k not in out,"DUPLICATE_KEY")
            out[k]=v
        return out
    def constant(_):
        raise Reject("NONFINITE")
    with Path(path).open("rb") as f:
        raw=f.read(MAX_PACKET+1)
    require(0<len(raw)<=MAX_PACKET,"PACKET_SIZE")
    try:
        return json.loads(raw,object_pairs_hook=unique,parse_constant=constant)
    except (ValueError,RecursionError,UnicodeError) as exc:
        raise Reject("INVALID_JSON") from exc

def write_json(path,obj):
    data=canonical(obj)
    require(len(data)<=MAX_PACKET,"PACKET_SIZE")
    with Path(path).open("xb") as f:
        f.write(data+b"\n")

def crypto(key,data,signature=None):
    # OpenSSL implements Ed25519. No shell, fixed executable and arguments.
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)
        (p/"message").write_bytes(data)
        if signature is None:
            cmd=["openssl","pkeyutl","-sign","-rawin","-inkey",str(key),"-in",str(p/"message")]
        else:
            (p/"signature").write_bytes(signature)
            cmd=["openssl","pkeyutl","-verify","-pubin","-rawin","-inkey",str(key),"-in",str(p/"message"),"-sigfile",str(p/"signature")]
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5,check=False)
        require(r.returncode==0,"SIGNATURE_FAILURE")
        return r.stdout

def body(packet):
    return {k:v for k,v in packet.items() if k!="signature"}

def make(kind,sender,recipient,payload,key,now=None,reply_to=None):
    now=int(time.time()) if now is None else now
    p={"version":VERSION,"type":kind,"id":uuid.uuid4().hex,"sender":sender,
       "recipient":recipient,"capability":CAP,"created":now,"expires":now+TTL,
       "reply_to":reply_to,"payload":payload,"payload_sha256":digest(payload)}
    p["signature"]=base64.b64encode(crypto(key,canonical(p))).decode("ascii")
    return p

def verify(p,local,peers,kind,now=None):
    now=int(time.time()) if now is None else now
    require(type(p) is dict and set(p)==FIELDS and len(canonical(p))<=MAX_PACKET,"ENVELOPE")
    require(p["version"]==VERSION and p["type"]==kind and p["capability"]==CAP,"PROTOCOL_OR_CAPABILITY")
    require(all(type(p[k]) is str and ID.fullmatch(p[k]) for k in ("id","sender","recipient")),"IDENTITY")
    require(p["recipient"]==local and p["sender"]!=local and p["sender"] in peers,"UNAUTHORIZED_PEER")
    require(type(p["created"]) is int and type(p["expires"]) is int and
            p["created"]<=now+10 and now<p["expires"] and 0<p["expires"]-p["created"]<=TTL,"EXPIRED_OR_CLOCK")
    require(p["payload_sha256"]==digest(p["payload"]),"PAYLOAD_HASH")
    require(type(p["signature"]) is str and len(p["signature"])==88,"SIGNATURE_FORMAT")
    try:
        sig=base64.b64decode(p["signature"],validate=True)
    except ValueError as e:
        raise Reject("SIGNATURE_FORMAT") from e
    require(len(sig)==64,"SIGNATURE_FORMAT")
    crypto(peers[p["sender"]],canonical(body(p)),sig)
    return p

def exact_fixture(f):
    require(type(f) is dict and set(f)=={"goal","release_gate","next_allowed"} and
            type(f["release_gate"]) is dict and set(f["release_gate"])=={"status"},"FIXTURE_SCHEMA")
    values=[f["goal"],f["release_gate"]["status"],f["next_allowed"]]
    require(all(type(x) is str and 0<len(x)<=256 for x in values),"FIXTURE_LIMIT")
    return {"goal":values[0],"gate":values[1],"next":values[2]}

def ledger(path,node,message_id,packet_hash,now):
    # Separate local state, no accepted-request replay, bounded storage.
    with sqlite3.connect(path,timeout=5) as db:
        db.execute("CREATE TABLE IF NOT EXISTS seen (node TEXT, id TEXT, digest TEXT, expires INTEGER, PRIMARY KEY(node,id))")
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM seen WHERE expires < ?",(now,))
        require(db.execute("SELECT count(*) FROM seen").fetchone()[0]<10000,"LEDGER_BUDGET")
        try:
            db.execute("INSERT INTO seen VALUES (?,?,?,?)",(node,message_id,packet_hash,now+TTL+20))
        except sqlite3.IntegrityError as exc:
            raise Reject("REPLAY") from exc

def request(local,peer,key,fixture,now=None):
    exact_fixture(fixture)
    return make("TASK_REQUEST",local,peer,{"fixture":fixture},key,now)

def receive(req,local,peers,key,db,now=None):
    now=int(time.time()) if now is None else now
    verify(req,local,peers,"TASK_REQUEST",now)
    require(req["reply_to"] is None and type(req["payload"]) is dict and set(req["payload"])=={"fixture"},"REQUEST_SCHEMA")
    answer=exact_fixture(req["payload"]["fixture"])
    ledger(db,local,req["id"],digest(req),now)
    return make("TASK_RESULT",local,req["sender"],{"request_sha256":digest(req),"answer":answer},key,now,req["id"])

def check(result,req,local,peers,db,now=None):
    now=int(time.time()) if now is None else now
    verify(result,local,peers,"TASK_RESULT",now)
    require(req["sender"]==local and result["sender"]==req["recipient"] and result["reply_to"]==req["id"],"REPLY_BINDING")
    require(now<req["expires"],"REQUEST_EXPIRED")
    require(result["payload"]=={"request_sha256":digest(req),"answer":exact_fixture(req["payload"]["fixture"])},"RESULT_ORACLE")
    ledger(db,local,result["id"],digest(result),now)
    return {"status":"VERIFIED_RECORDED_FIXTURE_ONLY","request_id":req["id"],"request_sha256":digest(req),"result_sha256":digest(result),"sender":result["sender"],"recipient":local,"action_authority":False}

def keys(root,name):
    priv=root/(name+".pem")
    pub=root/(name+".pub.pem")
    subprocess.run(["openssl","genpkey","-algorithm","ED25519","-out",str(priv)],check=True,capture_output=True,timeout=5)
    priv.chmod(0o600)
    subprocess.run(["openssl","pkey","-in",str(priv),"-pubout","-out",str(pub)],check=True,capture_output=True,timeout=5)
    return priv,pub

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.ak,self.ap=keys(self.root,"a")
        self.bk,self.bp=keys(self.root,"b")
        self.a={"b":self.bp};self.b={"a":self.ap}
        self.now=10000
        self.req=request("a","b",self.ak,FIXTURE,self.now)
    def receive(self,q=None,now=None):
        return receive(q or self.req,"b",self.b,self.bk,self.root/"b.db",self.now if now is None else now)
    def check(self,r,q=None):
        return check(r,q or self.req,"a",self.a,self.root/"a.db",self.now)
    def test_both_directions(self):
        self.assertEqual(self.check(self.receive())["status"],"VERIFIED_RECORDED_FIXTURE_ONLY")
        q=request("b","a",self.bk,FIXTURE,self.now)
        r=receive(q,"a",self.a,self.ak,self.root/"a.db",self.now)
        self.assertEqual(check(r,q,"b",self.b,self.root/"b.db",self.now)["sender"],"a")
    def test_unknown_peer(self):
        with self.assertRaises(Reject): receive(self.req,"b",{},self.bk,self.root/"b.db",self.now)
    def test_signature(self):
        with self.assertRaises(Reject): self.receive({**self.req,"signature":base64.b64encode(b"x"*64).decode()})
    def test_tamper(self):
        with self.assertRaises(Reject): self.receive({**self.req,"payload":{"fixture":{"goal":"changed"}}})
    def test_wrong_recipient(self):
        with self.assertRaises(Reject): verify(self.req,"c",self.b,"TASK_REQUEST",self.now)
    def test_expired(self):
        with self.assertRaises(Reject): self.receive(now=self.now+TTL)
    def test_request_replay(self):
        self.receive()
        with self.assertRaises(Reject): self.receive()
    def test_result_replay(self):
        r=self.receive();self.check(r)
        with self.assertRaises(Reject): self.check(r)
    def test_wrong_reply(self):
        r=self.receive()
        q=request("a","b",self.ak,FIXTURE,self.now)
        with self.assertRaises(Reject): self.check(r,q)
    def test_wrong_answer_signed(self):
        r=make("TASK_RESULT","b","a",{"request_sha256":digest(self.req),"answer":{}},self.bk,self.now,self.req["id"])
        with self.assertRaises(Reject): self.check(r)
    def test_bad_capability(self):
        with self.assertRaises(Reject): self.receive({**self.req,"capability":"shell"})
    def test_extra_input(self):
        with self.assertRaises(Reject): request("a","b",self.ak,{**FIXTURE,"shell":"ignored"},self.now)
    def test_duplicate_json(self):
        p=self.root/"bad.json";p.write_text('{"x":1,"x":2}')
        with self.assertRaises(Reject): read_json(p)

def pilot():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); ak,ap=keys(root,"janus");bk,bp=keys(root,"peer")
        a="janus:research";b="janus:peer-fixture";now=int(time.time())
        receipts=[]
        for s,t,sk,tk,sp,tp in ((a,b,ak,bk,ap,bp),(b,a,bk,ak,bp,ap)):
            q=request(s,t,sk,FIXTURE,now);write_json(root/(s.split(":")[1]+".request.json"),q)
            q=read_json(root/(s.split(":")[1]+".request.json"))
            r=receive(q,t,{s:sp},tk,root/(t.split(":")[1]+".db"),now)
            write_json(root/(t.split(":")[1]+".result.json"),r)
            receipts.append(check(read_json(root/(t.split(":")[1]+".result.json")),q,s,{t:tp},root/(s.split(":")[1]+".db"),now))
        return {"schema":"janus.jfp.pilot_receipt.v1","status":"SIGNED_BIDIRECTIONAL_LOCAL_FILE_PILOT","receipts":receipts,
                "external_peers":0,"independent_operators":0,"network_transport":False,"production_authority":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["test","pilot","request","receive","verify"])
    p.add_argument("--local");p.add_argument("--peer")
    p.add_argument("--key",type=Path);p.add_argument("--peer-public-key",type=Path)
    p.add_argument("--fixture",type=Path);p.add_argument("--request",type=Path)
    p.add_argument("--result",type=Path);p.add_argument("--db",type=Path);p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.mode=="test":
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if a.mode=="pilot":
        print(json.dumps(pilot(),sort_keys=True));return 0
    require(a.local and a.peer and a.local!=a.peer and ID.fullmatch(a.local) and ID.fullmatch(a.peer) and a.out,"LOCAL_CONFIG")
    if a.mode=="request":
        x=request(a.local,a.peer,a.key,read_json(a.fixture))
    elif a.mode=="receive":
        x=receive(read_json(a.request),a.local,{a.peer:a.peer_public_key},a.key,a.db)
    else:
        x=check(read_json(a.result),read_json(a.request),a.local,{a.peer:a.peer_public_key},a.db)
    write_json(a.out,x)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
