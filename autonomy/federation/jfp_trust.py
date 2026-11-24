"""R7B.2 local trust and durable admission. No discovery-to-trust promotion.
Policy digest must be supplied by a trusted local operator, not a packet.
SQLite must reside on trusted durable storage; runner persistence is not implemented.
"""
import argparse
import base64
import hashlib
import json
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
import jfp_gateway as g

SCHEMA="janus.jfp.trust.v1"
PREFIX=bytes.fromhex("302a300506032b6570032100")
MAX_NODES=32

def fingerprint(pem):
    g.require(type(pem) is str and len(pem)<200,"KEY_FORMAT")
    lines=pem.splitlines()
    g.require(len(lines)==3 and lines[0]=="-----BEGIN PUBLIC KEY-----" and lines[2]=="-----END PUBLIC KEY-----","KEY_FORMAT")
    try:
        der=base64.b64decode(lines[1],validate=True)
    except ValueError as e:
        raise g.Reject("KEY_FORMAT") from e
    g.require(len(der)==44 and der.startswith(PREFIX),"KEY_ALGORITHM")
    return hashlib.sha256(der).hexdigest()

def policy_check(p,expected,local):
    g.require(type(p) is dict and set(p)=={"schema","local","epoch","nodes"},"POLICY_SCHEMA")
    g.require(g.digest(p)==expected,"POLICY_PIN")
    g.require(p["schema"]==SCHEMA and p["local"]==local and type(local) is str and g.ID.fullmatch(local),"POLICY_LOCAL")
    g.require(type(p["epoch"]) is int and 1<=p["epoch"]<=2**31-1,"EPOCH")
    g.require(type(p["nodes"]) is list and 1<=len(p["nodes"])<=MAX_NODES,"NODES")
    rows={}
    seen=set()
    for r in p["nodes"]:
        g.require(type(r) is dict and set(r)=={"id","public_key","fingerprint","state","not_before","not_after","capabilities"},"NODE_SCHEMA")
        n=r["id"]
        g.require(type(n) is str and g.ID.fullmatch(n) and n not in rows,"NODE_ID")
        fp=fingerprint(r["public_key"])
        g.require(fp==r["fingerprint"] and fp not in seen,"KEY_PIN")
        g.require(r["state"] in ("ACTIVE","REVOKED") and r["capabilities"]==[g.CAP],"CONSENT_SCOPE")
        g.require(type(r["not_before"]) is int and type(r["not_after"]) is int and 0<=r["not_before"]<r["not_after"]<=2**53-1,"KEY_VALIDITY")
        rows[n]=r
        seen.add(fp)
    g.require(local in rows,"LOCAL_MISSING")
    return rows

def active(rows,n,now):
    g.require(type(n) is str and n in rows,"UNADMITTED_NODE")
    r=rows[n]
    g.require(r["state"]=="ACTIVE" and r["not_before"]<=now<r["not_after"],"KEY_INACTIVE")
    return r

def key_path(row,root,name):
    p=root/(name+".pem")
    p.write_text(row["public_key"],encoding="ascii")
    return p

def own_key(row,key):
    p=subprocess.run(["openssl","pkey","-in",str(key),"-passin","pass:","-pubout","-outform","DER"],capture_output=True,timeout=5)
    g.require(p.returncode==0 and len(p.stdout)==44 and p.stdout.startswith(PREFIX) and hashlib.sha256(p.stdout).hexdigest()==row["fingerprint"],"LOCAL_KEY_MISMATCH")

def schema(db):
    db.execute("CREATE TABLE IF NOT EXISTS trust_head(local TEXT PRIMARY KEY,epoch INTEGER NOT NULL,digest TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS admission(local TEXT,token TEXT,expires INTEGER,PRIMARY KEY(local,token))")

def install(dbpath,p,pin,local):
    rows=policy_check(p,pin,local)
    with sqlite3.connect(dbpath,timeout=5) as db:
        schema(db)
        db.execute("BEGIN IMMEDIATE")
        prior=db.execute("SELECT epoch,digest FROM trust_head WHERE local=?",(local,)).fetchone()
        if prior:
            g.require(p["epoch"]>=prior[0],"POLICY_ROLLBACK")
            g.require(p["epoch"]!=prior[0] or pin==prior[1],"POLICY_EQUIVOCATION")
        db.execute("INSERT INTO trust_head VALUES (?,?,?) ON CONFLICT(local) DO UPDATE SET epoch=excluded.epoch,digest=excluded.digest",(local,p["epoch"],pin))
    return rows

def record(dbpath,p,pin,local,token,expiry,now):
    with sqlite3.connect(dbpath,timeout=5) as db:
        db.execute("BEGIN IMMEDIATE")
        g.require(db.execute("SELECT epoch,digest FROM trust_head WHERE local=?",(local,)).fetchone()==(p["epoch"],pin),"POLICY_CHANGED")
        db.execute("DELETE FROM admission WHERE expires<=?",(now,))
        g.require(db.execute("SELECT count(*) FROM admission").fetchone()[0]<10000,"LEDGER_BUDGET")
        try:
            db.execute("INSERT INTO admission VALUES (?,?,?)",(local,token,expiry))
        except sqlite3.IntegrityError as e:
            raise g.Reject("REPLAY") from e

def operate(mode,local,peer,p,pin,db,key=None,q=None,r=None,now=None):
    now=int(time.time()) if now is None else now
    rows=install(db,p,pin,local)
    g.require(local!=peer,"SELF_PEER")
    me=active(rows,local,now)
    other=active(rows,peer,now)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        mp=key_path(me,root,"local")
        pp=key_path(other,root,"peer")
        if mode=="request":
            own_key(me,key)
            q=g.request(local,peer,key,g.FIXTURE,now)
            record(db,p,pin,local,"OUT:"+q["id"],q["expires"]+20,now)
            return q
        if mode=="receive":
            own_key(me,key)
            g.verify(q,local,{peer:pp},"TASK_REQUEST",now)
            g.require(q["reply_to"] is None and q["payload"]=={"fixture":g.FIXTURE},"FIXTURE_SCOPE")
            answer=g.exact_fixture(q["payload"]["fixture"])
            out=g.make("TASK_RESULT",local,peer,{"request_sha256":g.digest(q),"answer":answer},key,now,q["id"])
            record(db,p,pin,local,"IN:"+peer+":"+q["id"],q["expires"]+20,now)
            return out
        g.require(mode=="verify","MODE")
        g.verify(q,peer,{local:mp},"TASK_REQUEST",now)
        g.verify(r,local,{peer:pp},"TASK_RESULT",now)
        g.require(q["reply_to"] is None and q["payload"]=={"fixture":g.FIXTURE},"FIXTURE_SCOPE")
        g.require(r["reply_to"]==q["id"] and r["payload"]=={"request_sha256":g.digest(q),"answer":g.exact_fixture(g.FIXTURE)},"RESULT_BINDING")
        # A differently signed result ID must not bypass one-result-per-request.
        record(db,p,pin,local,"RESULT:"+q["id"],q["expires"]+20,now)
        return {"status":"VERIFIED_TRUST_BOUND_FIXTURE_ONLY","local":local,"peer":peer,"policy_epoch":p["epoch"],"policy_sha256":pin,"request_sha256":g.digest(q),"result_sha256":g.digest(r),"action_authority":False}

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.ak,self.ap=g.keys(self.root,"a")
        self.bk,self.bp=g.keys(self.root,"b")
        self.now=10000
        def row(n,k):
            pem=k.read_text()
            return {"id":n,"public_key":pem,"fingerprint":fingerprint(pem),"state":"ACTIVE","not_before":9000,"not_after":11000,"capabilities":[g.CAP]}
        self.nodes=[row("a",self.ap),row("b",self.bp)]
        self.a={"schema":SCHEMA,"local":"a","epoch":1,"nodes":self.nodes}
        self.b={**self.a,"local":"b"}
    def call(self,mode,local="a",p=None,pin=None,key=None,q=None,r=None,now=None,peer=None):
        p=p if p is not None else (self.a if local=="a" else self.b)
        return operate(mode,local,peer or ("b" if local=="a" else "a"),p,pin if pin is not None else g.digest(p),self.root/(local+".db"),key if key is not None else (self.ak if local=="a" else self.bk),q,r,self.now if now is None else now)
    def exchange(self,local="a"):
        peer="b" if local=="a" else "a"
        q=self.call("request",local)
        r=self.call("receive",peer,q=q)
        return q,r
    def changed(self,local="a",epoch=2,**kw):
        p=self.a if local=="a" else self.b
        return {**p,"epoch":epoch,"nodes":[dict(self.nodes[0]),{**self.nodes[1],**kw}]}
    def test_both_directions(self):
        for local in ("a","b"):
            q,r=self.exchange(local)
            self.assertEqual(self.call("verify",local,q=q,r=r)["status"],"VERIFIED_TRUST_BOUND_FIXTURE_ONLY")
    def test_wrong_pin(self):
        with self.assertRaises(g.Reject):self.call("request",pin="0"*64)
    def test_unknown_peer(self):
        with self.assertRaises(g.Reject):self.call("request",peer="unknown")
    def test_wrong_private_key(self):
        with self.assertRaises(g.Reject):self.call("request",key=self.bk)
    def test_revocation_and_rollback(self):
        self.call("request")
        with self.assertRaises(g.Reject):self.call("request",p=self.changed(state="REVOKED"))
        with self.assertRaisesRegex(g.Reject,"POLICY_ROLLBACK"):self.call("request")
    def test_same_epoch_conflict(self):
        self.call("request")
        with self.assertRaisesRegex(g.Reject,"POLICY_EQUIVOCATION"):self.call("request",p=self.changed(epoch=1,not_after=12000))
    def test_expired_peer(self):
        with self.assertRaisesRegex(g.Reject,"KEY_INACTIVE"):self.call("request",p=self.changed(not_after=9999))
    def test_not_yet_active(self):
        with self.assertRaisesRegex(g.Reject,"KEY_INACTIVE"):self.call("request",p=self.changed(not_before=10001))
    def test_duplicate_key(self):
        with self.assertRaises(g.Reject):self.call("request",p=self.changed(public_key=self.nodes[0]["public_key"],fingerprint=self.nodes[0]["fingerprint"]))
    def test_scope_escalation(self):
        with self.assertRaises(g.Reject):self.call("request",p=self.changed(capabilities=[g.CAP,"shell"]))
    def test_request_replay_across_calls(self):
        q=self.call("request")
        self.call("receive","b",q=q)
        with self.assertRaisesRegex(g.Reject,"REPLAY"):self.call("receive","b",q=q)
    def test_second_result_new_id(self):
        q,r=self.exchange()
        self.call("verify",q=q,r=r)
        r2=g.make("TASK_RESULT","b","a",r["payload"],self.bk,self.now,q["id"])
        with self.assertRaisesRegex(g.Reject,"REPLAY"):self.call("verify",q=q,r=r2)
    def test_forged_own_request(self):
        q=g.request("a","b",self.bk,g.FIXTURE,self.now)
        r=g.make("TASK_RESULT","b","a",{"request_sha256":g.digest(q),"answer":g.exact_fixture(g.FIXTURE)},self.bk,self.now,q["id"])
        with self.assertRaises(g.Reject):self.call("verify",q=q,r=r)
    def test_altered_fixture(self):
        q=g.request("a","b",self.ak,{**g.FIXTURE,"goal":"different"},self.now)
        with self.assertRaisesRegex(g.Reject,"FIXTURE_SCOPE"):self.call("receive","b",q=q)
    def test_key_rotation_rejects_old_signature(self):
        q=self.call("request")
        nk,np=g.keys(self.root,"new")
        pem=np.read_text()
        p=self.changed("b",public_key=pem,fingerprint=fingerprint(pem))
        with self.assertRaises(g.Reject):self.call("receive","b",p=p,key=self.bk,q=q)
        # Old sender is revoked by replacing its public key in receiver policy.
        rows=[{**self.nodes[0],"public_key":pem,"fingerprint":fingerprint(pem)},self.nodes[1]]
        p={**self.b,"epoch":3,"nodes":rows}
        with self.assertRaisesRegex(g.Reject,"SIGNATURE_FAILURE"):self.call("receive","b",p=p,q=q)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("mode",choices=["test","request","receive","verify"])
    for name in ("local","peer","pin","policy","db","key","request","result","out"):a.add_argument("--"+name)
    x=a.parse_args()
    if x.mode=="test":
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    g.require(all((x.local,x.peer,x.pin,x.policy,x.db,x.out)),"CONFIG")
    p=g.read_json(x.policy)
    out=operate(x.mode,x.local,x.peer,p,x.pin,x.db,x.key,g.read_json(x.request) if x.request else None,g.read_json(x.result) if x.result else None)
    g.write_json(x.out,out)
    return 0
if __name__=="__main__":raise SystemExit(main())
