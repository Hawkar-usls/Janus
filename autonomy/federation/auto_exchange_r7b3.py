"""R7B.3 bounded automatic CI exchange. Public bundles only.
Each stage runs on a separate runner. Workflow outputs are the test trust root;
keys are disposable, not persistent peer identity. No network dispatch or writes.
"""
import base64
import json
import os
import tempfile
import time
from pathlib import Path
import jfp_gateway as g
import jfp_trust as t

NODES={"janus:r7b3":"Hawkar-usls/Janus","meta:r7b3":"Hawkar-usls/janus-meta-registry"}
RUN=os.environ.get("GITHUB_RUN_ID","")
ATTEMPT=os.environ.get("GITHUB_RUN_ATTEMPT","")
SCOPE="janus.jfp.auto_ci_bundle.v1"

def encode(x):
    raw=g.canonical(x)
    g.require(len(raw)<=g.MAX_PACKET,"BUNDLE_SIZE")
    return base64.b64encode(raw).decode("ascii")

def decode(s,stage,node):
    g.require(type(s) is str and 0<len(s)<=22000,"BUNDLE_SIZE")
    raw=base64.b64decode(s,validate=True)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"bundle.json";p.write_bytes(raw)
        b=g.read_json(p)
    g.require(type(b) is dict and set(b)=={"schema","stage","node","run","attempt","public_key","packet"},"BUNDLE_SCHEMA")
    g.require(b["schema"]==SCOPE and b["stage"]==stage and b["node"]==node and b["run"]==RUN and b["attempt"]==ATTEMPT,"BUNDLE_PROVENANCE")
    t.fingerprint(b["public_key"])
    return b

def bundle(stage,node,pub,packet):
    return {"schema":SCOPE,"stage":stage,"node":node,"run":RUN,"attempt":ATTEMPT,"public_key":pub.read_text(),"packet":packet}

def row(node,pem,now):
    return {"id":node,"public_key":pem,"fingerprint":t.fingerprint(pem),"state":"ACTIVE","not_before":now-600,"not_after":now+600,"capabilities":[g.CAP]}

def policy(local,other,local_pem,other_pem,now):
    return {"schema":t.SCHEMA,"local":local,"epoch":1,"nodes":[row(local,local_pem,now),row(other,other_pem,now)]}

def output(value):
    with open(os.environ["GITHUB_OUTPUT"],"a",encoding="utf-8") as f:
        f.write("bundle="+encode(value)+"\n")

def main():
    stage=os.environ["JFP_STAGE"];local=os.environ["JFP_NODE"]
    g.require(local in NODES and stage in ("request","receive","verify"),"LOCAL_SCOPE")
    g.require(os.environ["GITHUB_REPOSITORY"]=="Hawkar-usls/Janus" and RUN.isdigit() and ATTEMPT.isdigit(),"CALLER_SCOPE")
    peer=next(n for n in NODES if n!=local)
    now=int(time.time())
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        if stage=="request":
            key,pub=g.keys(root,"local")
            q=g.request(local,peer,key,g.FIXTURE,now)
            output(bundle(stage,local,pub,q))
            print("JFP_AUTO_REQUEST "+json.dumps({"node":local,"peer":peer,"id":q["id"],"sha256":g.digest(q),"run":RUN}))
        elif stage=="receive":
            req=decode(os.environ["JFP_REQUEST"],"request",peer)
            key,pub=g.keys(root,"local")
            p=policy(local,peer,pub.read_text(),req["public_key"],now)
            r=t.operate("receive",local,peer,p,g.digest(p),root/"state.db",key=key,q=req["packet"],now=now)
            output(bundle(stage,local,pub,r))
            print("JFP_AUTO_RESPONSE "+json.dumps({"node":local,"peer":peer,"request_id":req["packet"]["id"],"sha256":g.digest(r),"run":RUN}))
        else:
            req=decode(os.environ["JFP_REQUEST"],"request",local)
            res=decode(os.environ["JFP_RESPONSE"],"receive",peer)
            p=policy(local,peer,req["public_key"],res["public_key"],now)
            receipt=t.operate("verify",local,peer,p,g.digest(p),root/"state.db",q=req["packet"],r=res["packet"],now=now)
            # Verify persistent local replay guard in the same runner, no second result accepted.
            try:
                t.operate("verify",local,peer,p,g.digest(p),root/"state.db",q=req["packet"],r=res["packet"],now=now)
            except g.Reject as e:
                g.require(str(e)=="REPLAY","REPLAY_CONTROL")
            else:
                raise g.Reject("REPLAY_ACCEPTED")
            receipt.update({"run":RUN,"attempt":ATTEMPT,"request_id":req["packet"]["id"],"operator_relay":False,"ephemeral_stage_keys":True,"caller_context":os.environ["GITHUB_REPOSITORY"],"external_independent":False,"replay_rejected":True})
            print("JFP_AUTO_RESULT_VERIFIED "+json.dumps(receipt,sort_keys=True))
            output(receipt)

if __name__=="__main__":main()
