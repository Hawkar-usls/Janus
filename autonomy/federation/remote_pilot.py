"""R7B.1 public packet relay across two owned repos. No secrets or write token.
Ephemeral per-stage public keys are bound to CI logs by the operator, not
persistent peer identities. Only the fixed public fixture may be exchanged.
"""
import json, os, re, sys, tempfile, urllib.request, time
from pathlib import Path
import jfp_gateway as g

NODES={"Hawkar-usls/Janus":"janus:r7b1","Hawkar-usls/janus-meta-registry":"meta:r7b1"}
DIR="data/jfp-remote-pilot"
FIXTURE=g.FIXTURE

def need(x,msg):
    if not x: raise ValueError(msg)

def read_remote(repo,ref,name):
    need(repo in NODES and re.fullmatch("[0-9a-f]{40}",ref) and name in ("REQUEST","RESPONSE"),"REMOTE_SCOPE")
    url=f"https://raw.githubusercontent.com/{repo}/{ref}/{DIR}/{name}.json"
    with urllib.request.urlopen(url,timeout=15) as r:
        need(r.geturl()==url and r.status==200,"REMOTE_URL")
        raw=r.read(16385)
    need(len(raw)<=16384,"REMOTE_SIZE")
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"packet.json";p.write_bytes(raw)
        return g.read_json(p)

def unpack(b,repo,kind,root):
    need(type(b) is dict and set(b)=={"schema","repo","stage","public_key","packet","run_id","commit"},"BUNDLE")
    need(b["schema"]=="janus.jfp.remote_pilot.bundle.v1" and b["repo"]==repo and b["stage"]==kind,"BUNDLE_SCOPE")
    need(type(b["run_id"]) is str and b["run_id"].isdigit() and re.fullmatch("[0-9a-f]{40}",b["commit"]),"CI_ORIGIN")
    pem=b["public_key"]
    need(type(pem) is str and len(pem)<200 and pem.startswith("-----BEGIN PUBLIC KEY-----\n") and pem.endswith("-----END PUBLIC KEY-----\n"),"PEM")
    p=root/(kind+".pub.pem");p.write_text(pem)
    return b["packet"],p

def emit(repo,stage,pub,packet):
    b={"schema":"janus.jfp.remote_pilot.bundle.v1","repo":repo,"stage":stage,"public_key":pub.read_text(),
       "packet":packet,"run_id":os.environ["GITHUB_RUN_ID"],"commit":os.environ["GITHUB_SHA"]}
    print("JFP_PUBLIC_BUNDLE "+json.dumps(b,sort_keys=True,separators=(",",":"),ensure_ascii=False))

def main():
    repo=os.environ["GITHUB_REPOSITORY"]
    need(repo in NODES,"LOCAL_SCOPE")
    peer=next(r for r in NODES if r!=repo)
    cfg=g.read_json(Path(DIR)/"CONFIG.json")
    need(type(cfg) is dict and cfg.get("stage") in ("REQUEST","RESPONSE","VERIFY"),"CONFIG")
    stage=cfg["stage"]
    need(set(cfg)==({"stage"} if stage=="REQUEST" else {"stage","peer_ref"}),"CONFIG_FIELDS")
    a,b=NODES[repo],NODES[peer]
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        if stage=="REQUEST":
            key,pub=g.keys(root,"local")
            emit(repo,stage,pub,g.request(a,b,key,FIXTURE))
        elif stage=="RESPONSE":
            remote=read_remote(peer,cfg["peer_ref"],"REQUEST")
            q,pub_peer=unpack(remote,peer,"REQUEST",root)
            need(q.get("payload")=={"fixture":FIXTURE},"FIXTURE_ONLY")
            key,pub=g.keys(root,"local")
            r=g.receive(q,a,{b:pub_peer},key,root/"seen.db")
            emit(repo,stage,pub,r)
            print("JFP_REMOTE_REQUEST_VERIFIED "+json.dumps({"sender_repo":peer,"recipient_repo":repo,"peer_ref":cfg["peer_ref"],"request_id":q["id"],"request_sha256":g.digest(q),"result_sha256":g.digest(r)},sort_keys=True))
        else:
            own=g.read_json(Path(DIR)/"REQUEST.json")
            q,own_pub=unpack(own,repo,"REQUEST",root)
            # Verify own original envelope signature too; recipient is peer.
            g.verify(q,b,{a:own_pub},"TASK_REQUEST")
            remote=read_remote(peer,cfg["peer_ref"],"RESPONSE")
            r,pub_peer=unpack(remote,peer,"RESPONSE",root)
            out=g.check(r,q,a,{b:pub_peer},root/"seen.db")
            print("JFP_REMOTE_RESULT_VERIFIED "+json.dumps({**out,"request_repo":repo,"response_repo":peer,"peer_ref":cfg["peer_ref"],"peer_run":remote["run_id"],"local_run":os.environ["GITHUB_RUN_ID"],"ephemeral_keys":True,"operator_bound_public_key":True,"external_independent":False},sort_keys=True))
if __name__=="__main__":main()
