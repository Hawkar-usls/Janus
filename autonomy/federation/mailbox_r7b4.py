"""JFP R7B.4 one-shot owned-repo mailbox experiment.

The two GitHub PR runs exchange public fixture packets through their own research
branches. GitHub repository control is the pilot discovery root; Ed25519 keys are
ephemeral per run, pinned for the whole exchange, not permanent operator keys.
No external peers, general task execution, production writes or background loop.
"""
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
import jfp_gateway as g
import jfp_trust as t

BRANCH="research/jfp-r7b4-mailbox-20261124"
PILOT="JFP-R7B4-20261124-01"
NODES={"Hawkar-usls/Janus":"janus:r7b4","Hawkar-usls/janus-meta-registry":"meta:r7b4"}
DIR=Path("data/jfp-r7b4")
SCHEMA="janus.jfp.mailbox.v1"
MAX_WAIT=105

def run(args,cwd=None,timeout=15):
    r=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
    g.require(r.returncode==0,"COMMAND_FAILED:"+args[0]+":"+args[1])
    return r.stdout.decode().strip()

def context():
    repo=os.environ.get("GITHUB_REPOSITORY","")
    g.require(repo in NODES,"REPO_SCOPE")
    peer=next(x for x in NODES if x!=repo)
    return repo,peer,NODES[repo],NODES[peer]

def bundle(repo,pub,packet,stage):
    return {"schema":SCHEMA,"pilot":PILOT,"repo":repo,"stage":stage,"run_id":os.environ["GITHUB_RUN_ID"],"public_key":pub.read_text(),"packet":packet}

def validate(b,repo,stage):
    g.require(type(b) is dict and set(b)=={"schema","pilot","repo","stage","run_id","public_key","packet"},"BUNDLE_SCHEMA")
    g.require(b["schema"]==SCHEMA and b["pilot"]==PILOT and b["repo"]==repo and b["stage"]==stage,"BUNDLE_SCOPE")
    g.require(type(b["run_id"]) is str and b["run_id"].isdigit(),"RUN_ID")
    t.fingerprint(b["public_key"])
    return b

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    g.write_json(path,obj)

def publish(repo,names,message):
    g.require(run(["git","remote","get-url","origin"])=="https://github.com/"+repo or run(["git","remote","get-url","origin"])=="https://github.com/"+repo+".git","ORIGIN_SCOPE")
    g.require(all(x in ("REQUEST.json","RESPONSE.json","PEER_REQUEST.json","PEER_RESPONSE.json","QUEUE.json") for x in names),"WRITE_SCOPE")
    run(["git","add","--"]+[str(DIR/x) for x in names])
    run(["git","-c","user.name=JFP research pilot","-c","user.email=41898282+github-actions[bot]@users.noreply.github.com","commit","-m",message])
    run(["git","push","origin","HEAD:refs/heads/"+BRANCH],timeout=25)
    return run(["git","rev-parse","HEAD"])

def read_peer(peer,name,stage):
    g.require(peer in NODES and name in ("REQUEST.json","RESPONSE.json"),"READ_SCOPE")
    end=time.monotonic()+MAX_WAIT
    while time.monotonic()<end:
        try:
            # Public read from a clean directory, never use the caller's checkout credentials.
            with tempfile.TemporaryDirectory() as d:
                s=run(["git","ls-remote","--heads","https://github.com/"+peer+".git","refs/heads/"+BRANCH],cwd=d,timeout=10)
            parts=s.split()
            if len(parts)==2 and re.fullmatch("[0-9a-f]{40}",parts[0]) and parts[1]=="refs/heads/"+BRANCH:
                ref=parts[0]
                url="https://raw.githubusercontent.com/"+peer+"/"+ref+"/"+str(DIR/name)
                with urllib.request.urlopen(url,timeout=10) as r:
                    g.require(r.geturl()==url and r.status==200,"REMOTE_URL")
                    raw=r.read(g.MAX_PACKET+1)
                g.require(len(raw)<=g.MAX_PACKET,"BUNDLE_SIZE")
                with tempfile.TemporaryDirectory() as d:
                    p=Path(d)/"p.json";p.write_bytes(raw)
                    return validate(g.read_json(p),peer,stage),ref
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,subprocess.TimeoutExpired) as e:
            if isinstance(e,urllib.error.HTTPError) and e.code not in (404,429,500,502,503,504):raise
        time.sleep(3)
    raise g.Reject("PEER_MAILBOX_TIMEOUT:"+name)

def policy(local,peer,own,other,now):
    def row(n,pem):
        return {"id":n,"public_key":pem,"fingerprint":t.fingerprint(pem),"state":"ACTIVE","not_before":now-600,"not_after":now+600,"capabilities":[g.CAP]}
    return {"schema":t.SCHEMA,"local":local,"epoch":1,"nodes":[row(local,own),row(peer,other)]}

def exchange():
    repo,peer,local,other=context()
    g.require(os.environ.get("JFP_HEAD_BRANCH")==BRANCH and os.environ.get("JFP_HEAD_REPO")==repo,"EVENT_SCOPE")
    g.require(not DIR.exists(),"MAILBOX_ALREADY_EXISTS")
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);key,pub=g.keys(root,"local");db=root/"trust.db"
        q=g.request(local,other,key,g.FIXTURE)
        own=bundle(repo,pub,q,"REQUEST")
        write(DIR/"REQUEST.json",own)
        c1=publish(repo,["REQUEST.json"],"research(JFP): publish one signed public request")
        print("JFP_MAILBOX_REQUEST "+json.dumps({"repo":repo,"request_id":q["id"],"commit":c1}),flush=True)
        incoming,peer_req_ref=read_peer(peer,"REQUEST.json","REQUEST")
        now=int(time.time())
        p=policy(local,other,own["public_key"],incoming["public_key"],now);pin=g.digest(p)
        # One public key per peer for both directions, no key from response accepted.
        response=t.operate("receive",local,other,p,pin,db,key=key,q=incoming["packet"],now=now)
        out=bundle(repo,pub,response,"RESPONSE")
        write(DIR/"PEER_REQUEST.json",incoming)
        write(DIR/"RESPONSE.json",out)
        c2=publish(repo,["PEER_REQUEST.json","RESPONSE.json"],"research(JFP): persist inbound request and signed response")
        print("JFP_MAILBOX_RESPONSE "+json.dumps({"repo":repo,"request_id":incoming["packet"]["id"],"commit":c2,"peer_ref":peer_req_ref}),flush=True)
        returned,peer_res_ref=read_peer(peer,"RESPONSE.json","RESPONSE")
        g.require(returned["run_id"]==incoming["run_id"] and returned["public_key"]==incoming["public_key"],"PEER_KEY_OR_RUN_CHANGED")
        verified_at=int(time.time())
        receipt=t.operate("verify",local,other,p,pin,db,q=q,r=returned["packet"],now=verified_at)
        try:
            t.operate("verify",local,other,p,pin,db,q=q,r=returned["packet"],now=verified_at)
        except g.Reject as e:
            g.require(str(e)=="REPLAY","REPLAY_CONTROL")
        else:
            raise g.Reject("REPLAY_ACCEPTED")
        write(DIR/"PEER_RESPONSE.json",returned)
        queue={"schema":"janus.jfp.mailbox_queue.v1","pilot":PILOT,"repo":repo,"peer":peer,"state":"VERIFIED","verified_at":verified_at,"policy":p,"policy_sha256":pin,"request_sha256":g.digest(q),"incoming_sha256":g.digest(incoming["packet"]),"response_sha256":g.digest(response),"peer_response_sha256":g.digest(returned["packet"]),"peer_request_commit":peer_req_ref,"peer_response_commit":peer_res_ref,"receipt":receipt,"operator_relay":False,"replay_rejected":True,"permanent_signing_key":False,"action_authority":False}
        write(DIR/"QUEUE.json",queue)
        c3=publish(repo,["PEER_RESPONSE.json","QUEUE.json"],"research(JFP): persist verified mailbox checkpoint")
        print("JFP_MAILBOX_VERIFIED "+json.dumps({"repo":repo,"peer":peer,"commit":c3,"receipt":receipt,"operator_relay":False,"permanent_signing_key":False},sort_keys=True),flush=True)
        with open(os.environ["GITHUB_OUTPUT"],"a") as f:f.write("checkpoint="+c3+"\n")

def audit():
    # Archive replay only, never re-admit expired packets or execute a task.
    repo,peer,local,other=context()
    q=validate(g.read_json(DIR/"REQUEST.json"),repo,"REQUEST")
    i=validate(g.read_json(DIR/"PEER_REQUEST.json"),peer,"REQUEST")
    r=validate(g.read_json(DIR/"RESPONSE.json"),repo,"RESPONSE")
    x=validate(g.read_json(DIR/"PEER_RESPONSE.json"),peer,"RESPONSE")
    s=g.read_json(DIR/"QUEUE.json")
    g.require(s["schema"]=="janus.jfp.mailbox_queue.v1" and s["pilot"]==PILOT and s["state"]=="VERIFIED" and s["repo"]==repo and s["peer"]==peer,"QUEUE_SCOPE")
    g.require(q["public_key"]==r["public_key"] and i["public_key"]==x["public_key"] and q["run_id"]==r["run_id"] and i["run_id"]==x["run_id"],"ARCHIVE_IDENTITY")
    g.require(all(s[k]==g.digest(b["packet"]) for k,b in (("request_sha256",q),("incoming_sha256",i),("response_sha256",r),("peer_response_sha256",x))),"QUEUE_HASH")
    p=s["policy"];pin=s["policy_sha256"];at=s["verified_at"]
    g.require(type(at) is int and at<=int(time.time()) and s["action_authority"] is False and s["operator_relay"] is False,"ARCHIVE_SCOPE")
    rows=t.policy_check(p,pin,local)
    g.require(rows[local]["public_key"]==q["public_key"] and rows[other]["public_key"]==i["public_key"],"ARCHIVE_POLICY")
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        # Both signed request/response pairs, same archived keys. No live acceptance.
        receipt=t.operate("verify",local,other,p,pin,root/"archive.db",q=q["packet"],r=x["packet"],now=at)
        inverse={**p,"local":other}
        inverse_receipt=t.operate("verify",other,local,inverse,g.digest(inverse),root/"archive.db",q=i["packet"],r=r["packet"],now=at)
        g.require(receipt==s["receipt"] and inverse_receipt["action_authority"] is False,"ARCHIVE_RECEIPT")
    print("JFP_MAILBOX_ARCHIVE_REPLAY_PASS "+json.dumps({"repo":repo,"peer":peer,"checkpoint":run(["git","rev-parse","HEAD"]),"queue_sha256":g.digest(s),"two_signed_pairs":True,"archive_only":True,"live_admission":False,"permanent_signing_key":False},sort_keys=True))

if __name__=="__main__":
    import sys
    g.require(len(sys.argv)==2 and sys.argv[1] in ("exchange","audit"),"MODE")
    {"exchange":exchange,"audit":audit}[sys.argv[1]]()
