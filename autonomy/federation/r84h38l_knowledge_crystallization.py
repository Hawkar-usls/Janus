"""H38L: source-bound operational curriculum; no paper-content truth inference."""
import sys,copy,json,hashlib,shutil
from pathlib import Path
import torch
import r84h38j_fractal_self_learning as j
import r84h38i_fractal_reconstruction as i
import r84h38h_semantic_kaleidoscope as h
import r84h38e_semantic_invariance as e
import r84h38c_semantic_world_graph as g
ck=j.ck;read=j.read;write=j.write;sha=j.sha
# Finite templates are new synthetic renderings of real operational witnesses.
# The fixed relation model is not retrained; no scientific assertion is generated.
FORMS=[
("A was logged before B",0),("A was registered earlier than B",0),
("A was logged after B",1),("A was registered later than B",1),
("A was not logged before B",2),("A was never registered earlier than B",2),
("A was not logged after B",3),("A was never registered later than B",3),
("A was documented before B",0),("A was documented after B",1),
("A was not documented before B",2),("A was not documented after B",3)]
def preflight(exam,*old):
    ex=read(exam);refs=set()
    for p in old:
        x=read(p)
        for z in x.get("rows",[]):
            n=e.norm(z["text"],z["names"])
            if n:refs.add(n[0])
        for z in x.get("graphs",[]):
            for t in z["sentences"]+[z["claim"]]:
                n=e.norm(t,z["names"])
                if n:refs.add(n[0])
    hits=[]
    for z in ex["rows"]:
        n=e.norm(z["text"],z["names"])
        if n and n[0] in refs:hits.append(n[0])
    for z in ex["graphs"]:
        for t in z["sentences"]+[z["claim"]]:
            n=e.norm(t,z["names"])
            if n and n[0] in refs:hits.append(n[0])
    ck(not hits,"PRIOR_EXAM_OVERLAP:"+str(hits))
    print("R84H38L_PREFLIGHT",json.dumps({"old_exams":len(old),"overlap":len(hits),"exam_sha":sha(exam)}),flush=True)
def load_k(kcode):
    sys.path.insert(0,str(kcode.resolve()))
    import continuous_intake as k
    return k
def source_packets(k,queue,root):
    q=k.read(queue);s,m=k.source(root);ck(q["schema"]==k.SCHEMA and len(q["records"])<=k.CAP,"QUEUE")
    for dg,p in sorted(q["records"].items()):
        ck(k.checked(p)["witness"]["source_sha256"]==dg and k.verify_packet(p),"PACKET")
        w=p["witness"];ck(dg in s["active"] and s["generation"]>=w["generation"],"ACTIVE_SOURCE")
        original=k.read(root/f"generations/{w['generation']:06d}.json")
        ck(original["receipt_sha256"]==w["manifest_sha256"] and dg in original["documents"],"HISTORICAL_SOURCE")
        replay=k.verify_doc(root,dg,s["records"][dg],original,w["cortex_commit"])
        ck(replay==p,"PACKET_REPLAY")
        yield dg,p
def make_tasks(dg,p):
    a=p["derived"]["left"];b=p["derived"]["right"];fs=[{"id":f"s{k}",**z} for k,z in enumerate(p["atoms"])]
    for idx,(form,y) in enumerate(FORMS):
        # Pick a true positive or strict reverse-time negative, render in either order.
        forward=y in (0,1);lhs,rhs=(a,b) if forward else (b,a)
        claim={"left":lhs,"rel":"BEFORE","right":rhs,"positive":forward}
        surface=form.replace("A",rhs if y%2 else lhs).replace("B",lhs if y%2 else rhs)
        n=e.norm(surface,[a,b]);ck(n is not None and e.decode(y,n[1])==claim,"RENDER_ROUNDTRIP")
        q={"schema":"janus.event.graph.v1","facts":fs,"claim":claim}
        proof=g.solve(q);opposite=g.solve({**q,"claim":{**claim,"positive":not claim["positive"]}})
        ck(proof["status"]=="ENTAILED" and opposite["status"]=="CONTRADICTED" and set(proof["proof"])=={"s0","s1"} and set(opposite["proof"])=={"s0","s1"},"GRAPH_PROOF")
        # Check order/polarity independently from the neural model.
        ck((claim["left"],claim["right"],claim["positive"])==((a,b,True) if y<2 else (b,a,False)),"TRUTH")
        yield {"text":n[0],"y":y,"surface":surface,"names":[a,b],"gold":claim,"origin":{"kind":"VERIFIED_OPERATIONAL_SYNTHETIC_RENDERING","source_sha256":dg,"source_receipt":p["receipt_sha256"],"proof":proof,"opposite":opposite,"scientific_truth":False,"training_allowed_in_research":True}}
def train(hdir,gdir,idir,jdir,demi,kcode,queue,source,out):
    g.witness();torch.set_num_threads(2);ck(not out.exists(),"OUT");shutil.copytree(jdir,out)
    hr,bank,ns=i.load_h(hdir);gm=i.load_g(gdir,hr["bank_sha"]);ir,im=j.load_i(idir);r=read(jdir/"receipt.json")
    ck(r["bank_sha"]==hr["bank_sha"] and r["i_receipt_sha"]==sha(idir/"receipt.json"),"J_PARENT")
    parent=j.load_models(jdir,r["active"]);rule=r["rule"];newrows=read(out/"new_rows.json")
    base=bank["train"]+read(idir/"repair_rows.json")+newrows;dev=bank["dev"];known={z["text"] for z in base+dev}
    cortex=e.cortex(demi);root=out/"cortex";state=cortex.load_state(root)
    ck(len(state["active"])==len(newrows)==r["verified_new"],"MEMORY_PARENT")
    k=load_k(kcode);packets=list(source_packets(k,queue,source));ck(bool(packets),"NO_SOURCE_PACKETS")
    all_tasks=[];unique={};source_links={}
    for dg,p in packets:
        for z in make_tasks(dg,p):
            all_tasks.append(z);source_links.setdefault(z["text"],[]).append(dg)
            if z["text"] in unique:ck(unique[z["text"]]["y"]==z["y"],"LABEL_CONFLICT")
            else:unique[z["text"]]=z
    # Reject a tampered negative claim with same two facts.
    sample=all_tasks[0];p=packets[0][1];wrong={**sample["gold"],"positive":not sample["gold"]["positive"]}
    ck(g.solve({"schema":"janus.event.graph.v1","facts":[{"id":f"s{k}",**z} for k,z in enumerate(p["atoms"])],"claim":wrong})["status"]=="CONTRADICTED","NEGATIVE_CONTROL")
    selected=[];counts={};trace=[]
    for t,z in sorted(unique.items()):
        if t in known:counts["duplicate_prior"]=counts.get("duplicate_prior",0)+1;continue
        q=i.trace(t,hr,ns,bank,parent,rule);ck(i.verify(q),"RETURN")
        trace.append({"text":t,"return":q,"source_links":source_links[t],"label":z["y"]})
        if q["children"][0]["value"] is None:counts["relation_hold"]=counts.get("relation_hold",0)+1;continue
        if q["value"]==z["y"]:counts["already_correct"]=counts.get("already_correct",0)+1;continue
        if q["children"][0]["value"]!=0:counts["relation_wrong"]=counts.get("relation_wrong",0)+1;continue
        # Train only component failures, using verified labels, not the model answer.
        origin={**z["origin"],"label":z["y"],"return_sha":q["sha256"]}
        ck(cortex.add_text(root,state,t.encode(),origin),"MEMORY_ADD")
        ck(not cortex.add_text(root,state,t.encode(),origin),"MEMORY_DEDUP")
        selected.append({"text":t,"y":z["y"],"origin":origin})
    before=[];after=[];lost=[];wrong=[];cpu=0;decision="NO_NEW_COMPONENT_REFERRALS";candidate=parent;active=r["active"];manifest=None
    def assess(ms):
        ps=[]
        for z in dev:
            q=i.trace(z["text"],hr,ns,bank,gm)
            if q["value"] is None:q=i.trace(z["text"],hr,ns,bank,ms,rule)
            ps.append(q["value"])
        return [x for x,(p,z) in enumerate(zip(ps,dev)) if p is not None and p==z["y"]],[x for x,(p,z) in enumerate(zip(ps,dev)) if p is not None and p!=z["y"]]
    if selected:
        manifest=cortex.publish(root,state,"H38L_VERIFIED_OPERATIONAL_CURRICULUM");cortex.write_json(root/"STATE.json",cortex.sealed(state))
        candidate,cpu=j.fit(parent,base+selected);before,_=assess(parent);after,wrong=assess(candidate);lost=sorted(set(before)-set(after))
        proposal=j.save_models(candidate,out,"h38l-proposal")
        if not wrong and not lost and len(after)>len(before):decision="SHADOW_ACCEPT";active=proposal
        else:decision="ROLLBACK_KEEP_PARENT"
    newrows+=selected;last=j.save_models(candidate,out,"h38l-last")
    state=cortex.load_state(root);ck(len(state["active"])==len(newrows),"MEMORY_RELOAD")
    for z in newrows:
        dg=hashlib.sha256(z["text"].encode()).hexdigest();ck(cortex.read_object(root,dg,state["records"][dg]["size"]).decode()==z["text"],"MEMORY_READBACK")
    cycle={"stage":"H38L","source_packets":len(packets),"rendered_tasks":len(all_tasks),"unique_normalized_tasks":len(unique),"selected":len(selected),"skip_reasons":counts,"before":before,"after":after,"wrong":wrong,"lost":lost,"decision":decision,"cpu":cpu,"memory_generation":state["generation"],"active":active}
    r={**r,"schema":"h38l.train.v1","cycles":r["cycles"]+[cycle],"verified_new":len(newrows),"active":active,"last":last,"exam_read":False,"production":False}
    write(out/"receipt.json",r);write(out/"new_rows.json",newrows);write(out/"source_tasks.json",all_tasks);write(out/"source_returns.json",trace)
    print("R84H38L_TRAIN",json.dumps({"cycle":cycle,"new_knowledge_rows":len(selected),"scientific_claims_admitted":0,"source_queue_sha":sha(queue),"parent_receipt_sha":sha(jdir/"receipt.json"),"memory_readback":len(newrows),"production":False},sort_keys=True),flush=True)
if __name__=="__main__":
    {"train":train,"preflight":preflight}[sys.argv[1]](*map(Path,sys.argv[2:]))
