"""H38A: disclosed diagnostic, no tuning, no new weights or deployment."""
import sys,re
from pathlib import Path
from safetensors.torch import load_file
import r84h37_polarity_independence as h

# Conservative explicit clause parser. Only canonical asserted clauses;
# unsupported syntax, multiple facts and ambiguous/unknown text abstain.
RULES={
"cached_available":(r"entity is absent from the cache",r"entity is present in the cache"),
"persist_first":(r"entity is acknowledged before persistence",r"entity is persisted before acknowledgement"),
"multi_holder":(r"entity allows only one holder",r"entity allows multiple holders"),
"full_replica":(r"entity is partitioned across nodes",r"entity is fully replicated on every node"),
"newer_recovery":(r"entity recovers to an older version",r"entity recovers to a newer version"),
"all_or_none":(r"entity permits partial commit",r"entity prohibits partial commit"),
}
def parse(t):
    t=h.key(t)
    if not h.scope(t):return None
    found=[h.STATES.index((a,v)) for a,pair in RULES.items() for v,p in enumerate(pair) if re.fullmatch(p,t)]
    return found[0] if len(found)==1 else None

def load_models(folder,cls):
    rec=h.read(folder/"receipt.json");models=[]
    for z in rec["runs"]:
        p=folder/("verify-%d.safetensors"%z["seed"])
        h.ck(h.sha(p)==z["sha"],"WEIGHT_PIN")
        m=cls();m.load_state_dict(load_file(str(p)));models.append(m.eval())
    return rec,models

def outcome(g,y):
    if g is None:return "abstain"
    if y==h.HOLD:return "unknown_false"
    if g==y:return "correct"
    if g==(y^1):return "opposite"
    return "other_property"

def main(own,old,out):
    h.offline();out.mkdir(parents=True,exist_ok=False)
    r,models=load_models(own,h.V37);o,older=load_models(old,h.Verifier)
    tr=h.read(own/"train.json");dev=h.read(own/"dev.json")
    h.ck(h.digest(tr)==r["train_sha"] and h.digest(dev)==r["dev_sha"] and not r["exam_read"],"TRAIN_PIN")
    h.ck(r["train_sha"]=="9924343a64c416b278507357abaa34597757482fc2f10f21af1fee4da96ef167","H37C_TRAIN")
    here=Path(__file__).parent
    exams={33:"02e40d9360d717b0c8f3e5e8f420bdc5003e272b",34:"582ea0e9aa40c545af77964042121f7a95eb0b0a",36:"69823b8919ae31c23fae2f1f150feeaefdd28027",37:h.EXAM}
    report={};details={}
    for n,pin in exams.items():
        path=here/("R84H%d_FROZEN_EXAM.json"%n);h.ck(h.blob(path)==pin,"EXAM_PIN")
        ex=h.read(path);rows=[(z["text"],h.STATES.index((z["fact"]["attribute"],z["fact"]["value"]))) for z in ex["statements"]]+[(t,h.HOLD) for t in ex["unknown"]]
        norm=h.corpus(ex);valid=[t for t,k in norm.items() if k];ts=[norm[t][1] for t in valid]
        p,_=h.prob37(models,ts);q,_=h.probabilities(older,ts);lookup=dict(zip(valid,zip(p,q)))
        def choices(t):
            if t not in lookup:return {"h37":None,"h36":None,"parser":None,"agreement":None}
            a,b=lookup[t];s=norm[t][1]
            x=h.accepted(a,r["rule"],True,s);y=h.accepted(b,o["rule"],False,s)
            x=x[0] if len(x)==1 else None;y=y[0] if len(y)==1 else None;d=parse(s)
            return {"h37":x,"h36":y,"parser":d,"agreement":x if x is not None and x==d else None}
        scores={mode:{k:0 for k in ("correct","opposite","other_property","unknown_false","abstain","pairs_correct","pairs_wrong","pairs_abstain")} for mode in ("h37","h36","parser","agreement")}
        dr=[]
        for t,y in rows:
            g=choices(t);item={"text":t,"target":y,"accepted":g,"outcomes":{k:outcome(v,y) for k,v in g.items()}}
            if t in lookup:
                p37,p36=lookup[t];item["scores_h37"]=p37;item["scores_h36"]=p36
            for k,v in item["outcomes"].items():scores[k][v]+=1
            dr.append(item)
        pair_rows=[]
        for z in ex["pairs"]:
            h.ck(h.exact.relation({"schema":"janus.relation.v1","facts":z["facts"],"claim":z["claim"]})==z["label"],"PAIR_ORACLE")
            a,b=choices(z["premise"]),choices(z["hypothesis"]);got={}
            for mode in scores:
                x,y=a[mode],b[mode]
                if x is None or y is None:answer=None
                else:
                    ax,vx=h.STATES[x];ay,vy=h.STATES[y]
                    answer=h.exact.relation({"schema":"janus.relation.v1","facts":[{"subject":norm[z["premise"]][0],"attribute":ax,"value":vx}],"claim":{"subject":norm[z["hypothesis"]][0],"attribute":ay,"value":vy}})
                got[mode]=answer
                scores[mode]["pairs_abstain" if answer is None else "pairs_correct" if answer==z["label"] else "pairs_wrong"]+=1
            pair_rows.append({"premise":z["premise"],"hypothesis":z["hypothesis"],"target":z["label"],"answers":got})
        report[str(n)]={"facts_n":len(ex["statements"]),"unknown_n":len(ex["unknown"]),"pairs_n":len(ex["pairs"]),"scores":scores}
        details[str(n)]={"rows":dr,"pairs":pair_rows}
    h.ck(report["37"]["scores"]["h37"]["correct"]==7 and report["37"]["scores"]["h37"]["opposite"]==3 and report["37"]["scores"]["h37"]["unknown_false"]==0,"H37C_REPLAY")
    # Tiny self-check proves only the literal grammar, not general semantics.
    for a,pair in RULES.items():
        for v,t in enumerate(pair):h.ck(parse(t)==h.STATES.index((a,v)),"PARSER_CONTROL")
    for t in ("entity may or may not be present in the cache","entity is not present in the cache","entity is present in the cache and is absent from the cache"):
        h.ck(parse(t) is None,"ABSTAIN_CONTROL")
    result={"schema":"r84h38a.diagnostic.v1","status":"DIAGNOSTIC_ONLY","production":False,"training":False,"new_exam":False,"teacher_loaded":False,"scope":"Disclosed H33/H34/H36/H37, canonical literal parser only; no threshold change, no external replication, no promotion","parser_controls":15,"exams":report}
    h.write(out/"diagnostic.json",result);h.write(out/"cases.json",details);h.log("R84H38A_RESULT",result)
    for row in details["37"]["rows"]:
        if row["outcomes"]["h37"] in ("opposite","other_property","unknown_false"):h.log("R84H38A_ERROR",row)
if __name__=="__main__":main(*map(Path,sys.argv[1:]))
