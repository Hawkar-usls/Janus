"""R5.1: bounded document-disjoint registry semantic transfer experiment.
Original implementation; no execution of document text, external code or effects.
No claims of external blindness or neural student improvement.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import project_context_r4 as r4

MANIFEST_BLOB = "714b0c2eb23372335ae98e4bcecba025775112c0"
COMMIT = "e85c19c31bdad729f06961d64de36d8f3a620483"
LABELS = {"GOAL","OPEN","NEXT","HISTORY","EVIDENCE","BOUNDARY","STATUS","AUTHORITY","UNKNOWN"}

def words(x):
    return re.findall(r"[a-z][a-z0-9]*",x.lower())[:160]

def deref(doc, path):
    if type(path) is not list or not 1 <= len(path) <= 8:
        raise ValueError("BAD_POINTER")
    for key in path:
        if type(key) is not str or type(doc) is not dict or key not in doc:
            raise ValueError("UNKNOWN_POINTER")
        doc = doc[key]
    return doc

def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,
                           separators=(",",":"),allow_nan=False).encode()).hexdigest()

def feats(path, value, contextual):
    key = path[-1]
    result = {"key:"+x for x in words(key)}
    if contextual:
        result |= {"parent:"+x for x in words(" ".join(path[:-1]))}
        result.add("type:"+type(value).__name__)
        if isinstance(value,str):
            result |= {"value:"+x for x in words(value[:900])}
            if re.fullmatch("[0-9a-f]{40,64}",value):
                result.add("shape:digest")
            if value.startswith("https://"):
                result.add("shape:url")
        elif isinstance(value,dict):
            result |= {"child:"+x for x in words(" ".join(list(value)[:24]))}
        elif isinstance(value,list) and value and isinstance(value[0],dict):
            result |= {"item:"+x for x in words(" ".join(list(value[0])[:24]))}
    return result

def fit(rows, contextual):
    counts = defaultdict(Counter)
    classes = Counter()
    for row in rows:
        label = row["role"]
        classes[label] += 1
        counts[label].update(feats(row["pointer"],row["value"],contextual))
    vocab = set().union(*(set(c) for c in counts.values()))
    return counts,classes,vocab

def predict(model,path,value,contextual):
    counts,classes,vocab = model
    fs = feats(path,value,contextual) & vocab
    if not fs:
        return "UNKNOWN"
    # Bernoulli-like smoothed feature evidence; fixed before exam.
    scores={}
    for label,c in counts.items():
        p=classes[label]/sum(classes.values())
        scores[label]=math.log(p)+sum(math.log((c[f]+1)/(classes[label]+2)) for f in fs)
    rank=sorted(scores,key=lambda label:(-scores[label],label))
    if len(rank)<2 or scores[rank[0]]-scores[rank[1]]<0.5:
        return "UNKNOWN"
    return rank[0]

def rule(path,value):
    t=set(words(path[-1]))
    if t & {"execute","permission","override"}:
        return "UNKNOWN"
    if t & {"boundary","boundaries","limits","ceiling","policy","policies","control","rule","firewalls"}:
        return "BOUNDARY"
    if t & {"next","recommendation"}:
        return "NEXT"
    if t & {"gate","gates","unresolved","pending","custodian","obligations"}:
        return "OPEN"
    if t & {"history","historical","parents","preceding","repeat"}:
        return "HISTORY"
    if t & {"authority"}:
        return "AUTHORITY"
    if t & {"goal","purpose","aim","objective","question","summary"}:
        return "GOAL"
    if t & {"status","maturity","verdict"}:
        return "STATUS"
    if t & {"source","sources","metrics","sha","commit","receipt","audit","ci","evidence","seal","digest"}:
        return "EVIDENCE"
    return "UNKNOWN"

def load(root, spec):
    raw=r4.verified_bytes(root,spec["path"],spec["blob"])
    doc=r4.strict_object(raw)
    schema=doc.get("schema",doc.get("schema_id"))
    if type(schema) is not str:
        raise ValueError("SCHEMA_MISSING")
    rows=[]
    for f in spec["fields"]:
        if f["role"] not in LABELS:
            raise ValueError("LABEL_SCOPE")
        v=deref(doc,f["pointer"])
        rows.append({"schema":schema,"path":spec["path"],"blob":spec["blob"],
                     "pointer":f["pointer"],"value":v,"role":f["role"],
                     "value_sha256":digest(v)})
    return schema,rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    a=p.parse_args()
    raw=a.manifest.read_bytes()
    if r4.blob(raw)!=MANIFEST_BLOB:
        raise ValueError("FROZEN_SPLIT_DRIFT")
    spec=r4.strict_object(raw)
    if spec["commit"]!=COMMIT or len(spec["documents"])!=9:
        raise ValueError("SOURCE_SCOPE_DRIFT")
    paths=[x["path"] for x in spec["documents"]]
    blobs=[x["blob"] for x in spec["documents"]]
    if len(set(paths))!=9 or len(set(blobs))!=9:
        raise ValueError("DUPLICATE_DOCUMENT")
    train=[]
    train_schemas=set()
    for d in spec["documents"]:
        if d["partition"]=="TRAIN":
            s,rows=load(a.registry,d)
            train_schemas.add(s)
            train.extend(rows)
    assert len(train)==42 and len(train_schemas)==6
    # Models fit before examination documents are read.
    models={k:fit(train,c) for k,c in (("key_only",False),("contextual",True))}
    frozen=digest({k:{"counts":{label:dict(c) for label,c in m[0].items()},
                       "classes":dict(m[1]),"vocab":sorted(m[2])}
                   for k,m in models.items()})
    exam=[]; exam_schemas=set()
    for d in spec["documents"]:
        if d["partition"]=="EXAM":
            s,rows=load(a.registry,d)
            exam_schemas.add(s)
            exam.extend(rows)
    assert len(exam)==24 and len(exam_schemas)==3 and not train_schemas & exam_schemas
    seen_keys={x["pointer"][-1] for x in train}
    predictions=[]
    for row in exam:
        path,val=row["pointer"],row["value"]
        # Candidate field interpretations only; exact pointer and source provenance
        # are separate from inferred roles. No predicted role grants authority.
        predictions.append({"document":row["path"],"blob":row["blob"],
            "pointer":path,"value_sha256":row["value_sha256"],"expected":row["role"],
            "key_seen":path[-1] in seen_keys,
            "key_only":predict(models["key_only"],path,val,False),
            "contextual":predict(models["contextual"],path,val,True),
            "rules":rule(path,val),"action_authority":False})
    scores={}
    for k in ("key_only","contextual","rules"):
        scores[k]={"correct":sum(x[k]==x["expected"] for x in predictions),
                   "unknown_false_positive":sum(x["expected"]=="UNKNOWN" and x[k]!="UNKNOWN" for x in predictions),
                   "unseen_key_correct":sum(not x["key_seen"] and x[k]==x["expected"] for x in predictions)}
    result={"schema":"janus.registry.r51.real_document_exam_receipt.v1",
            "source_commit":COMMIT,"manifest_blob":MANIFEST_BLOB,
            "train_documents":6,"train_fields":42,"exam_documents":3,"exam_fields":24,
            "unseen_keys":sum(not x["key_seen"] for x in predictions),
            "model_digest":frozen,"scores":scores,"predictions":predictions,
            "schema_disjoint":True,"same_owner_developer_seen_exam":True,
            "external_blind":False,"actual_neural_weights_trained":False,
            "live_authority":False,"production_promotion":False,
            "status":"RESEARCH_ONLY__NO_AUTOMATIC_ROUTE_PROMOTION"}
    print(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False))

if __name__=="__main__":
    main()
