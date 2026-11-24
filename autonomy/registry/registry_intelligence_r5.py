"""R5 bounded CPU registry literacy experiment; not a production router.
Uses R4 verified source bytes, real Cortex methods on a disposable index,
and a read-only Beetle direction contract. No dispatch, model deployment,
network, or mutation of the durable Cortex or Beetle state.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import project_context_r4 as r4

EXAM = "4325de47d1ba4851de063537e29894f7b947460e"
BEETLE_BLOB = "ff9f381551f8d2d169e004eac5a3a379278689c5"
CORTEX_BLOB = "626b74379e195d2e72d7c5b727198f6a3e297f04"
# Human-reviewed labels for actual fields only. No test data in training.
CURRICULUM = {
    "COSMOS_SETI": [
        (("goal",), "GOAL"), (("open_gates",), "OPEN"),
        (("next_allowed",), "NEXT"), (("preceding",), "HISTORY"),
        (("current", "receipt"), "EVIDENCE"),
        (("current", "receipt_commit"), "EVIDENCE"),
        (("active_experiment", "receipt"), "EVIDENCE"),
        (("active_experiment", "receipt_commit"), "EVIDENCE"),
    ],
    "TRUMP": [
        (("current_frontier", "correct_question"), "GOAL"),
        (("current_frontier", "gate"), "OPEN"),
        (("current_frontier", "preregistration_status"), "OPEN"),
        (("2026_09_14_c023r_cache_fiber_result", "next_allowed_action"), "NEXT"),
        (("historical_v1_1", "verdict"), "HISTORY"),
        (("must_not_repeat_as_new",), "HISTORY"),
        (("latest_composite_authority", "scientific_seal_commit"), "EVIDENCE"),
        (("historical_v1_1", "fail_seal"), "EVIDENCE"),
        (("historical_v1_1", "postmortem_seal"), "EVIDENCE"),
    ],
}
# Separate source-based extraction oracle: strict pointer traversal and value hash.
# This validates exact extraction, not the human-assigned semantic role.
def oracle(doc, path):
    if not isinstance(path, (list, tuple)) or not 1 <= len(path) <= 8:
        raise ValueError("POINTER_SCOPE")
    node = doc
    for key in path:
        if type(key) is not str or type(node) is not dict or key not in node:
            raise ValueError("POINTER_NOT_FOUND")
        node = node[key]
    return node

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                 separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def tokens(key):
    return set(re.findall(r"[a-z]+", key.casefold())) - {"as", "the", "of", "and"}

def train(rows):
    # Small trainable token classifier (multinomial NB with learned counts).
    counts = defaultdict(Counter)
    classes = Counter()
    exact = defaultdict(set)
    for row in rows:
        key = row["path"][-1]
        label = row["role"]
        counts[label].update(tokens(key))
        classes[label] += 1
        exact[key].add(label)
    vocab = set().union(*(set(c) for c in counts.values()))
    return counts, classes, vocab, exact

def predict(model, key):
    counts, classes, vocab, exact = model
    if key in exact and len(exact[key]) == 1:
        return next(iter(exact[key]))
    ts = tokens(key) & vocab
    if not ts:
        return "UNKNOWN"
    scores = {}
    for label, c in counts.items():
        scores[label] = math.log(classes[label] / sum(classes.values())) + sum(
            math.log((c[t] + 1) / (sum(c.values()) + len(vocab))) for t in ts)
    ordered = sorted(scores, key=scores.get, reverse=True)
    if scores[ordered[0]] - scores[ordered[1]] < 0.3:
        return "UNKNOWN"
    return ordered[0]

def rules(key):
    # Explicit rule-only comparison; no authority or execution based on output.
    t = tokens(key)
    if t & {"execute", "permission", "override"}:
        return "UNKNOWN"
    match = []
    for label, words in {
        "GOAL": {"goal","aim","objective"},
        "OPEN": {"open","gate","gates","pending","unresolved","obligations"},
        "NEXT": {"next"},
        "HISTORY": {"historical","past","repeat","preceding"},
        "EVIDENCE": {"receipt","seal","digest","evidence"},
    }.items():
        if t & words:
            match.append(label)
    return match[0] if len(match)==1 else "UNKNOWN"

def run(meta, fund, beetle, demi, exam, out):
    # Validate all exact R4 bytes and build historical contexts first.
    contexts = {p:r4.read_project(meta,fund,p) for p in r4.ROUTES}
    rows, raw_docs = [], []
    for p, labels in CURRICULUM.items():
        route = r4.ROUTES[p]
        root = meta if route["repo"] == r4.META else fund
        raw = r4.verified_bytes(root,route["state"],route["state_blob"])
        doc = r4.strict_object(raw)
        raw_docs.append((raw, route["repo"], route["commit"], route["state"]))
        for path, role in labels:
            value = oracle(doc,path)
            rows.append({"project":p,"path":list(path),"role":role,
                         "source_blob":route["state_blob"],"value_sha256":digest(value),
                         "partition":"TRAIN_ONLY","source_authority":False})
    route = r4.ROUTES["COSMOS_SETI"]
    raw_docs.append((r4.verified_bytes(meta,route["journal"],route["journal_blob"]),
                     route["repo"],route["commit"],route["journal"]))
    # Genuine Memory Cortex integration, isolated temporary source-only index.
    r4.verified_bytes(demi,"janus_model/memory_cortex_growth.py",CORTEX_BLOB)
    from janus_model import memory_cortex_growth as g
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state = g.new_state()
        for raw, repo, commit, path in raw_docs:
            g.add_text(root,state,raw,{"repository":repo,"commit":commit,"path":path})
        g.publish(root,state,"R5_DISPOSABLE_PUBLIC_SOURCE_REPLAY")
        g.write_json(root/"STATE.json",g.sealed(state))
        memory = g.query(root,"open_gates next_allowed",limit=3)
        assert memory["status"]=="VERIFIED_SOURCE_POINTERS_ONLY"
        assert memory["citations"]
        assert all(c["content_is_untrusted_data"] is True for c in memory["citations"])
        mem_receipt = {"generation":memory["generation"],"indexed_documents":memory["indexed_documents"],
                       "citations":memory["citations"],"production_state_written":False}
    # Read actual Beetle direction; proposal only, no queue or workflow dispatch.
    direction_doc = r4.strict_object(r4.verified_bytes(beetle,"data/beetle/directions.json",BEETLE_BLOB))
    direction = next(x for x in direction_doc["directions"] if x["id"]=="memory_cortex_incremental_sources")
    assert direction["enabled"] is True
    assert direction["memory_cortex_consumer"]["repository"]=="Hawkar-usls/Janus-Demiurge"
    beetle_receipt = {"direction_id":direction["id"],"direction_blob":BEETLE_BLOB,
         "proposal":"Find provenance and incremental retrieval research for unresolved registry schemas",
         "source_kind":direction["memory_cortex_consumer"]["source_kind"],
         "dispatched":False,"candidate_is_evidence":False}
    # Freeze learned state before reading exam. No labels or cases in train.
    model = train(rows)
    counts, classes, vocab, exact = model
    model_receipt = {"counts":{k:dict(v) for k,v in counts.items()},
                     "classes":dict(classes),"vocab":sorted(vocab),
                     "exact":{k:sorted(v) for k,v in exact.items()}}
    model_sha = digest(model_receipt)
    assert not out.exists()
    out.mkdir(parents=True)
    (out/"TRAIN_ONLY.json").write_text(json.dumps(rows,indent=2)+"\n")
    (out/"STUDENT.json").write_text(json.dumps(model_receipt,indent=2)+"\n")
    exam_raw = exam.read_bytes()
    assert r4.blob(exam_raw)==EXAM
    cases = r4.strict_object(exam_raw)["cases"]
    assert len(cases)==20 and len({c["id"] for c in cases})==20
    seen_keys = set(exact)
    # A new schema layout can contain known field names; report overlap explicitly.
    results=[]
    for c in cases:
        k=c["path"][-1]
        value=oracle(c["document"],c["path"])
        exact_guess=next(iter(exact[k])) if k in exact and len(exact[k])==1 else "UNKNOWN"
        learned=predict(model,k)
        rule=rules(k)
        results.append({"id":c["id"],"expected":c["expected"],"learned":learned,
                        "exact":exact_guess,"rules":rule,"key_seen_in_training":k in seen_keys,
                        "value_sha256":digest(value),"action_authority":False})
    scores={k:sum(x[k]==x["expected"] for x in results) for k in ("learned","exact","rules")}
    unknown_false_positive=sum(x["expected"]=="UNKNOWN" and x["learned"]!="UNKNOWN" for x in results)
    report={"schema":"janus.registry.r5.experimental_receipt.v1",
      "train_examples":len(rows),"train_sha256":digest(rows),"student_sha256":model_sha,
      "exam_cases":len(results),"exam_blob":EXAM,"scores":scores,
      "unseen_key_cases":sum(not x["key_seen_in_training"] for x in results),
      "unknown_false_positive":unknown_false_positive,
      "memory_cortex":mem_receipt,"beetle":beetle_receipt,
      "trinity":{"principle":"Separate source oracle from model predictions",
                 "actual_qwen_smollm_inference":False,"actual_trinity_runtime_connected":False,
                 "semantic_labels_human_authored":True,"source_pointer_oracle_only":True},
      "r4_context_status":{p:c["status"] for p,c in contexts.items()},
      "results":results,"external_exam":False,"production_promotion":False,
      "student_is_small_statistical_classifier_not_JANUS_neural_weights":True,
      "status":"EXPERIMENT_ONLY__EXTERNAL_VALIDATION_AND_RUNTIME_INTEGRATION_HOLD"}
    (out/"R5_RESULT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("train_examples","exam_cases","unseen_key_cases","scores","unknown_false_positive","status")},sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("meta","fund","beetle","demi","exam","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    run(a.meta,a.fund,a.beetle,a.demi,a.exam,a.out)
