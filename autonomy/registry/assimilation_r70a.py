"""R7.0A: offline, source-pinned factual composition preflight, not model training."""
import hashlib
import json
import sys
from pathlib import Path

SOURCES = {
    "memory": ("data/autonomy_map/JANUS-R69-LIVING-MEMORY-EVIDENCE.json", "aee43f5804d5c15a76c333fc9e020d7d619b28b8"),
    "trinity": ("research/essence/trinity_parental_school/R14_REAL_CPU_RESULT_20261123.json", "99cd8a480e7a11fb9a0d3fe0386738069ec9da1a"),
}
HOLD = "HOLD"
def strict_pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate JSON key")
        out[k] = v
    return out

def load(root, key):
    path, pin = SOURCES[key]
    p = root / path
    if p.is_symlink() or p.stat().st_size > 1000000:
        raise ValueError("unsafe source")
    raw = p.read_bytes()
    if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest() != pin:
        raise ValueError("source pin mismatch")
    return json.loads(raw, object_pairs_hook=strict_pairs)

def select(docs, ref):
    source, *parts = ref
    v = docs[source]
    for part in parts:
        v = v[part]
    if type(v) not in (int, bool, str):
        raise ValueError("non-scalar")
    return v

def answer(docs, task, compose):
    op, *refs = task
    try:
        if op not in {"get", "difference", "sum", "equal", "both_false"}:
            return HOLD
        if op == "get" and len(refs) == 1:
            return select(docs, refs[0])
        if not compose or len(refs) != 2:
            return HOLD
        a, b = (select(docs, ref) for ref in refs)
        if op in {"difference", "sum"} and type(a) is type(b) is int:
            return a-b if op == "difference" else a+b
        if op == "equal" and type(a) is type(b):
            return a == b
        if op == "both_false" and type(a) is type(b) is bool:
            return not a and not b
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return HOLD

# Developer-authored diagnostic cases, all public; never an independent exam.
M = "memory"
T = "trinity"
TASKS = [
    (("get", (M,"result","transition_pass")), 2),
    (("get", (T,"fifth","total_cases")), 64),
    (("get", (T,"real_teacher_probe","mathematically_verified_accepted")), 0),
    (("get", (T,"governance","production_JANUS_checkpoint_updated")), False),
    (("difference",(T,"fifth","total_oracle_matches"),(T,"fifth","total_format_only_matches")), 16),
    (("difference",(T,"real_teacher_probe","training_only_proposals"),(T,"real_teacher_probe","mathematically_verified_accepted")), 24),
    (("equal",(M,"organs",0,"generation"),(M,"organs",1,"generation")), True),
    (("both_false",(M,"production_deployed"),(T,"governance","production_JANUS_checkpoint_updated")), True),
    (("sum",(M,"result","transition_pass"),(M,"result","planner_reopen_pass")), 4),
    (("equal",(M,"result","registry_document_reads"),(T,"real_teacher_probe","mathematically_verified_accepted")), True),
    (("get",(M,"unreported_live_generation")), HOLD),
    (("execute",(T,"next")), HOLD),
    (("difference",(M,"action_authority"),(T,"fifth","total_cases")), HOLD),
    (("get",(M,"organs",0)), HOLD),
]
def score(docs, compose):
    values = [answer(docs, task, compose) for task, expected in TASKS]
    return sum(type(v) is type(expected) and v == expected for v, (_, expected) in zip(values, TASKS))

def main():
    docs = {"memory": load(Path(sys.argv[1]), "memory"), "trinity": load(Path(sys.argv[2]), "trinity")}
    lookup = score(docs, False)
    composition = score(docs, True)
    assert composition == len(TASKS), "diagnostic mismatch"
    assert answer(docs, ("get", (M,"result","transition_pass")), True) == 2
    # Strong deterministic composition is the actual reference, not retrieval alone.
    report = {"schema":"janus.r70a.preflight.v1","sources_verified":2,"public_diagnostic_cases":len(TASKS),
              "lookup_only_correct":lookup,"typed_composition_correct":composition,
              "strong_reference_correct":composition,"neural_training":False,
              "independent_exam":False,"neural_gain_established":False,
              "production_changed":False,"action_authority":False,
              "status":"ENGINEERING_PREFLIGHT_PASS__INDEPENDENT_EXAM_HOLD"}
    print("R70A_PREFLIGHT "+json.dumps(report, sort_keys=True))
if __name__ == "__main__":
    main()
