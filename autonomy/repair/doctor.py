#!/usr/bin/env python3
"""JANUS R3: bounded FIRST-PARTY diagnostic self-repair organ (proposal-only).

Recognizes a concrete deficit in the independent R1 health receipt, derives
one allowlisted source patch from a fixed recipe, tests it against preregistered
internal cases and preserves source/patch/evaluation lineage. It never executes
text from observations, edits the checked-out source, fetches or runs foreign
code, creates a fork, merges a PR or marks a degraded service repaired.
"""
from __future__ import annotations
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

SCHEMA = "janus.autonomy.r3.repair_proposal.v1"
BASE_BLOB_SHA1 = "238b0767e5298dbd1723ed38dfef60cbfabed67f"
BASE_COMMIT = "376fdc5fe16b36071136290cec715724b21ffe86"
TARGET = "autonomy/pillars/engine.py"
EXAM = "autonomy/repair/R3_FROZEN_NEW_CASES.json"
SOURCE_HEALTH_SNAPSHOT = "c12e95b93548657f536501583aaa5c5d274ae8cfc8c26c0c6a7b34579ba6d9b3"
NEW_ROUTES = {
    "CHECKS_UNKNOWN": "REQUIRE_CHECK_RUN_EVIDENCE",
    "LATEST_ACTION_NOT_AT_HEAD": "REQUERY_LATEST_ACTION_AT_HEAD",
    "REPOSITORY_OBSERVATIONS_INCOMPLETE": "RETRY_BOUNDED_REPOSITORY_OBSERVATION"
}
OLD_ANCHOR = '      "CHECKS_NOT_VISIBLE":"CONFIGURE_OBSERVER_READ_SCOPE",\n'
NEW_ANCHOR = OLD_ANCHOR + "".join(
    f'      "{k}":"{v}",\n' for k,v in NEW_ROUTES.items()
)
RECOVERY_LIMIT = 50_000


class RepairHold(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise RepairHold(message)


def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False).encode("utf-8")


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def blob_sha(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii")+raw).hexdigest()


def read_bounded(path: Path, max_bytes=RECOVERY_LIMIT):
    require(path.is_file() and not path.is_symlink()
            and 0<path.stat().st_size<=max_bytes,"R3_UNTRUSTED_PATH")
    return path.read_bytes()


def strict_json(raw):
    def unique(pairs):
        result={}
        for k,v in pairs:
            require(k not in result,"R3_JSON_DUPLICATE")
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(RepairHold("R3_JSON_NONFINITE")))


def mapping_from_source(source):
    require(type(source) is str and len(source)<=RECOVERY_LIMIT,"R3_SOURCE_BUDGET")
    tree=ast.parse(source,filename=TARGET)
    functions=[node for node in tree.body
               if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
               and node.name=="health_triage"]
    require(len(functions)==1,"R3_HEALTH_TRIAGE_MISSING")
    assignments=[
        a for a in ast.walk(functions[0]) if isinstance(a,ast.Assign)
        and len(a.targets)==1 and isinstance(a.targets[0],ast.Name)
        and a.targets[0].id=="mapping"
    ]
    require(len(assignments)==1,"R3_AMBIGUOUS_MAPPING")
    mapping=ast.literal_eval(assignments[0].value)
    require(type(mapping) is dict and len(mapping)<=16
            and all(type(k) is str and type(v) is str for k,v in mapping.items()),
            "R3_MAPPING_SCOPE")
    return mapping


def baseline_from_candidate(current):
    require(type(current) is str and current.count(NEW_ANCHOR)==1,
            "R3_ROLLBACK_ANCHOR_MISMATCH")
    base=current.replace(NEW_ANCHOR,OLD_ANCHOR,1)
    require(blob_sha(base.encode("utf-8"))==BASE_BLOB_SHA1,
            "R3_REVERTED_BASELINE_BLOB_MISMATCH")
    return base


def diagnose(health, base_source):
    require(type(health) is dict
            and health.get("schema")=="janus.autonomy.health.v1"
            and health.get("snapshot_sha256")==SOURCE_HEALTH_SNAPSHOT
            and health.get("health_sha256")==digest({k:v for k,v in health.items()
                                                      if k!="health_sha256"})
            and health.get("integrity_gate")=="PASS"
            and health.get("observation_gate")=="HOLD"
            and health.get("operational_gate")=="NOT_ASSESSED_REQUIRES_SERVICE_SPECIFIC_PROBES"
            and health.get("independent_admission_gate")=="HOLD_EXTERNAL_REVIEW_REQUIRED"
            and health.get("execution_allowed") is False
            and health.get("full_autonomy_proven") is False,
            "R3_HEALTH_TRUST_GATE_HOLD")
    reasons=health.get("observation_reasons")
    require(type(reasons) is list and len(reasons)==41
            and all(type(x) is str and 1<=len(x)<=240 for x in reasons),
            "R3_WRONG_HISTORICAL_COHORT")
    base_mapping=mapping_from_source(base_source)
    unknown={}
    for reason in reasons:
        suffix=reason.rsplit(":",1)[-1]
        if suffix not in base_mapping:
            unknown[suffix]=unknown.get(suffix,0)+1
    require(unknown=={"CHECKS_UNKNOWN":3,"LATEST_ACTION_NOT_AT_HEAD":1,
                      "REPOSITORY_OBSERVATIONS_INCOMPLETE":1},
            "R3_UNKNOWN_SOURCE_DIAGNOSIS_UNREVIEWED")
    return {"status":"R3_DIAGNOSIS_VERIFIED_5_UNCLASSIFIED_REASONS",
            "source_snapshot_sha256":health["snapshot_sha256"],
            "source_health_sha256":health["health_sha256"],
            "unclassified_reason_counts":unknown,
            "unclassified_total":sum(unknown.values()),
            "reported_operational_health":"NOT_ASSESSED",
            "service_fault_repaired":False}


def construct_candidate(base_source):
    raw=base_source.encode("utf-8")
    require(blob_sha(raw)==BASE_BLOB_SHA1,"R3_UNEXPECTED_BASELINE_BLOB")
    require(base_source.count(OLD_ANCHOR)==1
            and NEW_ANCHOR not in base_source,
            "R3_PATCH_ANCHOR_NOT_UNIQUE")
    base_mapping=mapping_from_source(base_source)
    require(not set(NEW_ROUTES).intersection(base_mapping),"R3_ROUTE_ALREADY_PRESENT")
    proposed=base_source.replace(OLD_ANCHOR,NEW_ANCHOR,1)
    expected={**base_mapping,**NEW_ROUTES}
    require(mapping_from_source(proposed)==expected,
            "R3_ADDITIONAL_UNAPPROVED_CHANGE")
    require(blob_sha(baseline_from_candidate(proposed).encode("utf-8"))==BASE_BLOB_SHA1,
            "R3_ROLLBACK_NOT_BYTE_EXACT")
    diff="".join(difflib.unified_diff(
        base_source.splitlines(keepends=True),
        proposed.splitlines(keepends=True),
        fromfile="a/"+TARGET,tofile="b/"+TARGET
    ))
    require(diff.count("\n+")<=4 and diff.count("\n-")<=2
            and len(diff)<1800,"R3_PATCH_EXCEEDS_ALLOWLIST")
    return proposed,diff


def classify(mapping,reason):
    require(type(reason) is str and len(reason)<=240,"R3_REASON_BUDGET")
    return mapping.get(reason.rsplit(":",1)[-1],"UNCLASSIFIED")


def evaluate_frozen(exam,base_source,candidate):
    require(type(exam) is dict
            and exam.get("schema")=="janus.r3.repair_eval.frozen.v1"
            and exam.get("case_count")==24
            and exam.get("never_train_on_this_suite") is True
            and exam.get("external_custodian") is False,
            "R3_UNTRUSTED_FROZEN_SUITE")
    cases=exam.get("cases")
    require(type(cases) is list and len(cases)==24
            and len({case.get("id") for case in cases if type(case) is dict})==24,
            "R3_FROZEN_SUITE_COLLISION")
    old=mapping_from_source(base_source);new=mapping_from_source(candidate)
    parent_right=0;child_right=0;lost=0;gained=0
    for c in cases:
        require(type(c) is dict and set(c)=={"id","reason","expected_action","partition","authority"}
                and c["partition"]=="INTERNAL_FROZEN_NEW_ENGINEERING_CASES"
                and c["authority"] is False
                and type(c["reason"]) is str
                and type(c["expected_action"]) is str,
                "R3_FROZEN_CASE_MUTATION")
        p=classify(old,c["reason"])==c["expected_action"]
        q=classify(new,c["reason"])==c["expected_action"]
        parent_right+=p;child_right+=q;lost+=p and not q;gained+=q and not p
    result={"new_internal_cases":len(cases),"parent_correct":parent_right,
            "candidate_correct":child_right,"newly_correct":gained,
            "lost_parent_correct":lost,
            "evaluation_kind":"STATIC_INTERNAL_DIAGNOSTIC_ROUTING_NOT_NEURAL",
            "externally_blind":False,
            "candidate_review_ready":lost==0 and gained>=1 and child_right>parent_right}
    return result


def propose(health,baseline,exam,working_candidate):
    findings=diagnose(health,baseline)
    candidate,patch=construct_candidate(baseline)
    require(candidate==working_candidate,
            "R3_WORKTREE_DIFFERS_FROM_GENERATED_PATCH")
    outcome=evaluate_frozen(exam,baseline,candidate)
    require(outcome["candidate_review_ready"] and outcome["lost_parent_correct"]==0,
            "R3_REPAIR_REGRESSION")
    proposed={
        "schema":SCHEMA,
        "status":"FIRST_PARTY_DIAGNOSTIC_PATCH_PAIRED_PASS__REVIEW_ONLY",
        "source_commit":BASE_COMMIT,
        "source_path":TARGET,
        "source_git_blob_sha1":BASE_BLOB_SHA1,
        "candidate_git_blob_sha1":blob_sha(candidate.encode("utf-8")),
        "repair_recipe":"ALLOWLISTED_MISSING_DIAGNOSTIC_ROUTE_LITERALS_ONLY",
        "diagnosis":findings,
        "patch_sha256":hashlib.sha256(patch.encode()).hexdigest(),
        "patch_text":patch,
        "frozen_exam_sha256":digest(exam),
        "paired_evaluation":outcome,
        "dependency_search":"REUSED_NATIVE_JANUS_HEALTH_TRIAGE__NO_FOREIGN_CODE_IMPORTED",
        "rollback":{"strategy":"REVERT_THIS_DRAFT_CHANGE",
                    "restore_source_git_blob_sha1":BASE_BLOB_SHA1},
        "constraints":{"new_files_executed_from_external_repo":False,
                       "working_models_changed":False,
                       "external_forks_created":0,
                       "current_49_repo_operational_health_proven":False,
                       "auto_merge":False,
                       "trusted_external_approval":False,
                       "production_activated":False,
                       "self_modifying_workflow_executed":False},
    }
    proposed["receipt_sha256"]=digest(proposed)
    return proposed


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--health",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--exam",type=Path,default=Path(EXAM))
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    require(not args.out.exists() and not args.out.is_symlink(),
            "R3_OUTPUT_OVERWRITE_DENIED")
    report=propose(
       strict_json(read_bounded(args.health)),
       read_bounded(args.baseline).decode("utf-8"),
       strict_json(read_bounded(args.exam)),
       read_bounded(args.candidate).decode("utf-8"))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","repair_recipe",
                       "source_git_blob_sha1","candidate_git_blob_sha1")},
                     sort_keys=True))


if __name__=="__main__":
    main()
