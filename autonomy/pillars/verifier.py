"""Pillars R2: separately coded structural receipt verifier (not external custody).

Does not import the candidate planner. This can detect drift/rebinding and
unsafe authority fields, but authors who control both codepaths can still
defeat it. Production trust needs protected external verification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ZERO="0"*64
HEXDIGEST=re.compile(r"[0-9a-f]{64}\Z")


def seal(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),
                                 ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def verify(plan,catalog,needs,policy):
    if not all(type(x) is dict for x in (plan,catalog,needs,policy)):
        raise ValueError("VERIFY_INVALID_INPUT")
    if plan.get("schema")!="janus.autonomy.pillars.research_plan.v1":
        raise ValueError("VERIFY_SCHEMA")
    expected=seal({k:v for k,v in plan.items() if k!="receipt_sha256"})
    if plan.get("receipt_sha256")!=expected or not HEXDIGEST.fullmatch(expected):
        raise ValueError("VERIFY_RECEIPT_TAMPER")
    if plan.get("catalog_digest")!=seal(catalog) or plan.get("needs_digest")!=seal(needs):
        raise ValueError("VERIFY_INPUT_REBINDING")
    if plan.get("source_evidence")!=needs.get("source_evidence"):
        raise ValueError("VERIFY_OBSERVATION_REBINDING")
    if (plan.get("status")!="OFFLINE_PROPOSALS_GENERATED__ACTIVATION_AND_AUTONOMY_HOLD" or any(
        plan.get(k) is not True for k in ("no_model_training","no_foreign_code_run",
              "no_fork_or_repo_created","no_self_approved_evaluation","janus_main_unchanged")) or
        plan.get("action_authority") is not False or
        plan.get("external_admission")!="HOLD_NOT_OBTAINED" or
        plan.get("p03_checkpoint")!="HOLD_UNCHANGED"):
        raise ValueError("VERIFY_AUTHORITY_ESCALATION")
    if policy.get("mode")!="PROPOSALS_ONLY" or policy.get("external_effects") is not False or (
        policy.get("limits") or {}).get("max_new_repositories")!=0:
        raise ValueError("VERIFY_POLICY_DRIFT")
    gaps=needs.get("needs")
    pillars=catalog.get("pillars")
    if type(gaps) is not list or type(pillars) is not list or len(gaps)!=16 or len(pillars)!=13:
        raise ValueError("VERIFY_SOURCE_COUNT_DRIFT")
    pmap={p["id"]:p for p in pillars}
    if len(pmap)!=13 or any(p["license_bytes_verified"] is not False or
             p["action_authority"] is not False for p in pillars):
        raise ValueError("VERIFY_LICENSE_OR_SOURCE_HOLD")
    score=lambda x:4*x["impact"]+2*x["urgency"]-3*x["risk"]-x["cost"]
    selection=sorted(gaps,key=lambda x:(-score(x),x["id"]))[:policy["limits"]["max_proposals"]]
    blue=plan.get("selected_blueprints")
    events=plan.get("gene_capsule_events")
    if type(blue) is not list or type(events) is not list or (
      len(blue)!=len(events)) or len(blue)!=len(selection) or len(blue)>4:
        raise ValueError("VERIFY_BUDGET_OR_LIST_DRIFT")
    prev=ZERO
    for i,(need,item,event) in enumerate(zip(selection,blue,events)):
        if (item.get("organ_id")!="candidate_"+need["id"] or item.get("kind")!="MODULE_PROPOSAL_ONLY" or (
            item.get("capability")!=need["capability"]) or
            item.get("priority_score")!=score(need) or
            item.get("source_map_git_commit")!=needs["source_evidence"]["source_commit"] or
            item.get("module_path")!="modules/candidates/"+need["id"]+"/"):
            raise ValueError("VERIFY_SELECTION_OR_CANDIDATE_DRIFT")
        for key in ("has_passed_independent_tests","review_is_authenticated",
                    "execution_allowed","fork_allowed","production_promotion"):
            if item.get(key) is not False:
                raise ValueError("VERIFY_PROPOSAL_PROMOTION")
        if (item.get("test_partition")!="SYNTHETIC_ENGINEERING_ONLY__NEW_EXTERNAL_EXAM_REQUIRED" or (
            item.get("training_partition")!="TRAIN_ONLY_NOT_EXECUTED") or
            item.get("state")!="CATALOGUED_NOT_BORN"):
            raise ValueError("VERIFY_FALSE_TRAINING")
        refs=[{"repository":pmap[p]["repo"],"pinned_commit":pmap[p]["commit_sha"],
               "license_status":pmap[p]["import_policy"]} for p in need["pillars"]]
        if item.get("patterns_from_pinned_sources")!=refs:
            raise ValueError("VERIFY_PILLAR_REBINDING")
        pure={k:v for k,v in item.items() if k!="gene_sha256"}
        gene=seal(pure)
        if item.get("gene_sha256")!=gene:
            raise ValueError("VERIFY_GENE_REBINDING")
        if (event.get("sequence")!=i or event.get("previous_head")!=prev or (
            event.get("gene_sha256")!=gene) or
            event.get("kind")!="GENE_AND_CAPSULE_PROPOSED" or
            event.get("capsule_status")!="UNEVALUATED_NO_NEURAL_GAIN" or
            event.get("candidate_id")!=item["organ_id"]):
            raise ValueError("VERIFY_EVENT_CHAIN")
        current=seal({k:v for k,v in event.items() if k!="head"})
        if event.get("head")!=current:
            raise ValueError("VERIFY_EVENT_SEAL")
        prev=current
    if plan.get("event_head")!=prev or plan.get("not_selected_count")!=len(gaps)-len(blue):
        raise ValueError("VERIFY_HEAD_OR_COUNT")
    return {
        "schema":"janus.autonomy.pillars.structural_verifier.v1",
        "status":"OFFLINE_PLAN_STRUCTURALLY_CONSISTENT__ADMISSION_HOLD",
        "proposals_checked":len(blue),
        "event_head_verified":prev,
        "separate_codepath_not_independent_authority":True,
        "external_witness_attested":False,
        "may_activate_organs":False,
        "may_fork_external_code":False,
        "may_promote_models":False,
    }


def main():
    a=argparse.ArgumentParser()
    for arg in ("plan","catalog","needs","policy","out"):
        a.add_argument("--"+arg,type=Path,required=True)
    p=a.parse_args()
    def read(path):
        return json.loads(path.read_text())
    result=verify(read(p.plan),read(p.catalog),read(p.needs),read(p.policy))
    if p.out.exists():
        raise ValueError("VERIFY_OUTPUT_EXISTS")
    p.out.parent.mkdir(parents=True,exist_ok=True)
    p.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
