"""JANUS Pillars R2 — source-pinned, budgeted plan for organ evolution.

Converts a reviewed *partial* 49-repository map into an inert growth backlog.
All candidate inputs are DATA. This code does not fetch/install/execute foreign
code, fork, dispatch, merge PRs, modify checkpoints or train a model.

This is a design-level port of independent public architecture patterns, NOT a
copy of third-party source code or proof of learned self-improvement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

from autonomy import admission

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "CATALOG.json"
NEEDS = HERE / "NEEDS.json"
ZERO = "0" * 64
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}\Z")
ID = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")
ALLOWED = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
RESTRICTED = {"GPL-3.0", "NOASSERTION", None}
STATUSES = {"RESEARCH_ONLY", "NOT_DEMONSTRATED"}
EXPECTED_PILLARS = 13
EXPECTED_NEEDS = 16


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("utf-8")


def sha(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def require(ok, code):
    if not ok:
        raise ValueError(code)


def exact_fields(obj, names):
    require(type(obj) is dict and set(obj)==set(names.split()), "PILLARS_FIELDS_DRIFT")


def read_json(path, max_bytes=100_000):
    p=Path(path)
    require(p.is_file() and not p.is_symlink() and 0<p.stat().st_size<=max_bytes,
            "PILLARS_INVALID_INPUT_FILE")
    raw=p.read_bytes()
    def pairs(seq):
        out={}
        for k,v in seq:
            require(k not in out, "PILLARS_DUPLICATE_JSON_KEY")
            out[k]=v
        return out
    def no_inf(_):
        raise ValueError("PILLARS_NONFINITE_JSON")
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=no_inf)


def validate_catalog(doc):
    exact_fields(doc,"schema frozen_at_utc source intent license_scope code_copied foreign_code_executed fork_created policy_reuse pillars")
    require(doc["schema"]=="janus.autonomy.pillars.catalog.v1" and
            doc["source"]=="PUBLIC_GITHUB_REPOSITORY_METADATA_AND_DEFAULT_BRANCH_HEAD_ONLY" and
            doc["intent"]=="ARCHITECTURE_REFERENCE_AND_NONEXECUTING_GAP_PLANNING" and
            doc["license_scope"]=="METADATA_ONLY__LICENSE_BYTES_HISTORY_AND_TRANSITIVE_DEPS_UNVERIFIED" and
            doc["code_copied"] is False and doc["foreign_code_executed"] is False and
            doc["fork_created"] is False and doc["policy_reuse"]=="autonomy/policy.json",
            "PILLARS_UNVERIFIED_SOURCE_AUTHORITY")
    items=doc["pillars"]
    require(type(items) is list and len(items)==EXPECTED_PILLARS,"PILLARS_UNEXPECTED_COUNT")
    seen=set();repos=set()
    for p in items:
        exact_fields(p,"id repo canonical_url default_branch commit_sha license_metadata_spdx license_bytes_verified dependency_licenses_verified import_policy capability_patterns evidence_level action_authority")
        require(type(p["id"]) is str and ID.fullmatch(p["id"]) is not None
                and p["id"] not in seen,"PILLARS_ID_COLLISION")
        seen.add(p["id"])
        require(type(p["repo"]) is str and REPO.fullmatch(p["repo"]) is not None
                and p["repo"] not in repos and
                not p["repo"].startswith("Hawkar-usls/") and
                p["canonical_url"]=="https://github.com/"+p["repo"] and
                type(p["default_branch"]) is str and
                p["default_branch"] in ("main","master") and
                type(p["commit_sha"]) is str and SHA40.fullmatch(p["commit_sha"]),
                "PILLARS_INVALID_UPSTREAM_BINDING")
        repos.add(p["repo"])
        status=p["license_metadata_spdx"]
        require(status in ALLOWED | RESTRICTED, "PILLARS_LICENSE_METADATA_DRIFT")
        expected=("PATTERN_ONLY__CODE_IMPORT_NEEDS_BYTE_LEVEL_REVIEW"
                  if status in ALLOWED else
                  "PATTERN_ONLY__CODE_IMPORT_HOLD_LICENSE_OR_POLICY")
        require(p["import_policy"]==expected and
                p["license_bytes_verified"] is False and
                p["dependency_licenses_verified"] is False and
                p["action_authority"] is False and
                p["evidence_level"]=="PUBLIC_DESCRIPTION_AND_REPOSITORY_METADATA_NOT_INDEPENDENT_FUNCTIONAL_REPLICATION",
                "PILLARS_UNVERIFIED_LICENSE_OR_AUTHORITY")
        patterns=p["capability_patterns"]
        require(type(patterns) is list and len(patterns)==2 and
                len(set(patterns))==2 and
                all(type(x) is str and ID.fullmatch(x) for x in patterns),
                "PILLARS_INVALID_CAPABILITY")
    return {p["id"]:p for p in items}


def validate_needs(doc, pillars):
    exact_fields(doc,"schema source_evidence incomplete_is_not_proof_of_absence needs")
    require(doc["schema"]=="janus.autonomy.pillars.needs.v1" and
            doc["incomplete_is_not_proof_of_absence"] is True,
            "NEEDS_SCOPE_DRIFT")
    e=doc["source_evidence"]
    exact_fields(e,"repository source_commit current_state_git_blob_sha contract_git_blob_sha observed_public_repositories health_integrity observed_health autonomy_phase source_is_frozen_not_live")
    require(e["repository"]=="Hawkar-usls/janus-meta-registry" and
            all(type(e[k]) is str and SHA40.fullmatch(e[k]) for k in
                ("source_commit","current_state_git_blob_sha","contract_git_blob_sha")) and
            e["observed_public_repositories"]==49 and
            e["health_integrity"]=="PASS" and e["observed_health"]=="HOLD" and
            e["autonomy_phase"]=="P0.3" and
            e["source_is_frozen_not_live"] is True,
            "NEEDS_OBSERVATION_SCOPE_DRIFT")
    needs=doc["needs"]
    require(type(needs) is list and len(needs)==EXPECTED_NEEDS,"NEEDS_UNEXPECTED_COUNT")
    seen=set()
    for n in needs:
        exact_fields(n,"id capability observed_status impact urgency risk cost pillars goal requires_human_activation")
        require(type(n["id"]) is str and ID.fullmatch(n["id"]) and
                n["id"] not in seen and
                type(n["capability"]) is str and
                re.fullmatch(r"[A-Z][A-Z0-9_]{5,80}",n["capability"]) and
                n["observed_status"] in STATUSES and
                type(n["requires_human_activation"]) is bool and
                n["requires_human_activation"] is True and
                type(n["goal"]) is str and 10<=len(n["goal"])<=300,
                "NEEDS_UNREVIEWED_CAPABILITY")
        seen.add(n["id"])
        for k,lo,hi in (("impact",1,10),("urgency",1,10),
                        ("risk",1,5),("cost",1,5)):
            require(type(n[k]) is int and lo<=n[k]<=hi,"NEEDS_BAD_BUDGET")
        require(type(n["pillars"]) is list and len(n["pillars"])==2 and
                len(set(n["pillars"]))==2 and
                all(p in pillars for p in n["pillars"]), "NEEDS_MISSING_PILLAR")
    return needs


def need_score(n):
    return 4*n["impact"] + 2*n["urgency"] - 3*n["risk"] - n["cost"]


def plan(catalog, gaps, policy):
    source=validate_catalog(catalog)
    needs=validate_needs(gaps,source)
    admission.validate_policy(policy)
    require(policy["mode"]=="PROPOSALS_ONLY" and
            policy["kill_switch"] is False and
            policy["external_effects"] is False and
            policy["limits"]["max_new_repositories"]==0 and
            policy["limits"]["max_proposals"]<=4,
            "PILLARS_EFFECT_BOUNDARY")
    selected=sorted(needs,key=lambda x:(-need_score(x),x["id"]))[:policy["limits"]["max_proposals"]]
    parent=sha({"catalog":catalog,"needs":gaps,
                "policy_sha256":sha(policy)})
    proposals=[];events=[];head=ZERO
    for i,n in enumerate(selected):
        refs=[{"repository":source[p]["repo"],
               "pinned_commit":source[p]["commit_sha"],
               "license_status":source[p]["import_policy"]}
              for p in n["pillars"]]
        proposed={
          "schema":"janus.autonomy.pillars.organ_blueprint.v1",
          "organ_id":"candidate_"+n["id"],
          "parent_repo":"Hawkar-usls/Janus",
          "kind":"MODULE_PROPOSAL_ONLY",
          "module_path":"modules/candidates/"+n["id"]+"/",
          "capability":n["capability"],
          "reason":n["goal"],
          "source_map_git_commit":gaps["source_evidence"]["source_commit"],
          "patterns_from_pinned_sources":refs,
          "priority_score":need_score(n),
          "typed_inputs":["OBSERVATION_RECEIPT_V1","TRUSTED_POLICY_V1"],
          "typed_outputs":["CANDIDATE_CAPSULE_V1"],
          "resource_budget":{"cpu_seconds":60,"memory_mib":256,"disk_mib":100},
          "test_partition":"SYNTHETIC_ENGINEERING_ONLY__NEW_EXTERNAL_EXAM_REQUIRED",
          "training_partition":"TRAIN_ONLY_NOT_EXECUTED",
          "rollback":"CLOSE_DRAFT_OR_REVERT_EXPLICIT_REVIEWED_COMMIT",
          "state":"CATALOGUED_NOT_BORN",
          "has_passed_independent_tests":False,
          "review_is_authenticated":False,
          "execution_allowed":False,
          "fork_allowed":False,
          "production_promotion":False,
        }
        gene=sha(proposed)
        event={"sequence":i,"previous_head":head,"kind":"GENE_AND_CAPSULE_PROPOSED",
               "gene_sha256":gene,"capsule_status":"UNEVALUATED_NO_NEURAL_GAIN",
               "candidate_id":proposed["organ_id"],"source_graph_digest":parent}
        event["head"]=sha(event)
        head=event["head"]
        proposals.append({**proposed,"gene_sha256":gene})
        events.append(event)
    outcome={
      "schema":"janus.autonomy.pillars.research_plan.v1",
      "status":"OFFLINE_PROPOSALS_GENERATED__ACTIVATION_AND_AUTONOMY_HOLD",
      "catalog_digest":sha(catalog),"needs_digest":sha(gaps),
      "source_evidence":gaps["source_evidence"],
      "fixed_ranked_gap_count":len(needs),
      "selected_blueprints":proposals,
      "gene_capsule_events":events,
      "event_head":head,
      "not_selected_count":len(needs)-len(proposals),
      "maximum_proposals":policy["limits"]["max_proposals"],
      "no_model_training":True,"no_foreign_code_run":True,
      "no_fork_or_repo_created":True,
      "no_self_approved_evaluation":True,
      "janus_main_unchanged":True,
      "external_admission":"HOLD_NOT_OBTAINED",
      "p03_checkpoint":"HOLD_UNCHANGED",
      "action_authority":False,
    }
    outcome["receipt_sha256"]=sha(outcome)
    return outcome


def health_triage(health):
    """Use existing R1 verifier result as evidence; never invert HOLD to PASS."""
    exact_fields(health, "schema snapshot_sha256 verified_at integrity_gate observation_gate integrity_reasons observation_reasons operational_gate independent_admission_gate execution_allowed full_autonomy_proven authority_delta health_sha256")
    require(health["schema"]=="janus.autonomy.health.v1" and
            health["integrity_gate"]=="PASS" and
            health["observation_gate"]=="HOLD" and
            health["operational_gate"]=="NOT_ASSESSED_REQUIRES_SERVICE_SPECIFIC_PROBES" and
            health["independent_admission_gate"]=="HOLD_EXTERNAL_REVIEW_REQUIRED" and
            health["execution_allowed"] is False and
            health["full_autonomy_proven"] is False and
            health["authority_delta"]==0,"PILLARS_UNTRUSTED_HEALTH")
    # Verify the input by the original health verifier's canonical encoding,
    # but do not misrepresent a same-owner hash as external authentication.
    expected_health=hashlib.sha256(json.dumps(
        {k:v for k,v in health.items() if k!="health_sha256"},
        sort_keys=True,separators=(",",":"),ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()
    require(health["health_sha256"]==expected_health,
            "PILLARS_HEALTH_DIGEST_MISMATCH")
    # reason strings are untrusted DATA; only count fixed suffix categories.
    mapping={
      "MEMBER_LINK_UNVERIFIED":"VERIFY_MEMBERSHIP_SOURCE",
      "CHECKS_STALE":"RECHECK_FRESH_HEAD_STATUS",
      "CHECKS_NOT_ALL_SUCCESS":"DIAGNOSE_FAILING_CHECKS",
      "LATEST_RUN_NOT_SUCCESS":"INVESTIGATE_CI_REGRESSION",
      "CHECKS_NOT_VISIBLE":"CONFIGURE_OBSERVER_READ_SCOPE",
    }
    counts={name:0 for name in mapping}
    unclassified=0
    reasons=health["observation_reasons"]
    require(type(reasons) is list and len(reasons)<=1000 and
            all(type(x) is str and len(x)<=300 for x in reasons),"PILLARS_BAD_REASONS")
    for reason in reasons:
        suffix=reason.rsplit(":",1)[-1]
        if suffix in counts: counts[suffix]+=1
        else: unclassified+=1
    out={"schema":"janus.autonomy.pillars.health_triage.v1",
      "status":"HEALTH_HOLD_TRIAGED_NOT_REPAIRED",
      "source_snapshot_sha256":health["snapshot_sha256"],
      "source_health_sha256":health["health_sha256"],
      "snapshot_only_not_live":True,
      "observed_reason_count":len(reasons),
      "reason_counts":counts,"unclassified_reasons":unclassified,
      "next_actions":[mapping[k] for k,v in sorted(counts.items(),
                   key=lambda kv:(-kv[1],kv[0])) if v>0],
      "no_ci_mutation":True,"no_issue_posted":True,
      "no_workflow_run_dispatched":True,"admission_allowed":False}
    out["receipt_sha256"]=sha(out)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--catalog",type=Path,default=CATALOG)
    p.add_argument("--needs",type=Path,default=NEEDS)
    p.add_argument("--policy",type=Path,default=Path("autonomy/policy.json"))
    p.add_argument("--health",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    output=plan(read_json(args.catalog),read_json(args.needs),
                read_json(args.policy))
    if args.health:
        output["health_triage"]=health_triage(read_json(args.health,1_000_000))
        output["receipt_sha256"]=sha({k:v for k,v in output.items() if k!="receipt_sha256"})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    require(not args.out.exists() and not args.out.is_symlink(),
            "PILLARS_OUTPUT_EXISTS")
    args.out.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":output["status"],
                      "proposed":len(output["selected_blueprints"]),
                      "external_admission":output["external_admission"],
                      "receipt_sha256":output["receipt_sha256"]},sort_keys=True))


if __name__=="__main__":
    main()
