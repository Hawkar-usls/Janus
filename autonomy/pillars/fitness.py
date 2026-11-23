"""JANUS R2 prospective generation selector: strict paired regression protection.

This is an experiment admission *API*, not a trained model, evaluator oracle,
external custodian or production deployer. Internal improvement never grants
graduation; production still requires an independently protected verifier,
untouched external test cohorts, real rollback and operator sign-off.
"""
from __future__ import annotations

import hashlib
import json
import re

SHA64 = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,63}\Z")


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,
           separators=(",",":"),allow_nan=False).encode()).hexdigest()


class FitnessHold(ValueError):
    pass


def require(ok,reason):
    if not ok:
        raise FitnessHold(reason)


def examine(record):
    require(type(record) is dict and set(record)=={
        "schema","parent_checkpoint_sha256","candidate_checkpoint_sha256",
        "benchmark_sha256","partition","training_on_benchmark",
        "external_independent_custodian","parent_runtime_kind",
        "candidate_runtime_kind","cases"},"FITNESS_CONTRACT_DRIFT")
    require(record["schema"]=="janus.pillars.paired_fitness.v1" and
            record["partition"]=="PUBLIC_INTERNAL_SANDBOX_NOT_GRADUATION" and
            record["training_on_benchmark"] is False and
            record["external_independent_custodian"] is False,
            "FITNESS_EXAM_LEAK_OR_AUTHORITY")
    for k in ("parent_checkpoint_sha256","candidate_checkpoint_sha256","benchmark_sha256"):
        require(type(record[k]) is str and SHA64.fullmatch(record[k]) is not None,
                "FITNESS_UNBOUND_SOURCE")
    require(record["parent_checkpoint_sha256"]!=record["candidate_checkpoint_sha256"],
            "FITNESS_IDENTICAL_GENERATION")
    require(record["parent_runtime_kind"]==record["candidate_runtime_kind"]==
            "SAME_CPU_GITHUB_RUNNER_CLASS","FITNESS_COMPARABILITY")
    cases=record["cases"]
    require(type(cases) is list and len(cases)==12,"FITNESS_SAMPLE_SCOPE")
    ids=set()
    for c in cases:
        require(type(c) is dict and set(c)=={
             "case_id","parent_correct","candidate_correct",
             "parent_cpu_ms","candidate_cpu_ms"},"FITNESS_CASE_FIELDS")
        ident=c["case_id"]
        require(type(ident) is str and ID.fullmatch(ident) and
                ident not in ids,"FITNESS_CASE_ID")
        ids.add(ident)
        for k in ("parent_correct","candidate_correct"):
            require(type(c[k]) is bool,"FITNESS_INVALID_TRUTH_VALUE")
        for k in ("parent_cpu_ms","candidate_cpu_ms"):
            require(type(c[k]) is int and 1<=c[k]<=120_000,"FITNESS_INVALID_RUNTIME")
    parent=sum(c["parent_correct"] for c in cases)
    child=sum(c["candidate_correct"] for c in cases)
    lost=sum(c["parent_correct"] and not c["candidate_correct"] for c in cases)
    gain=sum(c["candidate_correct"] and not c["parent_correct"] for c in cases)
    parent_cpu=sum(c["parent_cpu_ms"] for c in cases)
    child_cpu=sum(c["candidate_cpu_ms"] for c in cases)
    # Preserve every parent skill. Baseline has independent requirement to be
    # nontrivial, otherwise fake zeros would inflate candidate wins.
    promising=(parent>=5 and lost==0 and gain>=2 and
               child>parent and child_cpu<=int(parent_cpu*1.2))
    result={
       "schema":"janus.pillars.paired_fitness_result.v1",
       "status":("INTERNAL_CANDIDATE_PROMISING__EXTERNAL_GRADUATION_HOLD"
                 if promising else "HOLD_INTERNAL_FUNCTIONAL_GAIN_OR_REGRESSION"),
       "source_record_sha256":digest(record),
       "benchmark_sha256":record["benchmark_sha256"],
       "parent_correct":parent,
       "candidate_correct":child,
       "newly_solved":gain,
       "lost_previously_solved":lost,
       "parent_cpu_ms":parent_cpu,
       "candidate_cpu_ms":child_cpu,
       "same_cpu_class_reported_not_independently_measured":True,
       "test_set_is_internal_and_known":True,
       "new_external_exam_batches_passed":0,
       "may_replace_parent":False,
       "may_promote_weights":False,
       "may_spawn_organ":False,
       "independent_verification_completed":False,
       "training_weights_modified":False,
    }
    result["receipt_sha256"]=digest(result)
    return result
