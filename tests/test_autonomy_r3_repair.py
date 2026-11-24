"""R3 first-party self-repair controls; no dynamic foreign code execution."""
from __future__ import annotations
import copy
from pathlib import Path
import unittest

from autonomy.repair import doctor
from autonomy.pillars import engine

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/doctor.TARGET
EXAM=ROOT/doctor.EXAM


def historical_health():
    reasons=[]
    for label,count in (
        ("MEMBER_LINK_UNVERIFIED",21),
        ("CHECKS_STALE",10),
        ("CHECKS_NOT_ALL_SUCCESS",5),
        ("CHECKS_UNKNOWN",3),
        ("LATEST_ACTION_NOT_AT_HEAD",1),
        ("REPOSITORY_OBSERVATIONS_INCOMPLETE",1),
    ):
        for i in range(count):
            reasons.append(f"Hawkar-usls/fixture-{i}:{label}" if label!="REPOSITORY_OBSERVATIONS_INCOMPLETE" else label)
    h={
       "schema":"janus.autonomy.health.v1",
       "snapshot_sha256":doctor.SOURCE_HEALTH_SNAPSHOT,
       "verified_at":"2026-11-23T16:42:32Z",
       "integrity_gate":"PASS","observation_gate":"HOLD",
       "integrity_reasons":[],"observation_reasons":reasons,
       "operational_gate":"NOT_ASSESSED_REQUIRES_SERVICE_SPECIFIC_PROBES",
       "independent_admission_gate":"HOLD_EXTERNAL_REVIEW_REQUIRED",
       "execution_allowed":False,"full_autonomy_proven":False,"authority_delta":0,
    }
    h["health_sha256"]=doctor.digest(h)
    return h


def files():
    changed=TARGET.read_text()
    base=doctor.baseline_from_candidate(changed)
    suite=doctor.strict_json(EXAM.read_bytes())
    return base,changed,suite


class R3RepairTests(unittest.TestCase):
    def test_original_source_blob_and_exact_rollback(self):
        old,new,_=files()
        self.assertEqual(doctor.blob_sha(old.encode()),doctor.BASE_BLOB_SHA1)
        self.assertNotEqual(doctor.blob_sha(new.encode()),doctor.BASE_BLOB_SHA1)
        self.assertEqual(doctor.mapping_from_source(old)["CHECKS_STALE"],
                         doctor.mapping_from_source(new)["CHECKS_STALE"])
        self.assertEqual(len(doctor.mapping_from_source(new)),
                         len(doctor.mapping_from_source(old))+3)

    def test_original_real_failure_is_reproduced_from_signed_fixture(self):
        old,_,_=files()
        diagnosis=doctor.diagnose(historical_health(),old)
        self.assertEqual(diagnosis["unclassified_total"],5)
        self.assertEqual(diagnosis["unclassified_reason_counts"],{
            "CHECKS_UNKNOWN":3,
            "LATEST_ACTION_NOT_AT_HEAD":1,
            "REPOSITORY_OBSERVATIONS_INCOMPLETE":1,
        })
        self.assertFalse(diagnosis["service_fault_repaired"])

    def test_candidate_is_generated_from_exact_one_source_edit(self):
        old,new,_=files()
        candidate,patch=doctor.construct_candidate(old)
        self.assertEqual(candidate,new)
        self.assertIn('+"      "CHECKS_UNKNOWN"',patch.replace("+      ","+\"      "))
        self.assertLess(len(patch),1800)
        self.assertEqual(doctor.baseline_from_candidate(candidate),old)

    def test_new_preregistered_exam_improves_without_regression(self):
        old,new,suite=files()
        result=doctor.evaluate_frozen(suite,old,new)
        self.assertEqual(result["new_internal_cases"],24)
        self.assertEqual(result["parent_correct"],15)
        self.assertEqual(result["candidate_correct"],24)
        self.assertEqual(result["newly_correct"],9)
        self.assertEqual(result["lost_parent_correct"],0)
        self.assertTrue(result["candidate_review_ready"])
        self.assertFalse(result["externally_blind"])

    def test_real_janus_triage_now_classifies_all_41_reasons(self):
        r=engine.health_triage(historical_health())
        self.assertEqual(r["observed_reason_count"],41)
        self.assertEqual(r["unclassified_reasons"],0)
        self.assertEqual(r["reason_counts"]["CHECKS_UNKNOWN"],3)
        self.assertEqual(r["reason_counts"]["LATEST_ACTION_NOT_AT_HEAD"],1)
        self.assertEqual(r["reason_counts"]["REPOSITORY_OBSERVATIONS_INCOMPLETE"],1)
        self.assertEqual(r["status"],"HEALTH_HOLD_TRIAGED_NOT_REPAIRED")
        self.assertFalse(r["admission_allowed"])

    def test_unrelated_unknown_reason_still_not_auto_repaired(self):
        old,new,_=files()
        self.assertEqual(doctor.classify(doctor.mapping_from_source(new),
                         "Hawkar-usls/Janus:FAKE_HEALTH_APPROVAL"),"UNCLASSIFIED")
        self.assertEqual(doctor.classify(doctor.mapping_from_source(old),
                         "Hawkar-usls/Janus:FAKE_HEALTH_APPROVAL"),"UNCLASSIFIED")
        h=historical_health()
        h["observation_reasons"].append("Hawkar-usls/a:NEW_UNKNOWN")
        h["health_sha256"]=doctor.digest({k:v for k,v in h.items() if k!="health_sha256"})
        with self.assertRaisesRegex(doctor.RepairHold,"WRONG_HISTORICAL_COHORT"):
            doctor.diagnose(h,old)

    def test_fake_health_pass_and_invalid_signatures_are_rejected(self):
        old,_,_=files()
        for key,value in (("integrity_gate","HOLD"),
                          ("observation_gate","PASS"),
                          ("independent_admission_gate","PASS"),
                          ("execution_allowed",True),
                          ("snapshot_sha256","1"*64)):
            h=historical_health()
            h[key]=value
            h["health_sha256"]=doctor.digest({k:v for k,v in h.items() if k!="health_sha256"})
            with self.subTest(key=key),self.assertRaisesRegex(doctor.RepairHold,"HEALTH_TRUST_GATE"):
                doctor.diagnose(h,old)
        h=historical_health()
        h["observation_reasons"][0]="tampered"
        with self.assertRaisesRegex(doctor.RepairHold,"HEALTH_TRUST_GATE"):
            doctor.diagnose(h,old)

    def test_source_injection_and_wrong_base_sha_are_rejected(self):
        old,new,_=files()
        for source in (old.replace("CHECKS_STALE","HACKED",1),
                       old+"# trailing injected code\n",
                       new):
            with self.subTest(source=source[-15:]),self.assertRaises(doctor.RepairHold):
                doctor.construct_candidate(source)

    def test_exam_tampering_and_candidate_divergence_are_rejected(self):
        old,new,exam=files()
        bad=copy.deepcopy(exam)
        bad["cases"][0]["expected_action"]="FAKE_APPROVAL"
        result=doctor.evaluate_frozen(bad,old,new)
        self.assertFalse(result["candidate_review_ready"] if result["lost_parent_correct"] else result["candidate_correct"]==24)
        with self.assertRaisesRegex(doctor.RepairHold,"WORKTREE_DIFFERS"):
            doctor.propose(historical_health(),old,exam,new+"# unaudited change\n")

    def test_report_has_no_self_grant_or_effects(self):
        old,new,exam=files()
        report=doctor.propose(historical_health(),old,exam,new)
        self.assertEqual(report["status"],"FIRST_PARTY_DIAGNOSTIC_PATCH_PAIRED_PASS__REVIEW_ONLY")
        self.assertEqual(report["source_git_blob_sha1"],doctor.BASE_BLOB_SHA1)
        self.assertEqual(report["rollback"]["restore_source_git_blob_sha1"],doctor.BASE_BLOB_SHA1)
        for k in ("working_models_changed","current_49_repo_operational_health_proven",
                  "auto_merge","trusted_external_approval","production_activated",
                  "self_modifying_workflow_executed"):
            self.assertFalse(report["constraints"][k],k)
        self.assertEqual(report["constraints"]["external_forks_created"],0)

    def test_frozen_suite_commit_and_non_authoritative_partition(self):
        self.assertEqual(doctor.blob_sha(EXAM.read_bytes()),doctor.FROZEN_EXAM_BLOB_SHA1)
        _,_,exam=files()
        self.assertEqual(exam["case_count"],24)
        self.assertFalse(exam["external_custodian"])
        self.assertTrue(exam["never_train_on_this_suite"])
        self.assertTrue(all(x["authority"] is False for x in exam["cases"]))


if __name__=="__main__":
    unittest.main()
