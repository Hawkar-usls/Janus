"""Adversarial contract tests; no network, models, exam data or fork API."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from autonomy import admission as a

ROOT = Path(__file__).resolve().parents[1]
LICENSE = (b"MIT License\nPermission is hereby granted, free of charge, to any person.\n"
           b"THE SOFTWARE IS PROVIDED AS IS.\n")


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "autonomy/policy.json").read_text())
        self.files = {"LICENSE": LICENSE, "src/example.py": b"def add(a, b): return a + b\n"}
        self.candidate = {
            "id": "bounded-math", "repo": "example-author/bounded-math",
            "source_url": "https://github.com/example-author/bounded-math",
            "branch": "main", "branch_is_default": True, "commit_sha": "a" * 40,
            "license": {"spdx": "MIT", "path": "LICENSE",
                        "blob_sha": hashlib.sha1(b"blob " + str(len(LICENSE)).encode() + b"\0" + LICENSE).hexdigest(),
                        "sha256": hashlib.sha256(LICENSE).hexdigest()},
            "source_provenance": {"commit_url": "https://github.com/example-author/bounded-math/commit/" + "a" * 40,
                                  "tree_sha": "b" * 40},
            "bundle": a.bundle_manifest(self.files, self.policy), "proposer": "JANUS",
            "purpose": "A bounded arithmetic helper proposal"}
        self.ledger = a.new_ledger(self.policy, ["existing-reader"])
        self.seq = 0

    def event(self, action, payload, subject="bounded-math"):
        self.seq += 1
        return {"request_id": "request-" + str(self.seq), "action": action,
                "subject": subject, "payload": payload}

    def apply(self, action, payload, subject="bounded-math"):
        self.ledger = a.transition(self.ledger, self.event(action, payload, subject),
                                   expected_head=self.ledger["head"], policy=self.policy)
        return self.ledger

    def review(self, subject):
        return {"subject_sha256": subject, "reviewer": "ExternalReviewer",
                "decision": "APPROVE_PROPOSAL_ONLY", "evidence_sha256": "c" * 64,
                "source": "EXTERNAL_REVIEW_RECEIPT",
                "authentication": "REQUIRES_OPERATOR_VERIFICATION"}

    def reviewed(self):
        self.apply("catalog", self.candidate)
        self.apply("quarantine", self.files)
        self.apply("review", self.review(a.digest(self.candidate)))

    def fork_payload(self):
        return {"destination_owner": "Hawkar-usls", "destination_name": "quarantined-math",
                "source_commit_sha": "a" * 40}

    def organ(self, kind="module"):
        receipt = {"source_sha256": "d" * 64, "result": "PASS", "verifier": "ExternalTestVerifier",
                   "evidence_sha256": "e" * 64}
        spec = {"id": "diagnostic-helper", "kind": kind, "parent_repo": "Hawkar-usls/Janus",
                "module_path": "modules/diagnostic_helper", "function_id": "bounded-diagnostic-helper",
                "inputs": ["PUBLIC_METADATA"], "outputs": ["DIAGNOSTIC_PROPOSAL"],
                "proposer": "JANUS", "source_sha256": "d" * 64, "depth": 1,
                "resources": {"cpu_seconds": 30, "memory_mib": 128, "disk_mib": 10},
                "test_receipt": copy.deepcopy(receipt), "health_receipt": copy.deepcopy(receipt),
                "training_data_policy": "TRAIN_ONLY_NO_EXAM_ACCESS",
                "rollback": {"operation": "REVERT_DRAFT_PR", "parent_commit_sha": "f" * 40}}
        spec["health_receipt"]["verifier"] = "ExternalHealthVerifier"
        spec["review"] = self.review(a.digest(spec))
        return spec

    def test_full_intake_remains_inert_and_explicitly_unauthenticated(self):
        self.reviewed()
        self.apply("fork_proposal", self.fork_payload())
        row = self.ledger["candidates"]["bounded-math"]
        self.assertEqual(row["state"], "FORK_PROPOSED")
        self.assertFalse(row["execution_allowed"])
        self.assertFalse(self.ledger["reviews_authenticated"])
        self.assertTrue(row["operator_verification_required"])
        self.assertEqual(self.ledger["counters"], {"proposals": 1, "forks": 1, "new_repositories": 0})
        self.assertEqual(self.ledger["candidates"]["bounded-math"]["scan"]["scope"],
                         "SUPPLIED_BYTES_ONLY_NOT_HISTORY_OR_COMPLETE_TREE")
        self.assertNotIn("def add", json.dumps(self.ledger))

    def test_mutation_does_not_change_input(self):
        previous = copy.deepcopy(self.ledger)
        self.apply("catalog", self.candidate)
        self.assertEqual(previous["head"], a.ZERO)
        self.assertEqual(previous["candidates"], {})

    def test_quarantine_cannot_be_skipped(self):
        self.apply("catalog", self.candidate)
        with self.assertRaises(a.Hold):
            self.apply("fork_proposal", self.fork_payload())
        with self.assertRaises(a.Hold):
            self.apply("review", self.review(a.digest(self.candidate)))

    def test_default_branch_source_hash_license_source_constraints(self):
        mutations = [("branch_is_default", False), ("commit_sha", "main"),
                     ("source_url", "https://evil.invalid/source"), ("branch", "../main"),
                     ("repo", "Hawkar-usls/already-owned"), ("permissions", ["write"])]
        for key, value in mutations:
            candidate = copy.deepcopy(self.candidate); candidate[key] = value
            with self.subTest(key=key), self.assertRaises(a.Hold):
                self.apply("catalog", candidate)
        candidate = copy.deepcopy(self.candidate); candidate["license"]["spdx"] = "UNKNOWN"
        with self.assertRaises(a.Hold): self.apply("catalog", candidate)

    def test_license_hash_and_text_are_bound(self):
        self.apply("catalog", self.candidate)
        changed = dict(self.files); changed["LICENSE"] = b"unlicensed"
        with self.assertRaises(a.Hold): self.apply("quarantine", changed)
        candidate = copy.deepcopy(self.candidate); candidate["license"]["blob_sha"] = "0" * 40
        with self.assertRaises(a.Hold): a.scan_bundle(self.files, candidate, self.policy)
        changed["LICENSE"] = b"This file contains no license grant"
        candidate["bundle"] = a.bundle_manifest(changed, self.policy)
        raw = changed["LICENSE"]
        candidate["license"].update(sha256=hashlib.sha256(raw).hexdigest(),
                                    blob_sha=hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest())
        with self.assertRaises(a.Hold): a.scan_bundle(changed, candidate, self.policy)

    def test_secret_findings_hold_without_recording_secret_bytes(self):
        self.files["src/example.py"] = b"token = 'ghp_" + b"A" * 30 + b"'"
        self.candidate["bundle"] = a.bundle_manifest(self.files, self.policy)
        self.apply("catalog", self.candidate); self.apply("quarantine", self.files)
        self.assertFalse(self.ledger["candidates"]["bounded-math"]["scan"]["clear"])
        self.assertNotIn("ghp_", json.dumps(self.ledger))
        with self.assertRaises(a.Hold): self.apply("review", self.review(a.digest(self.candidate)))

    def test_executable_or_lifecycle_content_is_quarantined(self):
        for path, content in ((".github/workflows/ci.yml", b"on: push"),
                              ("setup.py", b"print('install')"), ("run.sh", b"echo hi"),
                              ("package.json", b'{"scripts":{"postinstall":"bad"}}'),
                              ("artifact.bin", b"\x00\x01")):
            files = dict(self.files); files[path] = content
            candidate = copy.deepcopy(self.candidate); candidate["bundle"] = a.bundle_manifest(files, self.policy)
            with self.subTest(path=path):
                scan = a.scan_bundle(files, candidate, self.policy)
                self.assertFalse(scan["clear"])
                self.assertIn(path, scan["unsafe_findings"])

    def test_path_traversal_and_bounded_bytes(self):
        for path in ("../escape", "/absolute", "a/../escape", "a\\b", "c:drive", ".git/config"):
            with self.subTest(path=path), self.assertRaises(a.Hold):
                a.bundle_manifest({path: b"data"}, self.policy)
        with self.assertRaises(a.Hold):
            a.bundle_manifest({"large": b"x" * (1048576 + 1)}, self.policy)

    def test_self_review_and_stale_review_rejected(self):
        self.apply("catalog", self.candidate); self.apply("quarantine", self.files)
        review = self.review(a.digest(self.candidate)); review["reviewer"] = "janus"
        with self.assertRaises(a.Hold): self.apply("review", review)
        review = self.review("0" * 64)
        with self.assertRaises(a.Hold): self.apply("review", review)

    def test_destination_and_source_cannot_drift(self):
        self.reviewed()
        for key, value in (("destination_owner", "other-owner"), ("source_commit_sha", "b" * 40),
                           ("destination_name", "../unsafe")):
            payload = self.fork_payload(); payload[key] = value
            with self.subTest(key=key), self.assertRaises(a.Hold): self.apply("fork_proposal", payload)

    def test_compare_and_swap_and_idempotency(self):
        event = self.event("catalog", self.candidate)
        first = a.transition(self.ledger, event, expected_head=a.ZERO, policy=self.policy)
        self.assertEqual(a.transition(first, event, expected_head=a.ZERO, policy=self.policy), first)
        event2 = self.event("review", self.review(a.digest(self.candidate)))
        with self.assertRaises(a.Hold): a.transition(first, event2, expected_head=a.ZERO, policy=self.policy)
        event["payload"]["purpose"] = "changed"
        with self.assertRaises(a.Hold): a.transition(first, event, expected_head=a.ZERO, policy=self.policy)

    def test_kill_switch_blocks_forward_work_but_allows_revocation(self):
        self.apply("catalog", self.candidate)
        self.policy["kill_switch"] = True
        with self.assertRaises(a.Hold): self.apply("quarantine", self.files)
        self.apply("revoke", {"reason": "Emergency operator stop"})
        self.assertEqual(self.ledger["candidates"]["bounded-math"]["state"], "REVOKED")
        with self.assertRaises(a.Hold): self.apply("revoke", {"reason": "Do not consume audit on terminal proposals"})

    def test_audit_and_state_tamper_rejected(self):
        self.reviewed()
        for target in ("audit", "state", "authority"):
            ledger = copy.deepcopy(self.ledger)
            if target == "audit": ledger["audit"][0]["previous_head"] = "f" * 64
            elif target == "state": ledger["candidates"]["bounded-math"]["state"] = "ACTIVE"
            else: ledger["execution_allowed"] = True
            with self.subTest(target=target), self.assertRaises(a.Hold): a.verify_ledger(ledger, self.policy)

    def test_preloaded_zero_head_cannot_skip_catalog_or_quarantine(self):
        forged = copy.deepcopy(self.ledger)
        forged["candidates"]["bounded-math"] = {
            "state": "REVIEW_ASSERTED_UNVERIFIED", "candidate": self.candidate,
            "execution_allowed": False}
        with self.assertRaises(a.Hold): a.verify_ledger(forged, self.policy)
        event = self.event("fork_proposal", self.fork_payload())
        with self.assertRaises(a.Hold):
            a.transition(forged, event, expected_head=a.ZERO, policy=self.policy)
        # Even a self-consistent hash of a skipped transition must fail replay.
        fabricated = a._reduce(forged, event, expected_head=a.ZERO, policy=self.policy)
        with self.assertRaises(a.Hold): a.verify_ledger(fabricated, self.policy)

    def test_audit_replay_rejects_reordered_or_removed_required_transition(self):
        self.reviewed(); self.apply("fork_proposal", self.fork_payload())
        forged = copy.deepcopy(self.ledger)
        forged["audit"].pop(1)
        # Rehashing a forged sequence cannot invent the missing quarantine.
        previous = a.ZERO
        for i, row in enumerate(forged["audit"]):
            row["sequence"] = i; row["previous_head"] = previous
            row["head"] = a.digest({k: v for k, v in row.items() if k != "head"})
            previous = row["head"]
        forged["head"] = previous
        with self.assertRaises(a.Hold): a.verify_ledger(forged, self.policy)

    def test_duplicate_source_and_cross_type_identifier_rejected(self):
        self.apply("catalog", self.candidate)
        duplicate = copy.deepcopy(self.candidate); duplicate["id"] = "alias"
        with self.assertRaises(a.Hold): self.apply("catalog", duplicate, "alias")
        organ = self.organ(); organ["id"] = "bounded-math"
        organ["review"] = self.review(a.digest({k: v for k, v in organ.items() if k != "review"}))
        with self.assertRaises(a.Hold): self.apply("organ_proposal", organ)

    def test_module_birth_is_proposal_only_and_repository_budget_is_zero(self):
        spec = self.organ(); self.apply("organ_proposal", spec, spec["id"])
        self.assertEqual(self.ledger["organs"][spec["id"]]["state"], "PROPOSED_NOT_ACTIVE")
        self.assertFalse(self.ledger["organs"][spec["id"]]["activation_allowed"])
        self.assertFalse(self.ledger["execution_allowed"])
        self.ledger = a.new_ledger(self.policy)
        spec = self.organ("repository")
        with self.assertRaises(a.Hold): self.apply("organ_proposal", spec, spec["id"])

    def test_existing_function_exam_access_self_gate_and_depth_block_birth(self):
        for key, value in (("function_id", "existing-reader"), ("depth", 2),
                           ("training_data_policy", "EXAM_TUNING"), ("permissions", {"write": True})):
            spec = self.organ(); spec[key] = value
            spec["review"] = self.review(a.digest({k: v for k, v in spec.items() if k != "review"}))
            with self.subTest(key=key), self.assertRaises(a.Hold): self.apply("organ_proposal", spec, spec["id"])
        spec = self.organ(); spec["health_receipt"]["verifier"] = "JANUS"
        spec["review"] = self.review(a.digest({k: v for k, v in spec.items() if k != "review"}))
        with self.assertRaises(a.Hold): self.apply("organ_proposal", spec, spec["id"])

    def test_birth_test_health_and_review_are_separate_unverified_roles(self):
        for role in ("health_receipt", "review"):
            spec = self.organ()
            if role == "health_receipt":
                spec["health_receipt"]["verifier"] = spec["test_receipt"]["verifier"]
            else:
                spec["review"]["reviewer"] = spec["health_receipt"]["verifier"]
            subject = a.digest({k: v for k, v in spec.items() if k != "review"})
            spec["review"]["subject_sha256"] = subject
            with self.subTest(role=role), self.assertRaises(a.Hold):
                self.apply("organ_proposal", spec, spec["id"])

    def test_integer_bool_nan_and_policy_escalation_rejected(self):
        for value in (True, -1, 17, 1.5, float("nan")):
            policy = copy.deepcopy(self.policy); policy["limits"]["max_candidates"] = value
            with self.subTest(value=value), self.assertRaises(a.Hold): a.validate_policy(policy)
        for key, value in (("external_effects", True), ("authority_delta", True),
                           ("mode", "AUTONOMOUS_EXECUTION"), ("allowed_licenses", [["MIT"]])):
            policy = copy.deepcopy(self.policy); policy[key] = value
            with self.subTest(key=key), self.assertRaises(a.Hold): a.validate_policy(policy)

    def test_proposal_budget_is_lifetime_and_rollback_never_refunds(self):
        self.policy["limits"]["max_proposals"] = 1
        self.ledger = a.new_ledger(self.policy)
        spec = self.organ(); self.apply("organ_proposal", spec, spec["id"])
        self.apply("rollback", {"reason": "Withdraw draft proposal"}, spec["id"])
        self.assertEqual(self.ledger["counters"]["proposals"], 1)
        other = self.organ(); other["id"] = "second"; other["function_id"] = "distinct-second"
        other["review"] = self.review(a.digest({k: v for k, v in other.items() if k != "review"}))
        with self.assertRaises(a.Hold): self.apply("organ_proposal", other, other["id"])


if __name__ == "__main__":
    unittest.main()
