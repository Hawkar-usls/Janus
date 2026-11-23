"""Inert dependency/organ proposals. There is deliberately no effect executor.

Receipts and identity strings are untrusted assertions. Structural review checks
do not authenticate people or establish scientific independence. An external
operator must verify evidence, persist/CAS the ledger against an anchored head,
and separately authorize any real fork, repository creation or activation.
The API never downloads, imports, installs or executes candidate content.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import PurePosixPath

OWNER = "Hawkar-usls"
ZERO = "0" * 64
HARD_LIMITS = {"max_candidates": 16, "max_proposals": 4, "max_depth": 1,
               "max_forks": 1, "max_new_repositories": 1,
               "max_artifact_bytes": 1048576, "max_files": 128,
               "max_events": 128}
LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}


class Hold(ValueError):
    """An incomplete or unsafe proposal is held without changing the ledger."""


def require(condition, reason):
    if not condition:
        raise Hold(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def fields(value, names):
    require(type(value) is dict and set(value) == set(names.split()), "FIELDS_MISMATCH")


def text(value, max_length=256):
    require(type(value) is str and 0 < len(value) <= max_length and
            not any(ord(c) < 32 for c in value), "INVALID_TEXT")
    return value


def sha(value, size=64):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(size) + "}", value), "INVALID_HASH")
    return value


def integer(value, maximum):
    require(type(value) is int and 0 <= value <= maximum, "INVALID_INTEGER")
    return value


def identity(value):
    require(type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", value), "INVALID_IDENTITY")
    return value


def repo(value, external=False):
    require(type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}", value), "INVALID_REPOSITORY")
    require(value.split("/")[1] not in {".", ".."}, "INVALID_REPOSITORY")
    if external:
        require(value.split("/")[0].lower() != OWNER.lower(), "NOT_EXTERNAL_UPSTREAM")
    else:
        require(value.split("/")[0] == OWNER, "WRONG_OWNER")
    return value


def safe_path(value):
    text(value)
    require("\\" not in value and ":" not in value and not value.startswith("/") and
            all(part not in {"", ".", "..", ".git"} for part in value.split("/")), "UNSAFE_PATH")
    return value


def validate_policy(policy):
    fields(policy, "schema mode kill_switch external_effects authority_delta allowed_licenses limits")
    require(policy["schema"] == "janus.autonomy.policy.v1" and policy["mode"] == "PROPOSALS_ONLY", "WRONG_POLICY")
    require(type(policy["kill_switch"]) is bool and policy["external_effects"] is False and
            type(policy["authority_delta"]) is int and policy["authority_delta"] == 0, "AUTHORITY_ESCALATION")
    licenses = policy["allowed_licenses"]
    require(type(licenses) is list and licenses and
            all(type(x) is str and x in LICENSES for x in licenses) and
            len(licenses) == len(set(licenses)), "LICENSE_POLICY")
    fields(policy["limits"], " ".join(HARD_LIMITS))
    for name, ceiling in HARD_LIMITS.items():
        integer(policy["limits"][name], ceiling)
    return policy


def _policy_digest(policy):
    # The emergency stop can change without rewriting a historical ledger.
    return digest({k: v for k, v in policy.items() if k != "kill_switch"})


def new_ledger(policy, existing_functions=()):
    validate_policy(policy)
    require(type(existing_functions) in {tuple, list} and len(existing_functions) <= 128,
            "EXISTING_FUNCTIONS_LIMIT")
    known = sorted(text(x, 100) for x in existing_functions)
    require(len(known) == len(set(known)), "DUPLICATE_FUNCTION")
    return {"schema": "janus.autonomy.admission_ledger.v1", "policy_sha256": _policy_digest(policy),
            "head": ZERO, "candidates": {}, "organs": {}, "existing_functions": known,
            "counters": {"proposals": 0, "forks": 0, "new_repositories": 0},
            "audit": [], "execution_allowed": False, "reviews_authenticated": False}


def _state_digest(ledger):
    return digest({k: v for k, v in ledger.items() if k not in {"audit", "head"}})


def verify_ledger(ledger, policy):
    validate_policy(policy)
    fields(ledger, "schema policy_sha256 head candidates organs existing_functions counters audit execution_allowed reviews_authenticated")
    require(ledger["schema"] == "janus.autonomy.admission_ledger.v1" and
            ledger["policy_sha256"] == _policy_digest(policy), "POLICY_DRIFT")
    require(ledger["execution_allowed"] is False and ledger["reviews_authenticated"] is False,
            "FORGED_AUTHORITY")
    require(type(ledger["audit"]) is list and len(ledger["audit"]) <= policy["limits"]["max_events"], "AUDIT_LIMIT")
    # Reconstruct the complete state from a pristine genesis. Hash-checking a
    # final snapshot alone would permit preloaded candidates to skip quarantine.
    replayed = new_ledger(policy, ledger["existing_functions"])
    replay_policy = copy.deepcopy(policy)
    replay_policy["kill_switch"] = False  # emergency stop is not retroactive
    for item in ledger["audit"]:
        fields(item, "sequence request_id payload_sha256 action subject previous_head state_sha256 event head")
        require(type(item["sequence"]) is int and item["sequence"] == len(replayed["audit"]), "AUDIT_SEQUENCE")
        replayed = _reduce(replayed, item["event"], expected_head=replayed["head"], policy=replay_policy)
        require(replayed["audit"][-1] == item, "AUDIT_REPLAY_MISMATCH")
    require(replayed == ledger, "STATE_REPLAY_MISMATCH")
    require(type(ledger["candidates"]) is dict and len(ledger["candidates"]) <= policy["limits"]["max_candidates"], "CANDIDATE_LIMIT")
    require(type(ledger["organs"]) is dict, "ORGAN_TYPE")
    fields(ledger["counters"], "proposals forks new_repositories")
    for counter, limit in (("proposals", "max_proposals"), ("forks", "max_forks"),
                           ("new_repositories", "max_new_repositories")):
        integer(ledger["counters"][counter], policy["limits"][limit])
    return True


def validate_candidate(candidate, policy):
    fields(candidate, "id repo source_url branch branch_is_default commit_sha license source_provenance bundle proposer purpose")
    require(re.fullmatch(r"[a-z][a-z0-9-]{0,63}", text(candidate["id"], 64)), "INVALID_ID")
    upstream = repo(candidate["repo"], external=True)
    require(candidate["source_url"] == "https://github.com/" + upstream, "NONCANONICAL_SOURCE")
    branch = text(candidate["branch"], 100)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", branch) and ".." not in branch and
            not branch.endswith(("/", ".", ".lock")) and "//" not in branch, "UNSAFE_BRANCH")
    require(candidate["branch_is_default"] is True, "DEFAULT_BRANCH_UNVERIFIED")
    commit = sha(candidate["commit_sha"], 40)
    fields(candidate["license"], "spdx path blob_sha sha256")
    require(candidate["license"]["spdx"] in policy["allowed_licenses"], "LICENSE_HOLD")
    safe_path(candidate["license"]["path"]); sha(candidate["license"]["blob_sha"], 40)
    sha(candidate["license"]["sha256"])
    fields(candidate["source_provenance"], "commit_url tree_sha")
    require(candidate["source_provenance"]["commit_url"] == candidate["source_url"] + "/commit/" + commit, "SOURCE_COMMIT_MISMATCH")
    sha(candidate["source_provenance"]["tree_sha"], 40)
    fields(candidate["bundle"], "sha256 bytes file_count")
    sha(candidate["bundle"]["sha256"])
    integer(candidate["bundle"]["bytes"], policy["limits"]["max_artifact_bytes"])
    require(0 < integer(candidate["bundle"]["file_count"], policy["limits"]["max_files"]), "EMPTY_BUNDLE")
    identity(candidate["proposer"]); text(candidate["purpose"], 500)


def bundle_manifest(files, policy):
    """Hash only bounded supplied bytes; never extract archives or follow paths."""
    validate_policy(policy)
    require(type(files) is dict and 0 < len(files) <= policy["limits"]["max_files"] and
            all(type(name) is str for name in files), "FILE_LIMIT")
    rows, total = [], 0
    for name, content in sorted(files.items()):
        safe_path(name)
        require(type(content) is bytes, "BYTES_REQUIRED")
        total += len(content)
        require(total <= policy["limits"]["max_artifact_bytes"], "BUNDLE_TOO_LARGE")
        rows.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    return {"sha256": digest(rows), "bytes": total, "file_count": len(rows)}


def scan_bundle(files, candidate, policy):
    """Conservative static hints, not a security certification or malware verdict.

    A real trusted executor must re-fetch the exact commit, inspect all tree
    modes (including symlinks/submodules), independently rescan secrets/history
    and verify the license's legal applicability before approving any effect.
    """
    validate_candidate(candidate, policy)
    manifest = bundle_manifest(files, policy)
    require(manifest == candidate["bundle"], "BUNDLE_HASH_MISMATCH")
    license_path = candidate["license"]["path"]
    require(license_path in files and hashlib.sha256(files[license_path]).hexdigest() ==
            candidate["license"]["sha256"], "LICENSE_BYTES_MISMATCH")
    raw = files[license_path]
    git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    require(git_blob == candidate["license"]["blob_sha"], "LICENSE_GIT_BLOB_MISMATCH")
    license_signatures = {"MIT": (b"Permission is hereby granted", b"THE SOFTWARE IS PROVIDED"),
                          "Apache-2.0": (b"Apache License", b"Version 2.0"),
                          "BSD-2-Clause": (b"Redistribution and use", b"THIS SOFTWARE IS PROVIDED"),
                          "BSD-3-Clause": (b"Redistribution and use", b"THIS SOFTWARE IS PROVIDED",
                                           b"Neither the name")}
    require(all(part in raw for part in license_signatures[candidate["license"]["spdx"]]),
            "LICENSE_TEXT_REQUIRES_REVIEW")
    secrets, unsafe = [], []
    patterns = (rb"-----BEGIN [A-Z ]*PRIVATE KEY-----", rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
                rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b", rb"\bAKIA[A-Z0-9]{16}\b",
                rb"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*[\"']?[^\s\"']{12,}")
    for name, content in sorted(files.items()):
        if any(re.search(pattern, content) for pattern in patterns):
            secrets.append(name)
        lower = name.lower()
        if (lower.startswith((".github/workflows/", ".git/")) or
                PurePosixPath(lower).name in {"setup.py", ".gitmodules", ".gitattributes", ".npmrc"} or
                b"\x00" in content or lower.endswith((".exe", ".dll", ".so", ".ps1", ".sh"))):
            unsafe.append(name)
        if lower.endswith("package.json"):
            try:
                package = json.loads(content)
                if type(package) is not dict or package.get("scripts"):
                    unsafe.append(name)
            except (ValueError, UnicodeError):
                unsafe.append(name)
    return {"schema": "janus.autonomy.static_scan.v1", "candidate_sha256": digest(candidate),
            "bundle_sha256": manifest["sha256"], "commit_sha": candidate["commit_sha"],
            "secret_findings": sorted(set(secrets)), "unsafe_findings": sorted(set(unsafe)),
            "scope": "SUPPLIED_BYTES_ONLY_NOT_HISTORY_OR_COMPLETE_TREE", "executed": False,
            "security_certified": False, "clear": not secrets and not unsafe}


def _review(receipt, subject, proposer):
    fields(receipt, "subject_sha256 reviewer decision evidence_sha256 source authentication")
    require(receipt["subject_sha256"] == subject, "REVIEW_HASH_MISMATCH")
    require(identity(receipt["reviewer"]).lower() != proposer.lower(), "SELF_REVIEW")
    require(receipt["decision"] == "APPROVE_PROPOSAL_ONLY" and
            receipt["source"] == "EXTERNAL_REVIEW_RECEIPT" and
            receipt["authentication"] == "REQUIRES_OPERATOR_VERIFICATION", "REVIEW_SCOPE")
    sha(receipt["evidence_sha256"])


def _organ(spec, policy, ledger):
    fields(spec, "id kind parent_repo module_path function_id inputs outputs proposer source_sha256 depth resources test_receipt health_receipt training_data_policy rollback review")
    require(re.fullmatch(r"[a-z][a-z0-9-]{0,63}", text(spec["id"], 64)), "INVALID_ID")
    require(type(spec["kind"]) is str and spec["kind"] in {"module", "repository"}, "ORGAN_KIND")
    repo(spec["parent_repo"]); safe_path(spec["module_path"])
    require(spec["module_path"].startswith("modules/"), "MODULE_PATH_SCOPE")
    function = text(spec["function_id"], 100)
    require(function not in ledger["existing_functions"] and
            all(x["spec"]["function_id"] != function for x in ledger["organs"].values()), "DUPLICATE_FUNCTION")
    for interface in (spec["inputs"], spec["outputs"]):
        require(type(interface) is list and 0 < len(interface) <= 8 and
                all(type(x) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", x) for x in interface), "UNTYPED_INTERFACE")
    proposer = identity(spec["proposer"]); source = sha(spec["source_sha256"])
    integer(spec["depth"], policy["limits"]["max_depth"])
    fields(spec["resources"], "cpu_seconds memory_mib disk_mib")
    for name, ceiling in (("cpu_seconds", 60), ("memory_mib", 256), ("disk_mib", 100)):
        require(0 < integer(spec["resources"][name], ceiling), "RESOURCE_BUDGET")
    for name in ("test_receipt", "health_receipt"):
        receipt = spec[name]
        fields(receipt, "source_sha256 result verifier evidence_sha256")
        require(receipt["source_sha256"] == source and receipt["result"] == "PASS" and
                identity(receipt["verifier"]).lower() != proposer.lower(), "INDEPENDENT_GATE_REQUIRED")
        sha(receipt["evidence_sha256"])
    require(spec["test_receipt"]["verifier"].lower() != spec["health_receipt"]["verifier"].lower(),
            "INDEPENDENT_HEALTH_ROLE_REQUIRED")
    require(spec["training_data_policy"] == "TRAIN_ONLY_NO_EXAM_ACCESS", "EXAM_FIREWALL")
    fields(spec["rollback"], "operation parent_commit_sha")
    require(spec["rollback"]["operation"] == "REVERT_DRAFT_PR", "ROLLBACK_REQUIRED")
    sha(spec["rollback"]["parent_commit_sha"], 40)
    review_subject = digest({k: v for k, v in spec.items() if k != "review"})
    _review(spec["review"], review_subject, proposer)
    require(spec["review"]["reviewer"].lower() not in
            {spec["test_receipt"]["verifier"].lower(), spec["health_receipt"]["verifier"].lower()},
            "INDEPENDENT_REVIEW_ROLE_REQUIRED")


def transition(ledger, event, *, expected_head, policy):
    """Return a new ledger; caller owns durable atomic storage and head anchoring.

    Supported actions: catalog, quarantine, review, fork_proposal,
    organ_proposal, revoke, rollback. No state grants execution authority.
    """
    verify_ledger(ledger, policy)
    fields(event, "request_id action subject payload")
    normalized = copy.deepcopy(event)
    if event["action"] == "quarantine":
        require(event["subject"] in ledger["candidates"], "UNKNOWN_SUBJECT")
        normalized["payload"] = scan_bundle(event["payload"],
                                              ledger["candidates"][event["subject"]]["candidate"], policy)
    result = _reduce(ledger, normalized, expected_head=expected_head, policy=policy)
    verify_ledger(result, policy)
    return result


def _scan_receipt(scan, candidate):
    fields(scan, "schema candidate_sha256 bundle_sha256 commit_sha secret_findings unsafe_findings scope executed security_certified clear")
    require(scan["schema"] == "janus.autonomy.static_scan.v1" and
            scan["candidate_sha256"] == digest(candidate) and
            scan["bundle_sha256"] == candidate["bundle"]["sha256"] and
            scan["commit_sha"] == candidate["commit_sha"], "SCAN_SOURCE_MISMATCH")
    require(scan["scope"] == "SUPPLIED_BYTES_ONLY_NOT_HISTORY_OR_COMPLETE_TREE" and
            scan["executed"] is False and scan["security_certified"] is False, "SCAN_SCOPE")
    for name in ("secret_findings", "unsafe_findings"):
        rows = scan[name]
        require(type(rows) is list and len(rows) <= 128 and all(type(x) is str for x in rows), "SCAN_FINDINGS")
        require(rows == sorted(set(rows)), "SCAN_FINDINGS")
        for path in rows:
            safe_path(path)
    require(type(scan["clear"]) is bool and
            scan["clear"] == (not scan["secret_findings"] and not scan["unsafe_findings"]), "SCAN_CLEAR_MISMATCH")


def _reduce(ledger, event, *, expected_head, policy):
    """Internal deterministic reducer over normalized, non-executable events."""
    fields(event, "request_id action subject payload")
    request_id = text(event["request_id"], 100)
    subject = text(event["subject"], 64)
    payload_hash = digest(event)
    for prior in ledger["audit"]:
        if prior["request_id"] == request_id:
            require(prior["payload_sha256"] == payload_hash, "IDEMPOTENCY_CONFLICT")
            return copy.deepcopy(ledger)
    require(expected_head == ledger["head"], "HEAD_CONFLICT")
    action, payload = event["action"], event["payload"]
    require(type(action) is str and action in {"catalog", "quarantine", "review", "fork_proposal",
                                              "organ_proposal", "revoke", "rollback"}, "UNKNOWN_ACTION")
    require(not policy["kill_switch"] or action in {"revoke", "rollback"}, "KILL_SWITCH")
    require(len(ledger["audit"]) < policy["limits"]["max_events"], "AUDIT_LIMIT")
    result = copy.deepcopy(ledger)
    if action == "catalog":
        validate_candidate(payload, policy)
        require(subject == payload["id"] and subject not in result["candidates"] and
                subject not in result["organs"], "DUPLICATE_ID")
        require(len(result["candidates"]) < policy["limits"]["max_candidates"], "CANDIDATE_LIMIT")
        require(all((x["candidate"]["repo"].lower(), x["candidate"]["commit_sha"]) !=
                    (payload["repo"].lower(), payload["commit_sha"]) for x in result["candidates"].values()), "DUPLICATE_SOURCE")
        result["candidates"][subject] = {"state": "CATALOG", "candidate": copy.deepcopy(payload),
                                         "execution_allowed": False}
    elif action == "organ_proposal":
        _organ(payload, policy, result)
        require(subject == payload["id"] and subject not in result["organs"] and
                subject not in result["candidates"], "DUPLICATE_ID")
        _spend(result, policy, "proposals", "max_proposals")
        if payload["kind"] == "repository":
            _spend(result, policy, "new_repositories", "max_new_repositories")
        result["organs"][subject] = {"state": "PROPOSED_NOT_ACTIVE", "spec": copy.deepcopy(payload),
                                     "activation_allowed": False, "execution_allowed": False}
    elif action in {"revoke", "rollback"}:
        fields(payload, "reason")
        text(payload["reason"], 500)
        require(subject in result["candidates"] or subject in result["organs"], "UNKNOWN_SUBJECT")
        row = result["candidates"].get(subject, result["organs"].get(subject))
        require(row["state"] not in {"REVOKED", "ROLLED_BACK_PROPOSAL"}, "TERMINAL_STATE")
        row["state"] = "REVOKED" if action == "revoke" else "ROLLED_BACK_PROPOSAL"
        row["reason"] = payload["reason"]
    else:
        require(subject in result["candidates"], "UNKNOWN_SUBJECT")
        row = result["candidates"][subject]
        candidate = row["candidate"]
        if action == "quarantine":
            require(row["state"] == "CATALOG", "ILLEGAL_TRANSITION")
            _scan_receipt(payload, candidate)
            row.update(state="QUARANTINED", scan=copy.deepcopy(payload))
        elif action == "review":
            require(row["state"] == "QUARANTINED" and row["scan"]["clear"] is True, "QUARANTINE_HOLD")
            _review(payload, digest(candidate), candidate["proposer"])
            row.update(state="REVIEW_ASSERTED_UNVERIFIED", review=copy.deepcopy(payload))
        elif action == "fork_proposal":
            fields(payload, "destination_owner destination_name source_commit_sha")
            require(row["state"] == "REVIEW_ASSERTED_UNVERIFIED", "REVIEW_REQUIRED")
            require(payload["destination_owner"] == OWNER and
                    re.fullmatch(r"[a-z][a-z0-9-]{0,63}", text(payload["destination_name"], 64)), "FORK_DESTINATION")
            require(payload["source_commit_sha"] == candidate["commit_sha"], "SOURCE_DRIFT")
            _spend(result, policy, "forks", "max_forks")
            _spend(result, policy, "proposals", "max_proposals")
            row.update(state="FORK_PROPOSED", fork=copy.deepcopy(payload),
                       operator_verification_required=True,
                       executor_preconditions=["AUTHENTICATE_EXTERNAL_REVIEW", "REFETCH_EXACT_COMMIT_AND_DEFAULT_BRANCH",
                                               "VERIFY_LICENSE_AND_COMPLETE_TREE_AND_HISTORY_SCANS",
                                               "NO_ACTIONS_OR_HOOKS_OR_CODE_EXECUTION", "VERIFY_POST_FORK_COMMIT",
                                               "SEPARATE_SCOPED_OPERATOR_AUTHORIZATION"])
        else:
            raise Hold("UNKNOWN_ACTION")
    audit = {"sequence": len(result["audit"]), "request_id": request_id,
             "payload_sha256": payload_hash, "action": action, "subject": subject,
             "previous_head": result["head"], "state_sha256": _state_digest(result),
             "event": copy.deepcopy(event)}
    audit["head"] = digest(audit)
    result["audit"].append(audit)
    result["head"] = audit["head"]
    return result


def _spend(ledger, policy, counter, limit):
    require(ledger["counters"][counter] < policy["limits"][limit], "BUDGET_EXHAUSTED_" + counter.upper())
    ledger["counters"][counter] += 1
