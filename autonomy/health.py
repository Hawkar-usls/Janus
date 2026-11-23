"""Independent, offline observation verifier. Never imports the collector.

PASS only describes metadata observations. It never grants execution/admission.
The trusted root must be supplied by the operator/CI, never by a report field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

OWNER = "Hawkar-usls"
SOURCE_PATHS = {
    "organism/JANUS_ORGANISM_v1.json",
    "organism/JANUS_REPOSITORY_ECOLOGY_v1.3.json",
    "organism/JANUS_REPOSITORY_ECOLOGY_v1.4.json",
    "organism/JANUS_ORGANISM_DELTA_2026-08-26_SIMPTOMAT_v1.json",
    "organism/JANUS_ORGANISM_ROUTE_REFINEMENT_2026-08-26_SIMPTOMAT_v1.json",
}
SHA = re.compile(r"[0-9a-f]{40}")
REPO = re.compile(r"Hawkar-usls/[A-Za-z0-9_.-]+")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def instant(value):
    if not isinstance(value, str):
        raise ValueError("timestamp missing")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone missing")
    return result


def verify(report, *, trusted_root, expected_source_commit, now=None, max_age_hours=24):
    """Return separate integrity, observation, operational and admission gates."""
    now = now or datetime.now(timezone.utc)
    integrity, observations = [], []
    def require(ok, message):
        if not ok:
            integrity.append(message)
    require(type(report) is dict, "REPORT_NOT_OBJECT")
    if type(report) is not dict:
        report = {}
    require(report.get("schema") == "janus.autonomy.inventory.v1", "SCHEMA")
    require(report.get("owner") == OWNER, "OWNER")
    require(report.get("action_authority") is False, "AUTHORITY")
    require(type(expected_source_commit) is str and bool(SHA.fullmatch(expected_source_commit)) and
            expected_source_commit != "0" * 40, "EXPECTED_SOURCE_COMMIT")
    try:
        require(report.get("snapshot_sha256") == digest({k: v for k, v in report.items()
                                                        if k != "snapshot_sha256"}), "DIGEST")
        age = (now - instant(report.get("observed_at"))).total_seconds()
        require(-300 <= age <= max_age_hours * 3600, "SNAPSHOT_STALE_OR_FUTURE")
    except (ValueError, TypeError, OverflowError):
        integrity.append("INVALID_DIGEST_OR_TIME")
    sources = report.get("sources", [])
    require(type(sources) is list and len(sources) == len(SOURCE_PATHS), "SOURCE_SET")
    seen = set()
    for source in sources if type(sources) is list else []:
        if type(source) is not dict:
            integrity.append("SOURCE_TYPE")
            continue
        path = source.get("path")
        require(type(path) is str and path in SOURCE_PATHS and path not in seen, "SOURCE_PATH")
        if type(path) is not str or path not in SOURCE_PATHS:
            continue
        seen.add(path)
        require(source.get("repository") == OWNER + "/Janus", "SOURCE_REPOSITORY")
        require(bool(SHA.fullmatch(str(source.get("commit_sha", "")))), "SOURCE_COMMIT")
        require(source.get("commit_sha") == expected_source_commit, "SOURCE_COMMIT_BINDING")
        try:
            raw = (Path(trusted_root) / path).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == source.get("sha256"), "SOURCE_BYTES:" + path)
        except OSError:
            integrity.append("TRUSTED_SOURCE_MISSING:" + path)
    require(seen == SOURCE_PATHS, "SOURCE_COVERAGE")
    complete = report.get("completeness", {})
    require(type(complete) is dict, "COMPLETENESS_TYPE")
    if type(complete) is not dict:
        complete = {}
    if complete.get("owner_inventory") is not True:
        observations.append("OWNER_INVENTORY_INCOMPLETE")
    if complete.get("repository_observations") is not True or complete.get("errors") != []:
        observations.append("REPOSITORY_OBSERVATIONS_INCOMPLETE")
    declared = report.get("declared_map", {})
    require(type(declared) is dict, "DECLARED_MAP_TYPE")
    if type(declared) is not dict:
        declared = {}
    members = declared.get("members", {}) if type(declared) is dict else {}
    require(type(members) is dict and bool(members), "MEMBERS")
    if type(members) is not dict:
        members = {}
    # Reconstruct canonical membership from trusted bytes, not collector labels.
    expected, expected_classes = {}, {}
    try:
        base = json.loads((Path(trusted_root) / "organism/JANUS_ORGANISM_v1.json").read_text(encoding="utf-8"))
        delta = json.loads((Path(trusted_root) / "organism/JANUS_ORGANISM_DELTA_2026-08-26_SIMPTOMAT_v1.json").read_text(encoding="utf-8"))
        refinement = json.loads((Path(trusted_root) / "organism/JANUS_ORGANISM_ROUTE_REFINEMENT_2026-08-26_SIMPTOMAT_v1.json").read_text(encoding="utf-8"))
        ecology = json.loads((Path(trusted_root) / "organism/JANUS_REPOSITORY_ECOLOGY_v1.3.json").read_text(encoding="utf-8"))
        expected_members = dict(base["members"])
        expected_members[delta["admit_member"]["organ_key"]] = delta["admit_member"]
        projected_members = {key: {"repository": value.get("repo"), "class": value["class"],
                                   "authority": value["authority"]} for key, value in expected_members.items()}
        require(members == projected_members, "MEMBER_CLASS_OR_AUTHORITY")
        expected_private = [{"slot": key, "class": value["class"]} for key, value in expected_members.items()
                            if value.get("repo") is None]
        expected_private += [{"slot": key, "class": value["class"]} for key, value in ecology["private_subtissues"].items()]
        require(declared.get("private_slots") == expected_private, "PRIVATE_SLOT_PROJECTION")
        for category in ("public_subtissues", "legacy_archives", "toolchain_dependencies", "external_research_instruments"):
            for item in ecology.get(category, {}).values():
                expected_classes[item["repo"]] = item["class"]
        profile = ecology["profile_navigation"]
        expected_classes[profile["repo"]] = profile["class"]
        expected_classes.update({value["repo"]: "ORGAN" for value in expected_members.values() if value.get("repo")})
        require(declared.get("classifications") == expected_classes, "ECOLOGY_CLASSIFICATIONS")
        expected = {key: value.get("repo") for key, value in base["members"].items()}
        admitted = delta["admit_member"]
        expected[admitted["organ_key"]] = admitted["repo"]
        require(declared.get("flows") == base["flows"] + delta["add_flows"], "CANONICAL_FLOW_COVERAGE")
        require(declared.get("refined_routes") == refinement["effective_routes"], "REFINED_ROUTE_GATES")
        require(declared.get("runtime_integration_verified") is False, "RUNTIME_OVERCLAIM")
        require(set(members) == set(expected), "DECLARED_MEMBER_SET")
        for key, repo in expected.items():
            require(type(members.get(key)) is dict and members[key].get("repository") == repo,
                    "DECLARED_MEMBER_BINDING:" + key)
    except (OSError, ValueError, KeyError, TypeError):
        integrity.append("TRUSTED_MAP_UNAVAILABLE")
    flows = declared.get("flows", [])
    require(type(flows) is list, "FLOWS_TYPE")
    for flow in flows if type(flows) is list else []:
        if type(flow) is not dict:
            integrity.append("FLOW_TYPE")
            continue
        require(type(flow.get("authority_delta")) is int and flow.get("authority_delta") == 0, "FLOW_AUTHORITY")
        targets = flow.get("to", [])
        require(type(targets) is list and type(flow.get("from")) is str and flow.get("from") in members and
                all(type(t) is str and t in members for t in targets), "FLOW_ENDPOINT")
    rows = report.get("repositories", [])
    require(type(rows) is list and 0 < len(rows) <= 100, "REPOSITORY_LIMIT")
    names = set()
    expected_organs = {repo: key for key, repo in expected.items() if repo}
    for row in rows if type(rows) is list else []:
        if type(row) is not dict:
            integrity.append("ROW_TYPE")
            continue
        repo = row.get("repository", "")
        require(type(repo) is str and bool(REPO.fullmatch(repo)) and repo not in names, "REPOSITORY_ID")
        if type(repo) is not str:
            continue
        names.add(repo)
        require(row.get("visibility") == "public", "PRIVATE_DISCLOSURE")
        require(row.get("discovery_state") in ("DECLARED", "CATALOG_ONLY"), "DISCOVERY_AUTHORITY")
        require(row.get("declared_class") == expected_classes.get(repo), "ROW_CLASS_BINDING:" + repo)
        require(row.get("discovery_state") == ("DECLARED" if repo in expected_classes else "CATALOG_ONLY"),
                "ROW_DISCOVERY_BINDING:" + repo)
        require(row.get("declared_organ") == expected_organs.get(repo), "ROW_ORGAN_BINDING:" + repo)
        if row.get("discovery_state") == "CATALOG_ONLY":
            require(not row.get("declared_organ"), "AUTO_PROMOTION")
        head = row.get("head_sha")
        if not isinstance(head, str) or not SHA.fullmatch(head) or head == "0" * 40:
            observations.append(repo + ":HEAD_UNKNOWN")
        # Runtime health cannot be inferred from an arbitrary successful workflow.
        if repo not in expected_organs:
            continue
        link = row.get("member_link", {})
        if type(link) is not dict:
            link = {}
        if (link.get("state") != "PRESENT" or link.get("commit_sha") != head or link.get("repository") != repo or
                link.get("organ_key") != expected_organs[repo] or
                link.get("schema") != "janus.organism.member_link.v1" or
                type(link.get("authority_delta")) is not int or link.get("authority_delta") != 0 or
                link.get("member_write_authority") is not False or link.get("cross_repo_write_default") != "DENY"):
            observations.append(repo + ":MEMBER_LINK_UNVERIFIED")
        actions = row.get("actions", {})
        if type(actions) is not dict:
            actions = {}
        if actions.get("state") != "OBSERVED" or actions.get("head_sha") != head:
            observations.append(repo + ":LATEST_ACTION_NOT_AT_HEAD")
        elif actions.get("status") != "completed" or actions.get("conclusion") != "success":
            observations.append(repo + ":LATEST_ACTION_NOT_SUCCESS")
        else:
            try:
                age = (now - instant(actions.get("updated_at"))).total_seconds()
                if not -300 <= age <= max_age_hours * 3600:
                    observations.append(repo + ":LATEST_ACTION_STALE")
            except (ValueError, TypeError):
                observations.append(repo + ":ACTION_TIME_UNKNOWN")
        checks = row.get("checks", {})
        if type(checks) is not dict:
            checks = {}
        runs = checks.get("runs")
        if (checks.get("state") != "OBSERVED" or checks.get("head_sha") != head or
                type(runs) is not list or not runs or len(runs) > 100):
            observations.append(repo + ":CHECKS_UNKNOWN")
        else:
            for check in runs:
                if type(check) is not dict or check.get("status") != "completed" or check.get("conclusion") != "success":
                    observations.append(repo + ":CHECKS_NOT_ALL_SUCCESS")
                    continue
                try:
                    age = (now - instant(check.get("completed_at"))).total_seconds()
                    if not -300 <= age <= max_age_hours * 3600:
                        observations.append(repo + ":CHECKS_STALE")
                except (ValueError, TypeError):
                    observations.append(repo + ":CHECK_TIME_UNKNOWN")
    require(type(complete.get("repository_count")) is int and complete.get("repository_count") == len(names), "REPOSITORY_COUNT")
    for key, member in members.items():
        if type(member) is dict and member.get("repository") and member["repository"] not in names:
            observations.append("DECLARED_REPOSITORY_MISSING:" + key)
    result = {"schema": "janus.autonomy.health.v1", "snapshot_sha256": report.get("snapshot_sha256"),
              "verified_at": now.isoformat(), "integrity_gate": "PASS" if not integrity else "HOLD",
              "observation_gate": "PASS" if not integrity and not observations else "HOLD",
              "integrity_reasons": sorted(set(integrity)), "observation_reasons": sorted(set(observations)),
              "operational_gate": "NOT_ASSESSED_REQUIRES_SERVICE_SPECIFIC_PROBES",
              "independent_admission_gate": "HOLD_EXTERNAL_REVIEW_REQUIRED",
              "execution_allowed": False, "full_autonomy_proven": False, "authority_delta": 0}
    result["health_sha256"] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--trusted-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--require-observation-pass", action="store_true")
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    if len(raw) > 4_000_000:
        raise SystemExit("SNAPSHOT_TOO_LARGE")
    result = verify(json.loads(raw), trusted_root=args.trusted_root,
                    expected_source_commit=args.expected_source_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("integrity_gate", "observation_gate", "independent_admission_gate")}))
    if result["integrity_gate"] != "PASS" or (args.require_observation_pass and result["observation_gate"] != "PASS"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
