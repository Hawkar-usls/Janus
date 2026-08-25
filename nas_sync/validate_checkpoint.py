#!/usr/bin/env python3
"""Fail-closed validator for sanitized JANUS NAS Git checkpoints.

Stdlib only. This validates the bounded public receipt surface; it does not
perform Git operations, deploy code, touch Docker, or grant runtime authority.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "janus.nas.checkpoint.v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
HOST = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
REPO = re.compile(r"^Hawkar-usls/[A-Za-z0-9_.-]+$")
ALLOWED_SYNC = {"PASS", "PARTIAL", "BLOCKED"}
ALLOWED_REPO = {
    "IN_SYNC", "FAST_FORWARDED", "LOCAL_AHEAD", "DIRTY_LOCAL",
    "DIVERGED", "SYNC_BLOCKED", "NOT_PRESENT",
}
ALLOWED_TESTS = {"PASS", "FAIL", "NOT_RUN", "NOT_AVAILABLE"}
FORBIDDEN_KEY_FRAGMENTS = {
    "token", "secret", "password", "passwd", "credential", "cookie",
    "session", "private_ip", "absolute_path", "env_value", "telegram",
    "database", "sqlite", "wal", "runtime_log",
}
SECRET_VALUE_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"\b(?:10|127|169\.254|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
]


def fail(msg: str) -> None:
    raise ValueError(msg)


def walk(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            low = str(key).lower()
            if any(fragment in low for fragment in FORBIDDEN_KEY_FRAGMENTS):
                fail(f"forbidden key at {path}.{key}")
            walk(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            walk(value, f"{path}[{i}]")
    elif isinstance(obj, str):
        for pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(obj):
                fail(f"secret/private-looking value at {path}")
        if obj.startswith(("/share/", "/home/", "/root/", "/mnt/", "C:\\", "D:\\")):
            fail(f"absolute local path at {path}")


def iso_utc(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None
    except ValueError:
        return False


def validate(payload: Any) -> None:
    if not isinstance(payload, dict):
        fail("root must be an object")
    walk(payload)
    required = {"schema", "timestamp_utc", "host_id", "canonical_janus_sha", "sync_status", "repos"}
    missing = required - payload.keys()
    if missing:
        fail(f"missing keys: {sorted(missing)}")
    allowed_root = required | {"summary"}
    extra = set(payload) - allowed_root
    if extra:
        fail(f"unexpected root keys: {sorted(extra)}")
    if payload["schema"] != SCHEMA:
        fail("wrong schema")
    if not iso_utc(payload["timestamp_utc"]):
        fail("timestamp_utc must be timezone-aware ISO-8601")
    if not isinstance(payload["host_id"], str) or not HOST.fullmatch(payload["host_id"]):
        fail("host_id must be a sanitized opaque identifier")
    if not isinstance(payload["canonical_janus_sha"], str) or not SHA40.fullmatch(payload["canonical_janus_sha"]):
        fail("canonical_janus_sha must be a 40-char lowercase SHA-1 commit id")
    if payload["sync_status"] not in ALLOWED_SYNC:
        fail("invalid sync_status")
    repos = payload["repos"]
    if not isinstance(repos, list) or len(repos) > 256:
        fail("repos must be a list with at most 256 entries")
    for i, item in enumerate(repos):
        if not isinstance(item, dict):
            fail(f"repos[{i}] must be an object")
        req = {"repo", "branch", "local_head", "remote_head", "status"}
        if req - item.keys():
            fail(f"repos[{i}] missing required keys")
        if set(item) - (req | {"tests"}):
            fail(f"repos[{i}] has unexpected keys")
        if not isinstance(item["repo"], str) or not REPO.fullmatch(item["repo"]):
            fail(f"repos[{i}].repo outside Hawkar-usls namespace")
        if not isinstance(item["branch"], str) or not (1 <= len(item["branch"]) <= 255):
            fail(f"repos[{i}].branch invalid")
        for field in ("local_head", "remote_head"):
            val = item[field]
            if val is not None and (not isinstance(val, str) or not SHA40.fullmatch(val)):
                fail(f"repos[{i}].{field} invalid")
        if item["status"] not in ALLOWED_REPO:
            fail(f"repos[{i}].status invalid")
        if "tests" in item and item["tests"] not in ALLOWED_TESTS:
            fail(f"repos[{i}].tests invalid")
    summary = payload.get("summary")
    if summary is not None:
        if not isinstance(summary, dict):
            fail("summary must be an object")
        allowed = {"repos_in_sync", "repos_dirty", "repos_diverged", "repos_blocked"}
        if set(summary) - allowed:
            fail("summary has unexpected keys")
        for key, value in summary.items():
            if not isinstance(value, int) or value < 0:
                fail(f"summary.{key} must be a non-negative integer")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    failed = False
    for raw in args.paths:
        path = Path(raw)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate(payload)
            print(f"PASS {path}")
        except Exception as exc:
            failed = True
            print(f"FAIL {path}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
