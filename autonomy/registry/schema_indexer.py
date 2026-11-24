#!/usr/bin/env python3
"""JANUS R3 Meta Registry Schema Explorer: public-data JSON catalog, read-only.

This inventories JSON metadata across a pinned *public* Git checkout. It does
not select a live authority from a filename, follow URLs, execute content,
run generated code, edit the repo, train a model or promote source files.
Only paths, schema labels and limited structure of ALLOWLISTED data/*.json
are retained. Large/invalid/unknown data are individually recorded as HOLD.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

from autonomy.registry import navigator as core

SCHEMA="janus.registry.public_json_schema_catalog.v1"
MAX_FILES=6_000
MAX_TOTAL_BYTES=48_000_000
MAX_FILE_BYTES=100_000
MAX_CANDIDATE_PATHS=220
MAX_FAMILY_TOP=100
MAX_KEY_NAMES=12
SAFE_SCHEMA=re.compile(r"[a-zA-Z0-9_.:/-]{1,140}\Z")
JSON_REL=re.compile(r"data/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.json\Z")
DIGEST_RE=re.compile(r"[0-9a-f]{40}\Z")
NAME_RE=re.compile(r"[A-Za-z0-9_.-]{1,180}\Z")
ROLE_WORDS={
  "CURRENT_STATE":("CURRENT-STATE","CURRENT_STATE"),
  "JOURNAL":("JOURNAL","DEVELOPMENT-JOURNAL"),
  "INDEX":("INDEX","CATALOG"),
  "RECEIPT":("RECEIPT","/receipts/"),
}
SKIP_SUBTREES=("data/autonomy_loop/receipts/","data/agent-inbox/private/")


def safe_category(path):
    up=path.upper()
    for category,needles in ROLE_WORDS.items():
        if any(needle.upper() in up for needle in needles):
            return category
    return "OTHER_JSON"


def public_metadata(root, rel):
    p=root/rel
    if not JSON_REL.fullmatch(rel) or p.is_symlink() or not p.is_file():
        return "HOLD_NOT_REGULAR_PUBLIC_DATA_JSON",None
    size=p.stat().st_size
    if size<1 or size>MAX_FILE_BYTES:
        return "HOLD_PER_FILE_BYTES",None
    raw=p.read_bytes()
    if len(raw)!=size:
        return "HOLD_RACE_CHANGED_FILE",None
    try:
        data=core.strict_json(raw)
    except (core.RegistryHold,UnicodeError,ValueError):
        return "HOLD_INVALID_JSON",None
    if type(data) is not dict:
        return "HOLD_NOT_OBJECT",None
    schema=data.get("schema")
    if type(schema) is not str or not SAFE_SCHEMA.fullmatch(schema):
        schema="UNDECLARED_OR_UNRECOGNIZED_SCHEMA"
    key_names=sorted(x for x in data if type(x) is str and NAME_RE.fullmatch(x))
    info={"schema_label":schema,
          "top_level_key_count":len(data),
          "top_level_key_names":key_names[:MAX_KEY_NAMES],
          "content_sha256":hashlib.sha256(raw).hexdigest(),
          "size_bytes":size}
    return "PARSED_SCHEMA_NOT_AUTHORITY",info


def make_catalog(root, expected_commit, at=None, max_files=MAX_FILES):
    root=Path(root).resolve(strict=True)
    if not DIGEST_RE.fullmatch(str(expected_commit)) or expected_commit!=core.MAIN_COMMIT:
        raise core.RegistryHold("INDEXER_UNPINNED_PUBLIC_TREE")
    data_dir=root/"data"
    if not data_dir.is_dir() or data_dir.is_symlink():
        raise core.RegistryHold("INDEXER_NO_PUBLIC_DATA_DIR")
    paths=sorted(p for p in data_dir.rglob("*.json") if p.is_file() or p.is_symlink())
    total=len(paths)
    if total>max_files or max_files>MAX_FILES:
        raise core.RegistryHold("INDEXER_FILE_COUNT_BUDGET_HOLD")
    families=Counter()
    categories=Counter()
    statuses=Counter()
    role_candidates=[]
    used=0
    parsed=0
    for p in paths:
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            statuses["HOLD_UNSAFE_LINK"]+=1
            continue
        rel=p.relative_to(root).as_posix()
        if any(rel.startswith(z) for z in SKIP_SUBTREES):
            statuses["SKIPPED_UNTRUSTED_OR_BULK_AREA"]+=1
            continue
        if not JSON_REL.fullmatch(rel):
            statuses["HOLD_INVALID_PATH"]+=1
            continue
        size=p.stat().st_size
        if size>MAX_FILE_BYTES or size<1:
            statuses["HOLD_PER_FILE_BYTES"]+=1
            continue
        if used+size>MAX_TOTAL_BYTES:
            statuses["HOLD_GLOBAL_BYTE_BUDGET"]+=1
            continue
        used+=size
        status,info=public_metadata(root,rel)
        statuses[status]+=1
        if info is None:
            continue
        parsed+=1
        category=safe_category(rel)
        categories[category]+=1
        families[info["schema_label"]]+=1
        if category!="OTHER_JSON" and len(role_candidates)<MAX_CANDIDATE_PATHS:
            role_candidates.append({
                "path":rel,
                "role_hint":category,
                "schema":info["schema_label"],
                "content_sha256":info["content_sha256"],
                "size_bytes":info["size_bytes"],
                "observed_only_not_current_authority":True,
            })
    # This inventory is not a real-time watcher: HEAD and source are pinned.
    report={
        "schema":SCHEMA,
        "status":"PUBLIC_JSON_SCHEMAS_INDEXED__ALL_DISCOVERIES_NONAUTHORITATIVE",
        "source_repository":"Hawkar-usls/janus-meta-registry",
        "source_commit_sha":expected_commit,
        "catalog_generated_at_utc":(at or datetime.now(timezone.utc)).isoformat(),
        "discovered_data_json_paths":total,
        "readable_schema_object_json":parsed,
        "observed_bytes":used,
        "bounded_global_budget_bytes":MAX_TOTAL_BYTES,
        "file_size_limit_bytes":MAX_FILE_BYTES,
        "files_requiring_other_review":total-parsed,
        "parse_status_counts":dict(sorted(statuses.items())),
        "category_counts":dict(sorted(categories.items())),
        "distinct_declared_and_undeclared_schemas":len(families),
        "schema_family_counts_top":dict(sorted(families.items(),
                                   key=lambda x:(-x[1],x[0]))[:MAX_FAMILY_TOP]),
        "role_path_candidates":role_candidates,
        "candidate_paths_truncated":len(role_candidates)>=MAX_CANDIDATE_PATHS,
        "complete_metadata_only_inventory":True,
        "all_schema_files_trusted":False,
        "all_filenames_are_live_projects":False,
        "source_text_executed":False,
        "source_file_writes":False,
        "automatic_project_admission":False,
        "automatic_fork_or_module_birth":False,
        "license_or_external_review_inferred":False,
        "next_step":"INDEPENDENT_REVIEW_OF_EXPLICIT_ROUTE_CANDIDATES_AGAINST_LIVE_STATE_AND_JOURNAL",
    }
    report["receipt_sha256"]=core.sha256(report)
    return report


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--root",type=Path,required=True)
    a.add_argument("--source-commit",required=True)
    a.add_argument("--out",type=Path,required=True)
    x=a.parse_args()
    report=make_catalog(x.root,x.source_commit)
    core.demand(not x.out.exists() and not x.out.is_symlink(),
                "INDEXER_OUTPUT_ALREADY_PRESENT")
    x.out.parent.mkdir(parents=True,exist_ok=True)
    x.out.write_text(json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":report["status"],
       "paths":report["discovered_data_json_paths"],
       "parsed":report["readable_schema_object_json"],
       "families":report["distinct_declared_and_undeclared_schemas"],
       "errors_or_skips":report["files_requiring_other_review"],
       "receipt_sha256":report["receipt_sha256"]},sort_keys=True))


if __name__=="__main__":
    main()
