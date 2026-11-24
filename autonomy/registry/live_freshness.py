#!/usr/bin/env python3
"""JANUS R3: read-only GitHub Meta Registry HEAD/index freshness witness.

Only two public GETs, one fixed hostname and one fixed canonical index path.
The result can prevent use of old maps; it cannot authorize new maps, copy
untrusted source code, merge PRs, promote models or lift P0.3. On any error,
unknown is a HOLD, never a fabricated "unchanged" result.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import ssl
from urllib.request import Request,build_opener,HTTPRedirectHandler,HTTPSHandler
from urllib.error import HTTPError,URLError

from autonomy.registry import navigator as core

URL_MAIN="https://api.github.com/repos/Hawkar-usls/janus-meta-registry/branches/main"
URL_INDEX_PREFIX="https://api.github.com/repos/Hawkar-usls/janus-meta-registry/contents/data/JANUS-CHAT-CONTEXT-INDEX.json?ref="
MAX_RESPONSE=100_000
HEX40=re.compile(r"[0-9a-f]{40}\Z")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*_args,**_kwargs):
        raise core.RegistryHold("R3_FRESHNESS_REDIRECT_DENIED")


def get_json(url, token=""):
    if url!=URL_MAIN and not (url.startswith(URL_INDEX_PREFIX)
                             and HEX40.fullmatch(url[len(URL_INDEX_PREFIX):])):
        raise core.RegistryHold("R3_UNALLOWLISTED_LIVE_URL")
    request=Request(url,method="GET",headers={
        "Accept":"application/vnd.github+json",
        "User-Agent":"JANUS-R3-Registry-ReadOnly",
        "X-GitHub-Api-Version":"2022-11-28",
        **({"Authorization":"Bearer "+token} if token else {})
    })
    opener=build_opener(NoRedirect(),HTTPSHandler(context=ssl.create_default_context()))
    with opener.open(request,timeout=8) as reply:
        if reply.status!=200:
            raise core.RegistryHold("R3_FRESHNESS_HTTP_NOT_200")
        raw=reply.read(MAX_RESPONSE+1)
    if len(raw)>MAX_RESPONSE:
        raise core.RegistryHold("R3_FRESHNESS_RESPONSE_SIZE")
    return core.strict_json(raw)


def check(fetch=get_json, token="", at=None):
    head=None
    blob=None
    checked=0
    try:
        item=fetch(URL_MAIN,token)
        checked+=1
        head=(item.get("commit") or {}).get("sha") if type(item) is dict else None
        if type(head) is not str or HEX40.fullmatch(head) is None:
            raise core.RegistryHold("R3_FRESHNESS_INVALID_MAIN_SHA")
        item=fetch(URL_INDEX_PREFIX+head,token)
        checked+=1
        if (type(item) is not dict or item.get("type")!="file"
            or item.get("path")!=core.INDEX):
            raise core.RegistryHold("R3_FRESHNESS_CONTENTS_IDENTITY")
        blob=item.get("sha")
        if type(blob) is not str or HEX40.fullmatch(blob) is None:
            raise core.RegistryHold("R3_FRESHNESS_INVALID_BLOB_SHA")
        status=("SOURCE_HEAD_AND_INDEX_MATCH_FROZEN_PIN__OTHER_SERVICES_UNCHECKED"
                if head==core.MAIN_COMMIT and blob==core.INDEX_BLOB else
                "HOLD_MAIN_OR_BOOT_INDEX_DRIFT__REVALIDATION_REQUIRED")
        error_kind=None
    except (core.RegistryHold,HTTPError,URLError,OSError,ValueError,TypeError) as exc:
        status="HOLD_LIVE_METADATA_UNAVAILABLE"
        error_kind=type(exc).__name__
    report={
        "schema":"janus.registry.r3_live_readonly_freshness.v1",
        "status":status,
        "checked_at_utc":(at or datetime.now(timezone.utc)).isoformat(),
        "expected_main_commit_sha":core.MAIN_COMMIT,
        "observed_main_commit_sha":head,
        "expected_index_blob_sha1":core.INDEX_BLOB,
        "observed_index_blob_sha1":blob,
        "api_get_calls_completed":checked,
        "api_get_budget":2,
        "error_type":error_kind,
        "full_organism_current_state_validated":False,
        "meta_all_json_rescanned_at_observed_head":False,
        "main_index_may_be_automatically_rewritten":False,
        "checkpoints_updated":False,
        "model_weights_updated":False,
        "new_foreign_code_executed":False,
        "external_review_completed":False,
        "p03_admission_granted":False,
        "next_step":("REVALIDATE_NEW_META_COMMIT_AND_SCHEMAS_BEFORE_REFERENCING"
                     if status.startswith("HOLD_MAIN_OR_BOOT_INDEX_DRIFT") else
                     "RECHECK_GITHUB_ACCESS_OR_METADATA_BEFORE_ACTION"
                     if status.startswith("HOLD_LIVE") else
                     "CHECK_OTHER_CURRENT_STATES_AND_FRESH_CI_SEPARATELY")
    }
    report["receipt_sha256"]=core.sha256(report)
    return report


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    result=check(token=os.environ.get("GITHUB_TOKEN",""))
    core.demand(not args.out.exists() and not args.out.is_symlink(),
                "R3_FRESHNESS_OUTPUT_EXISTS")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],
                      "observed_main_commit_sha":result["observed_main_commit_sha"],
                      "observed_index_blob_sha1":result["observed_index_blob_sha1"],
                      "receipt_sha256":result["receipt_sha256"]},sort_keys=True))


if __name__=="__main__":
    main()
