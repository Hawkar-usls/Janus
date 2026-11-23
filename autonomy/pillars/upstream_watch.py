"""GitHub public-metadata-only watcher for the 13 pinned external pillars.

Does NOT download archives/files, clone/fork/install modules, execute source
or accept upstream changes as authority. HEAD/metadata drift becomes HOLD.
Only fixed public api.github.com/ repos and branches from a validated catalog.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import ssl
import urllib.request
from urllib.error import HTTPError, URLError

from autonomy.pillars import engine

SCHEMA="janus.autonomy.pillars.upstream_watch.v1"
MAX_RESPONSE=100_000
MAX_REQUESTS=26
API="https://api.github.com"
OWNER_REPO=re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}\Z")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*_args,**_kwargs):
        raise ValueError("PILLARS_NO_REDIRECT")



def strict_decode(data):
    def unique(pairs):
        out={}
        for key,val in pairs:
            if key in out:
                raise ValueError("PILLARS_DUPLICATE_GITHUB_JSON")
            out[key]=val
        return out
    return json.loads(data,object_pairs_hook=unique,
        parse_constant=lambda _v: (_ for _ in ()).throw(ValueError("PILLARS_NONFINITE")))

def github_get(path, token=""):
    if not isinstance(path,str) or not re.fullmatch(
       r"/repos/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+(?:/branches/(?:main|master))?\Z",path):
        raise ValueError("PILLARS_UNTRUSTED_API_PATH")
    # Even a syntactically valid GitHub repository path must be in the
    # validated public source list. This prevents arbitrary private probes
    # if another caller ever invokes github_get directly.
    catalog=engine.read_json(engine.CATALOG)
    engine.validate_catalog(catalog)
    allowed={"/repos/"+p["repo"] for p in catalog["pillars"]}
    allowed.update("/repos/"+p["repo"]+"/branches/"+p["default_branch"]
                   for p in catalog["pillars"])
    if path not in allowed:
        raise ValueError("PILLARS_SOURCE_NOT_IN_ALLOWLIST")
    request=urllib.request.Request(API+path,headers={
      "Accept":"application/vnd.github+json",
      "User-Agent":"JANUS-Pillars-R2-ReadOnly",
      "X-GitHub-Api-Version":"2022-11-28",
      **({"Authorization":"Bearer "+token} if token else {})
    },method="GET")
    opener=urllib.request.build_opener(
        NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    with opener.open(request,timeout=5) as res:
        if res.status!=200 or res.headers.get("Content-Type","").split(";")[0]!="application/json":
            raise ValueError("PILLARS_PUBLIC_API_UNEXPECTED_RESPONSE")
        data=res.read(MAX_RESPONSE+1)
        if len(data)>MAX_RESPONSE:
            raise ValueError("PILLARS_METADATA_OVERSIZED")
        return strict_decode(data)


def compare(p, repo, branch):
    if not isinstance(repo,dict) or not isinstance(branch,dict):
        raise ValueError("PILLARS_BAD_PUBLIC_RESPONSE")
    name=repo.get("full_name")
    license=(repo.get("license") or {}).get("spdx_id")
    branch_name=repo.get("default_branch")
    head=(branch.get("commit") or {}).get("sha")
    checks={
      "repository_identity":name==p["repo"],
      "default_branch_identity":branch_name==p["default_branch"],
      "head_unchanged":head==p["commit_sha"],
      "license_metadata_unchanged":license==p["license_metadata_spdx"],
      "not_archived":repo.get("archived") is False,
      "not_disabled":repo.get("disabled") is False,
      "public_repository":repo.get("private") is False,
    }
    return {
      "pillar_id":p["id"],
      "repo":p["repo"],
      "status":"PIN_MATCH_METADATA_ONLY__CODE_IMPORT_STILL_HOLD"
               if all(checks.values()) else "HOLD_METADATA_DRIFT_OR_UNVERIFIED",
      "checks":checks,
      "frozen_commit_sha":p["commit_sha"],
      "observed_commit_sha":head if isinstance(head,str) and engine.SHA40.fullmatch(head) else None,
      "frozen_license_spdx":p["license_metadata_spdx"],
      "observed_license_spdx":license if isinstance(license,str) and len(license)<=64 else None,
      "license_bytes_verified":False,
      "source_code_executed":False,
      "auto_fork":False,
    }


def watch(catalog,fetch=github_get,token=""):
    entries=engine.validate_catalog(catalog)
    results=[]
    requests_used=0
    for p in catalog["pillars"]:
        path="/repos/"+p["repo"]
        # Sequential bounded API requests. Per-source failures hold only
        # that pillar; no speculative success and no fallback to HTML.
        try:
            repo=fetch(path,token);requests_used+=1
            branch=fetch(path+"/branches/"+p["default_branch"],token);requests_used+=1
            results.append(compare(p,repo,branch))
        except (ValueError,KeyError,TypeError,HTTPError,URLError,OSError) as exc:
            results.append({
                "pillar_id":p["id"],"repo":p["repo"],
                "status":"HOLD_PUBLIC_SOURCE_UNAVAILABLE",
                "checks":None,
                "frozen_commit_sha":p["commit_sha"],
                "observed_commit_sha":None,
                "frozen_license_spdx":p["license_metadata_spdx"],
                "observed_license_spdx":None,
                "license_bytes_verified":False,
                "source_code_executed":False,
                "auto_fork":False,
                "error_kind":type(exc).__name__,
            })
    if requests_used>MAX_REQUESTS or len(results)!=13:
        raise ValueError("PILLARS_REQUEST_BUDGET_EXCEEDED")
    match=sum(r["status"].startswith("PIN_MATCH") for r in results)
    result={
      "schema":SCHEMA,
      "status":"PUBLIC_PILLARS_METADATA_PIN_AUDIT_ONLY__NO_IMPORT",
      "observed_at_utc":datetime.now(timezone.utc).isoformat(),
      "catalog_sha256":engine.sha(catalog),
      "public_repositories_checked":len(results),
      "pin_matches":match,
      "held_for_review":len(results)-match,
      "requests_used":requests_used,
      "untrusted_source_was_executed":False,
      "license_bytes_verified":False,
      "forks_created":0,
      "auto_promoted":False,
      "results":results,
    }
    result["receipt_sha256"]=engine.sha(result)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",type=Path,default=engine.CATALOG)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    catalog=engine.read_json(args.catalog)
    import os
    result=watch(catalog,token=os.environ.get("GITHUB_TOKEN",""))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():
        raise ValueError("PILLARS_OVERWRITE_DENIED")
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"matched":result["pin_matches"],
     "held":result["held_for_review"],"requests":result["requests_used"]}))


if __name__=="__main__":
    main()
