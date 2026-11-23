#!/usr/bin/env python3
"""JANUS organism-wide, read-only coherence witness (public Github only).

The canonical Janus organism manifest and its reviewed Simptomat delta define
admitted organs. A .janus link by itself does NOT grant organ authority.

This auditor is distinct from the ESSENCE microkernel fleet audit: it checks
typed organism admission, dormant kernels, and the *ordered* cross-organ
Rowan cohort that currently blocks the existing autonomy roadmap P0.2.

No cross-repository write, no workflow dispatch, no GitHub Issue/PR mutation,
no private repo discovery, no model execution, no production/checkpoint changes.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, build_opener, HTTPRedirectHandler

OWNER="Hawkar-usls"
ID="JANUS_FEDERATED_ORGANISM"
MAX_FILE_BYTES=150_000
NAME=re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
HEX40=re.compile(r"^[0-9a-f]{40}$")
SOURCE_FILES={
    "canonical":("janus-meta-registry","data/rowan/CANONICAL_STATE.json"),
    "aura":("aura-oracle-tg","research/rowan/LATEST_HYPOTHESES.json"),
    "council":("Janus-Demiurge","research/rowan/LATEST_COUNCIL.json"),
    "coordination":("janus-meta-registry","data/rowan/coordination/LATEST_COORDINATION_STATE.json"),
    "cosmos":("Janus-Cosmos","data/integrations/rowan/LATEST.json"),
}
REPAIR_ORDER=[
    ("AURA","aura-oracle-tg","aura-rowan-falsifier-bridge.yml"),
    ("COUNCIL","Janus-Demiurge","janus-rowan-aura-council.yml"),
    ("COORDINATION","janus-meta-registry","rowan-coordination-autonomous-reconcile.yml"),
    ("COSMOS","Janus-Cosmos","cosmos-rowan-ecosystem-witness.yml"),
]


def canonical(o):
    return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def sha256_obj(o):
    return hashlib.sha256(canonical(o)).hexdigest()


def git_sha1(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def topology(root):
    base=json.loads((Path(root)/"organism/JANUS_ORGANISM_v1.json").read_text())
    delta=json.loads((Path(root)/"organism/JANUS_ORGANISM_DELTA_2026-08-26_SIMPTOMAT_v1.json").read_text())
    if base.get("organism_id")!=ID or base.get("schema")!="janus.organism.v1":
        raise ValueError("ORGANISM_TOPOLOGY_UNTRUSTED")
    design=base.get("design",{})
    if design.get("cross_repo_write_default")!="DENY" or design.get("private_repo_names_in_public_manifest") is not False:
        raise ValueError("ORGANISM_TOPOLOGY_AUTHORITY_ESCAPE")
    members=dict(base["members"])
    private=[k for k,v in members.items() if v.get("class")=="PRIVATE_GATED"]
    if len(private)!=1 or any(members[k].get("repo") is not None or members[k].get("mcp_default") is not False for k in private):
        raise ValueError("ORGANISM_PRIVATE_PLACEHOLDER_LEAK")
    if (delta.get("schema")!="janus.organism.admission_delta.v1"
        or delta.get("organism_id")!=ID
        or delta.get("status")!="ACTIVE_ADMISSION_DELTA"
        or delta.get("applies_to",{}).get("observed_version")!=base["version"]):
        raise ValueError("ORGANISM_DELTA_NOT_ADMITTED")
    added=delta["admit_member"]
    key=added["organ_key"]
    if key in members or added.get("mcp_default") is not False or added.get("authority")!="RESEARCH_DIAGNOSTIC_HYPOTHESIS_ONLY":
        raise ValueError("ORGANISM_DELTA_AUTHORITY_ESCALATION")
    members[key]=added
    allowed={}
    for key,val in members.items():
        repo=val.get("repo")
        if repo is None:
            continue
        if not isinstance(repo,str) or not repo.startswith(OWNER+"/") or not NAME.fullmatch(repo.split("/",1)[1]):
            raise ValueError("ORGANISM_MEMBER_REPO_MALFORMED")
        if repo in allowed or val.get("class") not in ("CORE","DOMAIN_LAB","INTERFACE_PHYSICAL"):
            raise ValueError("ORGANISM_DUPLICATE_OR_INVALID_MEMBER")
        allowed[repo]={"organ_key":key,"class":val["class"],"authority":val.get("authority"),
                       "role":val.get("role")}
    keys=set(members)
    for flow in base.get("flows",[])+delta.get("add_flows",[]):
        if flow.get("from") not in keys or not isinstance(flow.get("to"),list) or any(k not in keys for k in flow["to"]) or flow.get("authority_delta")!=0:
            raise ValueError("ORGANISM_FLOW_IMPLICIT_PRIVILEGE")
    return {"organism_id":ID,"canonical_version":base["version"],
            "public_admitted":allowed,"private_slots":len(private),
            "canonical_slots":len(base["members"]),"delta_admissions":1,
            "flow_count":len(base.get("flows",[]))+len(delta.get("add_flows",[]))}


def evaluate_link(repo,link,admitted):
    if link is None:
        return "MISSING_DECLARATION" if admitted else "TOOL_OR_UNCLASSIFIED"
    if not isinstance(link,dict):
        return "INVALID_LINK"
    if (link.get("schema")!="janus.organism.member_link.v1"
        or link.get("organism_id")!=ID
        or link.get("member_repository")!=repo
        or link.get("cross_repo_write_default")!="DENY"
        or link.get("authority_delta")!=0):
        return "INVALID_LINK"
    if admitted:
        if (link.get("organ_key")!=admitted["organ_key"]
            or link.get("role")!=admitted["role"]
            or link.get("authority")!=admitted["authority"]):
            return "ADMITTED_LINK_CONTRACT_DRIFT"
        return "ADMITTED_LINK_MATCH"
    return "EXTRA_DECLARATION_NO_ADMISSION"


def next_cohort_action(state):
    """No snapshot can skip an earlier failed organism organ in the chain."""
    if not all(isinstance(state.get(x),dict) for x in SOURCE_FILES):
        missing=[k for k in SOURCE_FILES if not isinstance(state.get(k),dict)]
        return {"status":"HOLD_UPSTREAM_UNKNOWN","missing":missing,"repair":None,
                "full_autonomy_proven":False,"authority_delta":0}
    meta=state["canonical"]
    run=meta.get("latest_durable_external_run")
    count=meta.get("last_completed_mutants")
    source_sha=meta.get("underlying_state_is_derived_from")
    if (meta.get("schema_id")!="janus.rowan.observer_state.v1"
        or type(run) is not int or run<=0 or type(count) is not int
        or not 0<=count<=8
        or not isinstance(source_sha,str) or not re.fullmatch(r"[a-f0-9]{64}",source_sha)):
        return {"status":"HOLD_CANONICAL_INVALID","repair":None,
                "full_autonomy_proven":False,"authority_delta":0}
    aura=state["aura"]
    council=state["council"]
    coordination=state["coordination"]
    cosmos=state["cosmos"]
    ar=aura.get("source",{})
    cs=council.get("sources",{})
    matched={
        "AURA":ar.get("upstream_run")==run
                and ar.get("state_sha256")==sha256_obj(meta)
                and ar.get("upstream_artifact_sha256")==source_sha
                and ar.get("completed_mutants")==count,
        "COUNCIL":cs.get("meta_registry",{}).get("canonical_run")==run
                and cs.get("meta_registry",{}).get("sha256")==sha256_obj(meta)
                and cs.get("aura_oracle",{}).get("packet_sha256")==sha256_obj(aura),
        "COORDINATION":coordination.get("meta_canonical_run")==run,
        "COSMOS":cosmos.get("canonical_rowan_run")==run,
    }
    observed={
        "canonical":run,
        "aura":ar.get("upstream_run"),
        "council":cs.get("meta_registry",{}).get("canonical_run"),
        "coordination":coordination.get("meta_canonical_run"),
        "cosmos":cosmos.get("canonical_rowan_run"),
    }
    for key,repo,workflow in REPAIR_ORDER:
        if not matched[key]:
            return {"status":"HOLD_SOURCE_COHORT_DRIFT",
                    "canonical_run":run,"observed_runs":observed,
                    "matched":matched,
                    "repair":{"organ":key,"repo":OWNER+"/"+repo,"existing_workflow":workflow,
                              "action":"REPAIR_SOURCE_BOUND_OUTPUT_THEN_REVERIFY",
                              "automatically_dispatched":False},
                    "full_autonomy_proven":False,"authority_delta":0}
    return {"status":"MATCHED_COHORT_IDS_ONLY__INDEPENDENT_WITNESSES_REQUIRED",
            "canonical_run":run,"observed_runs":observed,"matched":matched,
            "repair":{"organ":"WITNESS","repo":OWNER+"/janus-meta-registry",
                      "existing_workflow":"janus-rowan-ecosystem-sync-witness.yml",
                      "action":"VERIFY_FULL_SOURCE_AND_WORKFLOW_WITNESS","automatically_dispatched":False},
            "full_autonomy_proven":False,"authority_delta":0}


class BlockRedirect(HTTPRedirectHandler):
    def redirect_request(self,*a,**kw):
        raise ValueError("ORGANISM_UNEXPECTED_REDIRECT")


class PublicGithub:
    def __init__(self,token=""):
        self.opener=build_opener(BlockRedirect())
        self.token=token
    def get(self,path):
        if not path.startswith("/") or "//" in path or ".." in path:
            raise ValueError("ORGANISM_URL_OUT_OF_SCOPE")
        headers={"Accept":"application/vnd.github+json",
                 "User-Agent":"JANUS-Organism-Coherence/1"}
        if self.token:
            headers["Authorization"]="Bearer "+self.token
        request=Request("https://api.github.com"+path,headers=headers)
        with self.opener.open(request,timeout=25) as response:
            data=response.read(2_000_001)
        if not 0<len(data)<=2_000_000:
            raise ValueError("ORGANISM_API_RESPONSE_UNBOUNDED")
        return json.loads(data)
    def public_repos(self):
        # Public user endpoint only: private repository names are never fetched
        # or emitted by this workflow, regardless of token privileges.
        repos=[]
        for page in (1,2):
            batch=self.get(f"/users/{OWNER}/repos?per_page=100&type=owner&page={page}")
            if not isinstance(batch,list) or len(batch)>100:
                raise ValueError("ORGANISM_PUBLIC_REPO_PAGE_INVALID")
            for x in batch:
                repo=x.get("full_name")
                if (x.get("private") is True or x.get("visibility") not in ("public",None)
                    or not isinstance(repo,str) or not repo.startswith(OWNER+"/")
                    or not NAME.fullmatch(repo.split("/",1)[1])):
                    raise ValueError("ORGANISM_INVALID_PUBLIC_REPO")
                branch=x.get("default_branch","main")
                if not isinstance(branch,str) or not re.fullmatch(r"[A-Za-z0-9_.\-/]{1,100}",branch) or ".." in branch:
                    raise ValueError("ORGANISM_UNSAFE_DEFAULT_BRANCH")
                repos.append({"repository":repo,"default_branch":branch})
            if len(batch)<100:
                break
        if not repos or len(repos)!=len({x["repository"] for x in repos}):
            raise ValueError("ORGANISM_PUBLIC_INVENTORY_UNTRUSTED")
        return sorted(repos,key=lambda x:x["repository"].lower())
    def pin(self,repo,branch):
        name=repo.split("/")[-1]
        if repo!=OWNER+"/"+name or not NAME.fullmatch(name):
            raise ValueError("ORGANISM_PIN_REPO_NOT_ALLOWED")
        ref=self.get(f"/repos/{OWNER}/{name}/git/ref/heads/{quote(branch,safe='/')}")
        sha=ref.get("object",{}).get("sha")
        if not isinstance(sha,str) or not HEX40.fullmatch(sha):
            raise ValueError("ORGANISM_INVALID_SOURCE_HEAD")
        return sha
    def source(self,repo,sha,path):
        if not isinstance(sha,str) or not HEX40.fullmatch(sha):
            raise ValueError("ORGANISM_UNPINNED_READ")
        try:
            obj=self.get("/repos/"+repo+"/contents/"+quote(path,safe="/")+"?ref="+sha)
        except HTTPError as exc:
            if exc.code==404:
                return None
            raise
        if obj.get("type")!="file" or obj.get("encoding")!="base64":
            raise ValueError("ORGANISM_SOURCE_NOT_FILE")
        raw=base64.b64decode(obj["content"].replace("\n",""),validate=True)
        if not 0<len(raw)<=MAX_FILE_BYTES or git_sha1(raw)!=obj.get("sha"):
            raise ValueError("ORGANISM_BLOB_INTEGRITY_FAILED")
        value=json.loads(raw)
        if not isinstance(value,dict):
            raise ValueError("ORGANISM_SOURCE_NOT_OBJECT")
        return value


def perform(root,api):
    topo=topology(root)
    public=api.public_repos()
    admitted=topo["public_admitted"]
    def inspect(entry):
        repo=entry["repository"]
        try:
            sha=api.pin(repo,entry["default_branch"])
            link=api.source(repo,sha,".janus/JANUS_ORGANISM_LINK.json")
            manifest=api.source(repo,sha,".janus/essence/manifest.json")
            link_status=evaluate_link(repo,link,admitted.get(repo))
            if manifest is None:
                micro="MISSING"
            elif manifest.get("schema")!="janus.essence.microkernel/0.1":
                micro="UNKNOWN_SCHEMA"
            elif manifest.get("enabled") is False:
                micro="PRESENT_DISABLED"
            elif manifest.get("enabled") is True:
                micro="DECLARED_ENABLED_UNVERIFIED_EXECUTION"
            else:
                micro="INVALID_ENABLED_VALUE"
            return {"repository":repo,"commit_sha":sha,"link":link_status,"microkernel":micro}
        except (OSError,ValueError,TypeError,KeyError) as exc:
            return {"repository":repo,"link":"UNKNOWN_SOURCE_ERROR",
                    "microkernel":"UNKNOWN","error_class":type(exc).__name__}
    with ThreadPoolExecutor(max_workers=6) as pool:
        members=sorted(list(pool.map(inspect,public)),key=lambda x:x["repository"].lower())
    known={x["repository"] for x in public}
    missing_expected=sorted(set(admitted)-known)
    sources={}
    for tag,(name,path) in SOURCE_FILES.items():
        repo=OWNER+"/"+name
        try:
            branch=next(x["default_branch"] for x in public if x["repository"]==repo)
            commit=api.pin(repo,branch)
            data=api.source(repo,commit,path)
            sources[tag]=data
        except (OSError,ValueError,TypeError,StopIteration,KeyError):
            sources[tag]=None
    cohort=next_cohort_action(sources)
    summary={
        "public_repository_count":len(public),
        "public_admitted_organ_count":len(admitted),
        "private_organs_redacted":topo["private_slots"],
        "admitted_link_matches":sum(x["link"]=="ADMITTED_LINK_MATCH" for x in members),
        "missing_admitted":sum(x["link"]=="MISSING_DECLARATION" for x in members),
        "extra_link_no_admission":sum(x["link"]=="EXTRA_DECLARATION_NO_ADMISSION" for x in members),
        "invalid_or_drifted_links":sum(x["link"] in ("INVALID_LINK","ADMITTED_LINK_CONTRACT_DRIFT") for x in members),
        "source_unknown_members":sum(x["link"]=="UNKNOWN_SOURCE_ERROR" for x in members),
        "disabled_microkernel_manifests":sum(x["microkernel"]=="PRESENT_DISABLED" for x in members),
        "missing_microkernel_manifests":sum(x["microkernel"]=="MISSING" for x in members),
        "declared_enabled_microkernel_unverified":sum(x["microkernel"]=="DECLARED_ENABLED_UNVERIFIED_EXECUTION" for x in members),
        "missing_public_repos_for_admitted_organs":missing_expected,
    }
    receipt={
        "schema":"janus.organism.coherence_witness.v1",
        "organism_id":ID,
        "observed_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
        "status":"PUBLIC_TOPOLOGY_OBSERVED__AUTONOMY_NOT_FULLY_ATTESTED",
        "source_authority":"CANONICAL_MANIFEST_PLUS_EXPLICIT_DELTA_ONLY",
        "constitution":"WRITE_BACK_DEFAULT_DENY__MODEL_OUTPUT_NOT_EVIDENCE",
        "topology":{"canonical_slots":topo["canonical_slots"],
                    "delta_admissions":topo["delta_admissions"],
                    "public_admitted":len(admitted),
                    "private_placeholder_count":topo["private_slots"],
                    "flow_count":topo["flow_count"]},
        "summary":summary,"public_repositories":members,
        "priority_recovery":cohort,
        "private_repository_names_disclosed":False,
        "same_owner_internal_observation":True,
        "cross_repository_dispatch_performed":False,
        "production_weights_changed":False,
        "model_training_performed":False,
        "autonomous_merging_enabled":False,
        "full_autonomy_proven":False,
        "external_independent_validation":False,
    }
    receipt["sha256"]=sha256_obj(receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=Path("."))
    opts=parser.parse_args()
    report=perform(opts.root,PublicGithub(os.getenv("GH_TOKEN","")))
    if opts.out.exists() or opts.out.is_symlink():
        raise SystemExit("ORGANISM_AUDIT_RECEIPT_ALREADY_EXISTS")
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    opts.out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"public_repositories":report["summary"]["public_repository_count"],
                      "admitted_public_organs":report["summary"]["public_admitted_organ_count"],
                      "matched":report["summary"]["admitted_link_matches"],
                      "cohort":report["priority_recovery"]["status"],
                      "next":report["priority_recovery"].get("repair"),
                      "sha256":report["sha256"]},sort_keys=True,ensure_ascii=False))


if __name__=="__main__":
    main()
