#!/usr/bin/env python3
"""JANUS SCOBY & placenta ECM: two individually preregistered Rowan Boltz2 pilots.

One case and one action PER invocation; never dispatch a batch. Real source
sequences and pre-registered target selection are fixed before any prediction.
No culture, donor material, pathogen work or medicinal inference. Credentials
only via GitHub 'rowan-research' environment and never included in receipts.

Source: JANUS Meta Registry
data/rowan/protocols/ROWAN-SCOBY-KOMBUCHA-PLACENTAL-ECM-TWO-PROTEIN-
PILOT-PREREG-2026-10-26-v1.json
"""
from __future__ import annotations
import os,hashlib,importlib,json,math
from pathlib import Path
from budget_gate import finite_nonnegative
from probe import TARGET_ROWAN_FOLDER_UUID

TARGETS={
 "scoby_acsD":{
  "name":"JANUS-SCOBY-ACSD-P37719-20261026-R0",
  "uniprot":"P37719",
  "source":"https://rest.uniprot.org/uniprotkb/P37719.fasta",
  "sequence":"MTIFEKKPDFTLFLQTLSWEIDDQVGIEVRNELLREVGRGMGTRIMPPPCQTVDKLQIELNALLALIGWGTVTLELLSEDQSLRIVHENLPQVGSAGEPSGTWLAPVLEGLYGRWVTSQAGAFGDYVVTRDVDAEDLNAVPRQTIIMYMRVRSSAT",
  "sha256":"a7fa1a2d359ae60b7f88b4fefadf6502c9b578e3edb543ae01bf4ecca044adf9",
  "length":156,
  "scientific_limit":"single-chain confidence does not reproduce crystallographic octamer 3A8E or cellulose export"
 },
 "placenta_fn1":{
  "name":"JANUS-PLACENTA-ECM-FN1-TYPEII-355-463-20261026-R0",
  "uniprot":"P02751",
  "source":"https://rest.uniprot.org/uniprotkb/P02751.fasta",
  "sequence":"SNGEPCVLPFTYNGRTFYSCTTEGRQDGHLWCSTTSNYEQDQKYSFCTDHTVLVQTRGGNSNGALCHFPFLYNNHNYTDCTSEGRRDNMKWCGTTQNYDADQKFGFCPM",
  "sha256":"d8658e68f40439d7021df84a16c84fa7102a3cc939bd3a96b288b2944985ae0e",
  "length":109,
  "scientific_limit":"isolated FN1 two type-II repeats do not model placenta, native full FN1, BC binding or biocompatibility"
 }
}
CAP=10
WEEKLY_MIN=30
PAGE_SIZE=100
MAX_PAGES=15

class Hold(ValueError):pass
def sh(s):return hashlib.sha256(s.encode("ascii")).hexdigest()
def source(case):
 if case not in TARGETS:raise Hold("UNKNOWN_OR_NOT_PREREGISTERED_CASE")
 t=TARGETS[case]
 if sh(t["sequence"])!=t["sha256"] or len(t["sequence"])!=t["length"]:
  raise Hold("SOURCE_SHA_LENGTH_DRIFT")
 if not t["name"].startswith("JANUS-"):
  raise Hold("CASE_IDENTITY_DRIFT")
 return t
def check_weekly(api):
 u=api.whoami()
 if getattr(u,"individual_subscription",None) is not None:
  raise Hold("SUBSCRIPTION_STATUS_REVIEW")
 org=getattr(u,"organization",None)
 if org is None:raise Hold("NO_ORGANIZATION_WEEKLY_WALLET")
 w=finite_nonnegative(getattr(org,"weekly_credits",None))
 if w is None or w<WEEKLY_MIN:raise Hold("ORG_WEEKLY_BELOW_30_OR_UNKNOWN")
 return "ORG_WEEKLY_AVAILABLE_GE_30_AT_SUBMIT"
def existing(api,name):
 f=api.retrieve_folder(TARGET_ROWAN_FOLDER_UUID)
 if str(getattr(f,"uuid",None))!=TARGET_ROWAN_FOLDER_UUID:raise Hold("ROWAN_FOLDER_ID_DRIFT")
 matches=[]
 for page in range(MAX_PAGES):
  batch=api.list_workflows(parent_uuid=TARGET_ROWAN_FOLDER_UUID,page=page,size=PAGE_SIZE)
  if len(batch)>PAGE_SIZE:raise Hold("PROVIDER_PAGE_OVERFLOW")
  for w in batch:
   if str(getattr(w,"name",None))==name:
    if (str(getattr(w,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID
        or str(getattr(w,"workflow_type",None))!="protein_cofolding"):
     raise Hold("IDENTITY_CONFLICT_WRONG_PROVIDER_WORKFLOW")
    matches.append(w)
  if len(batch)<PAGE_SIZE:break
 else:raise Hold("WORKFLOW_PAGINATION_INCOMPLETE")
 if len(matches)>1:raise Hold("DUPLICATE_PROVIDER_CASES")
 return matches[0] if matches else None
def collect(workflow,name):
 try:
  w=workflow.fetch_latest()
  if str(getattr(w,"name",None))!=name or str(getattr(w,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
   raise Hold("COLLECT_SOURCE_PROVENANCE_DRIFT")
  if not w.done():return {"result_state":"PENDING_OR_IN_PROGRESS","additional_jobs":0}
  result=w.result(wait=False);scores=getattr(result,"scores",None)
  ptm=finite_nonnegative(getattr(scores,"ptm",None))
  lddt=finite_nonnegative(getattr(scores,"avg_lddt",None))
  credits=finite_nonnegative(getattr(w,"credits_charged",None))
  if ptm is None or lddt is None or ptm>1 or lddt>1:
   return {"result_state":"PROVIDER_SCORES_UNVERIFIED","additional_jobs":0,"charged_credits":credits}
  return {"result_state":"PROVIDER_SCORES_RETURNED","ptm":ptm,"avg_lddt":lddt,
          "charged_credits":credits,"additional_jobs":0}
 except Hold:raise
 except Exception:
  return {"result_state":"PROVIDER_RESULT_UNREADABLE","additional_jobs":0}
def execute(api,case,action,approved):
 t=source(case)
 if action not in ("submit","collect"):raise Hold("UNKNOWN_ACTION")
 prior=existing(api,t["name"])
 report={
  "schema_id":"janus.rowan.scoby_placenta_2x1.protein_job_receipt.v1",
  "case":case,"case_name":t["name"],"source_uniprot":t["uniprot"],
  "sequence_sha256":t["sha256"],"sequence_length":t["length"],
  "source_url":t["source"],
  "model":"BOLTZ_2","use_msa_server":True,"num_samples":1,
  "weekly_pool_min_preflight":WEEKLY_MIN,"max_credits_per_new_job":CAP,
  "new_jobs":0,"no_auto_retries":True,
  "scientific_scope":t["scientific_limit"],
  "SCOBY_microbiome_predicted":False,
  "placental_tissue_or_BC_composite_predicted":False,
  "experimental_material_performance_proven":False,
  "source_registry_protocol":"data/rowan/protocols/ROWAN-SCOBY-KOMBUCHA-PLACENTAL-ECM-TWO-PROTEIN-PILOT-PREREG-2026-10-26-v1.json"
 }
 if prior is not None:
  uid=str(getattr(prior,"uuid",""))
  if not uid:raise Hold("EXISTING_CASE_NO_UUID")
  report.update({"status":"CASE_ALREADY_EXISTS_NO_NEW_SUBMISSION",
                 "provider_uuid":uid,"provider_status":str(getattr(prior,"status",None))})
  if action=="collect":report.update(collect(prior,t["name"]))
  return report
 if action=="collect":
  report["status"]="NO_CASE_FOUND_NO_NEW_SUBMISSION"
  return report
 if approved is not True:raise Hold("USER_EXPLICIT_CREDIT_APPROVAL_MISSING")
 report["weekly_authorization"]=check_weekly(api)
 # Single selected CASE only; no iteration over TARGETS.
 w=api.submit_protein_cofolding_workflow(
  initial_protein_sequences=[t["sequence"]],model=api.CofoldingModel.BOLTZ_2,
  num_samples=1,use_msa_server=True,max_credits=CAP,
  folder_uuid=TARGET_ROWAN_FOLDER_UUID,name=t["name"],is_draft=False)
 uid=str(getattr(w,"uuid",""))
 if not uid:raise Hold("UNKNOWN_PROVIDER_UUID_DO_NOT_RETRY")
 report.update({"status":"ONE_PREREGISTERED_SCOBY_RELATED_PROTEIN_SUBMITTED",
                "new_jobs":1,"provider_uuid":uid,
                "provider_status":str(getattr(w,"status",None))})
 return report
def main():
 case=os.environ.get("JANUS_SCOBY_ROWAN_CASE")
 action=os.environ.get("JANUS_SCOBY_ROWAN_ACTION","collect")
 approved=os.environ.get("JANUS_SCOBY_ROWAN_SPEND_AUTH")=="TRUE"
 if not os.environ.get("ROWAN_API_KEY"):
  print("JANUS_SCOBY_ROWAN=HOLD_NO_API_KEY");return 2
 if action=="submit" and not approved:
  print("JANUS_SCOBY_ROWAN=HOLD_SUBMISSION_NOT_AUTHORIZED");return 2
 if action=="collect" and approved:
  print("JANUS_SCOBY_ROWAN=HOLD_COLLECT_MUST_NOT_AUTHORIZE_SPEND");return 2
 try:
  rowan=importlib.import_module("rowan")
  result=execute(rowan,case,action,approved)
  print("JANUS_SCOBY_ROWAN="+result["status"])
  print("JANUS_SCOBY_CASE="+str(case))
  print("JANUS_SCOBY_ROWAN_NEW_JOBS="+str(result["new_jobs"]))
  if result.get("result_state"):
   print("JANUS_SCOBY_ROWAN_RESULT_STATE="+str(result["result_state"]))
  if result.get("provider_uuid"):
   print("JANUS_SCOBY_ROWAN_PROVIDER_JOB_URL=https://labs.rowansci.com/protein-cofolding/"+result["provider_uuid"])
  out=os.environ.get("JANUS_SCOBY_ROWAN_RECEIPT")
  if out:
   dest=Path(out)
   if dest.exists():raise Hold("IMMUTABLE_RECEIPT_ALREADY_EXISTS")
   dest.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
  return 0
 except Hold as ex:
  print("JANUS_SCOBY_ROWAN=HOLD_"+str(ex));return 2
 except Exception:
  print("JANUS_SCOBY_ROWAN=HOLD_PROVIDER_UNCERTAIN_NO_AUTO_RETRY");return 3
if __name__=="__main__":raise SystemExit(main())
