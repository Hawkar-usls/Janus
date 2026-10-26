#!/usr/bin/env python3
"""One immutable source-pinned Rowan structural pilot for *mature* human milk LALBA.

Complements the older AcsD/FN1 separate single-protein pilots; never models
SCOBY, human tissue, maternal phenotype, affinity or a microbial composite.
Pre-registration: janus-meta-registry/data/rowan/protocols/
ROWAN-SCOBY-MILK-LALBA-MATURE123-SINGLE-PROTEIN-PREREG-2026-10-26-v1.json
"""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
from rowan_scoby_placenta_two_protein_pilots import (
    CAP, WEEKLY_MIN, TARGET_ROWAN_FOLDER_UUID, Hold, check_weekly, existing, collect
)

SEQUENCE = "KQFTKCELSQLLKDIDGYGGIALPELICTMFHTSGYDTQAIVENNESTEYGLFQISNKLWCKSSQVPQSRNICDISCDKFLDDDITDDIMCAKKILDIKGIDYWLAHKALCTEKLEQWLCEKL"
SHA256 = "f7e3b0b48179423a572873eabbb5614d878b0391e868fc8a2be8e0e10e32d9c4"
CASE = "milk_lalba_mature"
NAME = "JANUS-SCOBY-MILK-LALBA-P00709-MATURE123-20261026-R0"
PREREG = "data/rowan/protocols/ROWAN-SCOBY-MILK-LALBA-MATURE123-SINGLE-PROTEIN-PREREG-2026-10-26-v1.json"
SOURCE = "https://rest.uniprot.org/uniprotkb/P00709.json"

def validate_source():
    if len(SEQUENCE) != 123 or hashlib.sha256(SEQUENCE.encode("ascii")).hexdigest() != SHA256:
        raise Hold("UNIPROT_MATURE_CHAIN_LENGTH_OR_HASH_DRIFT")
    if CAP != 10 or WEEKLY_MIN != 30:
        raise Hold("BUDGET_POLICY_CHANGED")
    return True

def execute(api, case, action, credit_permission):
    validate_source()
    if case != CASE:
        raise Hold("UNREGISTERED_CASE")
    if action not in ("submit", "collect"):
        raise Hold("UNKNOWN_ACTION")
    if action == "collect" and credit_permission:
        raise Hold("COLLECTION_MUST_NOT_AUTHORIZE_SPEND")
    if action == "submit" and not credit_permission:
        raise Hold("EXPLICIT_WEEKLY_CREDIT_PERMISSION_REQUIRED")
    w = existing(api, NAME)  # unique exact name + folder, paginated and fail closed
    report = {
        "schema_id": "janus.rowan.scoby_milk_lalba_mature_single_structure_pilot.v1",
        "case": CASE, "case_name": NAME, "source_uniprot": "P00709",
        "source_url": SOURCE, "source_mature_chain_one_based": [20,142],
        "sequence_sha256": SHA256, "sequence_length":123,
        "source_prereg": PREREG, "model":"BOLTZ_2", "num_samples":1,
        "use_msa_server":True, "rowan_folder_uuid":TARGET_ROWAN_FOLDER_UUID,
        "weekly_minimum": WEEKLY_MIN, "max_credits_per_new_job":CAP,
        "new_jobs":0, "no_auto_retry":True,
        "milk_or_placental_tissue_modelled":False,
        "SCOBY_or_bacterial_cellulose_modelled":False,
        "experimental_fermentation_or_matrix_properties_measured":False,
        "raw_human_materials_involved":False, "protein_affinity_or_fitness_inferred":False,
        "cross_protein_ptm_performance_comparison_authorized":False,
        "shadow_weights_modified":False, "true_blind_validation":False
    }
    if w is not None:
        uid = str(getattr(w,"uuid",""))
        if not uid:
            raise Hold("EXISTING_PROVIDER_UUID_UNAVAILABLE")
        report.update({"status":"EXISTING_CASE_NO_ADDITIONAL_SPEND",
                       "provider_uuid":uid,"provider_status":str(getattr(w,"status",None))})
        if action == "collect":
            report.update(collect(w,NAME))
        return report
    if action == "collect":
        report["status"] = "NO_CASE_FOUND_ZERO_SPEND"
        return report
    report["weekly_credit_authorization"] = check_weekly(api)
    w = api.submit_protein_cofolding_workflow(
        initial_protein_sequences=[SEQUENCE],
        model=api.CofoldingModel.BOLTZ_2,
        num_samples=1,use_msa_server=True,max_credits=CAP,
        folder_uuid=TARGET_ROWAN_FOLDER_UUID,name=NAME,is_draft=False
    )
    uid = str(getattr(w,"uuid",""))
    if not uid:
        raise Hold("UNCONFIRMED_PROVIDER_SUBMISSION_DO_NOT_RETRY")
    report.update({"status":"ONE_MATURE_LALBA_PROVIDER_SUBMISSION",
                   "new_jobs":1,"provider_uuid":uid,
                   "provider_status":str(getattr(w,"status",None))})
    return report

def main():
    case = os.getenv("JANUS_SCOBY_LALBA_CASE", CASE)
    action = os.getenv("JANUS_SCOBY_LALBA_ACTION","collect")
    credit_permission = os.getenv("JANUS_SCOBY_LALBA_SPEND_AUTH") == "TRUE"
    if not os.getenv("ROWAN_API_KEY"):
        print("JANUS_LALBA=HOLD_NO_API_KEY")
        return 2
    try:
        if action == "submit" and not credit_permission:
            raise Hold("EXPLICIT_WEEKLY_CREDIT_PERMISSION_REQUIRED")
        result = execute(importlib.import_module("rowan"), case, action, credit_permission)
        print("JANUS_LALBA_STATUS=" + result["status"])
        print("JANUS_LALBA_NEW_JOBS=" + str(result["new_jobs"]))
        if "provider_uuid" in result:
            print("JANUS_LALBA_PROVIDER_URL=https://labs.rowansci.com/protein-cofolding/" + result["provider_uuid"])
        if "result_state" in result:
            print("JANUS_LALBA_RESULT_STATE=" + result["result_state"])
        dest = os.getenv("JANUS_SCOBY_LALBA_RECEIPT")
        if dest:
            p=Path(dest)
            if p.exists():
                raise Hold("IMMUTABLE_RECEIPT_EXISTS")
            p.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf8")
        return 0
    except Hold as e:
        print("JANUS_LALBA=HOLD_" + str(e))
        return 2
    except Exception:
        print("JANUS_LALBA=HOLD_UNKNOWN_PROVIDER_STATE_NO_AUTO_RETRY")
        return 3

if __name__ == "__main__":
    raise SystemExit(main())
