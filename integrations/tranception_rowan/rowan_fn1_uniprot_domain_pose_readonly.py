#!/usr/bin/env python3
"""Exploratory post-primary FN1 per-domain fold vs NMR relative pose (read-only).

The boundaries come from UniProt P02751, not the prior structural results.
Clearly second-stage exploratory; no Rowan submit calls, credits, or biology claims.
"""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import tempfile
from collections import OrderedDict
import numpy as np
from Bio.PDB import MMCIFParser
from Bio.SVDSuperimposer import SVDSuperimposer

from rowan_scoby_placenta_two_protein_pilots import existing,Hold
from rowan_fn1_1e88_1e8b_nmr_readonly import (
    TARGET, TARGET_NAME, EXPECTED_UUID, REFS, FULL_REF_SEQUENCE,
    sha, fetch, require_exact_reference_source, exact_CA_from_model,
    map_CA_to_target, MIN_MATCHED_CA, MAX_BYTES
)
EXPECTED_HASHES={
  "ROWAN":"67281aa7363212573cd8de4012dfe8a7345470ba09a4b561867103d1ea40b703",
  "1E88":"0b91f9074f4e24f327c10d4d254e9fce543c14b741910b5fe15fac342bbf1f25",
  "1E8B":"ad4960bfb9a4dcdb421892a577b5c0b32151f9f5fcbc60a86ec76d17fb873de6"
}
BOUNDARIES={"FNII_1":list(range(0,49)), "FNII_2":list(range(60,109))}
INTERDOMAIN_CA_DISTANCE_THRESHOLD=8.0

def kabsch(ref,pred):
    x=np.asarray(ref,dtype=float);y=np.asarray(pred,dtype=float)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<3:
        raise Hold("INVALID_CA_DOMAIN_GEOMETRY")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise Hold("NONFINITE_DOMAIN_COORDINATES")
    sup=SVDSuperimposer();sup.set(x,y);sup.run()
    rot,tran=sup.get_rotran()
    return float(sup.get_rms()),rot,tran

def evaluate_domain_geometry(ref,pred):
    results={}
    for name,indices in BOUNDARIES.items():
        valid=[i for i in indices if i in ref and i in pred]
        if len(valid)<45:
            raise Hold("INSUFFICIENT_DOMAIN_IDENTICAL_CA_"+name)
        r=np.array([ref[i] for i in valid],dtype=float)
        p=np.array([pred[i] for i in valid],dtype=float)
        rmsd,rot,tran=kabsch(r,p)
        results[name]={"n_CA":len(valid),"independently_fitted_CA_RMSD_A":round(rmsd,6)}
    # Orient both protein fragments only by fit to the FIRST type II repeat,
    # then measure second-repeat mismatch WITHOUT a second alignment.
    d1=[i for i in BOUNDARIES["FNII_1"] if i in ref and i in pred]
    d2=[i for i in BOUNDARIES["FNII_2"] if i in ref and i in pred]
    r1=np.array([ref[i] for i in d1]);p1=np.array([pred[i] for i in d1])
    r2=np.array([ref[i] for i in d2]);p2=np.array([pred[i] for i in d2])
    _,R,T=kabsch(r1,p1)
    moved2=p2@R+T
    anchored2=float(np.sqrt(np.mean(np.sum((moved2-r2)**2,axis=1))))
    local2=results["FNII_2"]["independently_fitted_CA_RMSD_A"]
    results["FNII_2_after_FNII_1_alignment"]={
        "CA_RMSD_A":round(anchored2,6),
        "minus_independent_FNII_2_fit_RMSD_A":round(anchored2-local2,6)
    }
    pred_dist=np.linalg.norm(p1[:,None,:]-p2[None,:,:],axis=2)
    ref_dist=np.linalg.norm(r1[:,None,:]-r2[None,:,:],axis=2)
    pi=pred_dist<=INTERDOMAIN_CA_DISTANCE_THRESHOLD
    ri=ref_dist<=INTERDOMAIN_CA_DISTANCE_THRESHOLD
    tp=int(np.sum(pi & ri));npred=int(np.sum(pi));nref=int(np.sum(ri))
    prec=tp/npred if npred else None
    recall=tp/nref if nref else None
    f1=(2*prec*recall/(prec+recall)) if prec is not None and recall is not None and (prec+recall)>0 else None
    results["interdomain_CA_contacts"]={
        "threshold_A":INTERDOMAIN_CA_DISTANCE_THRESHOLD,
        "candidate_pairs":int(len(d1)*len(d2)),
        "predicted_contacts":npred,
        "reference_contacts":nref,
        "true_positive_contacts":tp,
        "precision":round(prec,6) if prec is not None else None,
        "recall":round(recall,6) if recall is not None else None,
        "F1":round(f1,6) if f1 is not None else None
    }
    return results

def summary(models):
    fields={
        "FNII_1_RMSD":lambda m:m["FNII_1"]["independently_fitted_CA_RMSD_A"],
        "FNII_2_RMSD":lambda m:m["FNII_2"]["independently_fitted_CA_RMSD_A"],
        "FNII_2_anchored_RMSD":lambda m:m["FNII_2_after_FNII_1_alignment"]["CA_RMSD_A"],
        "FNII_2_relative_pose_excess_RMSD":lambda m:m["FNII_2_after_FNII_1_alignment"]["minus_independent_FNII_2_fit_RMSD_A"]
    }
    d={}
    for k,f in fields.items():
        values=[f(m) for m in models]
        d[k]={"min":round(float(np.min(values)),6),"median":round(float(np.median(values)),6),"max":round(float(np.max(values)),6)}
    return d

def run(api):
    if len(TARGET["sequence"])!=109 or sha(TARGET["sequence"].encode("ascii"))!=TARGET["sha256"]:
        raise Hold("FROZEN_INPUT_SEQUENCE_MISMATCH")
    older=existing(api,TARGET_NAME)
    if older is None or str(getattr(older,"uuid",""))!=EXPECTED_UUID:
        raise Hold("OLDER_FN1_PROVIDER_WORKFLOW_MISSING")
    w=older.fetch_latest()
    if not w.done():
        raise Hold("OLDER_FN1_NOT_COMPLETED")
    struct=w.result(wait=False).get_predicted_structure()
    if struct is None:raise Hold("ROWAN_ORIGINAL_STRUCTURE_NOT_AVAILABLE")
    with tempfile.TemporaryDirectory(prefix="janus-fn1-domains-") as tmp:
        p=Path(tmp);oldwd=Path.cwd()
        try:
            os.chdir(p)
            struct.download_structure(name="fn1_existing",file_format="mmcif")
        finally:
            os.chdir(oldwd)
        matches=list(p.glob("fn1_existing*"))
        if len(matches)!=1 or not 0<matches[0].stat().st_size<=MAX_BYTES:
            raise Hold("ROWAN_PREDICTION_FILE_INVALID")
        rowan_bytes=matches[0].read_bytes()
        if sha(rowan_bytes)!=EXPECTED_HASHES["ROWAN"]:
            raise Hold("ORIGINAL_ROWAN_COORDINATE_HASH_DRIFT")
        pred_structure=MMCIFParser(QUIET=True).get_structure("pred",str(matches[0]))
        pred_models=list(pred_structure.get_models())
        if len(pred_models)!=1:
            raise Hold("ROWAN_MORE_THAN_ONE_MODEL")
        pred_coords=map_CA_to_target(TARGET["sequence"],exact_CA_from_model(pred_models[0],"A"))
        results=[]
        for code,count in REFS:
            require_exact_reference_source(code,"https://data.rcsb.org/rest/v1/core/polymer_entity/"+code+"/1")
            content=fetch("https://files.rcsb.org/download/"+code+".cif")
            if sha(content)!=EXPECTED_HASHES[code]:
                raise Hold("RCSB_MMCIF_COORDINATES_CHANGED_"+code)
            fname=p/(code+".cif")
            fname.write_bytes(content)
            models=list(MMCIFParser(QUIET=True).get_structure(code,str(fname)).get_models())
            if len(models)!=count:
                raise Hold("EXPERIMENTAL_MODEL_COUNT_DRIFT")
            individual=[]
            for i,model in enumerate(models,1):
                ref_coords=map_CA_to_target(TARGET["sequence"],exact_CA_from_model(model,"A"))
                metric=evaluate_domain_geometry(ref_coords,pred_coords)
                metric["model_index_one_based"]=i
                individual.append(metric)
            results.append({
                "pdb_id":code,"model_count":len(models),
                "all_model_results":individual,
                "full_range_and_median":summary(individual)
            })
    return {
        "schema_id":"janus.rowan.fn1_109aa_exploratory_paired_domain_relative_pose_v1",
        "status":"PASS_REAL_ROWAN_FN1_UNIPROT_DOMAIN_GEOMETRY_ALL_NMR_MODELS",
        "date_utc":"2026-10-26",
        "analysis_role":"POSTHOC_EXPLORATORY_SEPARATE_FROM_PREVIOUS_FROZEN_GLOBAL_BENCHMARK",
        "source_protocol":"data/rowan/protocols/ROWAN-FN1-109AA-UNIPROT-DOMAIN-POSE-POSTHOC-EXPLORATORY-2026-10-26-v1.json",
        "source_rowan_job_uuid":EXPECTED_UUID,
        "source_hashes":EXPECTED_HASHES,
        "domain_source":"UniProt P02751: first type II 355–403; second type II 415–463; 11aa linker excluded from separate fits",
        "domain_0based_target_positions":{"FNII_1":[0,48],"LINKER":[49,59],"FNII_2":[60,108]},
        "results":results,
        "no_new_rowan_credits":0,"no_new_rowan_jobs":0,
        "same_publication_NMR_not_independent":True,
        "not_prospective_blind":True,
        "no_placental_ECM_or_cellulose_interaction_claim":True,
        "not_human_materials_study":True,
        "shadow_weights_updated":False
    }

def main():
    if not os.getenv("ROWAN_API_KEY"):
        print("JANUS_FN1_DOMAINS=HOLD_NO_READONLY_PROVIDER_KEY")
        return 2
    try:
        obj=run(importlib.import_module("rowan"))
        dest=os.getenv("JANUS_FN1_DOMAIN_RECEIPT")
        if not dest:raise Hold("NO_RECEIPT_PATH")
        p=Path(dest)
        if p.exists() or p.is_dir():raise Hold("IMMUTABLE_RECEIPT_EXISTS")
        p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
        for x in obj["results"]:
            d=x["full_range_and_median"]
            print("JANUS_FN1_DOMAINS_"+x["pdb_id"]+"_F1_RMSD="+str(d["FNII_1_RMSD"]["median"]))
            print("JANUS_FN1_DOMAINS_"+x["pdb_id"]+"_F2_RMSD="+str(d["FNII_2_RMSD"]["median"]))
            print("JANUS_FN1_DOMAINS_"+x["pdb_id"]+"_F2_ANCHORED_RMSD="+str(d["FNII_2_anchored_RMSD"]["median"]))
        print("JANUS_FN1_DOMAINS=PASS_EXPLORATORY_SOURCE_PINNED_0_CREDITS")
        return 0
    except (Hold,ValueError,KeyError,TypeError,IndexError,OSError) as err:
        print("JANUS_FN1_DOMAINS=HOLD_"+str(err))
        return 2
    except Exception:
        print("JANUS_FN1_DOMAINS=HOLD_UNCLASSIFIED_PROVIDER_REFERENCE_ERROR")
        return 3

if __name__=="__main__":raise SystemExit(main())
