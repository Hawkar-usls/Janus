#!/usr/bin/env python3
"""Read-only audit: previous Rowan FN1 109aa vs all NMR models of 1E88 and 1E8B.

Nothing is submitted to Rowan. No new Rowan credits. Same-publication NMR
models are not independent blind validation. No dECM or materials claim.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1
from Bio.SVDSuperimposer import SVDSuperimposer

from rowan_scoby_placenta_two_protein_pilots import TARGETS, existing, Hold

TARGET = TARGETS["placenta_fn1"]
TARGET_NAME = TARGET["name"]
EXPECTED_UUID = "74a70ba9-8fa2-4011-b89a-70c761cc632f"
PREREG = "data/rowan/protocols/ROWAN-FN1-TYPEII-1E88-1E8B-ALL-NMR-ENSEMBLE-109AA-PREREG-2026-10-26-v1.json"
FULL_REF_SEQUENCE = "YGHCVTDSGVVYSVGMQWLKTQGNKQMLCTCLGNGVSCQETAVTQTYGGNSNGEPCVLPFTYNGRTFYSCTTEGRQDGHLWCSTTSNYEQDQKYSFCTDHTVLVQTRGGNSNGALCHFPFLYNNHNYTDCTSEGRRDNMKWCGTTQNYDADQKFGFCPMA"
REFS = (("1E88",20),("1E8B",1))
MIN_MATCHED_CA = 100
MAX_BYTES=20_000_000

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def fetch(url):
    request=urllib.request.Request(url,headers={"User-Agent":"JANUS FN1 structural-read-only audit/1.0"})
    with urllib.request.urlopen(request,timeout=60) as stream:
        content=stream.read(MAX_BYTES+1)
    if not content or len(content)>MAX_BYTES:
        raise Hold("REFERENCE_BYTES_MISSING_OR_TOO_LARGE")
    return content

def require_exact_reference_source(reference, url):
    p=json.loads(fetch(url))
    full="".join(p["entity_poly"]["pdbx_seq_one_letter_code_can"].split())
    if len(full)!=160 or full!=FULL_REF_SEQUENCE or full.count(TARGET["sequence"])!=1 or full.index(TARGET["sequence"])!=50:
        raise Hold("REF_POLYMER_SEQUENCE_OFFSET_OR_CONTENT_DRIFT_"+reference)
    return full

def exact_CA_from_model(model, chain="A"):
    if chain not in model:
        raise Hold("REQUIRED_CHAIN_A_MISSING")
    residues=[]
    for r in model[chain].get_residues():
        if "CA" not in r:
            continue
        letter="M" if r.resname == "MSE" else seq1(r.resname,custom_map={"UNK":"X"})
        if len(letter)==1 and letter in "ACDEFGHIKLMNPQRSTVWY":
            xyz=np.asarray(r["CA"].get_coord(),dtype=float)
            if not np.all(np.isfinite(xyz)):
                raise Hold("NONFINITE_ORIGINAL_COORD")
            residues.append((letter,xyz))
    if not residues:
        raise Hold("REFERENCE_MODEL_NO_CA")
    return residues

def map_CA_to_target(target, residues):
    observed="".join(r[0] for r in residues)
    aln=PairwiseAligner(mode="local")
    aln.match_score=3
    aln.mismatch_score=-100
    aln.open_gap_score=-100
    aln.extend_gap_score=-100
    alns=aln.align(target,observed)
    if not alns:
        raise Hold("SEQUENCE_ALIGNMENT_MISSING")
    a=alns[0]
    mapped={}
    for ti,oi in zip(*a.indices):
        if ti>=0 and oi>=0 and target[ti]==observed[oi]:
            if int(ti) in mapped:
                raise Hold("SEQUENCE_POSITION_NOT_UNIQUE")
            mapped[int(ti)]=residues[int(oi)][1]
    if len(mapped)<MIN_MATCHED_CA or len(mapped)>len(target):
        raise Hold("INSUFFICIENT_EXACT_ALIGNED_CA_"+str(len(mapped)))
    return mapped

def compare_models(reference,pred,target_len):
    common=np.array(sorted(set(reference).intersection(pred)),dtype=int)
    if not MIN_MATCHED_CA<=len(common)<=target_len:
        raise Hold("INSUFFICIENT_COMMON_MATCHED_CA")
    ref=np.array([reference[int(i)] for i in common],dtype=float)
    pr=np.array([pred[int(i)] for i in common],dtype=float)
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(pr)):
        raise Hold("COORDINATES_NONFINITE")
    s=SVDSuperimposer()
    s.set(ref,pr);s.run()
    rot,tran=s.get_rotran()
    deviations=np.linalg.norm(pr@rot+tran-ref,axis=1)
    # Invariant pairwise geometry metrics; exclude local |i-j| < 4.
    ref_dist=np.linalg.norm(ref[:,None,:]-ref[None,:,:],axis=2)
    pred_dist=np.linalg.norm(pr[:,None,:]-pr[None,:,:],axis=2)
    mask=np.triu(abs(common[:,None]-common[None,:])>=4,k=1).astype(bool)
    rd=ref_dist[mask]
    pd=pred_dist[mask]
    if len(rd)<1:
        raise Hold("CONTACT_PAIRS_MISSING")
    ref_contacts=(rd<=8.0)
    pred_contacts=(pd<=8.0)
    tp=int(np.sum(ref_contacts & pred_contacts))
    fp=int(np.sum(~ref_contacts & pred_contacts))
    fn=int(np.sum(ref_contacts & ~pred_contacts))
    if (tp+fp)==0 or (tp+fn)==0:
        raise Hold("DEGENERATE_CONTACT_MATRIX")
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {
        "matched_exact_CA":int(len(common)),
        "target_CA_fraction":round(len(common)/target_len,6),
        "global_CA_RMSD_A":round(float(s.get_rms()),6),
        "GDT_TS_like_1_2_4_8_A":round(float(np.mean([np.mean(deviations<=v) for v in (1,2,4,8)])),6),
        "pairwise_dRMSD_A":round(float(np.sqrt(np.mean((pd-rd)**2))),6),
        "sequence_separated_pair_count":int(len(rd)),
        "reference_contacts_le8_A":int(np.sum(ref_contacts)),
        "pred_contacts_le8_A":int(np.sum(pred_contacts)),
        "contact_precision":round(float(precision),6),
        "contact_recall":round(float(recall),6),
        "contact_F1":round(float(f1),6)
    }

def aggregate(records):
    if not records:
        raise Hold("EMPTY_NMR_MODEL_ENSEMBLE")
    keys=("global_CA_RMSD_A","GDT_TS_like_1_2_4_8_A","pairwise_dRMSD_A","contact_F1","contact_precision","contact_recall","matched_exact_CA")
    return {
        key: {
            "median":round(float(np.median([r[key] for r in records])),6),
            "min":round(float(np.min([r[key] for r in records])),6),
            "max":round(float(np.max([r[key] for r in records])),6)
        } for key in keys
    }

def run(api):
    if len(TARGET["sequence"])!=109 or sha(TARGET["sequence"].encode("ascii"))!=TARGET["sha256"]:
        raise Hold("TARGET_SOURCE_SEQUENCE_DRIFT")
    if len(FULL_REF_SEQUENCE)!=160 or FULL_REF_SEQUENCE.index(TARGET["sequence"])!=50:
        raise Hold("FROZEN_REFERENCE_SEQUENCE_DRIFT")
    existing_job=existing(api,TARGET_NAME)
    if existing_job is None or str(getattr(existing_job,"uuid",""))!=EXPECTED_UUID:
        raise Hold("FROZEN_ORIGINAL_ROWAN_JOB_NOT_FOUND")
    job=existing_job.fetch_latest()
    if not job.done():
        raise Hold("PREEXISTING_FN1_ROWAN_JOB_NOT_COMPLETED")
    output=job.result(wait=False)
    structure=output.get_predicted_structure()
    if structure is None:
        raise Hold("ORIGINAL_ROWAN_PREDICTED_CIF_UNAVAILABLE")
    with tempfile.TemporaryDirectory(prefix="janus-rowan-fn1-nmr-") as d:
        root=Path(d)
        cwd=Path.cwd()
        try:
            os.chdir(root)
            structure.download_structure(name="fn1_old_prediction",file_format="mmcif")
        finally:
            os.chdir(cwd)
        result_cif=list(root.glob("fn1_old_prediction*"))
        if len(result_cif)!=1 or not 0<result_cif[0].stat().st_size<=MAX_BYTES:
            raise Hold("ORIGINAL_FN1_CIF_MISSING_OR_AMBIGUOUS")
        pb=result_cif[0].read_bytes()
        predicted=MMCIFParser(QUIET=True).get_structure("predicted",str(result_cif[0]))
        pred_models=list(predicted.get_models())
        if len(pred_models)!=1:
            raise Hold("ROWAN_PREDICTION_NOT_SINGLE_CONFORMER")
        pred_coords=map_CA_to_target(TARGET["sequence"],exact_CA_from_model(pred_models[0],"A"))
        refs=[]
        for pdb_id,expected_count in REFS:
            require_exact_reference_source(pdb_id,"https://data.rcsb.org/rest/v1/core/polymer_entity/"+pdb_id+"/1")
            cif_bytes=fetch("https://files.rcsb.org/download/"+pdb_id+".cif")
            ref_path=root/(pdb_id+".cif")
            ref_path.write_bytes(cif_bytes)
            reference_structure=MMCIFParser(QUIET=True).get_structure(pdb_id,str(ref_path))
            models=list(reference_structure.get_models())
            if len(models)!=expected_count:
                raise Hold("NMR_CONFORMER_COUNT_MISMATCH_"+pdb_id+"_GOT_"+str(len(models)))
            results=[]
            for i,refmodel in enumerate(models):
                refs_coords=map_CA_to_target(TARGET["sequence"],exact_CA_from_model(refmodel,"A"))
                metrics=compare_models(refs_coords,pred_coords,len(TARGET["sequence"]))
                metrics["model_index_one_based"]=i+1
                results.append(metrics)
            refs.append({
                "reference_pdb_id":pdb_id,
                "source_rcsb_mmcif_sha256":sha(cif_bytes),
                "model_count":len(results),
                "individual_model_results":results,
                "ensemble_median_full_min_max":aggregate(results)
            })
    return {
        "schema_id":"janus.rowan.fn1_1E88_20NMR_1E8B_1NMR_real_no_credit_structure_audit.v1",
        "date_utc":"2026-10-26",
        "status":"PASS_REAL_READONLY_FN1_AGAINST_ALL_PRESELECTED_NMR_MODELS",
        "preregistration_meta_registry_path":PREREG,
        "previous_rowan_uuid":EXPECTED_UUID,
        "previous_provider_ptm":0.696,
        "previous_provider_avg_lddt":0.864,
        "target_sequence_sha256":TARGET["sha256"],
        "target_sequence_length":109,
        "reference_full_polymer_length":160,
        "target_reference_offset_zero_based":50,
        "reference_extra_Nterminal_FnI_present":True,
        "reference_1E88_and_1E8B_same_publication":True,
        "not_independent_prospective_blind":True,
        "rowan_predicted_mmcif_sha256":sha(pb),
        "results":refs,
        "new_rowan_jobs":0,
        "new_rowan_credits_spent":0,
        "protein_stability_ddg_claim":False,
        "human_placenta_binding_or_materials_claim":False,
        "production_model_promoted":False
    }

def main():
    if not os.getenv("ROWAN_API_KEY"):
        print("JANUS_FN1_NMR=HOLD_MISSING_ROWAN_READONLY_CREDENTIAL")
        return 2
    try:
        obj=run(importlib.import_module("rowan"))
        out=os.getenv("JANUS_FN1_NMR_RECEIPT")
        if not out:
            raise Hold("NO_RECEIPT_OUTPUT_LOCATION")
        path=Path(out)
        if path.exists() or path.is_dir():
            raise Hold("IMMUTABLE_RECEIPT_ALREADY_EXISTS")
        path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
        for item in obj["results"]:
            x=item["ensemble_median_full_min_max"]
            print("JANUS_FN1_NMR_"+item["reference_pdb_id"]+"_MEDIAN_RMSD_A="+str(x["global_CA_RMSD_A"]["median"]))
            print("JANUS_FN1_NMR_"+item["reference_pdb_id"]+"_MEDIAN_CONTACT_F1="+str(x["contact_F1"]["median"]))
            print("JANUS_FN1_NMR_"+item["reference_pdb_id"]+"_MODELS="+str(item["model_count"]))
        print("JANUS_FN1_NMR=PASS_REAL_READONLY_NO_SPEND")
        print("JANUS_ROWAN_NEW_JOBS=0")
        return 0
    except (Hold,ValueError,KeyError,IndexError,OSError,TypeError) as err:
        print("JANUS_FN1_NMR=HOLD_"+str(err))
        return 2
    except Exception:
        print("JANUS_FN1_NMR=HOLD_UNCLASSIFIED_PROVIDER_OR_REFERENCE_ERROR")
        return 3

if __name__=="__main__":
    raise SystemExit(main())
