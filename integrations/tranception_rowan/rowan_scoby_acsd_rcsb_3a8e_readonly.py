#!/usr/bin/env python3
"""Zero Rowan-credit *read-only* AcsD 3A8E structure comparison.

No Rowan submit_protein_cofolding_workflow call. Native octamer geometry,
pore function, cellulose production, placental/milk interaction and biological
fitness are OUT OF SCOPE. Fixed reference chain A, no selecting best chain.
"""
from __future__ import annotations
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

from rowan_scoby_placenta_two_protein_pilots import (
    TARGETS, TARGET_ROWAN_FOLDER_UUID, existing, Hold
)

TARGET = TARGETS["scoby_acsD"]
EXPECTED_PROVIDER_UUID = "d6857380-d46b-4c27-bf79-faf761c9af36"
REFERENCE_URL = "https://files.rcsb.org/download/3A8E.cif"
REFERENCE_CHAIN = "A"
MIN_MATCHED_CA = 125
MAX_BYTES = 20_000_000

def digest(b):
    return hashlib.sha256(b).hexdigest()

def chain_ca(path, fixed_chain=REFERENCE_CHAIN):
    p = MMCIFParser(QUIET=True)
    structure = p.get_structure("structure", str(path))
    model = next(structure.get_models())
    if fixed_chain not in model:
        raise Hold("MISSING_REQUIRED_CHAIN_" + fixed_chain)
    out=[]
    for r in model[fixed_chain].get_residues():
        if "CA" not in r: continue
        aa=("M" if r.resname == "MSE" else seq1(r.resname, custom_map={"UNK":"X"}))
        if len(aa) != 1 or aa not in "ACDEFGHIKLMNPQRSTVWY": continue
        out.append((aa,np.asarray(r["CA"].get_coord(),dtype=float)))
    if not out:
        raise Hold("NO_CANONICAL_CA_COORDINATES")
    return out

def map_chain_to_target(seq, residues):
    observed = "".join(x[0] for x in residues)
    a = PairwiseAligner(mode="global")
    a.match_score=2
    a.mismatch_score=-4
    a.open_gap_score=-7
    a.extend_gap_score=-0.5
    aln=a.align(seq,observed)[0]
    mapping={}
    for ti,ri in zip(*aln.indices):
        if ti>=0 and ri>=0 and seq[ti]==residues[ri][0]:
            mapping[int(ti)]=np.asarray(residues[int(ri)][1],dtype=float)
    return mapping

def compare_arrays(ref, pred, length, cutoff=MIN_MATCHED_CA):
    common=sorted(set(ref).intersection(pred))
    if len(common)<cutoff or len(common)>length:
        raise Hold("INSUFFICIENT_SEQUENCE_IDENTICAL_CA_PAIRING")
    x=np.array([ref[i] for i in common],dtype=float)
    y=np.array([pred[i] for i in common],dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise Hold("INVALID_CA_COORDINATES")
    svd=SVDSuperimposer()
    svd.set(x,y);svd.run()
    rotation,translation=svd.get_rotran()
    d=np.linalg.norm((y@rotation+translation)-x,axis=1)
    return {
        "matched_CA_atoms":len(common),
        "target_sequence_length":length,
        "sequence_CA_coverage_fraction":round(len(common)/length,6),
        "global_CA_RMSD_A":round(float(svd.get_rms()),6),
        "GDT_TS_CA_1_2_4_8_A_fraction":round(float(np.mean([np.mean(d<=v) for v in (1,2,4,8)])),6),
        "median_CA_deviation_A":round(float(np.median(d)),6),
        "ca_fraction_within_2_A":round(float(np.mean(d<=2)),6),
        "ca_fraction_within_4_A":round(float(np.mean(d<=4)),6)
    }

def run(api):
    from rowan_scoby_placenta_two_protein_pilots import source
    source("scoby_acsD")
    old=existing(api,TARGET["name"])
    if old is None:
        raise Hold("ORIGINAL_ROWAN_WORKFLOW_MISSING")
    if str(getattr(old,"uuid","")) != EXPECTED_PROVIDER_UUID:
        raise Hold("ROWAN_JOB_UUID_NOT_ORIGINAL")
    old=old.fetch_latest()
    if not old.done():
        raise Hold("ROWAN_ACSD_JOB_NOT_DONE")
    result=old.result(wait=False)
    structure=result.get_predicted_structure()
    if structure is None:
        raise Hold("PREDICTED_STRUCTURE_NOT_ACCESSIBLE")
    with tempfile.TemporaryDirectory(prefix="janus-acsd-no-spend-") as tmp:
        tmp=Path(tmp)
        cwd=Path.cwd()
        try:
            os.chdir(tmp)
            structure.download_structure(name="acsd_rowan_readonly", file_format="mmcif")
        finally:
            os.chdir(cwd)
        cifs=list(tmp.glob("acsd_rowan_readonly*"))
        if len(cifs)!=1 or cifs[0].stat().st_size>MAX_BYTES:
            raise Hold("PREDICTED_MMCIF_NOT_UNAMBIGUOUS")
        pb=cifs[0].read_bytes()
        if not pb:raise Hold("EMPTY_ROWAN_MMCIF")
        req=urllib.request.Request(REFERENCE_URL,headers={"User-Agent":"JANUS-source-only-structural-audit/1.0"})
        with urllib.request.urlopen(req,timeout=40) as response:
            rb=response.read(MAX_BYTES+1)
        if not rb or len(rb)>MAX_BYTES:
            raise Hold("REFERENCE_PDB_MMCIF_INVALID_SIZE")
        ref_file=tmp/"3A8E.cif"
        ref_file.write_bytes(rb)
        # Fixed chain A for the crystal, not posthoc best-of-eight.
        source_seq=TARGET["sequence"]
        ref=map_chain_to_target(source_seq,chain_ca(ref_file,"A"))
        # Single-chain Rowan predictions should expose a one-character chain.
        # Require chain A explicitly; no choosing the best-fitting chain.
        pred=map_chain_to_target(source_seq,chain_ca(cifs[0],"A"))
        stats=compare_arrays(ref,pred,len(source_seq))
        return {
            "schema_id":"janus.rowan.scoby_acsd_3a8e_real_zero_credit_backbone_comparison.v1",
            "date_utc":"2026-10-26",
            "status":"PASS_REAL_READ_ONLY_ROWAN_PREDICTION_VS_3A8E_MONOMER_BACKBONE",
            "source_rowan_job_uuid":EXPECTED_PROVIDER_UUID,
            "source_rowan_job_name":TARGET["name"],
            "source_uniprot":TARGET["uniprot"],
            "source_sequence_sha256":TARGET["sha256"],
            "source_rowan_predicted_mmcif_sha256":digest(pb),
            "source_rcsb_structure_url":REFERENCE_URL,
            "source_rcsb_3a8e_mmcif_sha256":digest(rb),
            "experimental_reference_chain":"A",
            "matched_residue_rule":"Global sequence alignment; only identical residue symbols with CA; choose fixed chain A, no best-fit chain selection",
            "algorithm":"Biopython SVDSuperimposer Kabsch on exact sequence aligned CA",
            "precision":"6 decimal places in Angstrom/fractions; no confidence interval for single model and single crystallographic chain",
            "stats":stats,
            "new_rowan_jobs":0,
            "new_rowan_credits":0,
            "no_placental_human_sample":True,
            "no_cellulose_yield_or_pore_function_claim":True,
            "no_octamer_reconstruction_or_binding_claim":True,
            "single_monomer_only":True,
            "no_production_model_promotion":True
        }

def main():
    try:
        if not os.getenv("ROWAN_API_KEY"):
            raise Hold("ROWAN_READONLY_KEY_MISSING")
        result=run(importlib.import_module("rowan"))
        out=Path(os.environ.get("JANUS_ACSD_STRUCTURAL_READONLY_RECEIPT",""))
        if not str(out) or out.is_dir() or out.exists():
            raise Hold("MISSING_OR_EXISTING_RECEIPT_PATH")
        out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf8")
        print("JANUS_ROWAN_ACSD_REFERENCE=PASS")
        print("JANUS_ROWAN_ACSD_RMSD_A="+str(result["stats"]["global_CA_RMSD_A"]))
        print("JANUS_ROWAN_ACSD_MATCHED_CA="+str(result["stats"]["matched_CA_atoms"]))
        print("JANUS_ROWAN_NEW_JOBS=0")
        return 0
    except (Hold,ValueError,KeyError,OSError,IndexError,TypeError) as e:
        print("JANUS_ROWAN_ACSD_REFERENCE=HOLD_"+str(e))
        return 2
    except Exception:
        print("JANUS_ROWAN_ACSD_REFERENCE=HOLD_UNCLASSIFIED_SOURCE_OR_PROVIDER_READ_ERROR")
        return 3

if __name__=="__main__":
    raise SystemExit(main())
