#!/usr/bin/env python3
"""JANUS FN1: source-locked 2FN2 isolated FNII2 NMR + 3MQL X-ray benchmark.

Uses ONLY existing Rowan FN1 prediction, no Rowan submit, no credit spend.
2FN2 is single-domain: cannot test inter-domain orientation of isolated tandem pair.
3MQL includes FnI flanks: experimental X-ray, not pair-only context.
"""
from __future__ import annotations
import importlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser
from Bio.SVDSuperimposer import SVDSuperimposer
from rowan_scoby_placenta_two_protein_pilots import existing,Hold
from rowan_fn1_1e88_1e8b_nmr_readonly import (
    TARGET,TARGET_NAME,EXPECTED_UUID,MAX_BYTES,sha,fetch,exact_CA_from_model,
    map_CA_to_target,compare_models
)
from rowan_fn1_uniprot_domain_pose_readonly import evaluate_domain_geometry,kabsch

REF2="VLVQTRGGNSNGALCHFPFLYNNHNYTDCTSEGRRDNMKWCGTTQNYDADQKFGFCPMA"
REF3="YGDCVTDSGVVYSVGMQWLKTQGNKQMLCTCLGNGVSCQETAVTQTYGGNSNGEPCVLPFTYNGRTFYSCTTEGRQDGHLWCSTTSNYEQDQKYSFCTDHTVLVQTRGGNSNGALCHFPFLYNNHNYTDCTSEGRRDNMKWCGTTQNYDADQKFGFCPMAAHEEICTTNEGVMYRIGDQWDKQHDMGHMMRCTCVGNGRGEWTCIAYSQLRGTKHHHHHH"
FNII2=TARGET["sequence"][60:109]
ROWAN_ORIGINAL_SHA="67281aa7363212573cd8de4012dfe8a7345470ba09a4b561867103d1ea40b703"
PROTOCOL="data/rowan/protocols/ROWAN-FN1-ISOLATED-2FN2-AND-XRAY-3MQL-ORTHOGONAL-REFERENCE-GATE-2026-10-26-v1.json"

def source_polymer(pdb_id,expected):
    url="https://data.rcsb.org/rest/v1/core/polymer_entity/"+pdb_id+"/1"
    obj=json.loads(fetch(url))
    actual="".join(obj["entity_poly"]["pdbx_seq_one_letter_code_can"].split())
    if actual!=expected:
        raise Hold("RCSB_POLYMER_SEQUENCE_CHANGED_"+pdb_id)
    return actual

def subsegment_map(sequence,residues,required=45):
    """Map an exact known sequence to measured CAs; no best reference chain search."""
    observed="".join(q[0] for q in residues)
    a=PairwiseAligner(mode="local")
    a.match_score=3
    a.mismatch_score=-100
    a.open_gap_score=-100
    a.extend_gap_score=-100
    alignments=a.align(sequence,observed)
    if not alignments:
        raise Hold("MISSING_ISOLATED_FNII2_EXACT_SEQUENCE")
    ali=alignments[0]
    matched={}
    for si,oi in zip(*ali.indices):
        if si>=0 and oi>=0 and sequence[si]==observed[oi]:
            matched[int(si)]=np.array(residues[int(oi)][1],dtype=float)
    if not required<=len(matched)<=len(sequence):
        raise Hold("FNII2_2FN2_IDENTICAL_CA_COVERAGE_INSUFFICIENT")
    return matched

def domain_fit(ref,pred):
    idx=sorted(set(ref)&set(pred))
    if len(idx)<45:
        raise Hold("ISOLATED_2FN2_DOMAIN_PAIRING_TOO_SHORT")
    x=np.array([ref[i] for i in idx],dtype=float)
    y=np.array([pred[i] for i in idx],dtype=float)
    rmsd,rot,tran=kabsch(x,y)
    distances=np.linalg.norm((y@rot+tran)-x,axis=1)
    return {
        "matched_CA":len(idx),
        "CA_RMSD_A":round(rmsd,6),
        "GDT_TS_like_1_2_4_8_A":round(float(np.mean([np.mean(distances<=v) for v in (1,2,4,8)])),6)
    }

def summary(results,keys):
    return {k:{
        "median":round(float(np.median([x[k] for x in results])),6),
        "min":round(float(np.min([x[k] for x in results])),6),
        "max":round(float(np.max([x[k] for x in results])),6)
    } for k in keys}

def run(api):
    if len(TARGET["sequence"])!=109 or sha(TARGET["sequence"].encode("ascii"))!=TARGET["sha256"]:
        raise Hold("PREDICTION_INPUT_SEQUENCE_DRIFT")
    if len(REF2)!=59 or REF2.count(FNII2)!=1 or REF2.index(FNII2)!=9:
        raise Hold("ISOLATED_REFERENCE_SOURCE_FRAGMENT_MISMATCH")
    if len(REF3)!=220 or REF3.count(TARGET["sequence"])!=1 or REF3.index(TARGET["sequence"])!=50:
        raise Hold("XRAY_FULL_REFERENCE_SOURCE_FRAGMENT_MISMATCH")
    work=existing(api,TARGET_NAME)
    if work is None or str(getattr(work,"uuid",""))!=EXPECTED_UUID:
        raise Hold("ORIGINAL_ROWAN_JOB_UUID_NOT_CONFIRMED")
    work=work.fetch_latest()
    if not work.done():
        raise Hold("ROWAN_OLD_RESULT_NOT_FINISHED")
    structure=work.result(wait=False).get_predicted_structure()
    if structure is None:
        raise Hold("ORIGINAL_ROWAN_MM_CIF_NOT_ACCESSIBLE")
    result={}
    with tempfile.TemporaryDirectory(prefix="janus-fn1-orthogonal-") as tmp:
        root=Path(tmp);oldwd=Path.cwd()
        try:
            os.chdir(root)
            structure.download_structure(name="fn1_old",file_format="mmcif")
        finally:
            os.chdir(oldwd)
        files=list(root.glob("fn1_old*"))
        if len(files)!=1 or not 0<files[0].stat().st_size<=MAX_BYTES:
            raise Hold("PREVIOUS_PREDICTION_FILE_INCOMPLETE_OR_AMBIGUOUS")
        old_bytes=files[0].read_bytes()
        if sha(old_bytes)!=ROWAN_ORIGINAL_SHA:
            raise Hold("PREVIOUS_PREDICTION_CIF_HASH_DRIFT")
        pred_m=list(MMCIFParser(QUIET=True).get_structure("pred",str(files[0])).get_models())
        if len(pred_m)!=1:raise Hold("PREDICTED_FN1_MULTIPLE_CONFORMERS")
        pred=map_CA_to_target(TARGET["sequence"],exact_CA_from_model(pred_m[0],"A"))
        pred_second={i-60:pred[i] for i in range(60,109) if i in pred}
        for pdbid,reference_sequence,count in (("2FN2",REF2,20),("3MQL",REF3,1)):
            source_polymer(pdbid,reference_sequence)
            content=fetch("https://files.rcsb.org/download/"+pdbid+".cif")
            file=root/(pdbid+".cif")
            file.write_bytes(content)
            modes=list(MMCIFParser(QUIET=True).get_structure(pdbid,str(file)).get_models())
            if len(modes)!=count:
                raise Hold("REFERENCE_CONFORMER_COUNT_DRIFT_"+pdbid)
            rows=[]
            for i,model in enumerate(modes,1):
                original=exact_CA_from_model(model,"A")
                if pdbid=="2FN2":
                    ref=subsegment_map(FNII2,original)
                    metrics=domain_fit(ref,pred_second)
                else:
                    ref=map_CA_to_target(TARGET["sequence"],original)
                    metrics={
                        "full_FNII_pair_Kabsch":compare_models(ref,pred,109),
                        "UniProt_fnII_domains":evaluate_domain_geometry(ref,pred)
                    }
                rows.append({"model_index_one_based":i,**metrics})
            result[pdbid]={
                "experimental_method":"SOLUTION_NMR" if pdbid=="2FN2" else "X_RAY_DIFFRACTION",
                "mmcif_sha256":sha(content),
                "number_of_conformers":count,
                "all_conformers":rows,
                "all_models_summary": summary(rows,["CA_RMSD_A","GDT_TS_like_1_2_4_8_A"]) if pdbid=="2FN2" else None,
                "evidence_scope":"ISOLATED_FNII2_FOLD_ONLY_NO_INTERDOMAIN_GEOMETRY" if pdbid=="2FN2" else "FOUR_DOMAIN_XRAY_FNII_PAIR_WITH_FN_I_FLANKS"
            }
    return {
        "schema_id":"janus.rowan.fn1_2fn2_isolated_nmr_3mql_four_domain_xray_existing_model_audit.v1",
        "date_utc":"2026-10-26",
        "status":"PASS_SOURCE_PINNED_REAL_FN1_ORTHOGONAL_REFERENCE_COORDINATE_COMPARISON",
        "source_protocol":PROTOCOL,
        "old_rowan_job_uuid":EXPECTED_UUID,
        "old_rowan_cif_sha256":ROWAN_ORIGINAL_SHA,
        "source_109aa_sha256":TARGET["sha256"],
        "reference_results":result,
        "independent_isolated_pair_full_atomic_reference_found":False,
        "isolation_pair_NMR_biophysical_paper_PubMed":"10913243",
        "interpretation":"Only isolated FNII2 fold is validated without FnI by 2FN2; 3MQL checks all FNII pair geometry but has FnI flanks and H307D upstream. Neither establishes experimental isolated-pair 3D orientation.",
        "not_unseen_independent_model_validation":True,
        "new_rowan_jobs":0,"new_rowan_credits_spent":0,
        "no_materials_or_functional_prediction":True,
        "no_shadow_model_promotion":True
    }

def main():
    if not os.getenv("ROWAN_API_KEY"):
        print("JANUS_FN1_2FN2_3MQL=HOLD_NO_READONLY_ROWAN_KEY")
        return 2
    try:
        result=run(importlib.import_module("rowan"))
        path=os.getenv("JANUS_FN1_ORTHOGONAL_RECEIPT")
        if not path:raise Hold("NO_RECEIPT_PATH")
        out=Path(path)
        if out.exists() or out.is_dir():raise Hold("RECEIPT_ALREADY_EXISTS")
        out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
        print("JANUS_FN1_2FN2_3MQL=PASS_READ_ONLY")
        print("JANUS_FN1_2FN2_MEDIAN_CA_RMSD_A="+str(result["reference_results"]["2FN2"]["all_models_summary"]["CA_RMSD_A"]["median"]))
        xray=result["reference_results"]["3MQL"]["all_conformers"][0]
        print("JANUS_FN1_3MQL_GLOBAL_CA_RMSD_A="+str(xray["full_FNII_pair_Kabsch"]["global_CA_RMSD_A"]))
        print("JANUS_FN1_3MQL_FNII2_ANCHORED_CA_RMSD_A="+str(xray["UniProt_fnII_domains"]["FNII_2_after_FNII_1_alignment"]["CA_RMSD_A"]))
        print("JANUS_ROWAN_NEW_JOBS=0")
        return 0
    except (Hold,ValueError,TypeError,KeyError,IndexError,OSError) as e:
        print("JANUS_FN1_2FN2_3MQL=HOLD_"+str(e))
        return 2
    except Exception:
        print("JANUS_FN1_2FN2_3MQL=HOLD_UNCLASSIFIED_PROVIDER_OR_STRUCTURE_READ")
        return 3
if __name__=="__main__":
    raise SystemExit(main())
