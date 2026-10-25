#!/usr/bin/env python3
"""Read-only BBCl wildtype/K5A structure superposition on private Rowan PDBs.

Downloads structures transiently into a GitHub runner private tmp directory.
Only summary geometry is emitted; never upload PDBs, coordinates or workflow IDs.
No provider calculation, provider write, or model interpretation as fitness.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from probe import prepare, TARGET_ROWAN_FOLDER_UUID
from rowan_bbc1_autopilot import CASES, existing_by_name, BudgetHold

ROOT = Path(__file__).resolve().parent
AA3 = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"
))

class GeometryHold(ValueError):
    pass

def parse_pdb_ca(data: bytes, expected_sequence: str) -> np.ndarray:
    if not data or len(data) > 8_000_000:
        raise GeometryHold("PDB_MISSING_OR_TOO_LARGE")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise GeometryHold("PDB_NOT_ASCII") from exc
    chains: dict[str, list[tuple[tuple[str,str],str,list[float]]]] = {}
    seen: set[tuple[str,str,str]] = set()
    for line in text.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54 or line[12:16].strip() != "CA":
            continue
        if line[16:17] not in (" ","A"):
            continue
        ch = line[21:22]
        rid = (line[22:26],line[26:27])
        name = line[17:20].strip()
        if name not in AA3:
            raise GeometryHold("NONSTANDARD_CA_RESIDUE")
        key = (ch,*rid)
        if key in seen:
            raise GeometryHold("DUPLICATE_CA_RESIDUE")
        seen.add(key)
        try:
            xyz = [float(line[a:b]) for a,b in ((30,38),(38,46),(46,54))]
        except ValueError as exc:
            raise GeometryHold("BAD_PDB_COORDINATE") from exc
        if not all(math.isfinite(v) for v in xyz):
            raise GeometryHold("NONFINITE_COORDINATE")
        chains.setdefault(ch,[]).append((rid,AA3[name],xyz))
    candidates=[]
    for chain,atoms in chains.items():
        seq="".join(item[1] for item in atoms)
        if seq==expected_sequence:
            if len(atoms)!=len(expected_sequence):
                raise GeometryHold("CHAIN_LENGTH_MISMATCH")
            candidates.append(np.array([item[2] for item in atoms],dtype=float))
    if len(candidates)!=1:
        raise GeometryHold("EXPECTED_UNIQUE_64AA_CHAIN_NOT_FOUND")
    return candidates[0]

def kabsch_summary(wildtype: np.ndarray, mutant: np.ndarray)->dict:
    if wildtype.shape!=mutant.shape or wildtype.shape!=(64,3):
        raise GeometryHold("INVALID_WT_MUTANT_COORDINATE_DIMENSIONS")
    if not np.isfinite(wildtype).all() or not np.isfinite(mutant).all():
        raise GeometryHold("INVALID_COORDINATE_VALUES")
    p=wildtype-wildtype.mean(axis=0)
    q=mutant-mutant.mean(axis=0)
    u,_,vt=np.linalg.svd(p.T@q)
    reflection=np.linalg.det(u@vt)
    correction=np.diag([1.,1., -1. if reflection<0 else 1.])
    rotation=u@correction@vt
    aligned=p@rotation+mutant.mean(axis=0)
    per=np.linalg.norm(aligned-mutant,axis=1)
    if not np.isfinite(per).all():
        raise GeometryHold("NONFINITE_SUPERPOSITION_RESULT")
    return {
        "global_Calpha_rmsd_angstrom":round(float(np.sqrt(np.mean(per**2))),6),
        "mutation_position_5_Calpha_displacement_angstrom":round(float(per[4]),6),
        "positions_1_to_9_local_rmsd_angstrom":round(float(np.sqrt(np.mean(per[:9]**2))),6),
        "residues_superposed":64,
        "fit":"global_Kabsch_SVD_CA_no_scaling_no_reflections",
        "structural_fitness_claim":False,
        "one_prediction_per_arm":True,
    }

def get_pdb(api, case, workflow, tempdir):
    w=workflow.fetch_latest()
    if str(getattr(w,"parent_uuid",None))!=TARGET_ROWAN_FOLDER_UUID or not w.done():
        raise GeometryHold("WORKFLOW_NOT_FINISHED_OR_MOVED")
    result=w.result(wait=False)
    protein=result.get_predicted_structure()
    if protein is None:
        raise GeometryHold("NO_PREDICTED_STRUCTURE")
    sub=Path(tempdir)/case["arm"]
    sub.mkdir(parents=True,exist_ok=True)
    protein.download_pdb_file(path=sub,name=case["arm"])
    files=list(sub.rglob("*.pdb"))
    if len(files)!=1:
        raise GeometryHold("STRUCTURE_PDB_NOT_UNIQUE")
    data=files[0].read_bytes()
    return parse_pdb_ca(data,case["sequence"]),hashlib.sha256(data).hexdigest()

def analyze(api, manifest):
    api.whoami()
    folder=api.retrieve_folder(TARGET_ROWAN_FOLDER_UUID)
    if str(getattr(folder,"uuid",None))!=TARGET_ROWAN_FOLDER_UUID:
        raise GeometryHold("APPROVED_FOLDER_NOT_FOUND")
    cases=[prepare(manifest,arm) for arm in CASES]
    found=existing_by_name(api,TARGET_ROWAN_FOLDER_UUID,[c["id"] for c in cases])
    if any(len(found[c["id"]])!=1 for c in cases):
        raise GeometryHold("BOTH_DISTINCT_CASES_REQUIRED")
    with TemporaryDirectory(prefix="janus-rowan-private-") as tmp:
        coords, hashes={},{}
        for case in cases:
            coords[case["arm"]],hashes[case["arm"]]=get_pdb(api,case,found[case["id"]][0],tmp)
        summary=kabsch_summary(coords["wildtype"],coords["K5A"])
    return {
        "schema_id":"janus.rowan.bbc1.private_structure_superposition.v1",
        "assay_id":cases[0]["reference"],
        "model_family":"Boltz-2, as specified in the preregistered JANUS submission",
        "results_are_model_predictions_not_experimental_structures":True,
        "input_sequence_verified_by_pdb_residue_identity":True,
        "structure_data_sha256":hashes,
        "geometry":summary,
        "provenance_scope":"PRIVATE_PROVIDER_READ_ONLY_PDB_DOWNLOADED_TRANSIENTLY",
        "raw_structures_published":False,
        "free_energy_change_inferred":False,
        "fitness_change_inferred":False,
        "biological_replicates":0,
        "topa_independent_review":"OPEN",
    }

def main():
    if not os.environ.get("ROWAN_API_KEY"):
        print("JANUS_BBC1_GEOMETRY=NO_KEY");return 2
    try:
        manifest=json.loads((ROOT/"discovery_manifest.json").read_text(encoding="utf-8"))
        api=importlib.import_module("rowan")
        report=analyze(api,manifest)
    except (GeometryHold,BudgetHold) as exc:
        print("JANUS_BBC1_GEOMETRY_HOLD="+str(exc));return 0
    except Exception:
        print("JANUS_BBC1_GEOMETRY=API_OR_RUNTIME_BLOCKED");return 3
    g=report["geometry"]
    print("JANUS_BBC1_GEOMETRY=PASS_NO_BILLING")
    for key in ("global_Calpha_rmsd_angstrom",
                "mutation_position_5_Calpha_displacement_angstrom",
                "positions_1_to_9_local_rmsd_angstrom"):
        print("JANUS_BBC1_"+key.upper()+"="+str(g[key]))
    output=os.getenv("JANUS_BBC1_GEOMETRY_RECEIPT")
    if output:
        p=Path(output)
        if p.exists(): print("JANUS_BBC1_GEOMETRY=RECEIPT_EXISTS");return 3
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("x",encoding="utf-8") as f:
            json.dump(report,f,indent=2,allow_nan=False)
            f.write("\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
