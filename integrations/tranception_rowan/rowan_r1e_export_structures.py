"""Read-only R1E export of five already-completed public 1EY0 WT models."""
from pathlib import Path
import hashlib,json,os,re,math
from rowan_r1e_multisample import NAME,score_fields
from rowan_r1_cloud_pair import find,Hold
from rowan_1ey0_weekly_api import WT,WT_SHA256
EXPECTED={
"c04bd162896d58193480352f1ceed3e96b25d05a06b325800272fe70927cab89": (.94,.951),
"41c633dc51d567fb2ee4033bed5547b09af98fa854cc16bdd4d995464a615035": (.94,.950),
"b41a8a7fcebbbda1fbfc8c31387c6d73e796a6bcc13c2d85655307d8130ca03a": (.941,.948),
"1411ce784d375cdccbfa5baec7cf857ea6705f357609689f7d23af01252f1e3c": (.939,.948),
"c7c33531f3e90d8d84b5957dcad34816bd43b33d398faf858d818439f00b5c55": (.938,.945)}
AA=dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),"ARNDCQEGHILKMFPSTWYV"))
def sha(data):return hashlib.sha256(data).hexdigest()
def verify_atoms(raw):
    atoms=[s for s in raw.splitlines() if s.startswith("ATOM  ")]
    if not atoms:raise Hold("PDB_NO_ATOMS")
    residues={}
    ca={}
    for s in atoms:
        if len(s)<54 or s[17:20] not in AA:raise Hold("PDB_ATOM_FORMAT")
        coords=[float(s[i:i+8]) for i in (30,38,46)]
        if not all(math.isfinite(x) for x in coords):raise Hold("PDB_NONFINITE")
        key=(s[21],s[22:27]);aa=AA[s[17:20]]
        if key in residues and residues[key]!=aa:raise Hold("RESIDUE_CONFLICT")
        residues[key]=aa
        if s[12:16].strip()=="CA":ca[key]=ca.get(key,0)+1
    seq="".join(residues.values())
    if len(set(k[0] for k in residues))!=1 or seq!=WT or len(ca)!=len(WT) or any(v!=1 for v in ca.values()):
        raise Hold("PDB_SEQUENCE_OR_CA_MISMATCH")
    clean=("\n".join(atoms)+"\nEND\n").encode("ascii")
    return clean,len(atoms),len(residues)
def export(api,out):
    from probe import TARGET_ROWAN_FOLDER_UUID
    job=find(api,NAME)
    if job is None:raise Hold("EXISTING_JOB_MISSING")
    job=job.fetch_latest()
    if job.name!=NAME or str(job.parent_uuid)!=TARGET_ROWAN_FOLDER_UUID or str(job.status)!="Status.COMPLETED_OK" or not job.done():raise Hold("JOB_NOT_COMPLETED")
    if float(job.credits_charged)!=5.69:raise Hold("COST_CHANGED")
    rs=job.result(wait=False)
    p=list(rs.predictions)
    if len(p)!=5:raise Hold("COUNT")
    byhash={}
    for s in p:
        uid=str(s.predicted_structure_uuid or "");h=sha(uid.encode())
        if not uid or h not in EXPECTED or h in byhash:raise Hold("UUID_BINDING")
        conf=score_fields(s.scores)
        if (conf["ptm"],conf["avg_lddt"])!=EXPECTED[h]:raise Hold("SCORE_DRIFT")
        byhash[h]=uid
    out.mkdir(exist_ok=False)
    temp=out.parent/"r1e-pdb-raw";temp.mkdir(exist_ok=False)
    rows=[]
    for i,h in enumerate(EXPECTED,1):
        uid=byhash[h];protein=api.retrieve_protein(uid,workflow_uuid=str(job.uuid))
        name=f"1EY0_WT_R1E_sample_{i:02}"
        path=protein.download_structure(path=temp,name=name,workflow_uuid=str(job.uuid),file_format="pdb")
        if not Path(path).is_file() or Path(path).resolve()!= (temp/(name+".pdb")).resolve():raise Hold("PDB_EXPORT_PATH")
        clean,n_atoms,n_res=verify_atoms(Path(path).read_text(encoding="utf8"))
        dest=out/(name+".pdb")
        with dest.open("xb") as f:f.write(clean)
        rows.append({"sample":i,"structure_uuid_sha256":h,"pdb_sha256":sha(clean),"atom_count":n_atoms,"residue_count":n_res,"filename":dest.name,"ptm":EXPECTED[h][0],"avg_lddt":EXPECTED[h][1]})
    if len(set(x["pdb_sha256"] for x in rows))!=5:raise Hold("IDENTICAL_COORDINATE_EXPORTS")
    report={"status":"PASS_FIVE_EXPORTED_PDBS","case":NAME,"source_collection_run":"44531233894","sequence_sha256":WT_SHA256,"samples":rows,"existing_job_credits":5.69,"new_jobs":0,"new_computations":0,"scope":"SDK-converted ATOM coordinates; metadata stripped; no external structural validation or verified independent seeds"}
    (out/"manifest.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")
    print("R1E_COORDINATES=PASS_FIVE_EXPORTED_PDBS")
    print("R1E_COORDINATES_COUNT=5")
def main():
    try:
        import rowan
        from importlib.metadata import version
        if version("rowan-python")!="3.2.3" or version("stjames")!="0.0.289":raise Hold("SDK_DRIFT")
        export(rowan,Path(os.environ["R1E_EXPORT_DIR"]));return 0
    except Exception:
        print("R1E_COORDINATES_HOLD_NO_RETRY");return 2
if __name__=="__main__":raise SystemExit(main())
