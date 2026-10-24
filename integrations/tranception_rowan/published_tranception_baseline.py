#!/usr/bin/env python3
"""Attest the *published* Tranception/ProteinGym benchmark, without model execution.

This audits precomputed Spearman coefficients from official ProteinGym Git blob.
It is not a new Tranception forward pass. It must NEVER be merged with a
different assay subset (e.g. third-party 1084 single-mutant TSV).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = {
    "repo": "OATML-Markslab/ProteinGym",
    "path": "benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv",
    "git_blob_sha1": "b5affd05e5bf83eecf5d617ddb56168d7e80bc6f",
    "assay": "BBC1_YEAST_Tsuboyama_2023_1TG0",
}
SOURCE["git_blob_url"] = (
    "https://api.github.com/repos/" + SOURCE["repo"]
    + "/git/blobs/" + SOURCE["git_blob_sha1"]
)
METHODS = (
    "Tranception S no retrieval",
    "Tranception M no retrieval",
    "Tranception L no retrieval",
    "Tranception S",
    "Tranception M",
    "Tranception L",
    "TranceptEVE S",
    "TranceptEVE M",
    "TranceptEVE L",
)
PUBLIC_METADATA = ("Number of Mutants", "Selection Type", "UniProt ID")


class EvidenceError(ValueError):
    pass


def git_blob_sha(content: bytes) -> str:
    # Git object SHAs are not SHA-256 file hashes. Compute exact git blob SHA-1.
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def extract(content: bytes, *, expected_blob_sha: str = SOURCE["git_blob_sha1"]) -> dict:
    if not isinstance(content, bytes) or not 1 <= len(content) <= 3_000_000:
        raise EvidenceError("Missing or oversized upstream CSV")
    if git_blob_sha(content) != expected_blob_sha:
        raise EvidenceError("Official Git blob SHA-1 changed")
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    except UnicodeDecodeError as exc:
        raise EvidenceError("Upstream CSV decoding failed") from exc
    missing = set(( "DMS ID", *METHODS, *PUBLIC_METADATA)) - set(reader.fieldnames or [])
    if missing:
        raise EvidenceError("Missing official columns: " + ",".join(sorted(missing)))
    rows = [r for r in reader if r.get("DMS ID", "").strip() == SOURCE["assay"]]
    if len(rows) != 1:
        raise EvidenceError("Unique BBC1 benchmark row missing")
    r = rows[0]
    try:
        n = int(r["Number of Mutants"])
        values = {k: float(r[k]) for k in METHODS}
    except (TypeError, ValueError) as exc:
        raise EvidenceError("Invalid BBC1 benchmark numeric field") from exc
    if n != 2069:
        raise EvidenceError("Upstream benchmark population changed")
    if any(not math.isfinite(v) or not -1 <= v <= 1 for v in values.values()):
        raise EvidenceError("Invalid rank correlation")
    if r["Selection Type"] != "Stability":
        raise EvidenceError("Unexpected selection assay type")
    return {
        "schema_id": "janus.tranception.proteingym_published_bbc1_baseline.v1",
        "evidence_source": {
            "repository": SOURCE["repo"],
            "path": SOURCE["path"],
            "git_blob_sha1": expected_blob_sha,
            "source_kind": "OFFICIAL_PRECOMPUTED_PUBLISHED_AGGREGATE",
        },
        "assay_id": SOURCE["assay"],
        "number_of_mutants_reported_in_official_benchmark": n,
        "selection_type": r["Selection Type"],
        "uniprot_id": r["UniProt ID"],
        "spearman_rho": values,
        "reported_metrics_are_historical_not_new_predictions": True,
        "model_weights_executed_by_janus": False,
        "rowan_computation_executed": False,
        "directly_comparable_to_1084_single_variant_derivative": False,
        "benchmark_coverage_difference": (
            "Official benchmark says 2069 mutants. Separate pinned third-party "
            "single-substitution derivative contains 1084 unique sequences. "
            "Do not assume matching records or assay transformation."
        ),
        "topa_validation": "PENDING_PRIMARY_ASSAY_AND_MODEL_REPRODUCTION",
    }


def download_pinned_git_blob() -> bytes:
    req = Request(SOURCE["git_blob_url"], headers={
        "User-Agent": "JANUS-Evidence-Gate/1", "Accept": "application/vnd.github+json"
    })
    with urlopen(req, timeout=30) as response:
        raw = response.read(4_100_000)
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("sha") != SOURCE["git_blob_sha1"] or payload.get("encoding") != "base64":
        raise EvidenceError("Blob metadata mismatch")
    data = base64.b64decode(payload["content"], validate=False)
    if len(data) != payload.get("size"):
        raise EvidenceError("Blob size mismatch")
    return data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-file", type=Path, help="Optional already-fetched exact CSV")
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite immutable receipt")
    try:
        data = args.source_file.read_bytes() if args.source_file else download_pinned_git_blob()
        result = extract(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print("TRANCEPTION_OFFICIAL_BASELINE_SHA=PASS")
        print("OFFICIAL_BBC1_MUTANT_COUNT=" + str(result["number_of_mutants_reported_in_official_benchmark"]))
        for name in METHODS:
            print(name.replace(" ", "_").upper() + "=" + str(result["spearman_rho"][name]))
        print("ACTUAL_TRANCEPTION_OR_ROWAN_RUN=NO")
        return 0
    except (ValueError, OSError) as exc:
        print("TRANCEPTION_BASELINE_BLOCKED=" + type(exc).__name__)
        return 2
    except Exception:
        print("TRANCEPTION_BASELINE_REMOTE_OR_RUNTIME_ERROR")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
