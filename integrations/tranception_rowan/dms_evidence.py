#!/usr/bin/env python3
"""Validate a pinned public BBC1 experimental-assay derivative against JANUS.

Data source: public ProteinGYM-DMS *processed* single-substitution TSV.
This script does not run Rowan/Tranception or infer stability from pTM.
Outputs only measured label attribution and provenance; do not call it new science.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 1_000_000
SOURCE = {
    "repo": "genbio-ai/ProteinGYM-DMS",
    "dataset_commit": "cc6b0b56553e694dbf27dc84c6b37a8686e64b85",
    "path": "singles_substitutions/BBC1_YEAST_Tsuboyama_2023_1TG0.tsv",
    "sha256": "ca355a655381c95fe899d61a406251e0cd17d68bbabe287651d526fdeb821f87",
}
SOURCE["url"] = (
    f"https://huggingface.co/datasets/{SOURCE['repo']}/resolve/"
    f"{SOURCE['dataset_commit']}/{SOURCE['path']}"
)


class SourceIntegrityError(ValueError):
    pass


def validate_tsv(data: bytes, manifest: dict, *, expected_sha256: str = SOURCE["sha256"]) -> dict:
    if not isinstance(data, bytes) or len(data) > MAX_BYTES or not data:
        raise SourceIntegrityError("Missing/oversized raw assay bytes")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise SourceIntegrityError("Assay SHA256 differs from preregistered source")
    try:
        decoded = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceIntegrityError("Invalid assay UTF-8") from exc

    rows = csv.DictReader(io.StringIO(decoded), delimiter="\t")
    if rows.fieldnames is None or not {"sequences", "labels", "fold_id"} <= set(rows.fieldnames):
        raise SourceIntegrityError("Expected processed ProteinGym TSV fields missing")
    control = manifest["independent_control"]
    if control["assay_id"] != "BBC1_YEAST_Tsuboyama_2023_1TG0":
        raise SourceIntegrityError("Unrecognized assay context")
    wt, target = control["wildtype"], control["mutant"]
    if len(wt) != 64 or len(target) != 64 or wt[4] != "K" or target[4] != "A":
        raise SourceIntegrityError("Frozen 64aa K5A sequence not intact")
    if sum(a != b for a, b in zip(wt, target)) != 1:
        raise SourceIntegrityError("K5A is not a single mutation")

    found = []
    total = 0
    for row in rows:
        total += 1
        if total > 100_000:
            raise SourceIntegrityError("Unexpectedly large dataset")
        seq = (row.get("sequences") or "").strip()
        if seq != target:
            continue
        try:
            label = float(row["labels"])
            fold = int(row["fold_id"])
        except (TypeError, ValueError) as exc:
            raise SourceIntegrityError("Invalid target label/fold") from exc
        if not math.isfinite(label) or fold not in range(5):
            raise SourceIntegrityError("Nonfinite target label or unknown fold")
        found.append({"processed_label": label, "fold_id": fold})
    if len(found) > 1:
        raise SourceIntegrityError("Duplicate K5A sequences; cannot bind one label")
    state = "EXPERIMENTAL_DERIVATIVE_ROW_VERIFIED" if found else "K5A_ROW_ABSENT"
    return {
        "schema_id": "janus.tranception.bbc1.assay_provenance.v1",
        "status": state,
        "source": {
            "repository": SOURCE["repo"],
            "dataset_revision": SOURCE["dataset_commit"],
            "dataset_path": SOURCE["path"],
            "sha256": actual,
            "type": "THIRD_PARTY_PROCESSED_DMS_NOT_PRIMARY_RAW_OBSERVATION",
        },
        "assay_id": control["assay_id"],
        "variant": "K5A",
        "processed_row": found[0] if found else None,
        "rows_scanned": total,
        "row_count_for_variant": len(found),
        "biological_replicates_verified": False,
        "rowan_job_run": False,
        "tranception_job_run": False,
        "has_valid_two_arm_correlation": False,
        "protein_fitness_causal_claim": False,
        "warning": "A processed assay label is not a new experiment. A two-arm comparison cannot validate a general mutation-fitness predictor.",
    }


def get_pinned_dataset() -> bytes:
    # No credentials; only a public immutable revision and SHA-256 are accepted.
    request = Request(SOURCE["url"], headers={"User-Agent": "JANUS-science-provenance/1"})
    with urlopen(request, timeout=30) as response:
        blob = response.read(MAX_BYTES + 1)
    return blob


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assay-tsv", type=Path, help="Existing offline TSV bytes")
    p.add_argument("--download-pinned-public-assay", action="store_true")
    p.add_argument("--manifest", type=Path, default=ROOT / "discovery_manifest.json")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if bool(args.assay_tsv) == args.download_pinned_public_assay:
        p.error("Select exactly one of --assay-tsv or --download-pinned-public-assay")
    if args.output.exists():
        p.error("Refusing to overwrite immutable provenance output")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        blob = get_pinned_dataset() if args.download_pinned_public_assay else args.assay_tsv.read_bytes()
        receipt = validate_tsv(blob, manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print("JANUS_DMS_EVIDENCE_GATE=" + receipt["status"])
        # Absence is a valid negative result; do not fabricate a row.
        return 0
    except (OSError, ValueError) as e:
        print("JANUS_DMS_EVIDENCE_BLOCKED=" + type(e).__name__)
        return 2
    except Exception:
        print("JANUS_DMS_EVIDENCE_DOWNLOAD_OR_RUNTIME_FAILED")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
