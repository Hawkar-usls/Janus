#!/usr/bin/env python3
"""Blind, reproducible descriptive audit of published BBC1 single-mutant labels.

Do not interpret processed assay values as delta-G, Boltz structure-confidence,
disease causality, or de-novo JANUS measurements. No Rowan API, no credentials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median

from dms_evidence import SOURCE, SourceIntegrityError, get_pinned_dataset, MAX_BYTES

ROOT = Path(__file__).resolve().parent
ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def analyze(data: bytes, manifest: dict, *, sha256: str = SOURCE["sha256"]) -> dict:
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise SourceIntegrityError("BBC1 assay file missing or over permitted size")
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SourceIntegrityError("Dataset bytes changed since frozen public source")
    assay = manifest.get("independent_control", {})
    if assay.get("assay_id") != "BBC1_YEAST_Tsuboyama_2023_1TG0":
        raise SourceIntegrityError("Unexpected assay")
    wt = assay.get("wildtype", "")
    mut = assay.get("mutant", "")
    if not isinstance(wt, str) or len(wt) != 64 or len(mut) != 64 or wt[4] != "K" or mut[4] != "A":
        raise SourceIntegrityError("Mutant/reference mismatch")
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter="\t")
    if rows.fieldnames is None or not {"sequences", "labels", "fold_id"} <= set(rows.fieldnames):
        raise SourceIntegrityError("Required assay fields not found")

    labels: list[float] = []
    by_fold = [0] * 5
    found: list[float] = []
    seqs: set[str] = set()
    residue_counts: dict[str, int] = {}
    all_one_mutant = True
    for row in rows:
        seq = (row.get("sequences") or "").strip()
        if len(seq) != len(wt) or not set(seq).issubset(ALPHABET):
            raise SourceIntegrityError("Noncanonical / misaligned mutant")
        # Defensive: no label for same sequence twice.
        if seq in seqs:
            raise SourceIntegrityError("Duplicate protein sequence")
        seqs.add(seq)
        changes = [(i + 1, a, b) for i, (a, b) in enumerate(zip(wt, seq)) if a != b]
        if len(changes) != 1:
            all_one_mutant = False
        else:
            key = str(changes[0][0])
            residue_counts[key] = residue_counts.get(key, 0) + 1
        try:
            score = float(row["labels"])
            fold = int(row["fold_id"])
        except (TypeError, ValueError) as exc:
            raise SourceIntegrityError("Unparseable assay value or benchmark fold") from exc
        if not math.isfinite(score) or not 0 <= fold < 5:
            raise SourceIntegrityError("Invalid score or fold")
        labels.append(score)
        by_fold[fold] += 1
        if seq == mut:
            found.append(score)
        if len(labels) > 100_000:
            raise SourceIntegrityError("Too many rows")
    if not labels or len(found) != 1:
        raise SourceIntegrityError("No unique exact K5A observation")
    # Compute rank and p10/p90 from the full assay without any model predictions.
    ordered = sorted(labels)
    n = len(ordered)
    k5a = found[0]
    lower = sum(x < k5a for x in labels)
    ties = sum(x == k5a for x in labels)
    upper = sum(x > k5a for x in labels)
    if lower + ties + upper != n:
        raise SourceIntegrityError("Rank accounting failed")

    def interp(q: float) -> float:
        at = (n - 1) * q
        a, b = math.floor(at), math.ceil(at)
        return ordered[a] + (ordered[b] - ordered[a]) * (at - a)

    return {
        "schema_id": "janus.proteingym.bbc1_assay_population_audit.v1",
        "source_dataset_revision": SOURCE["dataset_commit"],
        "source_dataset_sha256": digest,
        "assay_id": assay["assay_id"],
        "source_kind": "THIRD_PARTY_PROCESSED_PUBLIC_DMS",
        "sequence_length": len(wt),
        "row_count": n,
        "unique_sequence_count": len(seqs),
        "all_rows_exactly_one_amino_acid_substitution": all_one_mutant,
        "mutated_position_count": len(residue_counts),
        "fold_counts": by_fold,
        "label_distribution": {
            "minimum": ordered[0], "p10": interp(0.1),
            "median": median(ordered), "p90": interp(0.9),
            "maximum": ordered[-1],
        },
        "K5A": {
            "processed_label": k5a,
            "rank_ascending": lower + 1,
            "number_below": lower,
            "number_tied_including_K5A": ties,
            "number_above": upper,
            "percent_below": lower / n * 100,
            "relative_position": "DESCRIPTIVE_ASSAY_RANK_ONLY",
        },
        "rowan_model_prediction": "NOT_RUN",
        "tranception_model_prediction": "NOT_RUN",
        "independent_replicates": "NOT_VERIFIED",
        "label_units_and_transform": "NOT_INDEPENDENTLY_AUDITED",
        "topa_scientific_review": "OPEN",
        "conclusion": "This is descriptive statistics of previously processed experimental data, NOT a validated structure-function mechanism.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-file", type=Path,
                        help="local exact pinned TSV; otherwise download from public Hub")
    parser.add_argument("--manifest", type=Path, default=ROOT / "discovery_manifest.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists, won't overwrite")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        data = args.source_file.read_bytes() if args.source_file else get_pinned_dataset()
        report = analyze(data, manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        print("BBC1_ASSAY_DESCRIPTIVE_AUDIT=PASS")
        print("ASSAY_ROWS=" + str(report["row_count"]))
        print("UNIQUE_SEQUENCES=" + str(report["unique_sequence_count"]))
        print("ALL_SINGLE_MUTANTS=" + str(report["all_rows_exactly_one_amino_acid_substitution"]))
        print("FOLD_COUNTS=" + ",".join(map(str, report["fold_counts"])))
        print("K5A_PROCESSED_LABEL=" + repr(report["K5A"]["processed_label"]))
        print("K5A_ASCENDING_RANK=" + str(report["K5A"]["rank_ascending"]))
        print("K5A_PERCENT_BELOW=" + repr(report["K5A"]["percent_below"]))
        print("ASSAY_MEDIAN=" + repr(report["label_distribution"]["median"]))
        print("ROWAN_AND_TRANCEPTION_REAL_RUN=NO")
        return 0
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        print("BBC1_AUDIT_BLOCKED=" + type(exc).__name__)
        return 2
    except Exception:
        print("BBC1_AUDIT_DOWNLOAD_OR_RUNTIME_ERROR")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
