#!/usr/bin/env python3
"""Independent paired reanalysis of *published* ProteinGym Tranception baselines.

Primary preregistered comparison: Tranception L retrieval vs L no retrieval.
Unit of inference = UniProt protein; averages multiple assays per UniProt.
S/M comparisons descriptive/exploratory only. No model weights or Rowan cloud run.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path
import random
import statistics

from published_tranception_baseline import (
    SOURCE, EvidenceError, git_blob_sha, download_pinned_git_blob,
)

PAIRS = {
    "L_PRIMARY": ("Tranception L no retrieval", "Tranception L"),
    "M_EXPLORATORY": ("Tranception M no retrieval", "Tranception M"),
    "S_EXPLORATORY": ("Tranception S no retrieval", "Tranception S"),
}
BOOTSTRAPS = 2000
SEED = 20261025


def num(cell: str | None) -> float | None:
    if cell is None or cell.strip() in {"", "NA", "N/A", "-", "nan", "NaN"}:
        return None
    try:
        v = float(cell)
    except ValueError as exc:
        raise EvidenceError("Invalid numeric rho") from exc
    if not math.isfinite(v) or not -1 <= v <= 1:
        raise EvidenceError("Rank correlation outside [-1,1]")
    return v


def quantile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    at = (len(vals) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return vals[lo] + (vals[hi] - vals[lo]) * (at - lo)


def protein_summary(proteins: dict[str, list[float]]) -> dict:
    means = [statistics.mean(proteins[k]) for k in sorted(proteins)]
    n = len(means)
    if not n:
        return {"status": "NO_COMPLETE_PROTEIN_PAIRS", "n_proteins": 0}
    win = sum(x > 0 for x in means)
    loss = sum(x < 0 for x in means)
    ties = n - win - loss
    rng = random.Random(SEED)
    boot = [statistics.mean(rng.choices(means, k=n)) for _ in range(BOOTSTRAPS)]
    return {
        "status": "PUBLISHED_PAIRED_ANALYSIS_NOT_MODEL_EXECUTION",
        "n_proteins": n,
        "n_assay_pairs": sum(map(len, proteins.values())),
        "mean_protein_delta_rho": statistics.mean(means),
        "median_protein_delta_rho": statistics.median(means),
        "winning_proteins": win,
        "losing_proteins": loss,
        "tied_proteins": ties,
        "bootstrap_mean_protein_delta_95pct_CI": [
            quantile(boot, 0.025), quantile(boot, 0.975)
        ],
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "ci_assumptions": (
            "Nonparametric resampling of UniProt-level mean deltas; "
            "proteins not guaranteed independent; NOT a causal or "
            "new experimental uncertainty interval"
        ),
    }


def analyze(data: bytes, *, expected_blob_sha: str = SOURCE["git_blob_sha1"],
            min_references: int = 1) -> dict:
    if not isinstance(data, bytes) or not 1 <= len(data) <= 2_000_000:
        raise EvidenceError("Unexpected official CSV size")
    if git_blob_sha(data) != expected_blob_sha:
        raise EvidenceError("Pinned official Git blob mismatch")
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    except UnicodeDecodeError as exc:
        raise EvidenceError("Bad CSV UTF-8") from exc
    cols = {"DMS ID", "UniProt ID", "Selection Type", "Number of Mutants"}
    for x, y in PAIRS.values():
        cols.update((x, y))
    if reader.fieldnames is None or not cols <= set(reader.fieldnames):
        raise EvidenceError("Missing official pairing/identification columns")

    paired = {k: defaultdict(list) for k in PAIRS}
    by_type = {k: defaultdict(lambda: defaultdict(list)) for k in PAIRS}
    counts = {"total_assays": 0, "distinct_DMS_ids": 0, "missing_uniprot_ids": 0}
    skipped = {k: 0 for k in PAIRS}
    seen_ids = set()
    bbc1 = None
    for row in reader:
        name = (row.get("DMS ID") or "").strip()
        if not name or name in seen_ids:
            raise EvidenceError("Empty or duplicate DMS ID")
        seen_ids.add(name)
        counts["total_assays"] += 1
        prot = (row.get("UniProt ID") or "").strip()
        selection = (row.get("Selection Type") or "").strip()
        if not prot:
            counts["missing_uniprot_ids"] += 1
        if name == SOURCE["assay"]:
            bbc1 = row
            if int(row["Number of Mutants"]) != 2069:
                raise EvidenceError("Published BBC1 cohort count drift")
        for pair, (baseline, improved) in PAIRS.items():
            before, after = num(row.get(baseline)), num(row.get(improved))
            if not prot or before is None or after is None:
                skipped[pair] += 1
                continue
            delta = after - before
            paired[pair][prot].append(delta)
            by_type[pair][selection][prot].append(delta)
    counts["distinct_DMS_ids"] = len(seen_ids)
    if not bbc1:
        raise EvidenceError("Frozen BBC1 control absent")
    if counts["total_assays"] < min_references:
        raise EvidenceError("Insufficient upstream reference rows")
    bbc1_deltas = {}
    for pair, (before, after) in PAIRS.items():
        a, b = num(bbc1.get(before)), num(bbc1.get(after))
        bbc1_deltas[pair] = None if a is None or b is None else b - a

    comparisons = {}
    for pair, values in paired.items():
        group_summaries = {}
        for sel, subgroups in sorted(by_type[pair].items()):
            if len(subgroups) >= 10:
                group_summaries[sel] = protein_summary(subgroups)
        comparisons[pair] = {
            **protein_summary(values),
            "excluded_assays_with_missing_pair_or_protein_id": skipped[pair],
            "selection_type_breakdown_exploratory_min_10_proteins": group_summaries,
        }
    return {
        "schema_id": "janus.tranception.proteingym.paired_retrieval_audit.v1",
        "evidence_type": "DERIVED_FROM_OFFICIAL_PRECOMPUTED_PUBLISHED_RESULTS",
        "source": {
            "repository": SOURCE["repo"], "path": SOURCE["path"],
            "exact_git_blob_sha1": expected_blob_sha,
        },
        "population": counts,
        "primary_preregistered_question": "Does retrieval improve Tranception L assay-level Spearman when averaged per UniProt?",
        "primary_statistical_unit": "PROTEIN_UNIPROT_ID",
        "comparisons": comparisons,
        "bbc1_delta_rho": bbc1_deltas,
        "scientific_qualifications": [
            "Only published per-assay Spearman coefficients, NOT per-mutant fitness scores",
            "No new Tranception model invocation and no Rowan computation",
            "Correlated proteins/families and assay methodology may violate bootstrap independence",
            "No claim of causal benefit or superiority on unseen proteins",
            "Bootstrap interval describes across-protein variability of published reported correlations",
            "Tranception S/M are secondary exploratory comparisons",
            "Official 2069-mutant BBC1 benchmark is not interchangeable with 1084 single-mutant derivative",
        ],
        "rowan_job_uuid": None,
        "model_weights_rerun": False,
        "topa_scientific_review": "OPEN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path,
                        help="Offline exact published original CSV (optional)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse overwriting frozen analysis")
    try:
        content = args.source_file.read_bytes() if args.source_file else download_pinned_git_blob()
        receipt = analyze(content, min_references=200)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        print("TRANCEPTION_PAIR_AUDIT=PASS")
        print("SOURCE_ASSAYS=" + str(receipt["population"]["total_assays"]))
        for model, report in receipt["comparisons"].items():
            print("MODEL=" + model)
            print("PROTEINS=" + str(report["n_proteins"]))
            print("PAIRED_ASSAYS=" + str(report["n_assay_pairs"]))
            print("MEAN_DELTA_RHO=" + repr(report["mean_protein_delta_rho"]))
            print("MEDIAN_DELTA_RHO=" + repr(report["median_protein_delta_rho"]))
            print("BOOTSTRAP_95CI=" + repr(report["bootstrap_mean_protein_delta_95pct_CI"]))
            print("WINS_LOSSES_TIES=" + repr([
                report["winning_proteins"], report["losing_proteins"],
                report["tied_proteins"]
            ]))
        print("ROWAN_LIVE_JOB=NOT_RUN")
        return 0
    except (ValueError, OSError) as exc:
        print("TRANCEPTION_PAIR_AUDIT_BLOCKED=" + type(exc).__name__)
        return 2
    except Exception:
        print("TRANCEPTION_PAIR_AUDIT_RUNTIME_ERROR")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
