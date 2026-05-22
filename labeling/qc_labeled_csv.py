#!/usr/bin/env python3
"""QC a labeled ET-EDU CSV for CBIR training/evaluation.

Checks:
- Required columns + concept columns exist
- Labels are binary-ish (0/1); reports non-binary values
- Per-split / per-role counts
- Per-concept positive counts and sparsity
- Query->gallery matchability for val/test (overlap and exact matches)

This is intentionally conservative: it reports issues and can optionally fail in --strict mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


META_COLS = ["Image_Path", "split", "retrieval_role"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC labeled CSV for ET-EDU-CBIR-V2")
    p.add_argument("--csv", default="data/ET-EDU-CBIR-V2/labels_template.csv", help="Labeled CSV")
    p.add_argument(
        "--concepts-file",
        default=None,
        help="Concepts file (defaults to concepts.txt next to CSV)",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write a JSON QC report.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit non-zero) if critical issues are detected.",
    )
    p.add_argument(
        "--valtest-splits",
        nargs="+",
        default=["val", "test"],
        help="Which splits to treat as evaluation (query/gallery).",
    )
    return p.parse_args()


def load_concepts(concepts_file: Path) -> list[str]:
    concepts = [ln.strip() for ln in concepts_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not concepts:
        raise ValueError(f"No concepts in: {concepts_file}")
    return concepts


def _to_binary_matrix(df: pd.DataFrame, concept_cols: list[str]) -> np.ndarray:
    mat = df[concept_cols].copy()
    for c in concept_cols:
        mat[c] = (mat[c].fillna(0).astype(float) > 0.5).astype(np.int8)
    return mat.to_numpy(dtype=np.int8)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    concepts_file = Path(args.concepts_file) if args.concepts_file else (csv_path.parent / "concepts.txt")
    if not concepts_file.exists():
        raise FileNotFoundError(f"Missing concepts file: {concepts_file}")

    concept_cols = load_concepts(concepts_file)

    df = pd.read_csv(csv_path)
    for c in META_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    missing_concepts = [c for c in concept_cols if c not in df.columns]
    if missing_concepts:
        raise ValueError("Missing concept columns: " + ", ".join(missing_concepts))

    # Detect non-binary values (before normalization).
    nonbinary: dict[str, dict[str, int]] = {}
    for c in concept_cols:
        series = df[c].copy()
        series = series.fillna(0)
        # Try parse numbers; keep NaNs as-is (already filled).
        vals = pd.to_numeric(series, errors="coerce").fillna(0)
        uniq = sorted(set(float(x) for x in vals.unique()))
        bad = [u for u in uniq if u not in (0.0, 1.0)]
        if bad:
            # Count a few examples.
            counts = vals.value_counts().to_dict()
            nonbinary[c] = {str(k): int(v) for k, v in counts.items()}

    # Normalize to {0,1} for computations.
    norm_df = df.copy()
    for c in concept_cols:
        norm_df[c] = (pd.to_numeric(norm_df[c], errors="coerce").fillna(0).astype(float) > 0.5).astype(int)

    report: dict[str, object] = {
        "csv": str(csv_path),
        "rows": int(len(norm_df)),
        "concepts": concept_cols,
        "splits": norm_df["split"].value_counts().to_dict(),
        "retrieval_roles": norm_df["retrieval_role"].value_counts().to_dict(),
        "nonbinary_columns": nonbinary,
    }

    # Per-concept stats
    pos_counts = {c: int(norm_df[c].sum()) for c in concept_cols}
    report["positive_counts"] = pos_counts
    report["positive_rates"] = {c: float(pos_counts[c] / max(1, len(norm_df))) for c in concept_cols}

    label_mat = _to_binary_matrix(norm_df, concept_cols)
    row_pos = label_mat.sum(axis=1)
    report["rows_all_zero"] = int((row_pos == 0).sum())
    report["rows_multi_label_ge2"] = int((row_pos >= 2).sum())

    # Query->gallery matchability (val/test)
    valtest = set(args.valtest_splits)
    eval_df = norm_df[norm_df["split"].isin(valtest)].copy()
    matchability: dict[str, dict[str, object]] = {}

    for split in sorted(eval_df["split"].unique()):
        split_df = eval_df[eval_df["split"] == split]
        q_df = split_df[split_df["retrieval_role"] == "query"].copy()
        g_df = split_df[split_df["retrieval_role"] == "gallery"].copy()

        if q_df.empty or g_df.empty:
            matchability[split] = {
                "queries": int(len(q_df)),
                "galleries": int(len(g_df)),
                "warning": "missing queries or galleries",
            }
            continue

        q_mat = _to_binary_matrix(q_df, concept_cols)
        g_mat = _to_binary_matrix(g_df, concept_cols)

        # overlap_count[i] = number of gallery items sharing >=1 positive label with query i
        overlap_counts = []
        exact_counts = []
        for i in range(q_mat.shape[0]):
            qv = q_mat[i]
            if qv.sum() == 0:
                overlap_counts.append(0)
                exact_counts.append(0)
                continue
            shared = (g_mat @ qv)  # number of shared labels per gallery
            overlap_counts.append(int((shared > 0).sum()))
            exact_counts.append(int((g_mat == qv).all(axis=1).sum()))

        overlap_counts_np = np.asarray(overlap_counts)
        exact_counts_np = np.asarray(exact_counts)

        matchability[split] = {
            "queries": int(len(q_df)),
            "galleries": int(len(g_df)),
            "queries_all_zero": int((q_mat.sum(axis=1) == 0).sum()),
            "queries_no_overlap": int((overlap_counts_np == 0).sum()),
            "queries_no_exact": int((exact_counts_np == 0).sum()),
            "overlap_median": float(np.median(overlap_counts_np)),
            "overlap_p10": float(np.percentile(overlap_counts_np, 10)),
            "overlap_p90": float(np.percentile(overlap_counts_np, 90)),
        }

    report["query_gallery_matchability"] = matchability

    # Print human-friendly summary
    print("QC summary")
    print(f"- CSV: {csv_path}")
    print(f"- Rows: {len(norm_df)} | Concepts: {len(concept_cols)}")
    print(f"- Splits: {report['splits']}")
    print(f"- Roles: {report['retrieval_roles']}")
    print(f"- Rows all-zero labels: {report['rows_all_zero']}")

    if nonbinary:
        print(f"- WARNING: non-binary values detected in {len(nonbinary)} concept columns")
        if args.strict:
            raise SystemExit(2)

    for split, srep in matchability.items():
        if "warning" in srep:
            print(f"- {split}: WARNING {srep['warning']} (q={srep['queries']}, g={srep['galleries']})")
        else:
            print(
                f"- {split}: q={srep['queries']} g={srep['galleries']} | "
                f"q_all_zero={srep['queries_all_zero']} q_no_overlap={srep['queries_no_overlap']} "
                f"(median overlap {srep['overlap_median']})"
            )

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"- Wrote JSON report: {out_json}")


if __name__ == "__main__":
    main()
