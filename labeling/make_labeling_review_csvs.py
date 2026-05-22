#!/usr/bin/env python3
"""Create human-friendly CSV slices for manual labeling (query-first) with extra context.

These review CSVs are meant for annotators (Excel/Sheets/Label Studio import), not necessarily
for training directly. The conversion script `build_edu_txt_from_labeled_csv.py` is concept-aware
(via concepts.txt) so it will ignore extra context columns if present.

Outputs (default):
- <dataset_root>/labeling_review/queries_val_test_review.csv
- <dataset_root>/labeling_review/gallery_val_test_review.csv
- <dataset_root>/labeling_review/train_review.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


META_COLS = ["Image_Path", "split", "retrieval_role"]
DEFAULT_CONTEXT_COLS = [
    "video_id",
    "camera_id",
    "timestamp_sec",
    "frame_idx",
    "track_id",
    "distance_bucket",
    "quality_score",
    "det_conf",
    "laplacian_var",
    "brightness",
    "contrast",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build review CSVs for labeling ET-EDU-CBIR-V2")
    p.add_argument(
        "--dataset-root",
        default="data/ET-EDU-CBIR-V2",
        help="Dataset root containing labels_template.csv + metadata.csv + concepts.txt",
    )
    p.add_argument(
        "--labels-csv",
        default=None,
        help="Path to labels_template.csv (defaults to <dataset-root>/labels_template.csv)",
    )
    p.add_argument(
        "--metadata-csv",
        default=None,
        help="Path to metadata.csv (defaults to <dataset-root>/metadata.csv)",
    )
    p.add_argument(
        "--concepts-file",
        default=None,
        help="Concepts file (defaults to <dataset-root>/concepts.txt)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (defaults to <dataset-root>/labeling_review)",
    )
    p.add_argument(
        "--context-cols",
        nargs="+",
        default=DEFAULT_CONTEXT_COLS,
        help="Metadata columns to include in review files.",
    )
    p.add_argument(
        "--valtest-splits",
        nargs="+",
        default=["val", "test"],
        help="Splits to treat as val/test for query/gallery review slices.",
    )
    return p.parse_args()


def load_concepts(concepts_file: Path) -> list[str]:
    concepts = [ln.strip() for ln in concepts_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not concepts:
        raise ValueError(f"No concepts in: {concepts_file}")
    return concepts


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    labels_csv = Path(args.labels_csv) if args.labels_csv else (dataset_root / "labels_template.csv")
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else (dataset_root / "metadata.csv")
    concepts_file = Path(args.concepts_file) if args.concepts_file else (dataset_root / "concepts.txt")
    out_dir = Path(args.out_dir) if args.out_dir else (dataset_root / "labeling_review")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")
    if not concepts_file.exists():
        raise FileNotFoundError(f"Missing concepts file: {concepts_file}")

    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_csv)
    meta_df = pd.read_csv(metadata_csv)

    for c in META_COLS:
        if c not in labels_df.columns:
            raise ValueError(f"labels CSV missing required column: {c}")

    concept_cols = load_concepts(concepts_file)
    missing = [c for c in concept_cols if c not in labels_df.columns]
    if missing:
        raise ValueError("labels CSV missing concept columns: " + ", ".join(missing))

    # Prepare metadata for join.
    if "image_path" not in meta_df.columns:
        raise ValueError("metadata CSV missing image_path")

    meta_keep = ["image_path"] + [c for c in args.context_cols if c in meta_df.columns]
    meta_slim = meta_df[meta_keep].copy().drop_duplicates(subset=["image_path"], keep="first")
    meta_slim = meta_slim.rename(columns={"image_path": "Image_Path"})

    merged = labels_df.merge(meta_slim, on="Image_Path", how="left")

    # Column order: meta, context, concepts
    context_cols = [c for c in args.context_cols if c in merged.columns]
    merged = merged[META_COLS + context_cols + concept_cols]

    valtest = set(args.valtest_splits)
    q = merged[(merged["retrieval_role"] == "query") & (merged["split"].isin(valtest))].copy()
    g = merged[(merged["retrieval_role"] == "gallery") & (merged["split"].isin(valtest))].copy()
    t = merged[merged["split"] == "train"].copy()

    # Helpful stable sort: by split, then camera/video, then time.
    sort_cols = [c for c in ["split", "camera_id", "video_id", "timestamp_sec", "frame_idx"] if c in merged.columns]
    if sort_cols:
        q = q.sort_values(sort_cols)
        g = g.sort_values(sort_cols)
        t = t.sort_values(sort_cols)

    q_path = out_dir / "queries_val_test_review.csv"
    g_path = out_dir / "gallery_val_test_review.csv"
    t_path = out_dir / "train_review.csv"

    q.to_csv(q_path, index=False)
    g.to_csv(g_path, index=False)
    t.to_csv(t_path, index=False)

    print("Done.")
    print(f"Queries (val/test): {len(q)} -> {q_path}")
    print(f"Gallery (val/test): {len(g)} -> {g_path}")
    print(f"Train: {len(t)} -> {t_path}")


if __name__ == "__main__":
    main()
