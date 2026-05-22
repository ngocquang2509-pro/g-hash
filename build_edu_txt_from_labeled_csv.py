#!/usr/bin/env python3
"""
Convert labeled ET-EDU CSV into txt files required by the current training loader.

Input CSV format:
- Image_Path
- split
- retrieval_role
- one or more concept columns with binary labels

Output files:
- train_img.txt
- train_label.txt
- test_img.txt
- test_label.txt
- concepts.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


META_COLUMNS = {"Image_Path", "split", "retrieval_role"}


def load_concepts(concepts_file: Path) -> list[str]:
    concepts: list[str] = []
    for line in concepts_file.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            concepts.append(name)
    if not concepts:
        raise ValueError(f"No concepts found in: {concepts_file}")
    return concepts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ET-EDU txt files from labeled CSV.")
    parser.add_argument(
        "--labeled-csv",
        default="data/ET-EDU-CBIR-V2/labels_template.csv",
        help="CSV with Image_Path, split, retrieval_role and binary concept columns.",
    )
    parser.add_argument(
        "--output-root",
        default="data/ET-EDU-CBIR-V2",
        help="Directory to write train/test txt files.",
    )
    parser.add_argument(
        "--concepts-file",
        default=None,
        help=(
            "Optional concepts.txt to define which columns are labels (and their order). "
            "If omitted, will try to find concepts.txt next to the CSV or in output-root; "
            "otherwise falls back to using all non-meta columns."
        ),
    )
    parser.add_argument(
        "--test-splits",
        nargs="+",
        default=["val", "test"],
        help="Split names that should be merged into test files.",
    )
    return parser.parse_args()


def write_txt(rows: pd.DataFrame, concept_cols: list[str], img_file: Path, label_file: Path) -> None:
    with img_file.open("w", encoding="utf-8") as f_img, label_file.open("w", encoding="utf-8") as f_lbl:
        for _, row in rows.iterrows():
            f_img.write(str(row["Image_Path"]).strip() + "\n")
            labels = [int(row[c]) for c in concept_cols]
            f_lbl.write(" ".join(str(v) for v in labels) + "\n")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.labeled_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing labeled CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Image_Path" not in df.columns:
        raise ValueError("CSV must include Image_Path column.")
    if "split" not in df.columns:
        raise ValueError("CSV must include split column.")

    concepts_file: Path | None = Path(args.concepts_file) if args.concepts_file else None
    if concepts_file is None:
        candidates = [
            csv_path.parent / "concepts.txt",
            output_root / "concepts.txt",
        ]
        concepts_file = next((p for p in candidates if p.exists()), None)

    if concepts_file is not None and concepts_file.exists():
        concept_cols = load_concepts(concepts_file)
        missing = [c for c in concept_cols if c not in df.columns]
        if missing:
            raise ValueError(
                "Missing concept columns from CSV (per concepts file): "
                + ", ".join(missing)
                + f"\nCSV: {csv_path}\nConcepts file: {concepts_file}"
            )
    else:
        concept_cols = [c for c in df.columns if c not in META_COLUMNS]
        if not concept_cols:
            raise ValueError("No concept columns found in CSV.")

    # Normalize labels to 0/1 ints.
    for c in concept_cols:
        df[c] = (df[c].fillna(0).astype(float) > 0.5).astype(int)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"].isin(args.test_splits)].copy()

    if train_df.empty:
        raise ValueError("No train rows found. Check split values in CSV.")
    if test_df.empty:
        print("Warning: no test rows found from requested test splits. Creating empty test files.")

    write_txt(
        rows=train_df,
        concept_cols=concept_cols,
        img_file=output_root / "train_img.txt",
        label_file=output_root / "train_label.txt",
    )
    write_txt(
        rows=test_df,
        concept_cols=concept_cols,
        img_file=output_root / "test_img.txt",
        label_file=output_root / "test_label.txt",
    )

    with (output_root / "concepts.txt").open("w", encoding="utf-8") as f:
        for c in concept_cols:
            f.write(c + "\n")

    print("Done.")
    print(f"Concepts: {len(concept_cols)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
