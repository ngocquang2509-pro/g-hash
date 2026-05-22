#!/usr/bin/env python3
"""Apply a Label Studio JSON export back into labels_template.csv.

Supported export shape: list[task]
- task['data']['image_path'] is used as the stable key (relative Image_Path)
- task['annotations'][*]['result'][*] contains the selected choices

If multiple annotations exist per task, you can choose a merge policy:
- first: use the first annotation
- majority: per-concept majority vote across annotations
- unanimous: per-concept AND across annotations (more conservative)

Also writes a disagreements CSV listing tasks where annotators disagree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


META_COLS = ["Image_Path", "split", "retrieval_role"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply Label Studio export to ET-EDU labels CSV")
    p.add_argument("--export-json", required=True, help="Label Studio exported JSON file")
    p.add_argument(
        "--labels-csv",
        default="data/ET-EDU-CBIR-V2/labels_template.csv",
        help="Base CSV to update",
    )
    p.add_argument(
        "--concepts-file",
        default=None,
        help="Concepts file (defaults to concepts.txt next to labels CSV)",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (defaults to <labels-csv>.labeled.csv)",
    )
    p.add_argument(
        "--from-name",
        default="concepts",
        help="Label Studio from_name for the Choices control",
    )
    p.add_argument(
        "--policy",
        choices=["first", "majority", "unanimous"],
        default="majority",
        help="How to merge multiple annotations per task",
    )
    p.add_argument(
        "--disagreements-csv",
        default=None,
        help="Optional output CSV listing tasks with annotator disagreement",
    )
    return p.parse_args()


def load_concepts(concepts_file: Path) -> list[str]:
    concepts = [ln.strip() for ln in concepts_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not concepts:
        raise ValueError(f"No concepts in: {concepts_file}")
    return concepts


def _choices_to_vec(choices: list[str], concept_cols: list[str]) -> np.ndarray:
    s = set(choices)
    return np.asarray([1 if c in s else 0 for c in concept_cols], dtype=np.int8)


def main() -> None:
    args = parse_args()
    export_path = Path(args.export_json)
    labels_csv = Path(args.labels_csv)

    if not export_path.exists():
        raise FileNotFoundError(export_path)
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)

    concepts_file = (
        Path(args.concepts_file)
        if args.concepts_file
        else (labels_csv.parent / "concepts.txt")
    )
    if not concepts_file.exists():
        raise FileNotFoundError(f"Missing concepts file: {concepts_file}")

    concept_cols = load_concepts(concepts_file)

    out_csv = Path(args.out_csv) if args.out_csv else labels_csv.with_suffix(labels_csv.suffix + ".labeled.csv")
    disagreements_csv = (
        Path(args.disagreements_csv)
        if args.disagreements_csv
        else (out_csv.parent / "labeling_review" / "disagreements.csv")
    )
    disagreements_csv.parent.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("Expected export JSON to be a list of tasks")

    key_to_vec: dict[str, np.ndarray] = {}
    disagreements: list[dict[str, object]] = []

    for task in tasks:
        if not isinstance(task, dict):
            continue
        data = task.get("data") or {}
        image_path = data.get("image_path")
        if not image_path:
            # Fallback: try parse from file URI (may not be stable)
            image_uri = data.get("image")
            if isinstance(image_uri, str) and "/images/" in image_uri:
                image_path = "images/" + image_uri.split("/images/", 1)[1]
        if not image_path:
            continue

        annotations = task.get("annotations") or []
        if not annotations:
            continue

        ann_vecs: list[np.ndarray] = []
        for ann in annotations:
            results = ann.get("result") or []
            selected: list[str] = []
            for r in results:
                if r.get("from_name") != args.from_name:
                    continue
                val = r.get("value") or {}
                if "choices" in val and isinstance(val["choices"], list):
                    selected = [str(x) for x in val["choices"]]
                    break
            ann_vecs.append(_choices_to_vec(selected, concept_cols))

        if not ann_vecs:
            continue

        # Detect disagreement.
        if len(ann_vecs) >= 2:
            base = ann_vecs[0]
            if any(not np.array_equal(v, base) for v in ann_vecs[1:]):
                disagreements.append(
                    {
                        "Image_Path": str(image_path),
                        "n_annotations": int(len(ann_vecs)),
                    }
                )

        mat = np.stack(ann_vecs, axis=0)  # (n, C)
        if args.policy == "first":
            final = mat[0]
        elif args.policy == "unanimous":
            final = (mat.sum(axis=0) == mat.shape[0]).astype(np.int8)
        else:  # majority
            final = (mat.sum(axis=0) > (mat.shape[0] / 2.0)).astype(np.int8)

        key_to_vec[str(image_path)] = final

    base_df = pd.read_csv(labels_csv)
    for c in META_COLS:
        if c not in base_df.columns:
            raise ValueError(f"Base labels CSV missing column: {c}")

    missing_concepts = [c for c in concept_cols if c not in base_df.columns]
    if missing_concepts:
        raise ValueError("Base labels CSV missing concept columns: " + ", ".join(missing_concepts))

    base_df = base_df.copy()
    base_df = base_df.set_index("Image_Path")

    applied = 0
    missing_keys = 0
    for k, vec in key_to_vec.items():
        if k not in base_df.index:
            missing_keys += 1
            continue
        for i, c in enumerate(concept_cols):
            base_df.at[k, c] = int(vec[i])
        applied += 1

    base_df = base_df.reset_index()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(out_csv, index=False)

    if disagreements:
        pd.DataFrame(disagreements).to_csv(disagreements_csv, index=False)

    print("Done.")
    print(f"- Applied labels: {applied}")
    print(f"- Tasks not found in base CSV: {missing_keys}")
    print(f"- Output CSV: {out_csv}")
    if disagreements:
        print(f"- Disagreements: {len(disagreements)} -> {disagreements_csv}")


if __name__ == "__main__":
    main()
