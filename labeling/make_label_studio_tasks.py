#!/usr/bin/env python3
"""Generate Label Studio tasks for ET-EDU-CBIR-V2 manual labeling.

Requires Label Studio to be installed separately if you want to use the UI.
This script only creates tasks JSON.

It embeds both:
- `image`: file:// URI to the local image (works with local file serving enabled)
- `image_path`: relative path from dataset root, used for stable merging back
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make Label Studio tasks JSON for ET-EDU-CBIR-V2")
    p.add_argument("--dataset-root", default="data/ET-EDU-CBIR-V2")
    p.add_argument("--labels-csv", default=None)
    p.add_argument("--metadata-csv", default=None)
    p.add_argument(
        "--out-json",
        default=None,
        help="Output tasks file (defaults to <dataset-root>/label_studio/tasks.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    labels_csv = Path(args.labels_csv) if args.labels_csv else (dataset_root / "labels_template.csv")
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else (dataset_root / "metadata.csv")
    out_json = Path(args.out_json) if args.out_json else (dataset_root / "label_studio" / "tasks.json")

    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_csv)
    meta_df = pd.read_csv(metadata_csv)

    if "Image_Path" not in labels_df.columns:
        raise ValueError("labels CSV missing Image_Path")
    if "image_path" not in meta_df.columns:
        raise ValueError("metadata CSV missing image_path")

    meta_slim = meta_df[
        [
            "image_path",
            "video_id",
            "camera_id",
            "timestamp_sec",
            "distance_bucket",
            "quality_score",
        ]
    ].copy()
    meta_slim = meta_slim.drop_duplicates(subset=["image_path"], keep="first")
    meta_slim = meta_slim.rename(columns={"image_path": "Image_Path"})

    merged = labels_df.merge(meta_slim, on="Image_Path", how="left")

    tasks = []
    for _, row in merged.iterrows():
        rel_path = str(row["Image_Path"]).strip()
        abs_path = (dataset_root / rel_path).resolve()
        if not abs_path.exists():
            # Skip missing images.
            continue

        data = {
            "image": abs_path.as_uri(),
            "image_path": rel_path,
            "split": str(row.get("split", "")),
            "retrieval_role": str(row.get("retrieval_role", "")),
            "camera_id": str(row.get("camera_id", "")),
            "video_id": str(row.get("video_id", "")),
            "timestamp_sec": float(row.get("timestamp_sec", 0.0)) if pd.notna(row.get("timestamp_sec", None)) else 0.0,
            "distance_bucket": str(row.get("distance_bucket", "")),
            "quality_score": float(row.get("quality_score", 0.0)) if pd.notna(row.get("quality_score", None)) else 0.0,
        }
        tasks.append({"data": data})

    out_json.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote tasks: {len(tasks)} -> {out_json}")


if __name__ == "__main__":
    main()
