import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ET-EDU train/test txt files from merged annotation CSV."
    )
    parser.add_argument("--csv-file", default="data/annotations_quality10k_merged.csv")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--target-train-ratio", type=float, default=0.8)
    parser.add_argument("--min-test-videos", type=int, default=2)
    return parser.parse_args()


def video_id_from_path(rel_path: str) -> str:
    name = Path(rel_path).name
    parts = name.split("_")
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[2]}"
    return "unknown"


def main():
    args = parse_args()
    csv_path = Path(args.csv_file)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]

    classes = header[1:]

    # Deduplicate by image path
    seen = set()
    uniq = []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        uniq.append(r)

    by_video = defaultdict(list)
    for r in uniq:
        by_video[video_id_from_path(r[0])].append(r)

    videos = list(by_video.keys())
    sizes = {v: len(by_video[v]) for v in videos}
    n_total = len(uniq)
    target_train = args.target_train_ratio * n_total

    best = None
    for k in range(1, len(videos)):
        for combo in itertools.combinations(videos, k):
            train_v = set(combo)
            test_v = len(videos) - len(train_v)
            if test_v < args.min_test_videos:
                continue
            train_n = sum(sizes[v] for v in train_v)
            score = abs(train_n - target_train)
            if best is None or score < best[0]:
                best = (score, train_v, train_n)

    if best is None:
        raise RuntimeError("Cannot find valid video-group split with current constraints.")

    _, train_videos, train_n = best
    train_rows, test_rows = [], []
    for vid, block in by_video.items():
        if vid in train_videos:
            train_rows.extend(block)
        else:
            test_rows.extend(block)

    with (out_root / "train_img.txt").open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(r[0] + "\n")
    with (out_root / "test_img.txt").open("w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(r[0] + "\n")
    with (out_root / "train_label.txt").open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(" ".join(r[1:]) + "\n")
    with (out_root / "test_label.txt").open("w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(" ".join(r[1:]) + "\n")
    with (out_root / "concepts.txt").open("w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")

    print("Done.")
    print(f"Unique rows: {n_total}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Train ratio: {len(train_rows) / max(1, n_total):.4f}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Test videos: {len(videos) - len(train_videos)}")


if __name__ == "__main__":
    main()
