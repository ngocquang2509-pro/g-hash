import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from PIL import Image
from tqdm import tqdm


BEHAVIORAL_CLASSES = [
    "using_phone",
    "dozing_off",
    "turning_sideways",
    "turning_back",
    "raising_hand",
    "opening_book",
    "reading",
    "writing",
    "listening",
    "head_down",
    "sitting",
    "standing",
    "walking",
    "interacting",
]

PROMPTS = [
    "a student holding and looking at a mobile phone",
    "a student sleeping or dozing off on the desk",
    "a student turning sideways to look at nearby classmates",
    "a student turning back to look behind",
    "a student raising their hand in class",
    "a student opening a book or notebook",
    "a student reading a book or paper",
    "a student writing notes with a pen",
    "a student listening attentively to lecture",
    "a student with head down on desk without writing",
    "a student sitting in class",
    "a student standing in class",
    "a student walking in the classroom",
    "students interacting or talking to each other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label remaining images with diversity filtering (low-duplicate selection)."
    )
    parser.add_argument("--input-dir", default="data/cropped_students_quality_10k")
    parser.add_argument("--metadata-csv", default="data/cropped_students_quality_10k/metadata.csv")
    parser.add_argument("--seed-csv", default="data/seed_annotations_quality10k.csv")
    parser.add_argument("--auto-output-csv", default="data/auto_annotations_quality10k_diverse.csv")
    parser.add_argument("--merged-output-csv", default="data/annotations_quality10k_merged.csv")
    parser.add_argument("--rel-prefix", default="cropped_students_quality_10k")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-new-images", type=int, default=0, help="0 = label all selected remaining images.")
    parser.add_argument("--max-per-class", type=int, default=2500)
    parser.add_argument("--frame-gap", type=int, default=220, help="Min frame distance in same video for hash compare.")
    parser.add_argument("--hash-threshold", type=int, default=6, help="Hamming threshold to treat as duplicate.")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def dct_phash(image_bgr, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype("float32"))
    dct_low = dct[:hash_size, :hash_size]
    median = float(dct_low[1:, 1:].mean())
    bits = 0
    for v in dct_low.flatten():
        bits = (bits << 1) | int(v > median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    # Works across Python/int variants (including numpy scalar ints).
    x = int(a) ^ int(b)
    return bin(x).count("1")


def read_seed(seed_csv: Path) -> Tuple[List[str], Dict[str, List[int]], Counter]:
    annotations: Dict[str, List[int]] = {}
    class_count = Counter()
    rows_out = []
    with seed_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            rel_path = row[0]
            labels = [int(x) for x in row[1:1 + len(BEHAVIORAL_CLASSES)]]
            annotations[rel_path] = labels
            rows_out.append(row)
            for idx, val in enumerate(labels):
                if val == 1:
                    class_count[BEHAVIORAL_CLASSES[idx]] += 1
    return header, annotations, class_count


def load_candidates(
    metadata_csv: Path,
    input_dir: Path,
    existing_paths: set,
    rel_prefix: str,
) -> List[Tuple[Path, str, str, int, float]]:
    """
    Return tuples: (full_path, rel_path, video_name, frame_idx, score)
    """
    candidates = []
    with metadata_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_path = Path(row["image_path"])
            if not full_path.exists():
                full_path = input_dir / full_path.name
            if not full_path.exists():
                continue
            rel_path = f"{rel_prefix}/{full_path.name}"
            if rel_path in existing_paths:
                continue
            video_name = row.get("video_name", "")
            frame_idx = int(row.get("frame_idx", "0"))
            score = float(row.get("score", "0"))
            candidates.append((full_path, rel_path, video_name, frame_idx, score))
    candidates.sort(key=lambda x: x[4], reverse=True)
    return candidates


def select_diverse(
    candidates: List[Tuple[Path, str, str, int, float]],
    frame_gap: int,
    hash_threshold: int,
) -> List[Tuple[Path, str]]:
    selected: List[Tuple[Path, str]] = []
    # For each video, keep recent selected hashes near by frame for duplicate checking.
    recent_by_video = defaultdict(list)
    exact_hash_seen = set()

    for full_path, rel_path, video_name, frame_idx, _ in tqdm(candidates, desc="Selecting diverse candidates"):
        img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ph = dct_phash(img)
        if ph in exact_hash_seen:
            continue

        recent = recent_by_video[video_name]
        filtered_recent = []
        is_dup = False
        for prev_frame, prev_hash in recent:
            if abs(frame_idx - prev_frame) <= frame_gap:
                filtered_recent.append((prev_frame, prev_hash))
                if hamming_distance(ph, prev_hash) <= hash_threshold:
                    is_dup = True
        recent_by_video[video_name] = filtered_recent
        if is_dup:
            continue

        selected.append((full_path, rel_path))
        exact_hash_seen.add(ph)
        recent_by_video[video_name].append((frame_idx, ph))

    return selected


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    metadata_csv = Path(args.metadata_csv)
    seed_csv = Path(args.seed_csv)
    auto_out_csv = Path(args.auto_output_csv)
    merged_out_csv = Path(args.merged_output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata csv: {metadata_csv}")
    if not seed_csv.exists():
        raise FileNotFoundError(f"Missing seed csv: {seed_csv}")

    header, existing_annotations, class_count = read_seed(seed_csv)
    existing_paths = set(existing_annotations.keys())
    print(f"Seed labeled images: {len(existing_paths)}")

    candidates = load_candidates(metadata_csv, input_dir, existing_paths, args.rel_prefix)
    print(f"Remaining candidates from metadata: {len(candidates)}")

    selected = select_diverse(
        candidates=candidates,
        frame_gap=args.frame_gap,
        hash_threshold=args.hash_threshold,
    )
    print(f"Diverse selected candidates: {len(selected)}")

    if args.max_new_images > 0:
        selected = selected[: args.max_new_images]
        print(f"After max_new_images cap: {len(selected)}")

    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    clip = hf_pipeline("zero-shot-image-classification", model=args.model, device=device)

    auto_out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_out_csv.parent.mkdir(parents=True, exist_ok=True)

    auto_rows: List[List[str]] = []
    for full_path, rel_path in tqdm(selected, desc="Auto labeling"):
        try:
            img = Image.open(full_path).convert("RGB")
        except Exception:
            continue

        try:
            results = clip(img, candidate_labels=PROMPTS)
        except Exception:
            continue

        score_map = {}
        for r in results:
            idx = PROMPTS.index(r["label"])
            score_map[BEHAVIORAL_CLASSES[idx]] = float(r["score"])

        sorted_cls = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        top_labels = [cls for cls, _ in sorted_cls[: args.top_k]]
        final_labels = [cls for cls in top_labels if class_count[cls] < args.max_per_class]
        if not final_labels:
            continue

        binary = [1 if cls in final_labels else 0 for cls in BEHAVIORAL_CLASSES]
        row = [rel_path] + binary
        auto_rows.append(row)

        for cls in final_labels:
            class_count[cls] += 1

    with auto_out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(auto_rows)

    merged_rows = []
    merged_seen = set()
    with seed_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if not row:
                continue
            if row[0] in merged_seen:
                continue
            merged_rows.append(row)
            merged_seen.add(row[0])
    for row in auto_rows:
        if row[0] in merged_seen:
            continue
        merged_rows.append(row)
        merged_seen.add(row[0])

    with merged_out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(merged_rows)

    print("\nDone.")
    print(f"Auto labeled rows: {len(auto_rows)}")
    print(f"Merged rows total: {len(merged_rows)}")
    print(f"Auto CSV:   {auto_out_csv}")
    print(f"Merged CSV: {merged_out_csv}")
    print("\nFinal class counts (merged estimate):")
    for cls in BEHAVIORAL_CLASSES:
        print(f"  {cls:22s}: {class_count[cls]}")


if __name__ == "__main__":
    main()
