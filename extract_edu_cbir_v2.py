#!/usr/bin/env python3
"""
ET-EDU CBIR Dataset Builder (V2)

Builds a person-crop dataset from classroom videos with:
1) person detection + IoU tracking,
2) keyframe sampling per track,
3) quality filtering,
4) global near-duplicate filtering,
5) split by video and query/gallery lists for CBIR.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    conf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ET-EDU CBIR dataset from videos.")
    parser.add_argument("--video-dir", default="data/ET-EDU", help="Directory containing source videos.")
    parser.add_argument("--output-root", default="data/ET-EDU-CBIR-V2", help="Output dataset directory.")
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="YOLO weights path.")

    parser.add_argument("--det-conf", type=float, default=0.60, help="Detector confidence threshold.")
    parser.add_argument("--det-fps", type=float, default=4.0, help="Detection FPS sampled from source videos.")
    parser.add_argument("--max-det-per-frame", type=int, default=20, help="Max detections per frame after sorting by confidence.")

    parser.add_argument("--track-iou-thr", type=float, default=0.35, help="IoU threshold for tracking association.")
    parser.add_argument("--track-max-age-sec", type=float, default=2.0, help="Max inactive age for tracks (seconds).")

    parser.add_argument("--keyframe-interval", type=float, default=1.0, help="Seconds between saved crops in the same track.")
    parser.add_argument("--max-per-track", type=int, default=3, help="Maximum kept crops per track.")
    parser.add_argument("--bbox-margin", type=float, default=0.12, help="Relative margin around person bbox before crop.")
    parser.add_argument("--min-scale-change", type=float, default=0.20, help="Scale change threshold to force-save within interval.")
    parser.add_argument("--min-center-change", type=float, default=0.08, help="Center shift threshold to force-save within interval.")

    parser.add_argument("--min-short-side", type=int, default=160, help="Minimum short side for valid crop.")
    parser.add_argument("--laplacian-min", type=float, default=120.0, help="Minimum Laplacian variance (sharpness).")
    parser.add_argument("--brightness-min", type=float, default=45.0, help="Minimum mean grayscale brightness.")
    parser.add_argument("--brightness-max", type=float, default=210.0, help="Maximum mean grayscale brightness.")
    parser.add_argument("--contrast-min", type=float, default=35.0, help="Minimum grayscale std-dev.")
    parser.add_argument("--black-ratio-max", type=float, default=0.15, help="Maximum ratio of near-black pixels.")
    parser.add_argument("--white-ratio-max", type=float, default=0.15, help="Maximum ratio of near-white pixels.")

    parser.add_argument("--hash-hamming", type=int, default=6, help="Max Hamming distance for global near-duplicate filtering.")
    parser.add_argument("--track-hist-dup-cos", type=float, default=0.94, help="Cosine threshold for in-track near-duplicate filtering.")

    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio by samples (grouped by video).")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio by samples (grouped by video).")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio by samples (grouped by video).")
    parser.add_argument("--gallery-query-ratio", type=int, default=10, help="Approximate gallery:query ratio in val/test.")

    parser.add_argument("--max-camera-ratio", type=float, default=0.40, help="Max fraction per camera after extraction (0..1].")
    parser.add_argument("--prune-unselected", action="store_true", help="Delete image files removed by camera rebalance.")

    parser.add_argument("--concepts-file", default="data/ET-EDU-CROPPED-PERSONS/concepts.txt", help="Optional concepts file to build label template.")

    parser.add_argument("--max-videos", type=int, default=0, help="Optional cap on processed videos (0 = all).")
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="Optional frame cap per video (0 = all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)


class SimpleIoUTracker:
    def __init__(self, iou_thr: float, max_age_steps: int):
        self.iou_thr = iou_thr
        self.max_age_steps = max(1, max_age_steps)
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, object]] = {}

    def update(self, bboxes: List[Tuple[int, int, int, int]], step_idx: int) -> List[int]:
        stale_ids = [
            tid
            for tid, data in self.tracks.items()
            if step_idx - int(data["last_step"]) > self.max_age_steps
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        if not bboxes:
            return []

        track_ids = list(self.tracks.keys())
        assigned_track_for_det: Dict[int, int] = {}

        if track_ids:
            iou_matrix = np.full((len(track_ids), len(bboxes)), -1.0, dtype=np.float32)
            for i, tid in enumerate(track_ids):
                tbox = self.tracks[tid]["bbox"]
                for j, bbox in enumerate(bboxes):
                    iou_matrix[i, j] = bbox_iou(tbox, bbox)

            used_tracks = set()
            used_dets = set()

            while True:
                flat_idx = int(np.argmax(iou_matrix))
                best_iou = float(iou_matrix.flat[flat_idx])
                if best_iou < self.iou_thr:
                    break

                i, j = np.unravel_index(flat_idx, iou_matrix.shape)
                tid = track_ids[i]

                if tid in used_tracks or j in used_dets:
                    iou_matrix[i, j] = -1.0
                    continue

                assigned_track_for_det[j] = tid
                used_tracks.add(tid)
                used_dets.add(j)
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        output_ids: List[int] = []
        for j, bbox in enumerate(bboxes):
            if j in assigned_track_for_det:
                tid = assigned_track_for_det[j]
            else:
                tid = self.next_id
                self.next_id += 1

            self.tracks[tid] = {"bbox": bbox, "last_step": step_idx}
            output_ids.append(tid)

        return output_ids


def dhash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for bit in diff.flatten().astype(np.uint8):
        bits = (bits << 1) | int(bit)
    return bits


def hamming_distance64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class MultiIndexHashFilter:
    def __init__(self):
        self.hashes: List[int] = []
        self.tables: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    @staticmethod
    def _segments(h: int) -> List[int]:
        return [(h >> (16 * i)) & 0xFFFF for i in range(4)]

    def is_duplicate(self, h: int, max_hamming: int) -> bool:
        candidates = set()
        for seg_idx, seg_val in enumerate(self._segments(h)):
            candidates.update(self.tables[(seg_idx, seg_val)])

        for idx in candidates:
            if hamming_distance64(h, self.hashes[idx]) <= max_hamming:
                return True
        return False

    def add(self, h: int) -> None:
        idx = len(self.hashes)
        self.hashes.append(h)
        for seg_idx, seg_val in enumerate(self._segments(h)):
            self.tables[(seg_idx, seg_val)].append(idx)


def compute_histogram(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 6, 6], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * margin))
    pad_y = int(round(bh * margin))

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(img_w, x2 + pad_x)
    ny2 = min(img_h, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def compute_bbox_changes(
    prev_bbox: Tuple[int, int, int, int] | None,
    cur_bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float]:
    if prev_bbox is None:
        return 1.0, 1.0

    px1, py1, px2, py2 = prev_bbox
    cx1, cy1, cx2, cy2 = cur_bbox

    prev_area = max(1.0, float((px2 - px1) * (py2 - py1)))
    cur_area = max(1.0, float((cx2 - cx1) * (cy2 - cy1)))
    scale_change = abs(cur_area - prev_area) / prev_area

    prev_cx = 0.5 * (px1 + px2)
    prev_cy = 0.5 * (py1 + py2)
    cur_cx = 0.5 * (cx1 + cx2)
    cur_cy = 0.5 * (cy1 + cy2)

    diag = max(1.0, math.hypot(frame_w, frame_h))
    center_change = math.hypot(cur_cx - prev_cx, cur_cy - prev_cy) / diag
    return scale_change, center_change


def classify_distance_bucket(bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[str, float]:
    x1, y1, x2, y2 = bbox
    area_ratio = float((x2 - x1) * (y2 - y1)) / float(max(1, frame_w * frame_h))
    if area_ratio >= 0.15:
        return "near", area_ratio
    if area_ratio >= 0.07:
        return "mid", area_ratio
    return "far", area_ratio


def detect_people(model: YOLO, frame_bgr: np.ndarray, conf_thr: float, max_det: int) -> List[Detection]:
    results = model.predict(frame_bgr, classes=[0], conf=conf_thr, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    detections = []
    for bbox, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(bbox=(x1, y1, x2, y2), conf=float(conf)))

    detections.sort(key=lambda d: d.conf, reverse=True)
    if max_det > 0:
        detections = detections[:max_det]
    return detections


def quality_metrics(crop_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    black_ratio = float(np.mean(gray <= 5))
    white_ratio = float(np.mean(gray >= 250))

    h, w = gray.shape
    return {
        "laplacian_var": lap_var,
        "brightness": brightness,
        "contrast": contrast,
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
        "width": int(w),
        "height": int(h),
        "gray": gray,
    }


def quality_ok(metrics: Dict[str, float], args: argparse.Namespace) -> Tuple[bool, str]:
    short_side = min(int(metrics["width"]), int(metrics["height"]))
    if short_side < args.min_short_side:
        return False, "small_crop"
    if metrics["laplacian_var"] < args.laplacian_min:
        return False, "low_sharpness"
    if metrics["brightness"] < args.brightness_min or metrics["brightness"] > args.brightness_max:
        return False, "bad_brightness"
    if metrics["contrast"] < args.contrast_min:
        return False, "low_contrast"
    if metrics["black_ratio"] > args.black_ratio_max:
        return False, "too_dark_pixels"
    if metrics["white_ratio"] > args.white_ratio_max:
        return False, "too_bright_pixels"
    return True, "ok"


def quality_score(metrics: Dict[str, float], det_conf: float) -> float:
    short_side = min(int(metrics["width"]), int(metrics["height"]))
    sharp_term = min(300.0, float(metrics["laplacian_var"])) / 300.0
    contrast_term = min(80.0, float(metrics["contrast"])) / 80.0
    size_term = min(256.0, float(short_side)) / 256.0
    return 0.35 * sharp_term + 0.25 * contrast_term + 0.20 * size_term + 0.20 * float(det_conf)


def list_videos(video_dir: Path) -> List[Path]:
    videos = [p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    videos.sort()
    return videos


def process_video(
    video_path: Path,
    model: YOLO,
    output_images_dir: Path,
    args: argparse.Namespace,
    hash_filter: MultiIndexHashFilter,
    sample_counter: itertools.count,
    global_stats: Counter,
) -> List[Dict[str, object]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        global_stats["video_open_failed"] += 1
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det_stride = max(1, int(round(fps / max(args.det_fps, 0.1))))
    max_age_steps = max(1, int(round(args.track_max_age_sec * max(args.det_fps, 0.1))))

    tracker = SimpleIoUTracker(iou_thr=args.track_iou_thr, max_age_steps=max_age_steps)
    track_state: Dict[int, Dict[str, object]] = {}
    accepted_rows: List[Dict[str, object]] = []

    processed_steps = 0
    accepted_in_video = 0
    frame_idx = -1

    pbar_total = frame_count if frame_count > 0 else None
    pbar_desc = f"Video {video_path.name}"
    pbar = tqdm(total=pbar_total, desc=pbar_desc, leave=False)

    while True:
        ret = cap.grab()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        if args.max_frames_per_video > 0 and frame_idx >= args.max_frames_per_video:
            break

        if frame_idx % det_stride != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret or frame is None:
            global_stats["skipped_decode_failed"] += 1
            continue

        processed_steps += 1
        detections = detect_people(model, frame, conf_thr=args.det_conf, max_det=args.max_det_per_frame)
        if not detections:
            continue

        bboxes = [d.bbox for d in detections]
        track_ids = tracker.update(bboxes, step_idx=processed_steps)

        frame_h, frame_w = frame.shape[:2]
        ts = float(frame_idx) / fps

        for det, track_id in zip(detections, track_ids):
            global_stats["candidates_seen"] += 1

            if args.max_per_track > 0:
                saved_for_track = int(track_state.get(track_id, {}).get("saved_count", 0))
                if saved_for_track >= args.max_per_track:
                    global_stats["rejected_track_quota"] += 1
                    continue

            if args.max_videos > 0 and accepted_in_video >= args.max_per_track * 1000000:
                # no-op placeholder: keep behavior explicit and avoid accidental hard stop.
                pass

            expanded = expand_bbox(det.bbox, frame_w, frame_h, margin=args.bbox_margin)
            x1, y1, x2, y2 = expanded
            if x2 <= x1 or y2 <= y1:
                global_stats["rejected_invalid_bbox"] += 1
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                global_stats["rejected_empty_crop"] += 1
                continue

            metrics = quality_metrics(crop)
            ok, reason = quality_ok(metrics, args)
            if not ok:
                global_stats[f"rejected_{reason}"] += 1
                continue

            tstate = track_state.setdefault(
                track_id,
                {
                    "saved_count": 0,
                    "last_saved_ts": -1e9,
                    "last_saved_bbox": None,
                    "last_saved_hist": None,
                },
            )

            scale_change, center_change = compute_bbox_changes(
                tstate["last_saved_bbox"],
                expanded,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            major_change = (scale_change >= args.min_scale_change) or (center_change >= args.min_center_change)

            elapsed = ts - float(tstate["last_saved_ts"])
            if int(tstate["saved_count"]) > 0 and elapsed < args.keyframe_interval and not major_change:
                global_stats["rejected_temporal_dense"] += 1
                continue

            hist = compute_histogram(crop)
            if tstate["last_saved_hist"] is not None:
                hist_sim = cosine_similarity(hist, tstate["last_saved_hist"])
                if hist_sim >= args.track_hist_dup_cos and not major_change:
                    global_stats["rejected_track_hist_dup"] += 1
                    continue

            gray = metrics["gray"]
            hval = dhash64(gray)
            if hash_filter.is_duplicate(hval, max_hamming=args.hash_hamming):
                global_stats["rejected_global_hash_dup"] += 1
                continue

            hash_filter.add(hval)

            distance_bucket, area_ratio = classify_distance_bucket(expanded, frame_w=frame_w, frame_h=frame_h)
            sample_idx = next(sample_counter)
            fname = (
                f"{distance_bucket}_{video_path.stem}_"
                f"f{frame_idx:06d}_t{track_id:04d}_{sample_idx:08d}.jpg"
            )
            rel_path = Path("images") / fname
            save_path = output_images_dir / fname

            if not cv2.imwrite(str(save_path), crop):
                global_stats["rejected_save_failed"] += 1
                continue

            qscore = quality_score(metrics, det_conf=det.conf)
            row = {
                "image_path": rel_path.as_posix(),
                "video_file": video_path.name,
                "video_id": video_path.stem,
                "camera_id": video_path.stem.split("_")[0],
                "frame_idx": frame_idx,
                "timestamp_sec": round(ts, 3),
                "track_id": track_id,
                "det_conf": round(float(det.conf), 5),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_area_ratio": round(float(area_ratio), 6),
                "distance_bucket": distance_bucket,
                "width": int(metrics["width"]),
                "height": int(metrics["height"]),
                "laplacian_var": round(float(metrics["laplacian_var"]), 4),
                "brightness": round(float(metrics["brightness"]), 4),
                "contrast": round(float(metrics["contrast"]), 4),
                "black_ratio": round(float(metrics["black_ratio"]), 6),
                "white_ratio": round(float(metrics["white_ratio"]), 6),
                "dhash_hex": f"{hval:016x}",
                "quality_score": round(float(qscore), 6),
            }
            accepted_rows.append(row)

            tstate["saved_count"] = int(tstate["saved_count"]) + 1
            tstate["last_saved_ts"] = ts
            tstate["last_saved_bbox"] = expanded
            tstate["last_saved_hist"] = hist

            accepted_in_video += 1
            global_stats["accepted"] += 1

    pbar.close()
    cap.release()
    return accepted_rows


def rebalance_by_camera(df: pd.DataFrame, max_ratio: float, seed: int) -> pd.DataFrame:
    if df.empty:
        return df

    if max_ratio <= 0:
        return df

    num_cameras = int(df["camera_id"].nunique())
    if num_cameras <= 1:
        return df

    max_ratio = min(float(max_ratio), 1.0)
    feasible_floor = 1.0 / float(num_cameras)
    effective_ratio = max(max_ratio, feasible_floor)

    total = len(df)
    cap = max(1, int(round(total * effective_ratio)))
    rng = np.random.default_rng(seed)

    keep_idx = []
    for camera_id, block in df.groupby("camera_id"):
        if len(block) <= cap:
            keep_idx.extend(block.index.tolist())
            continue

        # Keep highest quality first, then randomize tie region for diversity.
        block_sorted = block.sort_values(["quality_score", "timestamp_sec"], ascending=[False, True])
        head = block_sorted.iloc[: cap * 3]
        sampled = head.sample(n=cap, random_state=int(rng.integers(1, 10_000_000)))
        keep_idx.extend(sampled.index.tolist())

    keep_idx = sorted(set(keep_idx))
    return df.loc[keep_idx].copy().reset_index(drop=True)


def assign_split_by_video(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    ratios_sum = train_ratio + val_ratio + test_ratio
    if ratios_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    train_ratio /= ratios_sum
    val_ratio /= ratios_sum
    test_ratio /= ratios_sum

    total = len(df)
    target = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * test_ratio,
    }
    current = {"train": 0, "val": 0, "test": 0}

    grouped = df.groupby("video_id").size().to_dict()
    videos = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(videos)
    videos.sort(key=lambda v: grouped[v], reverse=True)

    assign = {}
    for vid in videos:
        split = max(current.keys(), key=lambda k: target[k] - current[k])
        assign[vid] = split
        current[split] += grouped[vid]

    out = df.copy()
    out["split"] = out["video_id"].map(assign)
    return out


def assign_query_gallery_roles(df: pd.DataFrame, gallery_query_ratio: int) -> pd.DataFrame:
    out = df.copy()
    out["retrieval_role"] = "train"

    ratio = max(1, int(gallery_query_ratio))
    for split in ["val", "test"]:
        mask = out["split"] == split
        if not mask.any():
            continue

        out.loc[mask, "retrieval_role"] = "gallery"
        block = out[mask].sort_values(["video_id", "timestamp_sec", "frame_idx"])

        for vid, vid_df in block.groupby("video_id"):
            n = len(vid_df)
            if n < 2:
                continue

            q_count = max(1, int(round(n / float(ratio + 1))))
            q_count = min(q_count, max(1, n - 1))
            chosen = np.linspace(0, n - 1, num=q_count, dtype=int)
            chosen_idx = vid_df.iloc[chosen].index
            out.loc[chosen_idx, "retrieval_role"] = "query"

    return out


def write_text_list(paths: List[str], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")


def create_label_template(df: pd.DataFrame, concepts_file: Path, out_file: Path) -> int:
    if not concepts_file.exists():
        return 0

    concepts = [line.strip() for line in concepts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not concepts:
        return 0

    templ = df[["image_path", "split", "retrieval_role"]].copy()
    templ.rename(columns={"image_path": "Image_Path"}, inplace=True)
    for c in concepts:
        templ[c] = 0

    out_file.parent.mkdir(parents=True, exist_ok=True)
    templ.to_csv(out_file, index=False)

    concepts_out = out_file.parent / "concepts.txt"
    with concepts_out.open("w", encoding="utf-8") as f:
        for c in concepts:
            f.write(c + "\n")

    return len(concepts)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    video_dir = Path(args.video_dir)
    output_root = Path(args.output_root)
    output_images = output_root / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    videos = list_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under: {video_dir}")

    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    print(f"Found {len(videos)} videos.")
    print(f"Output root: {output_root}")

    model = YOLO(args.yolo_weights)
    hash_filter = MultiIndexHashFilter()
    sample_counter = itertools.count(start=1)
    stats = Counter()

    all_rows: List[Dict[str, object]] = []
    for vid in videos:
        rows = process_video(
            video_path=vid,
            model=model,
            output_images_dir=output_images,
            args=args,
            hash_filter=hash_filter,
            sample_counter=sample_counter,
            global_stats=stats,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No crops passed filters. Try relaxing thresholds.")
        return

    df = pd.DataFrame(all_rows)
    before_rebalance = len(df)
    df = rebalance_by_camera(df, max_ratio=args.max_camera_ratio, seed=args.seed)
    after_rebalance = len(df)

    if args.prune_unselected and after_rebalance < before_rebalance:
        selected = set(df["image_path"].tolist())
        for rel_path in set(pd.DataFrame(all_rows)["image_path"].tolist()) - selected:
            full = output_root / rel_path
            if full.exists():
                full.unlink()

    df = assign_split_by_video(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    df = assign_query_gallery_roles(df, gallery_query_ratio=args.gallery_query_ratio)

    # Save metadata.
    metadata_csv = output_root / "metadata.csv"
    df = df.sort_values(["split", "video_id", "frame_idx", "track_id"]).reset_index(drop=True)
    df.to_csv(metadata_csv, index=False)

    # Write split text files.
    train_paths = df.loc[df["split"] == "train", "image_path"].tolist()
    val_query_paths = df.loc[(df["split"] == "val") & (df["retrieval_role"] == "query"), "image_path"].tolist()
    val_gallery_paths = df.loc[(df["split"] == "val") & (df["retrieval_role"] == "gallery"), "image_path"].tolist()
    test_query_paths = df.loc[(df["split"] == "test") & (df["retrieval_role"] == "query"), "image_path"].tolist()
    test_gallery_paths = df.loc[(df["split"] == "test") & (df["retrieval_role"] == "gallery"), "image_path"].tolist()

    write_text_list(train_paths, output_root / "train_img_unlabeled.txt")
    write_text_list(val_query_paths, output_root / "val_query_img.txt")
    write_text_list(val_gallery_paths, output_root / "val_gallery_img.txt")
    write_text_list(test_query_paths, output_root / "test_query_img.txt")
    write_text_list(test_gallery_paths, output_root / "test_gallery_img.txt")

    concept_count = create_label_template(
        df=df,
        concepts_file=Path(args.concepts_file),
        out_file=output_root / "labels_template.csv",
    )

    summary = {
        "videos_processed": len(videos),
        "candidates_seen": int(stats.get("candidates_seen", 0)),
        "accepted": int(stats.get("accepted", 0)),
        "removed_by_camera_rebalance": int(before_rebalance - after_rebalance),
        "metadata_rows": int(len(df)),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "retrieval_role_counts": {k: int(v) for k, v in df["retrieval_role"].value_counts().to_dict().items()},
        "distance_counts": {k: int(v) for k, v in df["distance_bucket"].value_counts().to_dict().items()},
        "camera_counts": {k: int(v) for k, v in df["camera_id"].value_counts().to_dict().items()},
        "video_counts": {k: int(v) for k, v in df["video_id"].value_counts().to_dict().items()},
        "rejected_reasons": {
            k: int(v)
            for k, v in stats.items()
            if k.startswith("rejected_") or k in {"video_open_failed"}
        },
        "label_template_concepts": concept_count,
        "train_img_unlabeled": str((output_root / "train_img_unlabeled.txt").as_posix()),
        "val_query_img": str((output_root / "val_query_img.txt").as_posix()),
        "val_gallery_img": str((output_root / "val_gallery_img.txt").as_posix()),
        "test_query_img": str((output_root / "test_query_img.txt").as_posix()),
        "test_gallery_img": str((output_root / "test_gallery_img.txt").as_posix()),
    }

    summary_file = output_root / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print("\nExtraction completed.")
    print(f"Accepted crops: {summary['accepted']}")
    print(f"Metadata rows: {summary['metadata_rows']}")
    print(f"Metadata: {metadata_csv}")
    print(f"Summary: {summary_file}")
    if concept_count > 0:
        print(f"Label template: {output_root / 'labels_template.csv'} ({concept_count} concepts)")


if __name__ == "__main__":
    main()
