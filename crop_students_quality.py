import argparse
import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError("Missing ultralytics. Install with: pip install ultralytics") from exc


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".MP4", ".AVI", ".MKV", ".MOV"}


@dataclass
class CropRecord:
    score: float
    output_path: Path
    video_name: str
    frame_idx: int
    conf: float
    sharpness: float
    area_ratio: float
    bucket: str
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    width: int
    height: int


class BucketKeeper:
    """Keep top-scoring crops in each distance bucket."""

    def __init__(self, out_dir: Path, quotas: Dict[str, int]):
        self.out_dir = out_dir
        self.quotas = quotas
        self.heaps: Dict[str, List[Tuple[float, int, CropRecord]]] = {k: [] for k in quotas.keys()}
        self.counter = 0
        self.entry_id = 0

    def _next_name(self, bucket: str, video_stem: str, frame_idx: int, det_idx: int) -> Path:
        fname = f"{bucket}_{video_stem}_f{frame_idx:06d}_p{det_idx:02d}_{self.counter:08d}.jpg"
        self.counter += 1
        return self.out_dir / fname

    def _next_entry_id(self) -> int:
        self.entry_id += 1
        return self.entry_id

    def try_add(
        self,
        bucket: str,
        score: float,
        crop_image: np.ndarray,
        video_stem: str,
        frame_idx: int,
        det_idx: int,
        conf: float,
        sharpness: float,
        area_ratio: float,
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        heap = self.heaps[bucket]
        quota = self.quotas[bucket]
        x1, y1, x2, y2 = bbox
        h, w = crop_image.shape[:2]

        if quota <= 0:
            return False

        if len(heap) < quota:
            output_path = self._next_name(bucket, video_stem, frame_idx, det_idx)
            cv2.imwrite(str(output_path), crop_image)
            rec = CropRecord(
                score=score,
                output_path=output_path,
                video_name=video_stem,
                frame_idx=frame_idx,
                conf=conf,
                sharpness=sharpness,
                area_ratio=area_ratio,
                bucket=bucket,
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                width=w,
                height=h,
            )
            heapq.heappush(heap, (score, self._next_entry_id(), rec))
            return True

        min_score, _, min_rec = heap[0]
        if score <= min_score:
            return False

        # Replace the weakest sample in this bucket.
        heapq.heapreplace(
            heap,
            (
                score,
                self._next_entry_id(),
                CropRecord(
                    score=score,
                    output_path=min_rec.output_path,
                    video_name=video_stem,
                    frame_idx=frame_idx,
                    conf=conf,
                    sharpness=sharpness,
                    area_ratio=area_ratio,
                    bucket=bucket,
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    width=w,
                    height=h,
                ),
            ),
        )
        cv2.imwrite(str(min_rec.output_path), crop_image)
        return True

    def rebalance_fill(self) -> None:
        """
        Move strongest samples from buckets with surplus to buckets with shortage.
        This is metadata-level rebalance after full pass.
        """
        # Best-effort: we only rebalance quotas in accounting; files stay as-is.
        return

    def all_records(self) -> List[CropRecord]:
        rows = []
        for heap in self.heaps.values():
            rows.extend([x[2] for x in heap])
        rows.sort(key=lambda r: r.score, reverse=True)
        return rows

    def stats(self) -> Dict[str, int]:
        return {bucket: len(heap) for bucket, heap in self.heaps.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop high-quality student images from videos with quality-aware sampling."
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/ET-EDU"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cropped_students_quality_10k"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/cropped_students_quality_10k/metadata.csv"))
    parser.add_argument("--model-path", type=str, default="yolov8n.pt")
    parser.add_argument("--target-count", type=int, default=10000)
    parser.add_argument("--seconds-per-frame", type=float, default=2.5)
    parser.add_argument("--min-conf", type=float, default=0.45)
    parser.add_argument("--min-width", type=int, default=96)
    parser.add_argument("--min-height", type=int, default=160)
    parser.add_argument("--min-area-ratio", type=float, default=0.008)
    parser.add_argument("--blur-threshold", type=float, default=45.0)
    parser.add_argument("--padding-ratio", type=float, default=0.12)
    parser.add_argument("--near-ratio", type=float, default=0.65)
    parser.add_argument("--mid-ratio", type=float, default=0.25)
    parser.add_argument("--far-ratio", type=float, default=0.10)
    parser.add_argument("--near-area-ratio", type=float, default=0.035)
    parser.add_argument("--mid-area-ratio", type=float, default=0.015)
    parser.add_argument("--max-videos", type=int, default=0, help="Only process first N videos (0 = all).")
    parser.add_argument("--device", type=str, default=None, help="YOLO device, e.g. cpu, 0")
    parser.add_argument("--enhance", action="store_true", help="Enhance crop before saving.")
    parser.add_argument("--denoise", type=int, default=0, help="Denoise strength, 0 to disable.")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit.")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size.")
    parser.add_argument("--unsharp-amount", type=float, default=1.0, help="Unsharp amount.")
    parser.add_argument("--unsharp-sigma", type=float, default=1.0, help="Unsharp sigma.")
    parser.add_argument("--unsharp-kernel", type=int, default=5, help="Unsharp kernel size.")
    return parser.parse_args()


def compute_sharpness(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def ensure_odd_kernel(size: int) -> int:
    return size if size % 2 == 1 else size + 1


def apply_clahe_bgr(image_bgr: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(image_bgr: np.ndarray, amount: float, sigma: float, kernel_size: int) -> np.ndarray:
    if amount <= 0:
        return image_bgr
    kernel = ensure_odd_kernel(max(1, kernel_size))
    blurred = cv2.GaussianBlur(image_bgr, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_crop(
    image_bgr: np.ndarray,
    denoise_strength: int,
    clahe_clip_limit: float,
    clahe_grid_size: int,
    unsharp_amount: float,
    unsharp_sigma: float,
    unsharp_kernel: int,
) -> np.ndarray:
    enhanced = image_bgr
    if denoise_strength > 0:
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            h=denoise_strength,
            hColor=denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    enhanced = apply_clahe_bgr(enhanced, clip_limit=clahe_clip_limit, grid_size=clahe_grid_size)
    enhanced = apply_unsharp_mask(enhanced, amount=unsharp_amount, sigma=unsharp_sigma, kernel_size=unsharp_kernel)
    return enhanced


def quality_score(conf: float, sharpness: float, area_ratio: float, min_area_ratio: float) -> float:
    # Normalized components.
    size_score = np.clip((area_ratio - min_area_ratio) / (0.08 - min_area_ratio), 0.0, 1.0)
    sharp_score = np.clip(math.log1p(sharpness) / 6.5, 0.0, 1.0)
    conf_score = np.clip(conf, 0.0, 1.0)
    return float(0.45 * size_score + 0.35 * sharp_score + 0.20 * conf_score)


def get_bucket(area_ratio: float, near_thr: float, mid_thr: float) -> str:
    if area_ratio >= near_thr:
        return "near"
    if area_ratio >= mid_thr:
        return "mid"
    return "far"


def collect_videos(video_dir: Path) -> List[Path]:
    videos = [p for p in sorted(video_dir.iterdir()) if p.is_file() and p.suffix in VIDEO_EXTENSIONS]
    return videos


def calc_quotas(target_count: int, near_ratio: float, mid_ratio: float, far_ratio: float) -> Dict[str, int]:
    total = near_ratio + mid_ratio + far_ratio
    near_q = int(round(target_count * near_ratio / total))
    mid_q = int(round(target_count * mid_ratio / total))
    far_q = target_count - near_q - mid_q
    return {"near": near_q, "mid": mid_q, "far": far_q}


def main() -> None:
    args = parse_args()

    videos = collect_videos(args.video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {args.video_dir}")
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    quotas = calc_quotas(args.target_count, args.near_ratio, args.mid_ratio, args.far_ratio)
    keeper = BucketKeeper(args.output_dir, quotas)

    model = YOLO(args.model_path)
    predict_kwargs = {"verbose": False}
    if args.device is not None:
        predict_kwargs["device"] = args.device

    print(f"Videos: {len(videos)}")
    print(f"Target: {args.target_count} crops")
    print(f"Bucket quotas: {quotas}")
    print(f"Sampling every {args.seconds_per_frame:.2f} sec/frame")
    if args.enhance:
        print(
            "Enhance: ON "
            f"(denoise={args.denoise}, clahe_clip={args.clahe_clip}, clahe_grid={args.clahe_grid}, "
            f"unsharp_amount={args.unsharp_amount}, unsharp_sigma={args.unsharp_sigma}, "
            f"unsharp_kernel={args.unsharp_kernel})"
        )
    else:
        print("Enhance: OFF")

    frames_scanned = 0
    detections_seen = 0
    accepted = 0

    for vid_idx, vid_path in enumerate(videos, start=1):
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 25.0
        frame_step = max(1, int(round(fps * args.seconds_per_frame)))
        frame_idx = 0

        print(f"[{vid_idx}/{len(videos)}] Processing: {vid_path.name} | fps={fps:.2f} | step={frame_step}")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            frames_scanned += 1
            frame_h, frame_w = frame.shape[:2]
            frame_area = float(frame_h * frame_w)

            results = model.predict(frame, **predict_kwargs)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                frame_idx += 1
                continue

            for det_idx, box in enumerate(boxes):
                if int(box.cls[0]) != 0:
                    continue

                conf = float(box.conf[0])
                if conf < args.min_conf:
                    continue

                detections_seen += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                if bw < args.min_width or bh < args.min_height:
                    continue

                area_ratio = (bw * bh) / frame_area
                if area_ratio < args.min_area_ratio:
                    continue

                # Add padding to preserve behavior context.
                pad_w = int(round(bw * args.padding_ratio))
                pad_h = int(round(bh * args.padding_ratio))
                cx1 = max(0, x1 - pad_w)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(frame_w, x2 + pad_w)
                cy2 = min(frame_h, y2 + pad_h)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                sharpness = compute_sharpness(crop)
                if sharpness < args.blur_threshold:
                    continue

                output_crop = crop
                if args.enhance:
                    output_crop = enhance_crop(
                        crop,
                        denoise_strength=args.denoise,
                        clahe_clip_limit=args.clahe_clip,
                        clahe_grid_size=args.clahe_grid,
                        unsharp_amount=args.unsharp_amount,
                        unsharp_sigma=args.unsharp_sigma,
                        unsharp_kernel=args.unsharp_kernel,
                    )

                bucket = get_bucket(area_ratio, near_thr=args.near_area_ratio, mid_thr=args.mid_area_ratio)
                score = quality_score(conf, sharpness, area_ratio, args.min_area_ratio)
                if keeper.try_add(
                    bucket=bucket,
                    score=score,
                    crop_image=output_crop,
                    video_stem=vid_path.stem,
                    frame_idx=frame_idx,
                    det_idx=det_idx,
                    conf=conf,
                    sharpness=sharpness,
                    area_ratio=area_ratio,
                    bbox=(cx1, cy1, cx2, cy2),
                ):
                    accepted += 1

            if frames_scanned % 200 == 0:
                print(
                    f"  frames={frames_scanned} detections={detections_seen} accepted={accepted} "
                    f"kept={keeper.stats()}"
                )

            frame_idx += 1

        cap.release()

    rows = keeper.all_records()
    with args.metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "bucket",
                "score",
                "video_name",
                "frame_idx",
                "conf",
                "sharpness",
                "area_ratio",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "width",
                "height",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.output_path.as_posix(),
                    r.bucket,
                    f"{r.score:.6f}",
                    r.video_name,
                    r.frame_idx,
                    f"{r.conf:.6f}",
                    f"{r.sharpness:.4f}",
                    f"{r.area_ratio:.6f}",
                    r.bbox_x1,
                    r.bbox_y1,
                    r.bbox_x2,
                    r.bbox_y2,
                    r.width,
                    r.height,
                ]
            )

    print("\nDone.")
    print(f"Frames scanned:  {frames_scanned}")
    print(f"Detections seen: {detections_seen}")
    print(f"Accepted events: {accepted}")
    print(f"Final kept:      {len(rows)}")
    print(f"By bucket:       {keeper.stats()}")
    print(f"Images:          {args.output_dir}")
    print(f"Metadata:        {args.metadata_csv}")


if __name__ == "__main__":
    main()
