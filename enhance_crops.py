import argparse
from pathlib import Path

import cv2
import numpy as np


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(files)


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


def enhance_image(
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
    enhanced = apply_unsharp_mask(
        enhanced,
        amount=unsharp_amount,
        sigma=unsharp_sigma,
        kernel_size=unsharp_kernel,
    )
    return enhanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance cropped student images using denoise + CLAHE + unsharp mask."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/cropped_students"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cropped_students_enhanced"))
    parser.add_argument("--recursive", action="store_true", help="Scan input directory recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output images if already exists.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all).")
    parser.add_argument("--denoise", type=int, default=0, help="Denoise strength, 0 to disable.")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit.")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size.")
    parser.add_argument("--unsharp-amount", type=float, default=1.1, help="Unsharp amount.")
    parser.add_argument("--unsharp-sigma", type=float, default=1.0, help="Gaussian sigma for unsharp mask.")
    parser.add_argument("--unsharp-kernel", type=int, default=5, help="Gaussian kernel size for unsharp mask.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    files = list_images(args.input_dir, recursive=args.recursive)
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        print("No input images found.")
        return

    processed = 0
    skipped = 0
    failed = 0

    print(f"Found {len(files)} images.")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    for idx, src_path in enumerate(files, start=1):
        rel_path = src_path.relative_to(args.input_dir)
        dst_path = args.output_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            print(f"[WARN] Cannot read image: {src_path}")
            continue

        enhanced = enhance_image(
            image,
            denoise_strength=args.denoise,
            clahe_clip_limit=args.clahe_clip,
            clahe_grid_size=args.clahe_grid,
            unsharp_amount=args.unsharp_amount,
            unsharp_sigma=args.unsharp_sigma,
            unsharp_kernel=args.unsharp_kernel,
        )

        ok = cv2.imwrite(str(dst_path), enhanced)
        if not ok:
            failed += 1
            print(f"[WARN] Cannot write image: {dst_path}")
            continue

        processed += 1
        if idx % 100 == 0 or idx == len(files):
            print(f"Progress {idx}/{len(files)} | processed={processed} skipped={skipped} failed={failed}")

    print("Done.")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":
    main()
