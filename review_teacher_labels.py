"""
Launcher UI review cho nhãn Teacher pseudo-label.

Mặc định:
  - Input CSV : data/teacher_pseudo_quality10k.csv
  - Data root : data
  - Output CSV: data/teacher_pseudo_quality10k_reviewed.csv

Có thể override qua command line:
  python review_teacher_labels.py --csv-file ... --save-path ...
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Launch review UI for teacher pseudo labels.")
    parser.add_argument("--csv-file", default="data/teacher_pseudo_quality10k.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--save-path", default="data/teacher_pseudo_quality10k_reviewed.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    csv_file = Path(args.csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_file}")

    cmd = [
        sys.executable,
        "review_labels.py",
        "--csv-file",
        args.csv_file,
        "--data-root",
        args.data_root,
        "--save-path",
        args.save_path,
    ]

    print("Launching teacher review UI...")
    print(f"  Input : {args.csv_file}")
    print(f"  Output: {args.save_path}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
