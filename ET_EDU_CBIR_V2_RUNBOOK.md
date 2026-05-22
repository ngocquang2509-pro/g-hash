# ET-EDU CBIR V2 Runbook

This runbook uses `extract_edu_cbir_v2.py` to build a cleaner, more diverse crop dataset from ET-EDU videos.

## 1) Quick Smoke Test

Run on 1 video to validate environment and output format.

```bash
python extract_edu_cbir_v2.py \
  --video-dir data/ET-EDU \
  --output-root data/ET-EDU-CBIR-V2-smoke \
  --max-videos 1 \
  --max-frames-per-video 1200
```

Expected outputs:

- `data/ET-EDU-CBIR-V2-smoke/images/*.jpg`
- `data/ET-EDU-CBIR-V2-smoke/metadata.csv`
- `data/ET-EDU-CBIR-V2-smoke/summary.json`
- `data/ET-EDU-CBIR-V2-smoke/train_img_unlabeled.txt`
- `data/ET-EDU-CBIR-V2-smoke/val_query_img.txt`
- `data/ET-EDU-CBIR-V2-smoke/val_gallery_img.txt`
- `data/ET-EDU-CBIR-V2-smoke/test_query_img.txt`
- `data/ET-EDU-CBIR-V2-smoke/test_gallery_img.txt`
- `data/ET-EDU-CBIR-V2-smoke/labels_template.csv` (if concepts file exists)

## 2) Full Extraction

Run on all ET-EDU videos with the recommended defaults.

```bash
python extract_edu_cbir_v2.py \
  --video-dir data/ET-EDU \
  --output-root data/ET-EDU-CBIR-V2
```

## 3) Tunable Controls

- More/less strict blur filter: `--laplacian-min 120`
- More/less dense temporal sampling: `--keyframe-interval 1.0`
- More/less per-track diversity: `--max-per-track 3`
- More/less duplicate filtering: `--hash-hamming 6`
- Stronger camera balancing: `--max-camera-ratio 0.40`

For current ET-EDU (3 camera IDs), `0.40` is usually safer than `0.25`.

## 4) Labeling Step (after extraction)

This script only builds high-quality, low-duplicate crops and split lists.

- Use `labels_template.csv` to annotate concepts.
- After annotation, run conversion:

```bash
python build_edu_txt_from_labeled_csv.py \
  --labeled-csv data/ET-EDU-CBIR-V2/labels_template.csv \
  --output-root data/ET-EDU-CBIR-V2
```

- The conversion creates `train_img.txt`, `train_label.txt`, `test_img.txt`, `test_label.txt`, `concepts.txt`.

## 5) Practical Notes

- Keep split by video to avoid leakage.
- Keep near-duplicate ratio under 10-12%.
- For CBIR evaluation, keep query/gallery lists generated from val/test splits.

## 6) Train with Dedicated Config

After label conversion is done, train with:

```bash
python train.py --config configs/et_edu_cbir_v2.yaml
```