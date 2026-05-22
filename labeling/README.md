# ET-EDU-CBIR-V2 Labeling (Max-Accuracy Workflow)

This folder contains helper scripts to make manual labeling accurate and auditable.

## Recommended process (accuracy-first)

1) **Label val/test queries first** (all images with `retrieval_role=query` and `split in {val,test}`)
2) Label val/test galleries
3) Label train images
4) Run QC after each phase
5) Export to training txt files

Why queries first? It prevents evaluation sets where queries have no relevant galleries.

## Option A — Label in spreadsheets (Excel/Google Sheets)

Generate review CSVs with extra context (camera/video/time/quality):

```bash
python labeling/make_labeling_review_csvs.py --dataset-root data/ET-EDU-CBIR-V2
```

Outputs:
- `data/ET-EDU-CBIR-V2/labeling_review/queries_val_test_review.csv`
- `data/ET-EDU-CBIR-V2/labeling_review/gallery_val_test_review.csv`
- `data/ET-EDU-CBIR-V2/labeling_review/train_review.csv`

You can label directly in `data/ET-EDU-CBIR-V2/labels_template.csv` as well.

## Option B — Label Studio (GUI)

1) Install Label Studio (optional):

```bash
pip install label-studio
```

2) Enable local file serving and start:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
label-studio start
```

3) Create tasks JSON:

```bash
python labeling/make_label_studio_tasks.py --dataset-root data/ET-EDU-CBIR-V2
```

4) In Label Studio:
- Create a project
- Paste the labeling config from `labeling/label_studio_config.xml`
- Import tasks from `data/ET-EDU-CBIR-V2/label_studio/tasks.json`

5) Export annotations as JSON and apply back:

```bash
python labeling/apply_label_studio_export.py \
  --export-json /path/to/label_studio_export.json \
  --labels-csv data/ET-EDU-CBIR-V2/labels_template.csv
```

This writes a labeled CSV alongside the original (suffix `.labeled.csv`).

## QC (always run)

```bash
python labeling/qc_labeled_csv.py --csv data/ET-EDU-CBIR-V2/labels_template.csv --out-json data/ET-EDU-CBIR-V2/qc_report.json
```

## Convert labeled CSV to training txt files

```bash
python build_edu_txt_from_labeled_csv.py \
  --labeled-csv data/ET-EDU-CBIR-V2/labels_template.csv \
  --output-root data/ET-EDU-CBIR-V2
```

Then train:

```bash
python train.py --config configs/et_edu_cbir_v2.yaml
```
