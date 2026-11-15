# Real Dataset Audit

## Source and acquisition

- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Owner:** Masoud Nickparvar
- **Kaggle identifier:** `masoudnickparvar/brain-tumor-mri-dataset` (dataset ID `1608934`)
- **Downloaded version:** 2
- **License displayed by Kaggle:** Attribution 4.0 International (CC BY 4.0)
- **Dataset card's stated count:** 7,200 (1,400 training and 400 testing per class)
- **Downloaded and audited count:** **7,200 valid images**
- **Local source:** KaggleHub cache, version directory ending in `versions/2`
- **Original data modified:** No

The source card describes version 2 as having removed duplicates and train/test overlap. The independent audit below found no byte-identical file groups crossing the published split, but did find decoded-pixel-identical images crossing it. This discrepancy is retained as a data-quality finding; it is not silently reconciled with the source claim.

## Counts

| Class | Published Training | Published Testing | Total |
|---|---:|---:|---:|
| glioma | 1,400 | 400 | 1,800 |
| meningioma | 1,400 | 400 | 1,800 |
| notumor | 1,400 | 400 | 1,800 |
| pituitary | 1,400 | 400 | 1,800 |
| **Total** | **5,600** | **1,600** | **7,200** |

All 7,200 class-directory files were supported, readable images. Corrupt image count: **0**. Unsupported file count: **0**. Empty expected class directories: **0**.

## Duplicate and split-leakage findings

- SHA-256-identical file groups crossing the published Training/Testing split: **0**.
- Exact decoded-RGB pixel duplicate groups crossing the published split: **111 groups**, containing 229 paths.
- The preparation step retained the original Testing files and excluded **115 Training-side files** whose decoded RGB pixels exactly matched an image in Testing.
- Training images with identical decoded RGB content were kept in the same group when forming Training and Validation.
- Verified exact decoded-RGB duplicate groups crossing the prepared Training/Validation/Testing splits: **0**.

The exact pixel hash includes image dimensions and RGB-decoded pixel bytes. It can identify identical decoded images even when file encodings differ; it is not a patient-identity check.

Near-duplicate screening used a 64-bit DCT perceptual hash (grayscale 32x32 input, low-frequency 8x8 coefficients; Hamming distance at most 4). It produced **5,531 candidate pairs** on the original published split, including **1,924 cross-split candidates**. On the prepared splits it produced 5,484 candidate pairs, including **2,740 cross-split candidates**. These are screening candidates, not confirmed duplicate images; common MRI structure can produce false positives. Candidate pairs are in `dataset_audit.json` and `real_split_summary.json` and need visual review. Near-duplicate separation is therefore not claimed.

## Dimensions and color modes

The source images vary in size: 381 distinct dimensions in Training and 203 in Testing. `512x512` is the most common dimension (4,012 Training and 1,002 Testing images). Full dimension-frequency tables are in `dataset_audit.json`.

| Split | Original mode counts |
|---|---|
| Training | L: 2,358; RGB: 3,238; RGBA: 3; P: 1 |
| Testing | RGB: 891; L: 709 |

The model loader converts images to RGB and resizes them to the model's configured input size.

## Patient independence and reproducibility

No reliable patient identifiers or group metadata were present. Patient-level separation is **not** claimed. The prepared validation split uses seed 42 and `StratifiedGroupKFold` with decoded-RGB exact duplicates as groups; its image lists are recorded in `real_split_manifest.csv`. The original 1,600-image Testing split remains untouched and is not used for training or model selection.

Files generated:

- `results/reports/dataset_audit.json` — full counts, image dimensions/modes, exact and near duplicate findings
- `results/reports/real_split_summary.json` — split counts and leakage checks
- `results/reports/real_split_manifest.csv` — each prepared path's source and assigned split
- `results/plots/dataset_class_distribution.png` — Training/Testing class distribution

The audit script is reproducible with:

```bash
python src/real_dataset_audit.py \
  --data-dir /path/to/brain-tumor-mri-dataset/versions/2 \
  --output-dir results
```

The data-preparation script creates symlinks in a separate directory; it does not modify or copy over the original files:

```bash
python src/prepare_real_splits.py \
  --data-dir /path/to/brain-tumor-mri-dataset/versions/2 \
  --output-dir /tmp/brain-tumor-mri-prepared \
  --report-dir results/reports \
  --seed 42
```
