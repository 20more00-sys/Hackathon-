# Brain Tumor MRI Classification

An educational four-class MRI image-classification prototype using ResNet50 transfer learning, held-out evaluation, optional Grad-CAM visualization, and a small Streamlit demo.

> **Medical disclaimer:** This project is not a medical device or diagnostic tool. It has not been clinically validated and must not be used for screening, diagnosis, treatment, triage, or patient-care decisions. Model scores and Grad-CAM visualizations are experimental only.

## For Judges – How to Run

### Option 1: Run the Colab notebook

Open [`Hackathon_Colab.ipynb`](./Hackathon_Colab.ipynb) in Google Colab, then select **Runtime → Restart session and run all**. The notebook installs its dependencies, downloads the dataset with KaggleHub, displays class counts, evaluates a majority-class baseline, and loads the repository checkpoint when available. If no checkpoint is present in the Colab runtime, it trains a short ResNet50 transfer-learning fallback. It then reports test metrics, plots confusion matrices, shows up to eight misclassified examples with a best-effort Grad-CAM, and saves `models/brain_tumor_model.keras`.

The notebook evaluates the Kaggle-provided `Testing` folder. This dataset has no patient identifiers; its image split does not establish patient-independent generalization.

### Option 2: Run the Streamlit app locally

From the repository root, install dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

The app defaults to the included checkpoint at `results/models/brain_tumor_model.keras`. If you use another compatible four-class Keras checkpoint, enter its path in the app. The app always displays the medical disclaimer; Grad-CAM is optional and may be unavailable for incompatible model architectures.

### Dataset credit

The notebook downloads [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar from Kaggle using `kagglehub`. The project records dataset version 2 and its CC BY 4.0 attribution. Do not redistribute MRI images without following the dataset license and source terms.

## Results in this repository

The existing validation-selected checkpoint reached **87.1% accuracy** and **0.868 macro F1** on the provided 1,600-image test split (400 images per class). A constant majority-class prediction scores **25.0%** on this balanced test set. Glioma recall was 73.0%, and meningioma recall was 77.8%.

These are dataset-specific experimental measurements from a **partial training run**; the configured full training schedule was not completed. The dataset audit also flagged 2,740 cross-split perceptual near-match candidates (not confirmed duplicates), and patient identifiers were unavailable. Do not interpret these results as clinical performance. See [the evaluation report](./results/reports/REAL_EVALUATION_REPORT.md), [training report](./results/reports/REAL_TRAINING_REPORT.md), and [one-page write-up](./docs/WRITEUP.md).

## Reproduce training locally

Download or otherwise provide the dataset with this layout:

```text
brain_tumor_data/
  Training/
    glioma/ meningioma/ notumor/ pituitary/
  Testing/
    glioma/ meningioma/ notumor/ pituitary/
```

Audit it and prepare exact-duplicate-safe train/validation splits while preserving the original test set:

```bash
python src/real_dataset_audit.py --data-dir brain_tumor_data --output-dir results
python src/prepare_real_splits.py \
  --data-dir brain_tumor_data \
  --output-dir results/prepared_splits \
  --report-dir results/reports \
  --seed 42
```

Train and evaluate the ResNet50 experiment:

```bash
python src/run_real_experiment.py \
  --data-dir brain_tumor_data \
  --prepared-dir results/prepared_splits \
  --output-dir results \
  --epochs 8 \
  --fine-tune-epochs 6 \
  --batch-size 32 \
  --img-size 224 \
  --seed 42
```

To run single-image inference or create a Grad-CAM visualization:

```bash
python src/predict.py \
  --model results/models/brain_tumor_model.keras \
  --image path/to/image.png \
  --gradcam-dir results/gradcam
```

## Project files

- `Hackathon_Colab.ipynb` — standalone Colab data, baseline, training, evaluation, and error-inspection workflow
- `src/` — reusable training, evaluation, prediction, dataset-audit, and Grad-CAM code
- `app/app.py` — Streamlit image upload demo
- `results/models/` — saved Keras checkpoints
- `results/plots/` and `results/reports/` — metrics, figures, and experiment documentation

Run the test suite with:

```bash
python -m pytest -q
```
