# Brain Tumor MRI Classification

This repository contains a corrected, reproducible ResNet50 pipeline for the four-class brain tumor MRI task.

## Installation

Use a Python 3.12+ environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

If using the bundled Python in this workspace:

```bash
/usr/local/python/3.13.8/bin/python -m pip install -r requirements.txt
```

## Dataset structure

The dataset must be organized as follows:

```text
brain_tumor_data/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

The expected class names are:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

## Downloading the real dataset

The verified initial experiment uses Kaggle dataset
[`masoudnickparvar/brain-tumor-mri-dataset`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset),
version 2 (CC BY 4.0). KaggleHub downloads to its cache and prints the resulting
path; it does not place data in the repository automatically.

```python
import kagglehub

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print("Path to dataset files:", path)
```

The downloaded version currently contains 7,200 images. Record the version and
license when reproducing results. Do not commit the downloaded MRI files.

Audit and prepare exact-duplicate-safe train/validation splits while preserving
the upstream Testing files:

```bash
python src/real_dataset_audit.py --data-dir /path/printed/by/kagglehub --output-dir results
python src/prepare_real_splits.py \
  --data-dir /path/printed/by/kagglehub \
  --output-dir /tmp/brain-tumor-mri-prepared \
  --report-dir results/reports --seed 42
```

After auditing and preparing splits, the experiment runner trains, selects a
checkpoint using validation data only, evaluates the untouched test split once,
and creates Grad-CAM examples:

```bash
python src/run_real_experiment.py \
  --data-dir /path/to/brain-tumor-mri-dataset/versions/2 \
  --prepared-dir results/prepared_splits \
  --output-dir results \
  --epochs 8 --fine-tune-epochs 6 --batch-size 32 --img-size 224 --seed 42
```

For resume behavior and actual dataset findings from the current workspace, see
`results/reports/REAL_TRAINING_REPORT.md` and
`results/reports/REAL_DATASET_AUDIT.md`. The saved current result is a partial
CPU training run, not the full schedule shown above.

## Environment setup

The project uses TensorFlow/Keras with ResNet50 and a clean ImageNet preprocessing pipeline. The model is trained with a transfer-learning stage followed by a controlled fine-tuning stage.

## Training commands

```bash
/usr/local/python/3.13.8/bin/python src/brain_tumor_pipeline.py --data-dir brain_tumor_data --output-dir artifacts --img-size 224 --batch-size 32 --validation-split 0.15
```

## Fine-tuning commands

The pipeline includes a safe two-stage workflow. The transfer-learning stage is saved first, then the final layers are selectively unfrozen.

```bash
/usr/local/python/3.13.8/bin/python src/brain_tumor_pipeline.py --data-dir brain_tumor_data --output-dir artifacts --img-size 224 --batch-size 32 --validation-split 0.15
```

This is the same command used for the main training workflow; fine-tuning is internal to the script when the first stage is stable.

## Evaluation commands

```bash
/usr/local/python/3.13.8/bin/python src/brain_tumor_pipeline.py --data-dir brain_tumor_data --output-dir results
```

Evaluation outputs are saved under:

- `results/plots/`
- `results/reports/`

## Prediction commands

```bash
/usr/local/python/3.13.8/bin/python src/predict.py --model artifacts/brain_tumor_model.keras --image path/to/image.png --gradcam-dir results/gradcam
```

The output includes the predicted class, per-class probability scores, confidence, and optional Grad-CAM artifacts. Predictions are experimental and not medical diagnoses.

## Grad-CAM instructions

```bash
/usr/local/python/3.13.8/bin/python src/gradcam.py --model artifacts/brain_tumor_model.keras --image path/to/image.png --output-dir results/gradcam
```

## Streamlit launch command

```bash
/usr/local/python/3.13.8/bin/python -m streamlit run app/app.py
```

## Results directory explanation

- `results/plots/`: confusion matrices, training history, ROC, and PR curves
- `results/reports/`: JSON summary and classification report
- `artifacts/`: saved models and training checkpoints

## Limitations

- This repository does not include the original Kaggle dataset files in the workspace.
- Real test accuracy and generalization can only be claimed with the actual MRI dataset and patient-aware splitting.
- GPU availability and dataset size may affect training runtime and convergence.
- Grad-CAM is interpretability support, not clinical proof.

## Medical disclaimer

This project is for research and educational use only. It is not a medical diagnostic system and should not be used to make clinical decisions.

## Validation

Run the project tests with:

```bash
/usr/local/python/3.13.8/bin/python -m pytest -q
```

Run CLI help with:

```bash
/usr/local/python/3.13.8/bin/python src/brain_tumor_pipeline.py --help
```
