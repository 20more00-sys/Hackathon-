# Real Evaluation Report

## Test protocol and limitations

The validation-selected checkpoint was evaluated on all **1,600 images** in the
original Kaggle version 2 Testing split. Test data was not used to select the
model or tune its hyperparameters. Predictions and ground-truth labels both
contained 1,600 entries.

This is an evaluation of a **partial training run**: the interrupted transfer
stage's history is unavailable, and the configured full schedule was not
completed. The dataset audit also found 2,740 cross-split perceptual near-match
candidates in the prepared splits. Those candidates are not all confirmed
duplicates, but they have not been visually resolved. No patient IDs were
available. Treat these metrics as dataset-specific experimental results, not
clinical evidence or patient-independent generalization.

## Metrics

| Metric | Test result |
|---|---:|
| Accuracy | 0.87125 |
| Macro F1 | 0.86791 |
| Weighted F1 | 0.86791 |
| Balanced accuracy | 0.87125 |
| Macro one-vs-rest ROC-AUC | 0.97213 |
| Macro one-vs-rest average precision (PR-AUC) | 0.93690 |

| Class | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| glioma | 0.93291 | 0.73000 | 0.81907 | 0.94766 | 0.90925 |
| meningioma | 0.80779 | 0.77750 | 0.79236 | 0.94821 | 0.86181 |
| notumor | 0.83648 | 0.99750 | 0.90992 | 0.99494 | 0.98309 |
| pituitary | 0.92235 | 0.98000 | 0.95030 | 0.99770 | 0.99345 |

Confusion matrix (rows: actual, columns: predicted; class order:
glioma, meningioma, notumor, pituitary):

```text
[[292, 65, 38,  5],
 [ 21, 311, 40, 28],
 [  0,  1, 399,  0],
 [  0,  8,  0, 392]]
```

## Grad-CAM

Grad-CAM generated non-constant heatmaps, original images, and overlays for
four real test examples:

| Actual class | Predicted class | Confidence |
|---|---|---:|
| glioma | notumor | 0.86722 |
| meningioma | meningioma | 0.75867 |
| notumor | notumor | 0.99948 |
| pituitary | pituitary | 0.65107 |

Artifacts are in `results/gradcam/<class>/`. Grad-CAM is an explanation
visualization, not proof of clinical validity.

## Historical baseline

The previously reported baseline is 60.81% accuracy. Its test dataset and
evaluation protocol could not be verified as equivalent to this version 2
dataset and split, so no direct comparison or improvement claim is made.

## Machine-readable results and plots

- `results/experiment_metrics.json`
- `results/reports/evaluation_summary.json`
- `results/reports/classification_report.json`
- `results/plots/confusion_matrix.png`
- `results/plots/normalized_confusion_matrix.png`
- `results/plots/roc_curves.png`
- `results/plots/precision_recall_curves.png`
- `results/plots/training_history.png`
