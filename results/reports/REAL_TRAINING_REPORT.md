# Real Training Report

## Status: experiment evaluation completed, configured training was partial

The first CPU training process was interrupted before its stage history was
written. It left a valid `transfer_best.keras` checkpoint, but its epoch count
and training history could not be recovered. The run was resumed from that
checkpoint, fine-tuned for one epoch, and then selected between the transfer
checkpoint and that fine-tuned checkpoint using validation loss only.

The requested full 8-epoch transfer plus 6-epoch fine-tuning schedule was **not
completed**. The first transfer-stage history is unavailable, and the resumed
fine-tuning stage ran one epoch rather than six. The outputs must be treated as
a real-data partial-training result, not a fully reproduced training schedule.

## Data and configuration

- Dataset: Kaggle `masoudnickparvar/brain-tumor-mri-dataset`, version 2, CC BY 4.0
- Source image count: 7,200
- Prepared Training: 4,701 images
- Prepared Validation: 784 images
- Untouched Testing: 1,600 images
- Seed: 42
- Input size: 224 x 224
- Batch size: 32
- Compute: CPU only; no GPU devices were available
- Exact train/test duplicates excluded from Training: 115 files
- Patient identifiers: unavailable; no patient-level separation claim

## Validation-based checkpoint selection

| Candidate | Validation categorical cross-entropy |
|---|---:|
| Resumed best transfer checkpoint | 0.125967 |
| One-epoch fine-tuned checkpoint | 0.161716 |

The transfer checkpoint had lower validation loss, so it was selected as
`results/models/brain_tumor_model.keras`. The fine-tuned checkpoint remains
available as `results/models/fine_tuned_best.keras`, but it was not selected.
The test set was not used for this selection.

## Artifacts

- `results/models/transfer_best.keras` — checkpoint from the interrupted
  transfer stage
- `results/models/fine_tuned_best.keras` — one-epoch fine-tuning candidate
- `results/models/brain_tumor_model.keras` — validation-selected checkpoint
- `results/reports/real_training_config.json` — run configuration
- `results/reports/fine_tuning_history.json` — history for the resumed epoch
- `results/reports/real_training.log` — command output and errors

The final evaluation was run once on the original 1,600-image Testing split.
See [REAL_EVALUATION_REPORT.md](./REAL_EVALUATION_REPORT.md). Its results are
subject to the split's reported near-duplicate risks and the absence of patient
group metadata.
