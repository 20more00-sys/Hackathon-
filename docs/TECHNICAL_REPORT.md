# Technical Report

## Initial baseline results

The original notebook trained a frozen ResNet50 model with ImageNet weights for four MRI classes. Earlier results were limited by weak class recall and incomplete fine-tuning control. The original training history showed the validation curve plateauing at around 60-62% accuracy, with poor Glioma and Meningioma recall.

## Previous model weaknesses

- Random augmentation was applied to validation data because the training and validation generators shared the same configuration.
- ResNet50 preprocessing used raw scaling rather than the ImageNet preprocessing function expected by the pretrained network.
- Fine-tuning was not done in a stable, checkpoint-aware, two-stage workflow.
- The code existed mainly as a notebook and lacked robust evaluation artifacts, save/load checks, and reusable inference paths.
- No clear separation existed between scientific workflows (dataset audit, training, evaluation) and production code paths.

## Corrections implemented

- Replaced `rescale=1./255` with `tf.keras.applications.resnet50.preprocess_input`.
- Split the generators into distinct training, validation, and test flows.
- Added class consistency and image-count validation checks.
- Added dataset auditing for class counts, corruption issues, duplicate filenames, and duplicate image hashing.
- Implemented safe transfer learning followed by controlled fine-tuning with checkpoint preservation.
- Added evaluation plots and JSON artifacts under `results/`.
- Added Grad-CAM and single-image prediction modules.
- Added a lightweight Streamlit UI and documentation updates.

## Experimental methodology

The current implementation follows a conservative workflow:

1. Audit the dataset and validate class structure.
2. Create separate data pipelines for training, validation, and test splits.
3. Train the classifier head while the pretrained ResNet50 backbone remains frozen.
4. Save the best transfer-learning checkpoint.
5. Unfreeze only the late backbone layers and recompile with a reduced learning rate.
6. Preserve the best fine-tune checkpoint and avoid overwriting a better checkpoint with a worse one.
7. Evaluate only on the held-out test set after training.

## Evaluation results

The repository provides a reproducible evaluation framework and reports, but no real MRI dataset was available in this environment. Results therefore remain synthetic or in the structure of a reproducible pipeline rather than clinical claims.

## Limitations

- Real model accuracy cannot be claimed without the actual brain-tumor MRI dataset.
- GPU availability was not guaranteed in this environment, so training behavior may differ from a GPU-enabled workstation.
- Grad-CAM highlights image regions, but it is not clinical proof of diagnosis.

## Future research directions

- Compare multiple backbones such as EfficientNet, DenseNet, and ConvNeXt variants.
- Evaluate realistic patient-level data split strategies when patient metadata is available.
- Add uncertainty estimation and calibration metrics for high-stakes deployment contexts.
- Perform external validation on independent clinical datasets.
