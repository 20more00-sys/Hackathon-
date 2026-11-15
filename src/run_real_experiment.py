from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

from src.brain_tumor_pipeline import (
    EXPECTED_CLASSES,
    build_resnet_model,
    create_generators,
    evaluate_model,
    train_model_stable,
)
from src.gradcam import generate_gradcam_visualization

KAGGLE_SOURCE = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
SEED = 42


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _combine_histories(training: dict[str, Any]) -> dict[str, list[float]]:
    first = training.get("transfer_learning", {})
    second = training.get("fine_tuning", {})
    return {
        key: list(first.get(key, [])) + list(second.get(key, []))
        for key in set(first) | set(second)
    }


def _fine_tune_from_checkpoint(
    checkpoint_path: Path,
    train_generator: Any,
    validation_generator: Any,
    output_dir: Path,
    epochs: int,
) -> tuple[tf.keras.Model, dict[str, list[float]], float, float]:
    model = tf.keras.models.load_model(str(checkpoint_path), compile=False)

    baseline_prediction = model.predict(validation_generator, verbose=0)
    baseline_loss = float(
        tf.keras.losses.categorical_crossentropy(
            tf.keras.utils.to_categorical(
                validation_generator.classes,
                num_classes=len(EXPECTED_CLASSES),
            ),
            baseline_prediction,
        ).numpy().mean()
    )

    head_start = next(
        (index for index, layer in enumerate(model.layers) if layer.name == "head_gap"),
        None,
    )
    if head_start is None:
        raise ValueError(f"Could not locate the classifier head in checkpoint {checkpoint_path}.")
    backbone_layers = model.layers[:head_start]
    head_layers = model.layers[head_start:]
    for layer in backbone_layers:
        layer.trainable = False
    for layer in backbone_layers[-20:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    for layer in head_layers:
        layer.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    checkpoint_best = output_dir / "fine_tuned_best.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=1e-6),
        ModelCheckpoint(
            filepath=str(checkpoint_best),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ]
    history_object = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=2,
    )
    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }
    best_fine_tune_loss = min(history.get("val_loss", []), default=float("inf"))
    if best_fine_tune_loss < baseline_loss and checkpoint_best.is_file():
        selected = tf.keras.models.load_model(str(checkpoint_best), compile=False)
    else:
        selected = tf.keras.models.load_model(str(checkpoint_path), compile=False)
    return selected, history, baseline_loss, best_fine_tune_loss


def run_experiment(
    dataset_root: str | Path,
    prepared_root: str | Path,
    output_dir: str | Path,
    epochs: int = 8,
    fine_tune_epochs: int = 6,
    batch_size: int = 32,
    img_size: int = 224,
    seed: int = SEED,
    resume_transfer: str | Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    prepared_root = Path(prepared_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    models_dir = output_dir / "models"
    gradcam_dir = output_dir / "gradcam"
    for path in (reports_dir, plots_dir, models_dir, gradcam_dir):
        path.mkdir(parents=True, exist_ok=True)

    audit_path = reports_dir / "dataset_audit.json"
    split_path = reports_dir / "real_split_summary.json"
    if not audit_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("Run real_dataset_audit.py and prepare_real_splits.py before training.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    split_summary = json.loads(split_path.read_text(encoding="utf-8"))
    if Path(audit["dataset_root"]).resolve() != dataset_root:
        raise ValueError("The audit report dataset root does not match --data-dir.")
    if Path(split_summary["prepared_root"]).resolve() != prepared_root:
        raise ValueError("The split report prepared root does not match --prepared-dir.")
    if split_summary["cross_split_exact_duplicate_group_count"] != 0:
        raise ValueError("Exact duplicate leakage remains across prepared splits.")
    if not (prepared_root / "Validation").is_dir():
        raise FileNotFoundError(f"Prepared Validation directory not found: {prepared_root / 'Validation'}")

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    training_config = {
        "dataset_source": KAGGLE_SOURCE,
        "dataset_version": 2,
        "seed": seed,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "validation_method": split_summary["validation_method"],
        "test_policy": "Untouched original Testing split; used only for final evaluation and Grad-CAM examples.",
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "resume_transfer_checkpoint": str(resume_transfer) if resume_transfer is not None else None,
    }
    _write_json(reports_dir / "real_training_config.json", training_config)
    _write_json(output_dir / "experiment_metrics.json", {
        "status": "training_in_progress",
        "dataset": audit["source"],
        "split_distribution": split_summary["split_distribution"],
        "split_total": split_summary["split_total"],
        "configuration": training_config,
    })

    train_generator, validation_generator, test_generator = create_generators(
        prepared_root,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        validation_dir=prepared_root / "Validation",
    )
    if list(train_generator.class_indices) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected training class mapping: {train_generator.class_indices}")
    if list(validation_generator.class_indices) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected validation class mapping: {validation_generator.class_indices}")
    if list(test_generator.class_indices) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected test class mapping: {test_generator.class_indices}")

    started_at = time.perf_counter()
    if resume_transfer is not None:
        transfer_path = Path(resume_transfer).expanduser().resolve()
        if not transfer_path.is_file():
            raise FileNotFoundError(f"Transfer checkpoint for resume not found: {transfer_path}")
        model, fine_tune_history, transfer_val_loss, fine_tune_val_loss = _fine_tune_from_checkpoint(
            transfer_path,
            train_generator,
            validation_generator,
            models_dir,
            epochs=fine_tune_epochs,
        )
        training = {
            "strategy": "resumed_interrupted_run_from_best_transfer_checkpoint",
            "selected_stage": "fine_tuning" if fine_tune_val_loss < transfer_val_loss else "transfer_learning",
            "transfer_learning": {},
            "fine_tuning": fine_tune_history,
            "best_transfer_validation_loss": transfer_val_loss,
            "best_fine_tuning_validation_loss": fine_tune_val_loss,
            "source_transfer_checkpoint": str(transfer_path),
        }
        model.save(models_dir / "brain_tumor_model.keras")
        _write_json(reports_dir / "fine_tuning_history.json", fine_tune_history)
    else:
        model, base_model = build_resnet_model(
            img_size=img_size,
            num_classes=len(EXPECTED_CLASSES),
            learning_rate=1e-3,
        )
        training = train_model_stable(
            model=model,
            base_model=base_model,
            train_generator=train_generator,
            val_generator=validation_generator,
            epochs=epochs,
            fine_tune_epochs=fine_tune_epochs,
            patience=3,
            output_dir=models_dir,
        )
    training_seconds = time.perf_counter() - started_at
    final_model_path = models_dir / "brain_tumor_model.keras"
    if not final_model_path.is_file():
        raise FileNotFoundError(f"Training completed without saving the selected model: {final_model_path}")

    history = _combine_histories(training)
    evaluation = evaluate_model(model, test_generator, history=history, output_dir=output_dir)

    gradcam_outputs = {}
    test_paths = list(test_generator.filepaths)
    test_labels = np.asarray(test_generator.classes)
    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        candidates = np.flatnonzero(test_labels == class_index)
        if not len(candidates):
            raise ValueError(f"No test image available for Grad-CAM class {class_name}.")
        image_path = test_paths[int(candidates[0])]
        gradcam_outputs[class_name] = generate_gradcam_visualization(
            image_path=image_path,
            model=model,
            output_dir=gradcam_dir / class_name,
            class_names=EXPECTED_CLASSES,
        )

    metrics = {
        "status": (
            "evaluation_completed_training_partial"
            if resume_transfer is not None
            else "completed"
        ),
        "dataset": {
            **audit["source"],
            "valid_image_count": audit["valid_image_count"],
            "training_distribution_original": audit["training_distribution"],
            "testing_distribution_original": audit["testing_distribution"],
        },
        "split_distribution": split_summary["split_distribution"],
        "split_total": split_summary["split_total"],
        "excluded_train_exact_test_duplicates": split_summary[
            "training_images_excluded_as_exact_test_duplicates"
        ],
        "cross_split_exact_duplicate_group_count": split_summary[
            "cross_split_exact_duplicate_group_count"
        ],
        "cross_split_near_duplicate_pair_count": split_summary[
            "cross_split_near_duplicate_pair_count"
        ],
        "patient_level_separation_claimed": False,
        "configuration": training_config,
        "training": {
            "workflow_completed": True,
            "full_configured_training_completed": resume_transfer is None,
            "real_training_executed": True,
            "duration_seconds": training_seconds,
            "strategy": training.get("strategy"),
            "selected_stage": training.get("selected_stage", "transfer_learning"),
            "checkpoint": str(final_model_path),
            "transfer_learning_epochs": (
                len(training.get("transfer_learning", {}).get("loss", []))
                if resume_transfer is None
                else None
            ),
            "interrupted_transfer_checkpoint_resume": resume_transfer is not None,
            "original_transfer_epoch_history_available": resume_transfer is None,
            "fine_tuning_epochs": len(training.get("fine_tuning", {}).get("loss", [])),
            "best_transfer_validation_loss": training.get("best_transfer_validation_loss"),
            "best_fine_tuning_validation_loss": training.get("best_fine_tuning_validation_loss"),
        },
        "evaluation": {
            key: evaluation[key]
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy",
                "roc_auc_macro",
                "per_class_roc_auc",
                "pr_auc_macro",
                "per_class_pr_auc",
                "class_names",
                "confusion_matrix",
            )
        },
        "test_images_evaluated": len(evaluation["y_true"]),
        "gradcam_on_real_test_images": gradcam_outputs,
        "historical_baseline": {
            "accuracy": 0.6081,
            "protocol_equivalence_verified": False,
            "direct_comparison_made": False,
        },
    }
    _write_json(output_dir / "experiment_metrics.json", metrics)
    _write_json(reports_dir / "training_history.json", training)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the real Kaggle MRI dataset.")
    parser.add_argument("--data-dir", required=True, help="Original Kaggle dataset root (version 2).")
    parser.add_argument("--prepared-dir", required=True, help="Prepared exact-duplicate-safe split directory.")
    parser.add_argument("--output-dir", default="results", help="Output directory for all experiment artifacts.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--resume-transfer",
        default=None,
        help="Resume from an existing best transfer checkpoint and fine-tune using validation data.",
    )
    args = parser.parse_args()
    result = run_experiment(
        args.data_dir,
        args.prepared_dir,
        args.output_dir,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        seed=args.seed,
        resume_transfer=args.resume_transfer,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
