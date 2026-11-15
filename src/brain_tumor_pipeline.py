from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EXPECTED_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.15
RESNET_PREPROCESSING = tf.keras.applications.resnet50.preprocess_input
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetAudit:
    dataset_root: str
    training_dir: str
    testing_dir: str
    class_names: list[str]
    counts: dict[str, int]
    training_distribution: dict[str, int]
    testing_distribution: dict[str, int]
    empty_dirs: list[str]
    missing_classes: list[str]
    corrupt_images: dict[str, list[str]]
    unsupported_formats: list[str]
    dimensions: dict[str, list[tuple[int, int, int]]]
    duplicate_filenames: dict[str, list[str]]
    duplicate_content_hashes: dict[str, list[str]]
    near_duplicate_pairs: list[list[str]]
    warnings: list[str]


def _normalise_class_name(name: str) -> str:
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "no_tumor": "notumor",
        "notumor": "notumor",
        "no_tumour": "notumor",
        "no_tumour_": "notumor",
        "notumour": "notumor",
    }
    return aliases.get(cleaned, cleaned)


def _safe_dimensions(image_path: Path) -> tuple[int, int, int] | None:
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            return rgb.size[0], rgb.size[1], 3
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _compute_image_hash(image_path: Path) -> str | None:
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb.resize((8, 8), resample=Image.Resampling.LANCZOS), dtype=np.uint8)
            gray = np.mean(arr, axis=2)
            bits = (gray > np.mean(gray)).astype(np.uint8)
            return hashlib.sha256(bits.tobytes()).hexdigest()
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _compute_near_duplicate_pairs(paths: list[Path]) -> list[list[str]]:
    pairs: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for i, left in enumerate(paths):
        for j in range(i + 1, len(paths)):
            right = paths[j]
            if left == right:
                continue
            left_hash = _compute_image_hash(left)
            right_hash = _compute_image_hash(right)
            if left_hash == right_hash:
                pair = sorted([str(left), str(right)])
                key = (pair[0], pair[1])
                if key not in seen:
                    seen.add(key)
                    pairs.append(pair)
    return pairs


def audit_dataset(dataset_root: str | Path) -> DatasetAudit:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    training_dir = root / "Training"
    testing_dir = root / "Testing"
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory missing: {training_dir}")
    if not testing_dir.exists():
        raise FileNotFoundError(f"Testing directory missing: {testing_dir}")

    training_distribution: dict[str, int] = {}
    testing_distribution: dict[str, int] = {}
    counts: dict[str, int] = {}
    corrupt_images: dict[str, list[str]] = {}
    unsupported_formats: list[str] = []
    dimensions: dict[str, list[tuple[int, int, int]]] = {}
    duplicate_filenames: dict[str, list[str]] = {}
    duplicate_content_hashes: dict[str, list[str]] = {}
    warnings: list[str] = []
    all_file_paths: list[Path] = []

    for split_name, split_dir in [("Training", training_dir), ("Testing", testing_dir)]:
        split_distribution: dict[str, int] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = _normalise_class_name(class_dir.name)
            filenames = sorted(p.name for p in class_dir.iterdir() if p.is_file())
            if not filenames:
                warnings.append(f"Empty class folder detected: {class_dir}")
            duplicates = [name for name in filenames if filenames.count(name) > 1]
            if duplicates:
                duplicate_filenames.setdefault(str(class_dir), sorted(set(duplicates)))

            valid_files: list[Path] = []
            corrupt: list[str] = []
            dim_list: list[tuple[int, int, int]] = []
            for file_name in filenames:
                image_path = class_dir / file_name
                if not _is_supported_image(image_path):
                    unsupported_formats.append(str(image_path))
                    continue
                all_file_paths.append(image_path)
                valid_files.append(image_path)
                dim = _safe_dimensions(image_path)
                if dim is None:
                    corrupt.append(file_name)
                    continue
                dim_list.append(dim)
            if corrupt:
                corrupt_images.setdefault(class_name, []).extend(corrupt)
            if dim_list:
                dimensions.setdefault(class_name, []).extend(dim_list)
            split_distribution[class_name] = len(valid_files)
            counts[class_name] = counts.get(class_name, 0) + len(valid_files)
        if split_name == "Training":
            training_distribution.update(split_distribution)
        else:
            testing_distribution.update(split_distribution)

    hash_map: dict[str, list[str]] = defaultdict(list)
    for image_path in all_file_paths:
        hash_value = _compute_image_hash(image_path)
        if hash_value is not None:
            hash_map[hash_value].append(str(image_path))
    duplicate_content_hashes = {key: value for key, value in hash_map.items() if len(value) > 1}

    near_duplicate_pairs = []
    for split_dir in [training_dir, testing_dir]:
        image_paths = [p for p in split_dir.rglob("*") if p.is_file() and _is_supported_image(p)]
        if image_paths:
            near_duplicate_pairs.extend(_compute_near_duplicate_pairs(image_paths))

    all_classes = sorted({key for key in counts})
    class_names = [cls for cls in EXPECTED_CLASSES if cls in all_classes]
    missing_classes = [cls for cls in EXPECTED_CLASSES if cls not in all_classes]

    if not class_names:
        raise ValueError(f"No valid class directories found under {training_dir} or {testing_dir}.")

    if set(class_names) != set(EXPECTED_CLASSES):
        warnings.append(
            "Dataset class names do not match the expected Brain Tumor MRI labels. "
            "Check for folder naming issues, such as 'no_tumor' vs 'notumor'."
        )

    if set(training_distribution.keys()) != set(EXPECTED_CLASSES) or set(testing_distribution.keys()) != set(EXPECTED_CLASSES):
        warnings.append(
            "Training and testing directories should each contain the expected 4 MRI classes. "
            "Missing or renamed classes may compromise model evaluation."
        )

    return DatasetAudit(
        dataset_root=str(root),
        training_dir=str(training_dir),
        testing_dir=str(testing_dir),
        class_names=class_names,
        counts=counts,
        training_distribution=training_distribution,
        testing_distribution=testing_distribution,
        empty_dirs=[warning for warning in warnings if "Empty class folder" in warning],
        missing_classes=missing_classes,
        corrupt_images=corrupt_images,
        unsupported_formats=sorted(set(unsupported_formats)),
        dimensions=dimensions,
        duplicate_filenames=duplicate_filenames,
        duplicate_content_hashes={key: value for key, value in duplicate_content_hashes.items()},
        near_duplicate_pairs=near_duplicate_pairs,
        warnings=warnings,
    )


def _assert_consistent_class_mapping(train_generator: Any, val_generator: Any, test_generator: Any) -> None:
    train_classes = list(train_generator.class_indices.keys())
    val_classes = list(val_generator.class_indices.keys())
    test_classes = list(test_generator.class_indices.keys())

    if train_classes != val_classes or val_classes != test_classes:
        raise AssertionError(
            "Class indexing mismatch across training, validation, and test data. "
            f"Train={train_classes}, Val={val_classes}, Test={test_classes}"
        )
    if set(train_classes) != set(EXPECTED_CLASSES):
        raise AssertionError(f"Unexpected class names: {train_classes}. Expected {EXPECTED_CLASSES}.")


def create_generators(
    dataset_root: str | Path,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    seed: int = 42,
    validation_dir: str | Path | None = None,
) -> tuple[Any, Any, Any]:
    root = Path(dataset_root)
    train_dir = root / "Training"
    test_dir = root / "Testing"

    if validation_dir is None and not 0.0 < validation_split < 1.0:
        raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}.")
    if validation_dir is not None and not Path(validation_dir).is_dir():
        raise FileNotFoundError(f"Validation directory not found: {validation_dir}")

    train_datagen_args = dict(
        preprocessing_function=RESNET_PREPROCESSING,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    if validation_dir is None:
        train_datagen_args["validation_split"] = validation_split
    train_datagen = ImageDataGenerator(**train_datagen_args)
    val_datagen_args = {"preprocessing_function": RESNET_PREPROCESSING}
    if validation_dir is None:
        val_datagen_args["validation_split"] = validation_split
    val_datagen = ImageDataGenerator(**val_datagen_args)
    test_datagen = ImageDataGenerator(preprocessing_function=RESNET_PREPROCESSING)

    train_flow_args = {"subset": "training"} if validation_dir is None else {}
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
        **train_flow_args,
    )
    if validation_dir is None:
        val_flow_dir = train_dir
        val_flow_args = {"subset": "validation"}
    else:
        val_flow_dir = Path(validation_dir)
        val_flow_args = {}
    val_generator = val_datagen.flow_from_directory(
        val_flow_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=seed,
        **val_flow_args,
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=seed,
    )

    if train_generator.samples == 0 or val_generator.samples == 0 or test_generator.samples == 0:
        raise ValueError(
            "A data generator has no samples after splitting. Increase per-class image counts "
            "or lower the validation_split value."
        )

    _assert_consistent_class_mapping(train_generator, val_generator, test_generator)
    if train_datagen.rotation_range != 15:
        raise AssertionError("Training augmentation is not configured correctly.")
    if val_datagen.rotation_range != 0 or test_datagen.rotation_range != 0:
        raise AssertionError("Validation and test generators must not apply random augmentation.")
    return train_generator, val_generator, test_generator


def build_resnet_model(
    img_size: int = DEFAULT_IMG_SIZE,
    num_classes: int = 4,
    learning_rate: float = 1e-3,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    weight_decay: float = 1e-4,
) -> tuple[Model, Model]:
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D(name="head_gap")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        name="classifier_dense",
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.5)(x)
    output_layer = Dense(num_classes, activation="softmax", name="brain_tumor_logits")(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def _build_training_callbacks(output_dir: str | Path | None = None, monitor: str = "val_loss") -> list[Any]:
    output_path = Path(output_dir) if output_dir else None
    callbacks: list[Any] = [
        EarlyStopping(monitor=monitor, patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor=monitor, factor=0.2, patience=2, min_lr=1e-6),
    ]
    if output_path is not None:
        callbacks.append(
            ModelCheckpoint(
                filepath=str(output_path / "transfer_best.keras"),
                monitor=monitor,
                mode="min",
                save_best_only=True,
                save_weights_only=False,
            )
        )
    return callbacks


def _safe_stage_history(history: Any) -> dict[str, list[float]]:
    safe = {}
    for key, values in history.history.items():
        cleaned = []
        for value in values:
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                cleaned.append(float(value))
            else:
                cleaned.append(float("nan"))
        safe[key] = cleaned
    return safe


def train_model_stable(
    model: Model,
    base_model: Model,
    train_generator: Any,
    val_generator: Any,
    epochs: int = 8,
    fine_tune_epochs: int = 6,
    patience: int = 3,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    if resume_from is not None and Path(resume_from).exists():
        model = tf.keras.models.load_model(str(resume_from), compile=False)
        base_model = next(
            (layer for layer in model.layers if isinstance(layer, Model)),
            base_model,
        )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]
    if output_path is not None:
        callbacks.append(
            ModelCheckpoint(
                filepath=str(output_path / "transfer_best.keras"),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=False,
            )
        )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=0,
    )
    stage_a_history = _safe_stage_history(history)

    val_loss = stage_a_history.get("val_loss", [])
    finite_val_loss = [value for value in val_loss if np.isfinite(value)]
    valid_first_stage = (
        len(finite_val_loss) == len(val_loss)
        and len(val_loss) > 1
        and min(finite_val_loss) < finite_val_loss[0]
    )
    if output_path is not None:
        model.save(output_path / "baseline_model.keras")
    transfer_best = output_path / "transfer_best.keras" if output_path is not None else None
    if transfer_best is not None and transfer_best.exists():
        best_transfer_model = tf.keras.models.load_model(str(transfer_best), compile=False)
        model.set_weights(best_transfer_model.get_weights())
        if output_path is not None:
            model.save(output_path / "best_transfer_model.keras")

    if output_path is not None:
        with (output_path / "transfer_learning_history.json").open("w", encoding="utf-8") as handle:
            json.dump(stage_a_history, handle, indent=2)

    if not valid_first_stage:
        if output_path is not None:
            model.save(output_path / "brain_tumor_model.keras")
            with (output_path / "fine_tuning_history.json").open("w", encoding="utf-8") as handle:
                json.dump({}, handle, indent=2)
            with (output_path / "training_history.json").open("w", encoding="utf-8") as handle:
                json.dump({"transfer_learning": stage_a_history, "fine_tuning": {}}, handle, indent=2)
        return {
            "transfer_learning": stage_a_history,
            "fine_tuning": {},
            "strategy": "skipped_fine_tuning_due_to_unstable_first_stage",
            "baseline_checkpoint": str(output_path / "baseline_model.keras") if output_path is not None else None,
            "best_transfer_checkpoint": str(transfer_best) if transfer_best is not None else None,
        }

    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]
    if output_path is not None:
        fine_tune_callbacks.append(
            ModelCheckpoint(
                filepath=str(output_path / "fine_tuned_best.keras"),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=False,
            )
        )

    fine_tune_history = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=val_generator,
        callbacks=fine_tune_callbacks,
        verbose=0,
    )
    stage_b_history = _safe_stage_history(fine_tune_history)

    best_fine_tune_loss = min(
        (value for value in stage_b_history.get("val_loss", []) if np.isfinite(value)),
        default=float("inf"),
    )
    best_transfer_loss = min(finite_val_loss, default=float("inf"))
    selected_stage = "fine_tuning" if best_fine_tune_loss < best_transfer_loss else "transfer_learning"

    if output_path is not None:
        best_ft_path = output_path / "fine_tuned_best.keras"
        if selected_stage == "fine_tuning" and best_ft_path.exists():
            best_ft_model = tf.keras.models.load_model(str(best_ft_path), compile=False)
            model.set_weights(best_ft_model.get_weights())
        elif transfer_best is not None and transfer_best.exists():
            best_transfer_model = tf.keras.models.load_model(str(transfer_best), compile=False)
            model.set_weights(best_transfer_model.get_weights())
        model.save(output_path / "brain_tumor_model.keras")
        with (output_path / "fine_tuning_history.json").open("w", encoding="utf-8") as handle:
            json.dump(stage_b_history, handle, indent=2)
        with (output_path / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "transfer_learning": stage_a_history,
                "fine_tuning": stage_b_history,
            }, handle, indent=2)

    return {
        "transfer_learning": stage_a_history,
        "fine_tuning": stage_b_history,
        "strategy": "stable_two_stage_training",
        "selected_stage": selected_stage,
        "best_transfer_validation_loss": best_transfer_loss,
        "best_fine_tuning_validation_loss": best_fine_tune_loss,
        "baseline_checkpoint": str(output_path / "baseline_model.keras") if output_path is not None else None,
        "best_transfer_checkpoint": str(transfer_best) if transfer_best is not None else None,
        "best_fine_tune_checkpoint": str(output_path / "fine_tuned_best.keras") if output_path is not None else None,
        "resume_from": str(resume_from) if resume_from is not None else None,
    }


def _ensure_prediction_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"Prediction lengths do not match ground truth: {len(y_true)} != {len(y_pred)}")


def _save_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            ax.text(col_idx, row_idx, int(cm[row_idx, col_idx]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_normalized_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    for row_idx in range(cm_norm.shape[0]):
        for col_idx in range(cm_norm.shape[1]):
            ax.text(col_idx, row_idx, f"{cm_norm[row_idx, col_idx]:.2f}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_training_history(history: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for split_key, title in [("accuracy", "Training Accuracy"), ("val_accuracy", "Validation Accuracy")]:
        if split_key in history:
            axes[0].plot(history[split_key], label=split_key)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_title("Accuracy")
    for split_key, title in [("loss", "Training Loss"), ("val_loss", "Validation Loss")]:
        if split_key in history:
            axes[1].plot(history[split_key], label=split_key)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].set_title("Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_roc_curves(y_true_one_hot: np.ndarray, probabilities: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(labels):
        y_true_bin = y_true_one_hot[:, idx]
        y_score = probabilities[:, idx]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        ax.plot(fpr, tpr, label=f"{label}")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_precision_recall_curves(y_true_one_hot: np.ndarray, probabilities: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(labels):
        y_true_bin = y_true_one_hot[:, idx]
        y_score = probabilities[:, idx]
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
        ax.plot(recall, precision, label=f"{label}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_model(model: Model, test_generator: Any, history: dict[str, list[float]] | None = None, output_dir: str | Path = "results") -> dict[str, Any]:
    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    reports_dir = output_path / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_generator.reset()
    probabilities = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())
    _ensure_prediction_length(np.asarray(y_true), np.asarray(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    y_true_one_hot = np.asarray(tf.keras.utils.to_categorical(y_true, num_classes=len(class_names)), dtype=np.float32)
    try:
        roc_auc_macro = roc_auc_score(y_true_one_hot, probabilities, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc_macro = None
    per_class_roc_auc = {}
    per_class_pr_auc = {}
    if probabilities.shape[1] == len(class_names):
        for idx, label in enumerate(class_names):
            try:
                per_class_roc_auc[label] = roc_auc_score(y_true_one_hot[:, idx], probabilities[:, idx])
            except ValueError:
                per_class_roc_auc[label] = None
            try:
                per_class_pr_auc[label] = average_precision_score(
                    y_true_one_hot[:, idx],
                    probabilities[:, idx],
                )
            except ValueError:
                per_class_pr_auc[label] = None
    valid_pr_auc = [value for value in per_class_pr_auc.values() if value is not None]
    pr_auc_macro = float(np.mean(valid_pr_auc)) if valid_pr_auc else None

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    _save_confusion_matrix(cm, class_names, plots_dir / "confusion_matrix.png")
    _save_normalized_confusion_matrix(cm, class_names, plots_dir / "normalized_confusion_matrix.png")
    if history is not None:
        _save_training_history(history, plots_dir / "training_history.png")
    _save_roc_curves(y_true_one_hot, probabilities, class_names, plots_dir / "roc_curves.png")
    _save_precision_recall_curves(y_true_one_hot, probabilities, class_names, plots_dir / "precision_recall_curves.png")

    with (reports_dir / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    summary = {
        "accuracy": float(accuracy),
        "precision": {label: float(val) for label, val in zip(class_names, precision)},
        "recall": {label: float(val) for label, val in zip(class_names, recall)},
        "f1_score": {label: float(val) for label, val in zip(class_names, f1)},
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_acc),
        "roc_auc_macro": None if roc_auc_macro is None else float(roc_auc_macro),
        "per_class_roc_auc": {label: (None if val is None else float(val)) for label, val in per_class_roc_auc.items()},
        "pr_auc_macro": pr_auc_macro,
        "per_class_pr_auc": {label: (None if val is None else float(val)) for label, val in per_class_pr_auc.items()},
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "ground_truth_length": int(len(y_true)),
        "prediction_length": int(len(y_pred)),
    }
    with (reports_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "probabilities": probabilities.tolist(),
        "class_names": class_names,
        "accuracy": float(accuracy),
        "precision": {label: float(val) for label, val in zip(class_names, precision)},
        "recall": {label: float(val) for label, val in zip(class_names, recall)},
        "f1_score": {label: float(val) for label, val in zip(class_names, f1)},
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_acc),
        "roc_auc_macro": None if roc_auc_macro is None else float(roc_auc_macro),
        "per_class_roc_auc": {label: (None if val is None else float(val)) for label, val in per_class_roc_auc.items()},
        "pr_auc_macro": pr_auc_macro,
        "per_class_pr_auc": {label: (None if val is None else float(val)) for label, val in per_class_pr_auc.items()},
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_training_summary(output_dir: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_training(
    data_dir: str | Path,
    output_dir: str | Path,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
) -> dict[str, Any]:
    audit_result = audit_dataset(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_audit_path = output_path / "reports" / "dataset_audit.json"
    dataset_audit_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(audit_result), handle, indent=2)

    train_generator, val_generator, test_generator = create_generators(
        dataset_root=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    model, base_model = build_resnet_model(
        img_size=img_size,
        num_classes=len(EXPECTED_CLASSES),
        learning_rate=1e-3,
    )

    training_history = train_model_stable(
        model=model,
        base_model=base_model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=8,
        fine_tune_epochs=6,
        output_dir=output_path,
    )

    evaluation = evaluate_model(model, test_generator, history=training_history.get("transfer_learning"), output_dir=output_path)
    summary = {
        "dataset_audit": asdict(audit_result),
        "training_history": training_history,
        "evaluation": evaluation,
    }
    save_training_summary(output_path, summary)
    return summary


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, evaluate, and report a ResNet50 MRI classification model.")
    parser.add_argument("--data-dir", type=str, default="brain_tumor_data", help="Root directory containing Training and Testing folders.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory for trained checkpoint files and reports.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Input image size for the model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for data generators.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction of the training split reserved for validation. Must be between 0 and 1.",
    )
    return parser


if __name__ == "__main__":
    args = _build_argument_parser().parse_args()
    run_training(
        args.data_dir,
        args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )
