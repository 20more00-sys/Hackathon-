from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.brain_tumor_pipeline import (
    EXPECTED_CLASSES,
    audit_dataset,
    build_resnet_model,
    create_generators,
    evaluate_model,
)
from src.gradcam import generate_gradcam_visualization
from src.prepare_real_splits import prepare_splits
from src.predict import predict_image, validate_image


def _make_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = color[0]
    image[:, :, 1] = color[1]
    image[:, :, 2] = color[2]
    Image.fromarray(image).save(path)


def _write_dataset(root: Path, train_count: int = 6, test_count: int = 4) -> None:
    for split, count in [("Training", train_count), ("Testing", test_count)]:
        for label in EXPECTED_CLASSES:
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for idx in range(count):
                _make_rgb_image(folder / f"sample_{idx}.png", (idx * 20 + 5, idx * 30 + 10, idx * 40 + 15))


def test_audit_dataset(tmp_path):
    dataset_root = tmp_path / "brain_tumor_data"
    _write_dataset(dataset_root)
    audit = audit_dataset(dataset_root)
    assert set(audit.class_names) == set(EXPECTED_CLASSES)
    assert audit.missing_classes == []
    assert sum(audit.counts.values()) > 0
    assert audit.training_distribution["notumor"] > 0


def test_missing_classes_are_reported(tmp_path):
    dataset_root = tmp_path / "brain_tumor_data"
    for split in ["Training", "Testing"]:
        for label in EXPECTED_CLASSES[:-1]:
            (dataset_root / split / label).mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(dataset_root)
    assert "pituitary" in audit.missing_classes


def test_corrupted_images_are_reported(tmp_path):
    dataset_root = tmp_path / "brain_tumor_data"
    _write_dataset(dataset_root)
    bad_path = dataset_root / "Training" / "glioma" / "bad.png"
    bad_path.write_bytes(b"not-an-image")
    audit = audit_dataset(dataset_root)
    assert "bad.png" in audit.corrupt_images.get("glioma", [])


def test_generators_and_model(tmp_path):
    dataset_root = tmp_path / "brain_tumor_data"
    _write_dataset(dataset_root)

    train_gen, val_gen, test_gen = create_generators(dataset_root, img_size=64, batch_size=8, validation_split=0.25)
    assert list(train_gen.class_indices.keys()) == list(val_gen.class_indices.keys()) == list(test_gen.class_indices.keys())
    assert set(train_gen.class_indices.keys()) == set(EXPECTED_CLASSES)
    assert train_gen.image_data_generator.rotation_range == 15
    assert val_gen.image_data_generator.rotation_range == 0
    assert test_gen.image_data_generator.rotation_range == 0

    model, base_model = build_resnet_model(img_size=64, num_classes=4, learning_rate=1e-3)
    assert model.output_shape[-1] == 4
    assert base_model.output_shape[-1] == 2048


def test_generators_accept_explicit_validation_directory(tmp_path):
    dataset_root = tmp_path / "dataset"
    validation_root = tmp_path / "validation"
    _write_dataset(dataset_root)
    for class_index, label in enumerate(EXPECTED_CLASSES):
        folder = validation_root / label
        folder.mkdir(parents=True)
        _make_rgb_image(folder / "validation.png", (class_index + 20, class_index + 40, class_index + 60))

    train_gen, val_gen, test_gen = create_generators(
        dataset_root,
        img_size=64,
        batch_size=8,
        validation_dir=validation_root,
    )
    assert train_gen.samples == 24
    assert val_gen.samples == 4
    assert test_gen.samples == 16
    assert val_gen.image_data_generator.rotation_range == 0


def test_prepared_splits_exclude_exact_test_duplicates(tmp_path):
    dataset_root = tmp_path / "source"
    for split, count in (("Training", 8), ("Testing", 2)):
        for class_index, label in enumerate(EXPECTED_CLASSES):
            folder = dataset_root / split / label
            folder.mkdir(parents=True)
            for image_index in range(count):
                array = np.zeros((64, 64, 3), dtype=np.uint8)
                array[:, :, 0] = 10 + class_index * 40
                array[:, :, 1] = 20 + image_index * 20
                array[:, :, 2] = 30 + class_index * 10 + image_index
                if split == "Testing" and image_index > 0:
                    array[:, :, 2] += 100
                Image.fromarray(array).save(folder / f"{image_index}.png")
    for class_index, label in enumerate(EXPECTED_CLASSES):
        (dataset_root / "Testing" / label / "0.png").write_bytes(
            (dataset_root / "Training" / label / "0.png").read_bytes()
        )

    result = prepare_splits(
        dataset_root,
        tmp_path / "prepared",
        tmp_path / "reports",
        seed=42,
        validation_folds=4,
    )
    assert result["training_images_excluded_as_exact_test_duplicates"] == 4
    assert result["cross_split_exact_duplicate_group_count"] == 0
    assert result["split_total"]["Testing"] == 8
    assert (tmp_path / "reports" / "real_split_manifest.csv").is_file()


def test_probability_sums_to_one(tmp_path):
    dataset_root = tmp_path / "brain_tumor_data"
    _write_dataset(dataset_root)
    train_gen, val_gen, test_gen = create_generators(dataset_root, img_size=64, batch_size=8, validation_split=0.25)
    model, _ = build_resnet_model(img_size=64, num_classes=4, learning_rate=1e-3)
    batch = next(test_gen)[0]
    probabilities = model.predict(batch, verbose=0)
    assert probabilities.shape[1] == len(EXPECTED_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_evaluation_outputs(tmp_path):
    dataset_root = tmp_path / "brain_tumor_data"
    _write_dataset(dataset_root)
    _, _, test_gen = create_generators(dataset_root, img_size=64, batch_size=8, validation_split=0.25)
    model, _ = build_resnet_model(img_size=64, num_classes=4, learning_rate=1e-3)
    result = evaluate_model(model, test_gen, history={"loss": [1.0, 0.5], "val_loss": [1.2, 0.8], "accuracy": [0.5, 0.7], "val_accuracy": [0.4, 0.6]}, output_dir=tmp_path / "results")
    assert result["accuracy"] >= 0.0
    assert len(result["class_names"]) == 4
    assert len(result["confusion_matrix"]) == 4
    assert all(len(row) == 4 for row in result["confusion_matrix"])
    assert (tmp_path / "results" / "plots" / "confusion_matrix.png").exists()
    assert (tmp_path / "results" / "reports" / "evaluation_summary.json").exists()


def test_model_save_and_load(tmp_path):
    model, _ = build_resnet_model(img_size=64, num_classes=4)
    model_path = tmp_path / "model.keras"
    model.save(model_path)
    loaded = tf.keras.models.load_model(model_path, compile=False)
    assert loaded.output_shape[-1] == 4


def test_gradcam_output(tmp_path):
    dataset_root = tmp_path / "brain_tumor_data"
    _write_dataset(dataset_root)
    _, _, test_gen = create_generators(dataset_root, img_size=64, batch_size=8, validation_split=0.25)
    model, base_model = build_resnet_model(img_size=64, num_classes=4, learning_rate=1e-3)
    sample_path = dataset_root / "Testing" / "glioma" / "sample_0.png"
    result = generate_gradcam_visualization(sample_path, model=model, base_model=base_model, output_dir=tmp_path / "gradcam")
    assert Path(result["heatmap_path"]).exists()
    assert Path(result["overlay_path"]).exists()
    assert result["predicted_class"] in EXPECTED_CLASSES


def test_predict_invalid_input(tmp_path):
    missing_image = tmp_path / "missing.png"
    model_path = tmp_path / "missing_model.keras"
    with pytest.raises(FileNotFoundError):
        predict_image(model_path, missing_image)

    bad_img_path = tmp_path / "bad.png"
    bad_img_path.write_bytes(b"not an image")
    model, _ = build_resnet_model(img_size=64, num_classes=4)
    valid_model_path = tmp_path / "valid_model.keras"
    model.save(valid_model_path)
    with pytest.raises(ValueError):
        predict_image(valid_model_path, bad_img_path)

    valid_img = tmp_path / "good.png"
    _make_rgb_image(valid_img, (10, 20, 30))
    array = validate_image(valid_img)
    assert array.shape[2] == 3


def test_json_summary_serializable(tmp_path):
    dataset_root = tmp_path / "brain_tumor_data"
    _write_dataset(dataset_root)
    summary = {"dataset": audit_dataset(dataset_root).counts, "expected": EXPECTED_CLASSES}
    payload = json.dumps(summary)
    assert isinstance(payload, str)
    assert len(summary["expected"]) == 4
