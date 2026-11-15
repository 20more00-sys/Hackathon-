from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

from src.real_dataset_audit import CLASSES, _BKTree, _perceptual_hash, _pixel_sha256

SPLITS = ("Training", "Validation", "Testing")


def _readable_images(root: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for class_name in CLASSES:
        for path in sorted((root / split / class_name).iterdir()):
            if not path.is_file():
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    pixel_hash = _pixel_sha256(image)
                    perceptual_hash = _perceptual_hash(image)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                continue
            records.append({
                "source_path": str(path),
                "source_split": split,
                "class": class_name,
                "pixel_hash": pixel_hash,
                "perceptual_hash": perceptual_hash,
            })
    return records


def _link_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def prepare_splits(
    dataset_root: str | Path,
    output_root: str | Path,
    report_dir: str | Path,
    seed: int = 42,
    validation_folds: int = 7,
    near_hash_distance: int = 4,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve()
    if dataset_root == output_root or dataset_root in output_root.parents:
        raise ValueError("Prepared split output must not be inside or equal to the original dataset directory.")
    if validation_folds < 2:
        raise ValueError("validation_folds must be at least 2.")
    if output_root.exists():
        raise FileExistsError(
            f"Prepared split output already exists; choose a new output directory: {output_root}"
        )
    for split in ("Training", "Testing"):
        if not (dataset_root / split).is_dir():
            raise FileNotFoundError(f"Required source split directory missing: {dataset_root / split}")

    original_training = _readable_images(dataset_root, "Training")
    original_testing = _readable_images(dataset_root, "Testing")
    test_pixel_hashes = {record["pixel_hash"] for record in original_testing}
    training_without_test_duplicates = [
        record for record in original_training
        if record["pixel_hash"] not in test_pixel_hashes
    ]
    excluded_test_duplicates = [
        record for record in original_training
        if record["pixel_hash"] in test_pixel_hashes
    ]
    if not training_without_test_duplicates:
        raise ValueError("No training images remain after excluding exact train/test duplicates.")

    labels = [record["class"] for record in training_without_test_duplicates]
    groups = [record["pixel_hash"] for record in training_without_test_duplicates]
    splitter = StratifiedGroupKFold(
        n_splits=validation_folds,
        shuffle=True,
        random_state=seed,
    )
    train_indices, validation_indices = next(
        splitter.split(training_without_test_duplicates, labels, groups)
    )
    train_records = [training_without_test_duplicates[index] for index in train_indices]
    validation_records = [training_without_test_duplicates[index] for index in validation_indices]
    final_records = {
        "Training": train_records,
        "Validation": validation_records,
        "Testing": original_testing,
    }

    for split in SPLITS:
        for class_name in CLASSES:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)

    manifest_path = report_dir / "real_split_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for split, records in final_records.items():
        for record in records:
            source_path = Path(record["source_path"])
            target_path = output_root / split / record["class"] / source_path.name
            _link_image(source_path, target_path)
            manifest_rows.append({
                "prepared_split": split,
                "class": record["class"],
                "source_split": record["source_split"],
                "source_path": str(source_path),
                "prepared_path": str(target_path),
                "decoded_rgb_sha256": record["pixel_hash"],
            })
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    distributions = {
        split: dict(Counter(record["class"] for record in records))
        for split, records in final_records.items()
    }
    split_hashes = {
        split: {record["pixel_hash"] for record in records}
        for split, records in final_records.items()
    }
    cross_split_exact_pairs: dict[str, list[str]] = defaultdict(list)
    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1:]:
            shared = split_hashes[left_split] & split_hashes[right_split]
            for value in shared:
                cross_split_exact_pairs[value].extend((left_split, right_split))

    tree = _BKTree()
    near_pairs: list[dict[str, Any]] = []
    for split, records in final_records.items():
        for record in records:
            for distance, other in tree.query(record["perceptual_hash"], near_hash_distance):
                if record["pixel_hash"] == other["pixel_hash"]:
                    continue
                pair = {
                    "path_a": other["source_path"],
                    "split_a": other["prepared_split"],
                    "class_a": other["class"],
                    "path_b": record["source_path"],
                    "split_b": split,
                    "class_b": record["class"],
                    "hamming_distance": distance,
                }
                near_pairs.append(pair)
            tree.add(
                record["perceptual_hash"],
                {
                    "source_path": record["source_path"],
                    "prepared_split": split,
                    "class": record["class"],
                    "pixel_hash": record["pixel_hash"],
                },
            )

    summary = {
        "dataset_root": str(dataset_root),
        "prepared_root": str(output_root),
        "seed": seed,
        "validation_method": f"StratifiedGroupKFold(n_splits={validation_folds}, first fold, shuffle=True)",
        "grouping_key": "SHA-256 of decoded RGB pixels plus image dimensions",
        "test_policy": "Original Testing split retained without removing or modifying any test files.",
        "training_images_excluded_as_exact_test_duplicates": len(excluded_test_duplicates),
        "excluded_train_duplicate_paths": [record["source_path"] for record in excluded_test_duplicates],
        "split_distribution": distributions,
        "split_total": {split: sum(values.values()) for split, values in distributions.items()},
        "cross_split_exact_duplicate_group_count": len(cross_split_exact_pairs),
        "cross_split_exact_duplicate_groups": dict(cross_split_exact_pairs),
        "near_duplicate_hash_method": (
            "64-bit pHash from grayscale 32x32 low-frequency DCT; "
            f"candidate pairs at Hamming distance <= {near_hash_distance}. "
            "These candidates may be false positives and need visual review."
        ),
        "near_duplicate_pair_count": len(near_pairs),
        "cross_split_near_duplicate_pair_count": sum(
            pair["split_a"] != pair["split_b"] for pair in near_pairs
        ),
        "near_duplicate_pairs": near_pairs,
        "patient_identifiers_available": False,
        "patient_level_separation_claimed": False,
        "manifest_path": str(manifest_path),
    }
    with (report_dir / "real_split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create exact-duplicate-safe real dataset splits.")
    parser.add_argument("--data-dir", required=True, help="Original dataset root with Training/ and Testing/.")
    parser.add_argument("--output-dir", required=True, help="Separate directory for symlinked prepared splits.")
    parser.add_argument("--report-dir", default="results/reports", help="Directory for split manifest and report.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-folds", type=int, default=7)
    args = parser.parse_args()
    summary = prepare_splits(
        args.data_dir,
        args.output_dir,
        args.report_dir,
        seed=args.seed,
        validation_folds=args.validation_folds,
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {
        "excluded_train_duplicate_paths",
        "cross_split_exact_duplicate_groups",
        "near_duplicate_pairs",
    }}, indent=2))


if __name__ == "__main__":
    main()
