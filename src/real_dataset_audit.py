from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _perceptual_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(grayscale, dtype=np.float32), norm="ortho")[:8, :8]
    low_frequency = coefficients.flatten()[1:]
    median = float(np.median(low_frequency))
    result = 0
    for value in low_frequency:
        result = (result << 1) | int(value > median)
    return result


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class _BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, list[dict[str, Any]], dict[int, Any]] | None = None

    def add(self, value: int, item: dict[str, Any]) -> None:
        if self.root is None:
            self.root = (value, [item], {})
            return
        node = self.root
        while True:
            node_value, entries, children = node
            distance = _hamming_distance(value, node_value)
            if distance == 0:
                entries.append(item)
                return
            if distance not in children:
                children[distance] = (value, [item], {})
                return
            node = children[distance]

    def query(self, value: int, radius: int) -> list[tuple[int, dict[str, Any]]]:
        if self.root is None:
            return []
        found: list[tuple[int, dict[str, Any]]] = []
        pending = [self.root]
        while pending:
            node_value, entries, children = pending.pop()
            distance = _hamming_distance(value, node_value)
            if distance <= radius:
                found.extend((distance, entry) for entry in entries)
            lower = distance - radius
            upper = distance + radius
            pending.extend(child for edge, child in children.items() if lower <= edge <= upper)
        return found


def audit_dataset(dataset_root: str | Path, output_dir: str | Path, near_hash_distance: int = 4) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    output_path = Path(output_dir)
    reports_dir = output_path / "reports"
    plots_dir = output_path / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    distribution: dict[str, dict[str, int]] = {}
    dimensions: dict[str, dict[str, int]] = defaultdict(Counter)
    modes: dict[str, dict[str, int]] = defaultdict(Counter)
    exact_bytes: dict[str, list[str]] = defaultdict(list)
    exact_pixels: dict[str, list[str]] = defaultdict(list)
    path_to_pixel_hash: dict[str, str] = {}
    path_to_split: dict[str, str] = {}
    corrupt_images: list[str] = []
    unsupported_files: list[str] = []
    empty_dirs: list[str] = []
    image_records: list[dict[str, Any]] = []
    file_count = 0

    for split in ("Training", "Testing"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Required split directory missing: {split_root}")
        distribution[split] = {}
        for class_name in CLASSES:
            class_root = split_root / class_name
            if not class_root.is_dir():
                distribution[split][class_name] = 0
                empty_dirs.append(str(class_root))
                continue
            files = sorted(path for path in class_root.iterdir() if path.is_file())
            valid_count = 0
            if not files:
                empty_dirs.append(str(class_root))
            for path in files:
                file_count += 1
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    unsupported_files.append(str(path))
                    continue
                try:
                    with Image.open(path) as image:
                        image.load()
                        source_mode = image.mode
                        width, height = image.size
                        byte_hash = _sha256_file(path)
                        pixel_hash = _pixel_sha256(image)
                        perceptual_hash = _perceptual_hash(image)
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                    corrupt_images.append(str(path))
                    continue

                valid_count += 1
                dimension_key = f"{width}x{height}"
                dimensions[split][dimension_key] += 1
                modes[split][source_mode] += 1
                exact_bytes[byte_hash].append(str(path))
                exact_pixels[pixel_hash].append(str(path))
                path_to_pixel_hash[str(path)] = pixel_hash
                path_to_split[str(path)] = split
                image_records.append(
                    {
                        "path": str(path),
                        "split": split,
                        "class": class_name,
                        "perceptual_hash": perceptual_hash,
                    }
                )
            distribution[split][class_name] = valid_count

    byte_duplicate_groups = [paths for paths in exact_bytes.values() if len(paths) > 1]
    pixel_duplicate_groups = [paths for paths in exact_pixels.values() if len(paths) > 1]
    cross_split_exact_byte_groups = [
        paths for paths in byte_duplicate_groups
        if len({path_to_split[path] for path in paths}) > 1
    ]
    cross_split_exact_pixel_groups = [
        paths for paths in pixel_duplicate_groups
        if len({path_to_split[path] for path in paths}) > 1
    ]

    tree = _BKTree()
    near_duplicate_pairs: list[dict[str, Any]] = []
    cross_split_near_duplicate_pairs: list[dict[str, Any]] = []
    for record in image_records:
        perceptual_hash = record["perceptual_hash"]
        for distance, other in tree.query(perceptual_hash, near_hash_distance):
            if record["path"] == other["path"]:
                continue
            if path_to_pixel_hash[record["path"]] == path_to_pixel_hash[other["path"]]:
                continue
            pair = {
                "path_a": other["path"],
                "path_b": record["path"],
                "hamming_distance": distance,
                "split_a": other["split"],
                "split_b": record["split"],
                "class_a": other["class"],
                "class_b": record["class"],
            }
            near_duplicate_pairs.append(pair)
            if other["split"] != record["split"]:
                cross_split_near_duplicate_pairs.append(pair)
        tree.add(perceptual_hash, record)

    summary: dict[str, Any] = {
        "dataset_root": str(root),
        "source": {
            "name": "Brain Tumor MRI Dataset",
            "owner": "Masoud Nickparvar",
            "kaggle_identifier": "masoudnickparvar/brain-tumor-mri-dataset",
            "kaggle_dataset_id": "1608934",
            "version": 2,
            "license": "Attribution 4.0 International (CC BY 4.0)",
            "url": "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        },
        "file_count_in_class_directories": file_count,
        "valid_image_count": len(image_records),
        "training_distribution": distribution["Training"],
        "testing_distribution": distribution["Testing"],
        "total_per_class": {
            name: distribution["Training"][name] + distribution["Testing"][name]
            for name in CLASSES
        },
        "corrupt_images": corrupt_images,
        "unsupported_files": unsupported_files,
        "empty_class_directories": empty_dirs,
        "dimensions_by_split": {split: dict(values) for split, values in dimensions.items()},
        "source_color_modes_by_split": {split: dict(values) for split, values in modes.items()},
        "exact_duplicate_byte_groups": byte_duplicate_groups,
        "exact_duplicate_decoded_rgb_groups": pixel_duplicate_groups,
        "cross_split_exact_duplicate_byte_groups": cross_split_exact_byte_groups,
        "cross_split_exact_duplicate_decoded_rgb_groups": cross_split_exact_pixel_groups,
        "near_duplicate_hash_method": (
            "64-bit pHash from the low-frequency 8x8 DCT of a grayscale 32x32 image; "
            f"pairs at Hamming distance <= {near_hash_distance}. Candidate pairs require visual review."
        ),
        "near_duplicate_pair_count": len(near_duplicate_pairs),
        "near_duplicate_pairs": near_duplicate_pairs,
        "cross_split_near_duplicate_pair_count": len(cross_split_near_duplicate_pairs),
        "cross_split_near_duplicate_pairs": cross_split_near_duplicate_pairs,
        "patient_identifiers_available": False,
        "patient_level_separation_claimed": False,
    }

    with (reports_dir / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CLASSES))
    width = 0.36
    ax.bar(x - width / 2, [distribution["Training"][name] for name in CLASSES], width, label="Training")
    ax.bar(x + width / 2, [distribution["Testing"][name] for name in CLASSES], width, label="Testing")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Valid image count")
    ax.set_title("Brain Tumor MRI Dataset Class Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "dataset_class_distribution.png", dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the downloaded four-class Brain Tumor MRI dataset.")
    parser.add_argument("--data-dir", required=True, help="Root containing Training/ and Testing/.")
    parser.add_argument("--output-dir", default="results", help="Directory for JSON reports and plots.")
    parser.add_argument("--near-hash-distance", type=int, default=4, help="Maximum pHash Hamming distance.")
    args = parser.parse_args()
    result = audit_dataset(args.data_dir, args.output_dir, args.near_hash_distance)
    print(json.dumps({key: value for key, value in result.items() if key not in {
        "exact_duplicate_byte_groups",
        "exact_duplicate_decoded_rgb_groups",
        "near_duplicate_pairs",
        "cross_split_near_duplicate_pairs",
    }}, indent=2))


if __name__ == "__main__":
    main()
