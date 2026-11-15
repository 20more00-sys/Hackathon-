from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.brain_tumor_pipeline import EXPECTED_CLASSES, RESNET_PREPROCESSING
from src.gradcam import GradCAM, generate_gradcam_visualization


def validate_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        image = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid image file: {path}") from exc
    return np.asarray(image, dtype=np.uint8)


def predict_image(model_path: str | Path, image_path: str | Path, gradcam_dir: str | Path | None = None) -> dict[str, object]:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = tf.keras.models.load_model(str(model_file), compile=False)
    image_array = validate_image(image_path)
    model_height, model_width = model.input_shape[1:3]
    resized = Image.fromarray(image_array).resize(
        (model_width, model_height),
        resample=Image.Resampling.BILINEAR,
    )
    image_batch = RESNET_PREPROCESSING(
        np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    )
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    prediction = {
        "predicted_class": EXPECTED_CLASSES[predicted_index],
        "confidence": confidence,
        "probabilities": {label: float(prob) for label, prob in zip(EXPECTED_CLASSES, probabilities)},
        "experimental_note": "Predictions are experimental and are not medical diagnoses.",
    }

    if gradcam_dir is not None:
        base_model = None
        if hasattr(model.layers[0], "get_layer"):
            base_model = model.layers[0]
        result = generate_gradcam_visualization(image_path, model, base_model=base_model, output_dir=gradcam_dir, class_names=EXPECTED_CLASSES)
        prediction["gradcam"] = {
            "original_path": result["original_path"],
            "heatmap_path": result["heatmap_path"],
            "overlay_path": result["overlay_path"],
        }
    return prediction


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference on a single MRI image with a saved ResNet50 model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved Keras model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the MRI image file.")
    parser.add_argument("--gradcam-dir", type=str, default=None, help="Optional directory for Grad-CAM visualizations.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    try:
        result = predict_image(args.model, args.image, gradcam_dir=args.gradcam_dir)
        print(json.dumps(result, indent=2))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
