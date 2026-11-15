from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from src.brain_tumor_pipeline import EXPECTED_CLASSES, RESNET_PREPROCESSING


class GradCAM:
    def __init__(self, model: tf.keras.Model, base_model: tf.keras.Model | None = None, last_conv_layer_name: str = "conv5_block3_out"):
        self.model = model
        self.base_model = base_model or self._find_nested_base_model()
        self.last_conv_layer_name = last_conv_layer_name
        self._last_conv_layer = self._resolve_last_conv_layer()
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self._last_conv_layer.output, self.model.output],
        )

    def _find_nested_base_model(self) -> tf.keras.Model:
        for layer in self.model.layers:
            if hasattr(layer, "get_layer") and hasattr(layer, "layers"):
                try:
                    layer.get_layer(self.last_conv_layer_name)
                    return layer
                except ValueError:
                    pass
        return self.model

    def _resolve_last_conv_layer(self):
        candidate = None
        if hasattr(self.base_model, "get_layer"):
            try:
                candidate = self.base_model.get_layer(self.last_conv_layer_name)
            except ValueError:
                candidate = None
        if candidate is not None:
            return candidate
        for layer in reversed(self.base_model.layers):
            if hasattr(layer, "output") and isinstance(layer.output, tf.Tensor):
                if layer.name.startswith("conv"):
                    return layer
        raise ValueError("Could not resolve the final convolutional layer for Grad-CAM.")

    def compute_heatmap(self, image_array: np.ndarray, class_index: int | None = None) -> np.ndarray:
        image_batch = np.expand_dims(image_array.astype(np.float32), axis=0)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_batch)
            if class_index is None:
                class_index = int(np.argmax(predictions[0]))
            class_score = predictions[:, class_index]
        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        heatmap = tf.reduce_sum(pooled_grads[:, None, None] * conv_outputs, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap[0].numpy()

    def overlay_heatmap(self, image_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        image = image_array.astype(np.float32)
        heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]), resample=Image.Resampling.BILINEAR))
        heatmap_colormap = plt.get_cmap("jet")(heatmap_resized / 255.0)
        heatmap_colormap = heatmap_colormap[:, :, :3]
        overlay = image * 0.6 + heatmap_colormap * 255.0 * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay


def generate_gradcam_visualization(
    image_path: str | Path,
    model: tf.keras.Model,
    base_model: tf.keras.Model | None = None,
    output_dir: str | Path | None = None,
    class_names: list[str] | None = None,
) -> dict[str, str | int | float | list[float]]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    class_names = class_names or EXPECTED_CLASSES
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image)
    model_height, model_width = model.input_shape[1:3]
    model_image = image.resize((model_width, model_height), resample=Image.Resampling.BILINEAR)
    resized = np.asarray(model_image, dtype=np.float32)
    input_tensor = RESNET_PREPROCESSING(np.expand_dims(resized, axis=0))
    probabilities = model.predict(input_tensor, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    conf = float(probabilities[predicted_index])

    gradcam = GradCAM(model=model, base_model=base_model, last_conv_layer_name="conv5_block3_out")
    heatmap = gradcam.compute_heatmap(resized, class_index=predicted_index)
    overlay = gradcam.overlay_heatmap(image_array, heatmap)

    output_dir = Path(output_dir) if output_dir is not None else Path("results") / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / f"{image_path.stem}_original.png"
    heatmap_path = output_dir / f"{image_path.stem}_heatmap.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"

    Image.fromarray(image_array.astype(np.uint8)).save(original_path)
    Image.fromarray((heatmap * 255).astype(np.uint8)).save(heatmap_path)
    Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": conf,
        "original_path": str(original_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "probabilities": [float(v) for v in probabilities],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for a brain MRI image.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained Keras model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the MRI image.")
    parser.add_argument("--output-dir", type=str, default="results/gradcam", help="Directory for Grad-CAM outputs.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    model = tf.keras.models.load_model(args.model, compile=False)
    base_model = None
    if hasattr(model.layers[0], "get_layer"):
        base_model = model.layers[0]
    result = generate_gradcam_visualization(args.image, model, base_model=base_model, output_dir=args.output_dir)
    print(result)
