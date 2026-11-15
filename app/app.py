from __future__ import annotations

import tempfile
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Streamlit is not installed. Install it with: pip install streamlit") from exc

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.brain_tumor_pipeline import EXPECTED_CLASSES, RESNET_PREPROCESSING
from src.gradcam import generate_gradcam_visualization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path("results") / "models" / "brain_tumor_model.keras"


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def predict_and_rank(model, image_array: np.ndarray) -> dict[str, object]:
    height, width = model.input_shape[1:3]
    if height is None or width is None:
        raise ValueError("The loaded model must have fixed image dimensions.")
    resized = Image.fromarray(image_array.astype(np.uint8)).resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )
    prepared = RESNET_PREPROCESSING(
        np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    )
    probabilities = np.asarray(model.predict(prepared, verbose=0)[0], dtype=np.float64)
    if probabilities.shape != (len(EXPECTED_CLASSES),) or not np.isfinite(probabilities).all():
        raise ValueError("The model returned invalid class probabilities.")
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_class": EXPECTED_CLASSES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            label: float(value)
            for label, value in zip(EXPECTED_CLASSES, probabilities)
        },
    }


st.set_page_config(page_title="Brain Tumor MRI Classifier")
st.title("Brain Tumor MRI Classifier")
st.warning(
    "Educational prototype only — not a medical device or diagnostic tool. "
    "Predictions and Grad-CAM images are not clinically validated and must not "
    "be used for screening, diagnosis, treatment, or patient-care decisions."
)


def _main() -> None:
    model_path_input = st.text_input(
        "Model path (relative to the project root or absolute)",
        value=str(DEFAULT_MODEL_PATH),
    )
    requested_path = Path(model_path_input).expanduser()
    model_path = requested_path if requested_path.is_absolute() else PROJECT_ROOT / requested_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        st.error(f"Model file not found: {model_path}")
        st.info(
            "The repository checkpoint is `results/models/brain_tumor_model.keras`. "
            "Run training or enter the path to a compatible four-class Keras model."
        )
        st.stop()

    try:
        model = load_model(str(model_path))
    except (OSError, ValueError, ImportError, tf.errors.OpError) as exc:
        st.error(f"Could not load the model at {model_path}: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an MRI image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )
    if uploaded_file is None:
        return

    try:
        with Image.open(uploaded_file) as uploaded_image:
            image = uploaded_image.convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        return

    st.image(array, caption="Uploaded MRI", use_container_width=True)
    try:
        result = predict_and_rank(model, array)
    except (ValueError, tf.errors.OpError) as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader("Experimental model output")
    st.metric("Predicted class", str(result["predicted_class"]))
    st.metric("Top-class score (not calibrated)", f"{result['confidence']:.1%}")
    probabilities = result["probabilities"]
    ranked_scores = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    st.dataframe(
        [
            {"Class": label, "Probability score": f"{score:.2%}"}
            for label, score in ranked_scores
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.bar_chart(
        {label: score for label, score in ranked_scores},
        y_label="Probability score",
    )

    try:
        with tempfile.TemporaryDirectory(prefix="brain-tumor-gradcam-") as temp_dir:
            temp_root = Path(temp_dir)
            image_path = temp_root / "uploaded.png"
            image.save(image_path)
            gradcam_result = generate_gradcam_visualization(
                image_path=image_path,
                model=model,
                output_dir=temp_root / "gradcam",
                class_names=EXPECTED_CLASSES,
            )
            with Image.open(gradcam_result["overlay_path"]) as overlay:
                st.image(
                    overlay.convert("RGB"),
                    caption="Grad-CAM overlay (qualitative only)",
                    use_container_width=True,
                )
    except (ValueError, KeyError, AttributeError, OSError, TypeError, tf.errors.OpError) as exc:
        st.info(
            "Grad-CAM is unavailable for this model or image; the prediction above "
            f"was still produced. Details: {exc}"
        )


_main()
