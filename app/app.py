from __future__ import annotations

from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Streamlit is not installed. Install it with: pip install streamlit") from exc

import numpy as np
import tensorflow as tf
from PIL import Image

from src.brain_tumor_pipeline import EXPECTED_CLASSES, RESNET_PREPROCESSING
from src.gradcam import generate_gradcam_visualization


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def predict_and_rank(model, image_array: np.ndarray) -> dict[str, object]:
    prepared = RESNET_PREPROCESSING(np.expand_dims(image_array.astype(np.float32), axis=0))
    probabilities = model.predict(prepared, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    payload = {
        "predicted_class": EXPECTED_CLASSES[predicted_index],
        "confidence": confidence,
        "probabilities": {label: float(value) for label, value in zip(EXPECTED_CLASSES, probabilities)},
    }
    return payload


st.set_page_config(page_title="Brain Tumor MRI Classifier")
st.title("Brain Tumor MRI Classifier")
st.caption("Experimental tool only. Predictions are not medical diagnoses.")

def _main():
    default_model = Path("artifacts/brain_tumor_model.keras")
    model_path = st.text_input("Model path", value=str(default_model))

    if not Path(model_path).exists():
        st.warning("No model file found yet. Train a model first, then provide the saved checkpoint path above.")
        st.stop()

    model = load_model(model_path)
    uploaded_file = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        array = np.asarray(image)
        st.image(array, caption="Uploaded MRI", use_container_width=True)
        result = predict_and_rank(model, array)
        st.subheader("Prediction")
        st.write(f"Predicted class: {result['predicted_class']}")
        st.write(f"Confidence: {result['confidence']:.2%}")
        st.bar_chart(result["probabilities"])

        try:
            temp_path = Path("/tmp") / uploaded_file.name
            Image.fromarray(array).save(temp_path)
            base_model = next((layer for layer in model.layers if hasattr(layer, "get_layer")), None)
            gradcam_result = generate_gradcam_visualization(
                image_path=temp_path,
                model=model,
                base_model=base_model,
                output_dir=Path("results/gradcam_app"),
                class_names=EXPECTED_CLASSES,
            )
            st.image(np.array(Image.open(gradcam_result["overlay_path"])), caption="Grad-CAM overlay", use_container_width=True)
        except Exception:
            st.info("Grad-CAM was not generated for this upload. The model is loaded successfully, but the visualization step is optional.")


_main()
