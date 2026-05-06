from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras
import json

from disease_solutions import get_solution
# Removed gpt_leaf_diagnosis imports as they are no longer needed

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

# You can keep these for reference, but the fallback triggers will be ignored
GPT_FALLBACK_RAW_CONFIDENCE_THRESHOLD = 0.50
MIN_CROP_CONFIDENCE_THRESHOLD = 0.55

PLANT_CLASS_PREFIXES = {
    "Bell Pepper": ("Pepper__bell",),
    "Potato": ("Potato",),
    "Tomato": ("Tomato",),
    "Ginger": ("Ginger",),
    "Maize": ("Maize",),
    "Pigeon_pea": ("Pigeon_pea",),
    "Turmeric": ("Turmeric",),
    "Banana": ("Banana",),
    "Chilli": ("Chilli",)
}

st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="Plant",
    layout="wide",
)

@st.cache_resource
def load_trained_model():
    return keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

def get_model_image_size(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    if height is None or width is None:
        return 224, 224
    return int(height), int(width)

def model_has_input_rescaling(model):
    return any(layer.name == "leaf_rescaling" for layer in model.layers)

def clean_label(class_name: str) -> str:
    return class_name.replace("___", " - ").replace("__", " ").replace("_", " ")

def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower()

def get_allowed_class_indexes(class_names, plant_name: str):
    prefixes = PLANT_CLASS_PREFIXES[plant_name]
    return [
        index
        for index, class_name in enumerate(class_names)
        if class_name.startswith(prefixes)
    ]

def get_best_class_for_selected_plant(predictions, class_names, plant_name: str):
    allowed_indexes = get_allowed_class_indexes(class_names, plant_name)
    if not allowed_indexes:
        raise ValueError(f"No model classes found for plant: {plant_name}")

    best_index = max(allowed_indexes, key=lambda index: predictions[index])
    crop_probability_total = float(np.sum(predictions[allowed_indexes]))
    raw_confidence = float(predictions[best_index])
    crop_confidence = (
        raw_confidence / crop_probability_total
        if crop_probability_total > 0
        else raw_confidence
    )
    return best_index, class_names[best_index], crop_confidence, raw_confidence

def preprocess_image(image: Image.Image, image_size, normalize_input: bool) -> np.ndarray:
    image = image.convert("RGB").resize(image_size)
    image_array = np.asarray(image, dtype=np.float32)
    if normalize_input:
        image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_leaf(image: Image.Image, file_name: str, model, class_names, plant_name: str):
    image_size = get_model_image_size(model)
    normalize_input = not model_has_input_rescaling(model)
    predictions = model.predict(
        preprocess_image(image, image_size, normalize_input),
        verbose=0,
    )[0]
    predicted_index, class_name, confidence, raw_confidence = get_best_class_for_selected_plant(
        predictions,
        class_names,
        plant_name,
    )
    status = "Healthy" if is_healthy(class_name) else "Infected"
    disease_name = "No disease detected" if status == "Healthy" else clean_label(class_name)

    # Added "Source" here directly since we aren't using the fallback function anymore
    return {
        "Image": file_name,
        "Selected Plant": plant_name,
        "Status": status,
        "Disease": disease_name,
        "Predicted Class": clean_label(class_name),
        "Crop Confidence": f"{confidence * 100:.2f}%",
        "Cure / Solution": get_solution(class_name),
        "Source": "Local Model"
    }

def render_result_card(result, image):
    status_color = "#15803d" if result["Status"] == "Healthy" else "#b91c1c"

    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            st.image(image, caption=result["Image"], width="stretch")
        with right:
            st.markdown(
                f"### <span style='color:{status_color}'>{result['Status']}</span>",
                unsafe_allow_html=True,
            )
            st.write(f"**Plant:** {result['Selected Plant']}")
            st.write(f"**Disease:** {result['Disease']}")
            st.write(f"**Predicted class:** {result['Predicted Class']}")
            st.write(f"**Crop confidence:** {result['Crop Confidence']}")
            st.write(f"**Source:** {result['Source']}")
            st.write("**Cure / Solution:**")
            st.info(result["Cure / Solution"])

def main():
    st.title("Local Plant Disease Prediction System")
    st.write(
        "Upload leaf images for local model analysis. "
        "This version runs entirely on your local dataset and model."
    )
    
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
        return

    if not CLASS_NAMES_PATH.exists():
        st.error(f"Class names file not found: {CLASS_NAMES_PATH.name}")
        return

    plant_name = st.selectbox(
        "Plant type",
        options=list(PLANT_CLASS_PREFIXES.keys()),
        index=1,
    )

    uploaded_files = st.file_uploader(
        "Upload leaf images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.warning("Please upload at least one leaf image.")
        return

    model = load_trained_model()
    class_names = load_class_names()
    results = []

    st.subheader("Prediction Results")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        # Call the local prediction directly, skipping the AI fallback function
        result = predict_leaf(
            image,
            uploaded_file.name,
            model,
            class_names,
            plant_name,
        )
        results.append(result)
        render_result_card(result, image)

    if results:
        result_table = pd.DataFrame(results)
        st.subheader("Summary")
        st.dataframe(result_table, width="stretch", hide_index=True)

        st.download_button(
            "Download Prediction Report",
            data=result_table.to_csv(index=False).encode("utf-8"),
            file_name="plant_disease_predictions.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()