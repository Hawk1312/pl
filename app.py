from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

from disease_solutions import get_solution
from gpt_leaf_diagnosis import (
    diagnose_leaf_with_gpt,
    get_ai_fallback_status,
    save_new_disease_image,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
NEW_DISEASE_DATASET_DIR = BASE_DIR / "new_disease_dataset"
GPT_FALLBACK_RAW_CONFIDENCE_THRESHOLD = 0.65
MIN_CROP_CONFIDENCE_THRESHOLD = 0.70

PLANT_CLASS_PREFIXES = {
    "Bell Pepper": ("Pepper__bell",),
    "Potato": ("Potato",),
    "Tomato": ("Tomato",),
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
        import json

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

    return {
        "Image": file_name,
        "Selected Plant": plant_name,
        "Status": status,
        "Disease": disease_name,
        "Predicted Class": clean_label(class_name),
        "Crop Confidence": f"{confidence * 100:.2f}%",
        "Cure / Solution": get_solution(class_name),
        "Needs GPT Review": (
            raw_confidence < GPT_FALLBACK_RAW_CONFIDENCE_THRESHOLD
            or confidence < MIN_CROP_CONFIDENCE_THRESHOLD
        ),
    }


def predict_leaf_with_gpt_fallback(image: Image.Image, file_name: str, model, class_names, plant_name: str):
    local_result = predict_leaf(image, file_name, model, class_names, plant_name)

    if not local_result["Needs GPT Review"]:
        local_result["Source"] = "Local model"
        local_result.pop("Needs GPT Review")
        return local_result

    try:
        gpt_result = diagnose_leaf_with_gpt(image)
        saved_path = save_new_disease_image(
            image,
            gpt_result,
            file_name,
            NEW_DISEASE_DATASET_DIR,
        )

        local_result.update(
            {
                "Selected Plant": gpt_result["plant"],
                "Status": gpt_result["status"],
                "Disease": gpt_result["disease"],
                "Predicted Class": "GPT vision diagnosis",
                "Crop Confidence": gpt_result["confidence"],
                "Cure / Solution": gpt_result["solution"],
                "Source": f"{gpt_result.get('provider', 'AI')} fallback",
                "Saved Review Image": str(saved_path.relative_to(BASE_DIR)),
            }
        )
    except Exception as error:
        fallback_status = get_ai_fallback_status()
        review_diagnosis = {
            "plant": plant_name,
            "disease": "Needs_review",
        }
        saved_path = save_new_disease_image(
            image,
            review_diagnosis,
            file_name,
            NEW_DISEASE_DATASET_DIR,
        )
        local_result.update(
            {
                "Status": "Unknown",
                "Disease": "Needs expert/GPT review",
                "Predicted Class": "Untrusted local prediction",
                "Crop Confidence": "Not reliable",
                "Cure / Solution": (
                    "The local model is unsure, so this image should not be treated "
                    f"as healthy or correctly classified. AI fallback could not run: {error}. "
                    "Review this image manually, then add the correct disease label before retraining."
                ),
                "Source": "AI fallback failed",
                "Saved Review Image": str(saved_path.relative_to(BASE_DIR)),
                "AI Debug": (
                    f".env exists={fallback_status['env_file_exists']}; "
                    f"OPENAI_API_KEY={fallback_status['openai_api_key_preview']}; "
                    f"GEMINI_API_KEY={fallback_status['gemini_api_key_preview']}; "
                    f"openai={fallback_status['openai_package_version']}; "
                    f"google-genai={fallback_status['google_genai_package_version']}; "
                    f"python={fallback_status['python_executable']}"
                ),
            }
        )

    local_result.pop("Needs GPT Review")
    return local_result


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
            if "Saved Review Image" in result:
                st.write(f"**Saved review image:** {result['Saved Review Image']}")
            if "AI Debug" in result:
                st.write(f"**AI debug:** {result['AI Debug']}")
            st.write("**Cure / Solution:**")
            st.info(result["Cure / Solution"])


def render_ai_debug_panel(fallback_status: dict):
    with st.expander("AI fallback debug", expanded=True):
        st.write("**Environment files**")
        st.json(
            {
                ".env path": str(fallback_status["env_file"]),
                ".env exists": fallback_status["env_file_exists"],
                ".env.example path": str(fallback_status["env_example_file"]),
                ".env.example exists": fallback_status["env_example_file_exists"],
                "python-dotenv available": fallback_status["dotenv_available"],
                "python-dotenv version": fallback_status["dotenv_package_version"],
            }
        )

        st.write("**API keys and models**")
        st.json(
            {
                "OPENAI_API_KEY": fallback_status["openai_api_key_preview"],
                "GEMINI_API_KEY": fallback_status["gemini_api_key_preview"],
                "OPENAI_LEAF_MODEL": fallback_status["openai_model"],
                "GEMINI_LEAF_MODEL": fallback_status["gemini_model"],
            }
        )

        st.write("**Python packages**")
        st.json(
            {
                "Python executable": fallback_status["python_executable"],
                "Python version": fallback_status["python_version"],
                "Working directory": fallback_status["working_directory"],
                "openai importable": fallback_status["openai_available"],
                "openai version": fallback_status["openai_package_version"],
                "google.genai importable": fallback_status["google_genai_available"],
                "google-genai version": fallback_status["google_genai_package_version"],
                "Platform": fallback_status["platform"],
            }
        )

        if not fallback_status["env_file_exists"]:
            st.error(
                "No .env file found. Rename .env.example to .env or create a new .env "
                "file in the same folder as app.py."
            )


def main():
    st.title("Plant Disease Prediction System")
    st.write(
        "Upload one or more infected leaf images. The model will predict whether "
        "the plant is healthy or infected, show the disease name, confidence "
        "scores, and print the cure or disease-management solution."
    )
    st.caption(
        "If the local model is unsure, AI fallback can diagnose the image and save it "
        "inside new_disease_dataset for later review and retraining."
    )

    fallback_status = get_ai_fallback_status()
    render_ai_debug_panel(fallback_status)
    if not (
        fallback_status["openai_api_key_set"]
        or fallback_status["gemini_api_key_set"]
    ):
        st.warning(
            "AI fallback is not configured. Add OPENAI_API_KEY or GEMINI_API_KEY "
            f"to {fallback_status['env_file'].name}."
        )
    elif fallback_status["gemini_api_key_set"] and not fallback_status["google_genai_available"]:
        st.warning(
            "Gemini fallback is configured, but google-genai is not installed in "
            "the Python environment running Streamlit."
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
        result = predict_leaf_with_gpt_fallback(
            image,
            uploaded_file.name,
            model,
            class_names,
            plant_name,
        )
        results.append(result)
        render_result_card(result, image)

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
