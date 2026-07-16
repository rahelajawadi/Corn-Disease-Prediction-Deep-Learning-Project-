import os

import numpy as np
import streamlit as st
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "model.tflite"
SAMPLES_DIR = "static"
IMAGE_SIZE = (255, 255)

CLASS_NAMES = {
    0: "Blight",
    1: "Common_Rust",
    2: "Grey_Leaf_Spot",
    3: "Healthy",
}

# Short descriptions shown after a prediction.
DISEASE_INFO = {
    "Blight": (
        "Northern corn leaf blight produces long, grayish-green to tan "
        "cigar-shaped lesions on the leaves. It thrives in cool, wet weather "
        "and can reduce yield if it reaches the upper leaves before tasseling. "
        "Manage with resistant hybrids, crop rotation, and fungicides."
    ),
    "Common_Rust": (
        "Common rust shows small, cinnamon-brown, powdery pustules scattered "
        "on both leaf surfaces. It spreads via airborne spores in cool, humid "
        "conditions. Most modern hybrids tolerate it well; severe cases can be "
        "treated with fungicides."
    ),
    "Grey_Leaf_Spot": (
        "Gray leaf spot causes rectangular, gray-to-tan lesions bounded by leaf "
        "veins. Favored by warm, humid weather and no-till residue, it can cause "
        "significant yield loss. Manage with resistant hybrids, rotation, "
        "residue management, and timely fungicides."
    ),
    "Healthy": (
        "No disease symptoms detected. The leaf appears healthy with uniform "
        "green coloration and no lesions or pustules."
    ),
}

st.set_page_config(page_title="Corn Disease Detection", page_icon="\U0001F33D", layout="centered")


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def predict(model, pil_image):
    """Run a prediction with the TFLite interpreter.

    The model already includes Resizing + Rescaling(1/255) layers, so we feed
    raw 0-255 pixel values and only handle channel/size shaping here.
    """
    input_details = model.get_input_details()[0]
    output_details = model.get_output_details()[0]

    image = pil_image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.array(image)
    input_image = image_array.reshape((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)).astype(
        input_details["dtype"]
    )

    model.set_tensor(input_details["index"], input_image)
    model.invoke()
    predictions = model.get_tensor(output_details["index"])[0]
    return predictions


def render_result(pil_image, predictions):
    predicted_index = int(np.argmax(predictions))
    predicted_name = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(pil_image, caption="Input image", use_container_width=True)
    with col_pred:
        st.markdown(f"### Prediction: `{predicted_name.replace('_', ' ')}`")
        st.metric("Confidence", f"{confidence:.2f}%")

    st.markdown("#### Confidence by class")
    scores = {
        CLASS_NAMES[i].replace("_", " "): float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }
    st.bar_chart(scores)

    st.markdown("#### About this result")
    st.info(DISEASE_INFO[predicted_name])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("\U0001F33D Corn Disease Detection")
st.write(
    "Upload a corn-leaf image (or pick a sample) to classify it as "
    "**Blight**, **Common Rust**, **Gray Leaf Spot**, or **Healthy**."
)

model = get_model()

tab_upload, tab_samples = st.tabs(["Upload image", "Try a sample"])

with tab_upload:
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    if uploaded is not None:
        pil_image = Image.open(uploaded)
        with st.spinner("Analyzing..."):
            predictions = predict(model, pil_image)
        render_result(pil_image, predictions)

with tab_samples:
    if os.path.isdir(SAMPLES_DIR):
        sample_files = sorted(
            f
            for f in os.listdir(SAMPLES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    else:
        sample_files = []

    if not sample_files:
        st.warning(f"No sample images found in '{SAMPLES_DIR}/'.")
    else:
        choice = st.selectbox("Pick a sample image", sample_files)
        if st.button("Classify sample"):
            pil_image = Image.open(os.path.join(SAMPLES_DIR, choice))
            with st.spinner("Analyzing..."):
                predictions = predict(model, pil_image)
            render_result(pil_image, predictions)
