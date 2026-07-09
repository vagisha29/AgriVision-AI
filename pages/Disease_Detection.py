# ==================================================
# Import Libraries
# ==================================================
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import zipfile
import requests


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ==================================================
# Load Model
# ==================================================
# ==================================================
# Load Model
# ==================================================

MODEL_FOLDER = "saved_model"
MODEL_ZIP = "crop_disease_model.keras.zip"

MODEL_URL = "https://huggingface.co/vagisha29/AgriVision/resolve/main/crop_disease_model.keras.zip"


def download_model():

    # Model already exists
    if os.path.exists(os.path.join(MODEL_FOLDER, "config.json")):
        return

    with st.spinner("Initializing AgriVision AI..."):

        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=300
        )

        response.raise_for_status()

        with open(MODEL_ZIP, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        # Extract the model
        os.makedirs(MODEL_FOLDER, exist_ok=True)

        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(MODEL_FOLDER)

        os.remove(MODEL_ZIP)
        # --------- DEBUG ---------
        st.write("Current directory:", os.listdir())

        if os.path.exists(MODEL_FOLDER):
            st.write("MODEL_FOLDER exists")
            st.write("Contents:", os.listdir(MODEL_FOLDER))
        else:
            st.write("MODEL_FOLDER does NOT exist")
        # -------------------------


download_model()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FOLDER)
st.write(tf.__version__)
st.write(tf.keras.__version__)

model = load_model()

with open("class_names.json", "r") as file:
    class_names = json.load(file)

IMAGE_SIZE = (224, 224)

# ==================================================
# Page Title
# ==================================================

st.markdown(
    """
    <h1 style='text-align:center; color:green;'>
        🌿 Disease Detection
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Upload a crop leaf image for disease analysis.")

st.divider()

# ==================================================
# Image Upload
# ==================================================

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    detect = st.button(
        "🌿 Detect Disease",
        use_container_width=True
    )

    if detect:

        with st.spinner("Analyzing leaf image..."):

            image = tf.keras.utils.load_img(
                uploaded_file,
                target_size=IMAGE_SIZE
            )

            image = tf.keras.utils.img_to_array(image)

            image = image / 255.0

            image = np.expand_dims(image, axis=0)

            prediction = model.predict(
                image,
                verbose=0
            )

            predicted_class = np.argmax(prediction)

            confidence = np.max(prediction)

            disease = class_names[predicted_class]

            disease = disease.replace("___", " - ")

            disease = disease.replace("__", " - ")

            disease = disease.replace("_", " ")

        st.success("Disease Detection Completed!")

        st.markdown("## 🌿 Disease Detected")

        st.markdown(
            f"""
            <div style="
                background-color:#E8F5E9;
                padding:20px;
                border-radius:10px;
                text-align:center;
                font-size:26px;
                color:green;
                font-weight:bold;">
                {disease}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("### Confidence")

        st.progress(float(confidence))

        st.write(f"**{confidence*100:.2f}%**")
