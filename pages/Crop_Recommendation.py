# ==================================================
# Import Libraries
# ==================================================

import streamlit as st
import pickle
import pandas as pd
# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)
# ==================================================
# Load Model
# ==================================================

@st.cache_resource
def load_models():

    with open("model1.pkl", "rb") as file:
        ml_model = pickle.load(file)

    with open("le_soil1.pkl", "rb") as file:
        le_soil = pickle.load(file)

    with open("le_crop1.pkl", "rb") as file:
        le_crop = pickle.load(file)

    return ml_model, le_soil, le_crop


ml_model, le_soil, le_crop = load_models()
    # ==================================================
# Page Title
# ==================================================

st.markdown(
    """
    <h1 style='text-align:center; color:green;'>
    🌾 Crop Recommendation
    </h1>
    """,
    unsafe_allow_html=True
)

st.write(
    "Please enter the following soil and environmental parameters."
)
st.info(
    "Please enter realistic soil and environmental values for accurate crop recommendation."
)

st.divider()

# ==================================================
# Input Fields
# ==================================================

col1, col2 = st.columns(2)

with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        format="%.1f",
        key="temperature"
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=200.0,
        step=1.0,
        format="%.1f",
        key="rainfall"
    )

    nitrogen = st.number_input(
        "Nitrogen (kg/ha)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=1.0,
        format="%.1f",
        key="nitrogen"
    )

    potassium = st.number_input(
        "Potassium (kg/ha)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=1.0,
        format="%.1f",
        key="potassium"
    )

with col2:

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=0.1,
        format="%.1f",
        key="humidity"
    )

    ph = st.number_input(
        "pH (0-14)",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1,
        format="%.1f",
        key="ph"
    )

    phosphorous = st.number_input(
        "Phosphorous (kg/ha)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=1.0,
        format="%.1f",
        key="phosphorous"
    )

    carbon = st.number_input(
        "Carbon (%)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.1f",
        key="carbon"
    )
soil_options = [
    s for s in le_soil.classes_
    if str(s) != "nan"
]

soil = st.selectbox(
    "Soil Type",
    soil_options,
    key="soil"
)
st.divider()
def reset_inputs():
    st.session_state.temperature = 0.0
    st.session_state.humidity = 0.0
    st.session_state.rainfall = 0.0
    st.session_state.ph = 0.0
    st.session_state.nitrogen = 0.0
    st.session_state.phosphorous = 0.0
    st.session_state.potassium = 0.0
    st.session_state.carbon = 0.0
    st.session_state.soil = soil_options[0]

button1, button2 = st.columns(2)

with button1:
    recommend = st.button(
        "🌱 Recommend Crop",
        use_container_width=True
    )

with button2:
    reset = st.button(
       "🔄 Reset",
        on_click=reset_inputs,
        use_container_width=True
    )



if recommend:

    with st.spinner("Predicting the most suitable crop..."):

        soil_number = le_soil.transform([soil])[0]

        new_data = pd.DataFrame([{
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "ph": ph,
            "nitrogen": nitrogen,
            "phosphorous": phosphorous,
            "potassium": potassium,
            "carbon": carbon,
            "soil": soil_number
        }])

        prediction = ml_model.predict(new_data)[0]
        crop_name = le_crop.inverse_transform([prediction])[0]

    st.success("Prediction Successful!")

    st.markdown("## ✅ Recommended Crop")
    st.markdown(
        f"""
        <div style="
            background-color:#E8F5E9;
            border:2px solid #4CAF50;
            padding:25px;
            border-radius:15px;
            text-align:center;
            font-size:30px;
            color:#2E7D32;
            font-weight:bold;">
            🌾 {crop_name}
        </div>
        """,
        unsafe_allow_html=True
    )
