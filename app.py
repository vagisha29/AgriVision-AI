import streamlit as st

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱",
    layout="wide"
)

# ----------------------------------------------------
# Title
# ----------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: green;'>
        🌱 AgriVision AI
    </h1>
    <h3 style='text-align: center;'>
        Intelligent Crop Recommendation & Disease Detection System
    </h3>
    """,
    unsafe_allow_html=True
)

st.divider()

# ----------------------------------------------------
# Welcome Section
# ----------------------------------------------------
st.header("Welcome!")

st.write(
    """
    AgriVision AI is an AI-powered platform designed to assist in modern agriculture
    through intelligent crop recommendation and disease detection.
    """
)

# ----------------------------------------------------
# Features
# ----------------------------------------------------
st.subheader("🔍 Features")

st.markdown("""
🌾 **Crop Recommendation**

Recommends the most suitable crop based on soil and environmental conditions.
""")

st.markdown("""
🌿 **Disease Detection**

Detects crop diseases from uploaded leaf images using Computer Vision.
""")

st.divider()

st.info("👉 Select a page from the navigation menu to get started.")