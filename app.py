import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Page Settings
# ----------------------------
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# ----------------------------
# Load Files
# ----------------------------
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🌱 Smart Crop Recommendation")
st.sidebar.info(
"""
This application recommends the most suitable crop
based on soil nutrients and environmental conditions.

Machine Learning Model:
✔ Random Forest
"""
)

# ----------------------------
# Main Title
# ----------------------------
st.title("🌾 Smart Crop Recommendation System")

st.write(
"Enter the following soil and weather parameters to predict the most suitable crop."
)

# ----------------------------
# Input Fields
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", value=80.0)
    ph = st.number_input("pH", value=6.5)
    rainfall = st.number_input("Rainfall (mm)", value=100.0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🌱 Predict Crop"):

    data = np.array([[N, P, K,
                      temperature,
                      humidity,
                      ph,
                      rainfall]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    crop = label_encoder.inverse_transform(prediction)

    st.success(f"✅ Recommended Crop: **{crop[0]}**")

# ----------------------------
# Accuracy
# ----------------------------
st.markdown("---")
st.subheader("📊 Model Performance")

st.info("Random Forest Accuracy: **99.31%**")

# ----------------------------
# About
# ----------------------------
st.markdown("---")

st.subheader("📖 About Project")

st.write("""
This project predicts the most suitable crop using Machine Learning.

Input Features

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

Algorithm Used

- Random Forest Classifier

Libraries

- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Joblib
""")