import streamlit as st
import numpy as np
import joblib

# Page Setup
st.set_page_config(
    page_title="Crop Predictor",
    page_icon="🌱",
    layout="centered"
)

# Load Model
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# App Title
st.title("🌱 Smart Crop Recommendation")
st.write("Enter soil and weather details to get the recommended crop.")


# Input Section

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0)
    P = st.number_input("Phosphorus (P)", min_value=0)
    K = st.number_input("Potassium (K)", min_value=0)
    temperature = st.number_input("Temperature (°C)", value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", value=80.0)
    ph = st.number_input("Soil pH", value=6.5)
    rainfall = st.number_input("Rainfall (mm)", value=100.0)


# Prediction

if st.button("🌾 Predict Crop"):

    input_data = np.array([[
        N, P, K,
        temperature,
        humidity,
        ph,
        rainfall
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"Recommended Crop: {prediction[0]}")
