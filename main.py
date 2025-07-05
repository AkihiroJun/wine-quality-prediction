import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load model and scaler
model = load('random_forest_model.pkl')
scaler = load('scaler.pkl')

st.title("Wine Quality Predictor 🍷")
st.write("Enter the chemical properties of the wine sample:")

# Input fields
features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
st.write("### Input Features")
input_data = []
for feature in features:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)


# Predict
if st.button("Predict Quality"):
    scaled = scaler.transform([input_data])
    prediction = model.predict(scaled)[0]
    confidence = model.predict_proba(scaled)[0][prediction]

    quality = "Good" if prediction == 1 else "Not Good"
    st.write(f"### Prediction: {quality}")
    st.write(f"Confidence Score: {confidence:.2f}")
