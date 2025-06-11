import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("usedcar_model.pkl")         
encoders = joblib.load("label_encoder.joblib")   

# Get feature names (must be stored in model manually if not present)
try:
    feature_names = model.feature_names_in_
except AttributeError:
    feature_names = list(encoders.keys()) + [col for col in model.feature_importances_ if col not in encoders]

st.title("🚗 Used Car Price Predictor")

user_input = {}

# Create input UI for each feature
for feature in feature_names:
    if feature in encoders:
        options = encoders[feature].classes_.tolist()
        selected = st.selectbox(f"Select {feature}", options)
        user_input[feature] = encoders[feature].transform([selected])[0]
    else:
        value = st.number_input(f"Enter {feature}", min_value=0)
        user_input[feature] = value

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Car Price: ₹ {round(prediction, 2)}")
