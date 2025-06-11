import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("model.joblib")
encoders = joblib.load("encoders.joblib")

# Get feature names (assuming stored during training)
feature_names = model.feature_names_in_  # Only works in sklearn >= 1.0

st.title("Used Car Price Predictor")

user_input = {}

# Create form for user inputs
for feature in feature_names:
    if feature in encoders:
        options = encoders[feature].classes_.tolist()
        selected = st.selectbox(f"Select {feature}", options)
        user_input[feature] = encoders[feature].transform([selected])[0]
    else:
        value = st.number_input(f"Enter {feature}", min_value=0)
        user_input[feature] = value

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Car Price: ₹ {round(prediction, 2)}")
