import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("model.joblib")
encoders = joblib.load("encoders.joblib")  # should be a dictionary

# Safely get feature names (only if the model has feature_names_in_ attribute)
if hasattr(model, 'feature_names_in_'):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = []  # fallback or raise an error if needed

st.title("Used Car Price Predictor")

# Example of getting user input
user_input = {}

for feature in feature_names:
    if feature in encoders:
        user_input[feature] = st.selectbox(f"Select {feature}", encoders[feature].classes_)
    else:
        user_input[feature] = st.text_input(f"Enter {feature}")

# Convert user input into a DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical inputs
for col, encoder in encoders.items():
    if col in input_df.columns:
        input_df[col] = encoder.transform(input_df[col])

# Predict
prediction = model.predict(input_df)[0]
st.success(f"Predicted Car Price: ₹{prediction:,.2f}")
