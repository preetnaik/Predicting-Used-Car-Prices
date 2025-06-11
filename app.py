import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("usedcar_model.pkl")         
encoders = joblib.load("label_encoder(1).joblib")

# Define feature names manually (must match the training data order)
feature_names = ['year', 'present_price', 'kms_driven', 'fuel_type', 'seller_type', 'transmission', 'owner']

st.title("Used Car Price Predictor")

# User input form
user_input = {}

user_input['year'] = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
user_input['present_price'] = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1, value=5.0)
user_input['kms_driven'] = st.number_input("KMs Driven", min_value=0, step=100, value=50000)
user_input['fuel_type'] = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
user_input['seller_type'] = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
user_input['transmission'] = st.selectbox("Transmission", ['Manual', 'Automatic'])
user_input['owner'] = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

# Prepare input for prediction
input_df = pd.DataFrame([user_input])

# Apply label encoding to categorical columns using the encoders dict
for col in input_df.columns:
    if col in encoders:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df[feature_names])[0]
    st.success(f"Estimated Price: ₹ {round(prediction, 2)} lakhs")
