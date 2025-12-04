import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("rf_model.joblib")

# Streamlit app
st.set_page_config(page_title="Insurance Customer Response Prediction", layout="centered")

st.title("Insurance Customer Response Prediction Dashboard")
st.write("Predict whether a customer will respond positively to an insurance policy offer.")

# Sidebar inputs
st.sidebar.header("Customer Details")

vehicle_damage = st.sidebar.selectbox("Vehicle Damage", ["Yes", "No"])
vehicle_age = st.sidebar.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
previously_insured = st.sidebar.selectbox("Previously Insured", ["Yes", "No"])

# Convert inputs to numeric encoding (same as preprocessing)
vehicle_damage_val = 1 if vehicle_damage == "Yes" else 0
vehicle_age_val = 1 if vehicle_age == "< 1 Year" else (2 if vehicle_age == "1-2 Year" else 3)
previously_insured_val = 1 if previously_insured == "Yes" else 0

# Create dataframe for prediction
input_data = pd.DataFrame([[vehicle_damage_val, previously_insured_val, vehicle_age_val]],
                          columns=["Vehicle_Damage", "Previously_Insured", "Vehicle_Age"])

st.subheader("Input Data Preview")
st.write(input_data)

# Prediction
if st.button("Predict Response"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Customer is **likely to respond positively** (Probability: {prob:.2f})")
    else:
        st.error(f"Customer is **unlikely to respond** (Probability: {prob:.2f})")
