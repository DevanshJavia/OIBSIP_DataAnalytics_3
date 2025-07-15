# app.py

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="💳 Fraud Detection Dashboard", layout="wide")
st.title("💳 Fraud Detection in Credit Card Transactions")

# Load model and scaler
model = joblib.load("fraud_rf_model.pkl")
scaler = joblib.load("fraud_scaler.pkl")

# Sidebar input
st.sidebar.header("🧾 Enter Transaction Details")
features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_data = []

for feature in features:
    value = st.sidebar.number_input(f"{feature}", value=0.0)
    input_data.append(value)

# Prediction
if st.sidebar.button("🔍 Detect Fraud"):
    X_input = pd.DataFrame([input_data], columns=features)
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {1 - prob:.2f})")
