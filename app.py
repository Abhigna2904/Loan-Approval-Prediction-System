# app.py

import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('loan_model.pkl')

st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details below to predict loan approval status.")

# Form inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=1)
Loan_Amount_Term = st.number_input("Loan Term (in days)", value=360)
Credit_History = st.selectbox("Credit History", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert to numeric for prediction
def preprocess_input():
    return [
        1 if Gender == "Male" else 0,
        1 if Married == "Yes" else 0,
        {"0": 0, "1": 1, "2": 2, "3+": 3}[Dependents],
        1 if Education == "Graduate" else 0,
        1 if Self_Employed == "Yes" else 0,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        1 if Credit_History == "Yes" else 0,
        {"Urban": 2, "Semiurban": 1, "Rural": 0}[Property_Area]
    ]

# Predict button
if st.button("Predict"):
    input_data = np.array([preprocess_input()])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
