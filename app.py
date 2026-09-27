import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Insurance Policy & Premium Predictor",
    page_icon="🛡️",
    layout="wide"
)

# --- Load Models & Pipeline Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        logistic_data = joblib.load('logistic_lapse_model.pkl')
        linear_data = joblib.load('linear_premium_model.pkl')
        return logistic_data, linear_data
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

logistic_data, linear_data = load_artifacts()

if logistic_data and linear_data:
    logreg = logistic_data['model']
    scaler = logistic_data['scaler']
    log_features_columns = logistic_data['columns']

    linreg = linear_data['model']
    lin_features_columns = linear_data['columns']

    # --- Feature Mapping Dictionaries ---
    health_map = {'Excellent': 0, 'Good': 1, 'Fair': 2}
    smoke_map = {'Non-smoker': 0, 'Former smoker': 1, 'Current smoker': 2}

    # --- Header & Description ---
    st.title("🛡️ Insurance Premium & Policy Lapse Predictor")
    st.markdown("Enter customer demographics and policy details to calculate predicted monthly premiums and assess policy lapse risk.")
    st.divider()

    # --- UI Layout: Input Form ---
    st.header("📋 Customer & Policy Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", min_value=18, max_value=80, value=40)
        gender = st.selectbox("Gender", options=['Male', 'Female', '__RARE_'])
        marital_status = st.selectbox("Marital Status", options=['Married', 'Single', 'Divorced', 'Widowed'])

    with col2:
        number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)
        annual_income = st.number_input("Annual Income (INR)", min_value=10000, max_value=1000000, value=75000, step=5000)
        health_status = st.selectbox("Health Status", options=['Excellent', 'Good', 'Fair'])

    with col3:
        smoking_status = st.selectbox("Smoking Status", options=['Non-smoker', 'Former smoker', 'Current smoker'])
        policy_type = st.selectbox("Policy Type", options=['Term Life', 'Whole Life', 'Universal Life', 'Variable Life'])
        coverage_amount = st.number_input("Coverage Amount (INR)", min_value=10000, max_value=5000000, value=300000, step=50000)
        tenure_months = st.number_input("Tenure (Months)", min_value=1, max_value=360, value=60, step=1)

    st.divider()

    # --- Prediction Execution ---
    if st.button("🚀 Calculate Predictions", use_container_width=True):
        health_score = health_map.get(health_status, 0)
        smoke_score = smoke_map.get(smoking_status, 0)

        # ----------------------------------------------------
        # Step 1: Linear Regression Inputs (Monthly Premium)
        # ----------------------------------------------------
        input_data_lin = {
            'age': [age],
            'number_of_dependents': [number_of_dependents],
            'annual_income': [annual_income],
            'health_score': [health_score],
            'smoke_score': [smoke_score],
            'coverage_amount': [coverage_amount],
            'gender': [gender],
            'marital_status': [marital_status],
            'policy_type': [policy_type]
        }
        df_lin_raw = pd.DataFrame(input_data_lin)
        df_lin_encoded = pd.get_dummies(df_lin_raw, columns=['gender', 'marital_status', 'policy_type'], drop_first=True)

        # Align with linear regression features
        final_input_lin = pd.DataFrame(columns=lin_features_columns)
        final_input_lin = pd.concat([final_input_lin, df_lin_encoded], ignore_index=True).fillna(0)
        final_input_lin = final_input_lin[lin_features_columns].astype(float)

        # Predict Monthly Premium
        predicted_monthly_premium = float(linreg.predict(final_input_lin)[0])

        # ----------------------------------------------------
        # Step 2: Logistic Regression Inputs (Lapse Risk)
        # ----------------------------------------------------
        annual_premium = predicted_monthly_premium * 12
        premium_to_income_pct = (annual_premium / annual_income * 100) if annual_income > 0 else 0.0

        input_data_log = {
            'age': [age],
            'number_of_dependents': [number_of_dependents],
            'annual_income': [annual_income],
            'health_score': [health_score],
            'smoke_score': [smoke_score],
            'coverage_amount': [coverage_amount],
            'monthly_premium': [predicted_monthly_premium],
            'premium_to_income_pct': [premium_to_income_pct],
            'tenure_months': [tenure_months],
            'gender': [gender],
            'marital_status': [marital_status],
            'policy_type': [policy_type]
        }
        df_log_raw = pd.DataFrame(input_data_log)
        df_log_encoded = pd.get_dummies(df_log_raw, columns=['gender', 'marital_status', 'policy_type'], drop_first=True)

        # Align with logistic regression features
        final_input_log = pd.DataFrame(columns=log_features_columns)
        final_input_log = pd.concat([final_input_log, df_log_encoded], ignore_index=True).fillna(0)
        final_input_log = final_input_log[log_features_columns].astype(float)

        # Scale features and calculate probability
        input_log_scaled = scaler.transform(final_input_log)
        lapse_probability = float(logreg.predict_proba(input_log_scaled)[:, 1][0])

        # ----------------------------------------------------
        # Step 3: Display Results
        # ----------------------------------------------------
        st.header("📊 Results")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric(
                label="Predicted Monthly Premium",
                value=f"₹{predicted_monthly_premium:,.2f}"
            )

        with res_col2:
            st.metric(
                label="Lapse Probability Risk",
                value=f"{lapse_probability:.2%}"
            )

        # Risk Classification Alert
        if lapse_probability >= 0.50:
            st.error("⚠️ **High Lapse Risk**: This policy shows a strong probability of cancellation/lapse. Consider providing retention incentives.")
        else:
            st.success("✅ **Low Lapse Risk**: This customer is likely to maintain their policy actively.")

else:
    st.warning("Please ensure both `logistic_lapse_model.pkl` and `linear_premium_model.pkl` are present in your project directory.")
