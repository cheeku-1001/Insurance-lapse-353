import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Load Models ---
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data

logistic_model_data = load_model('logistic_lapse_model.pkl')
linear_model_data = load_model('linear_premium_model.pkl')

logreg = logistic_model_data['model']
scaler = logistic_model_data['scaler']
log_features_columns = logistic_model_data['columns']

linreg = linear_model_data['model']
lin_features_columns = linear_model_data['columns']

# --- Feature Mapping ---
health_map = {'Excellent': 0, 'Good': 1, 'Fair': 2}
smoke_map = {'Non-smoker': 0, 'Former smoker': 1, 'Current smoker': 2}

# --- Streamlit App ---
st.set_page_config(page_title="Insurance Policy Prediction", layout="wide")
st.title("Insurance Policy Lapse & Premium Prediction")

st.markdown("Enter customer details to predict monthly premium and policy lapse probability.")

# --- Input Features ---
st.header("Customer Details")

# Layout for inputs
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 70, 40)
    gender = st.selectbox("Gender", ['Male', 'Female', '__RARE_'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced', 'Widowed'])

with col2:
    number_of_dependents = st.number_input("Number of Dependents", 0, 5, 1)
    annual_income = st.number_input("Annual Income (INR)", 30000, 200000, 75000, step=1000)
    health_status = st.selectbox("Health Status", ['Excellent', 'Good', 'Fair'])

with col3:
    smoking_status = st.selectbox("Smoking Status", ['Non-smoker', 'Former smoker', 'Current smoker'])
    policy_type = st.selectbox("Policy Type", ['Term Life', 'Whole Life', 'Universal Life', 'Variable Life'])
    coverage_amount = st.number_input("Coverage Amount (INR)", 50000, 1500000, 300000, step=50000)
    tenure_months = st.number_input("Tenure (Months)", 12, 120, 60)

# --- Prediction Logic ---
if st.button("Predict"):    
    # Map health and smoking status to scores
    health_score = health_map.get(health_status, -1) # -1 or raise error if invalid
    smoke_score = smoke_map.get(smoking_status, -1)  # -1 or raise error if invalid

    # Create a DataFrame for the current input for Linear Regression
    input_data_lin_raw = {
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
    input_df_lin = pd.DataFrame(input_data_lin_raw)

    # One-hot encode categorical features for Linear Model
    input_df_lin_encoded = pd.get_dummies(input_df_lin, columns=['gender', 'marital_status', 'policy_type'], drop_first=True)

    # Ensure all linear model training columns are present and in the correct order
    final_input_lin = pd.DataFrame(columns=lin_features_columns)
    final_input_lin = pd.concat([final_input_lin, input_df_lin_encoded], ignore_index=True)
    final_input_lin = final_input_lin.fillna(0) # Fill any missing dummy columns with 0
    final_input_lin = final_input_lin[lin_features_columns].astype(float) # Ensure correct order and type

    # Predict Monthly Premium
    predicted_monthly_premium = linreg.predict(final_input_lin)[0]
    
    # Calculate derived features for Logistic Regression
    annual_premium = predicted_monthly_premium * 12
    premium_to_income_pct = (annual_premium / annual_income) * 100 if annual_income > 0 else 0

    # Create a DataFrame for the current input for Logistic Regression
    input_data_log_raw = {
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
    input_df_log = pd.DataFrame(input_data_log_raw)

    # One-hot encode categorical features for Logistic Model
    input_df_log_encoded = pd.get_dummies(input_df_log, columns=['gender', 'marital_status', 'policy_type'], drop_first=True)

    # Ensure all logistic model training columns are present and in the correct order
    final_input_log = pd.DataFrame(columns=log_features_columns)
    final_input_log = pd.concat([final_input_log, input_df_log_encoded], ignore_index=True)
    final_input_log = final_input_log.fillna(0) # Fill any missing dummy columns with 0
    final_input_log = final_input_log[log_features_columns].astype(float) # Ensure correct order and type

    # Scale numeric features for Logistic Model
    input_log_scaled = scaler.transform(final_input_log)

    # Predict Lapse Probability
    lapse_probability = logreg.predict_proba(input_log_scaled)[:, 1][0]
    
    st.subheader("Prediction Results")
    st.metric(label="Predicted Monthly Premium (INR)", value=f"₹{predicted_monthly_premium:.2f}")
    st.metric(label="Predicted Lapse Probability", value=f"{lapse_probability:.2%}")

    if lapse_probability > 0.5:
        st.warning("This customer has a high probability of policy lapse.")
    else:
        st.success("This customer has a low probability of policy lapse.")

