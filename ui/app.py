import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="FedEx DCA Intelligent Allocation", layout="centered")

st.title("FedEx DCA Intelligent Allocation System")
st.write("AI-powered recovery prediction and DCA recommendation")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "recovery_model.pkl")

model = joblib.load(MODEL_PATH)

st.success("Model loaded successfully!")

st.header("Enter Case Details")

customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
customer_income = st.number_input("Customer Income", min_value=0, value=500000)
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
employment_duration = st.number_input("Employment Duration (months)", min_value=0, value=60)
loan_intent = st.selectbox(
    "Loan Intent",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
)
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Outstanding Amount", min_value=0, value=35000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
term_years = st.selectbox("Loan Term (years)", [1, 2, 3, 5])
historical_default = st.selectbox("Historical Default", ["Y", "N"])
cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=8)


input_df = pd.DataFrame([{
    "customer_age": customer_age,
    "customer_income": customer_income,
    "home_ownership": home_ownership,
    "employment_duration": employment_duration,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "term_years": term_years,
    "historical_default": historical_default,
    "cred_hist_length": cred_hist_length
}])


if st.button("Predict Recovery Probability"):
    prob = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result")
    st.write(f"Recovery Probability: **{prob:.2f}**")
