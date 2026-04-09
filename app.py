import streamlit as st
import pickle
import pandas as pd

# Set page configuration for a modern feel
st.set_page_config(page_title="Loan Intelligence", page_icon="💰", layout="centered")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load saved model and scaler
@st.cache_resource # Use caching so it doesn't reload every time you move a slider
def load_assets():
    with open('models/scaler_2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/best_loan_model_1.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

try:
    scaler, model = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run your training script first.")
    st.stop()

st.title("💰 Loan Eligibility Predictor")
st.markdown("Adjust the sliders below to see how applicant details affect approval probability.")

# Using tabs or expanders to organize the UI
with st.expander("👤 Personal Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        married = st.radio("Married", ["Yes", "No"], horizontal=True)
    with col2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    dependents = st.select_slider("Number of Dependents", options=[0, 1, 2, 3])

with st.expander("💵 Financial & Loan Details", expanded=True):
    # Sliders for ranges
    applicant_income = st.slider("Monthly Applicant Income ($)", 0, 25000, 5000, step=100)
    coapplicant_income = st.slider("Monthly Co-applicant Income ($)", 0, 15000, 0, step=100)
    
    col3, col4 = st.columns(2)
    with col3:
        loan_amount = st.slider("Loan Amount (in thousands $)", 10, 700, 150)
    with col4:
        loan_amount_term = st.select_slider("Term (Months)", options=[12, 36, 60, 84, 120, 180, 240, 300, 360], value=360)

    property_area = st.select_slider("Property Area", options=["Rural", "Semiurban", "Urban"], value="Semiurban")
    credit_history = st.segmented_control("Credit History Score", options=[0.0, 1.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor (0.0)", default=1.0)

# Preprocessing for prediction
gender_val = 1 if gender == "Male" else 0
married_val = 1 if married == "Yes" else 0
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0
property_mapping = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_val = property_mapping[property_area]
total_income = applicant_income + coapplicant_income

# Create DataFrame
input_data = pd.DataFrame([{
    'Gender': gender_val,
    'Married': married_val,
    'Dependents': dependents,
    'Education': education_val,
    'Self_Employed': self_employed_val,
    'Applicant_Income': applicant_income,
    'Loan_Amount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': property_val,
    'Total_Income': total_income
}])

st.markdown("---")

# Predict logic
features_scaled = scaler.transform(input_data)
prediction = model.predict(features_scaled)[0]
prob = model.predict_proba(features_scaled)[0][1]

# Visual Results
st.subheader("Results")
res_col1, res_col2 = st.columns([1, 2])

if prediction == 1:
    res_col1.metric("Status", "Approved ✅", delta="Qualified")
    res_col2.write(f"**Confidence:** {prob:.2%}")
    res_col2.progress(prob)
else:
    res_col1.metric("Status", "Rejected ❌", delta="- High Risk", delta_color="inverse")
    res_col2.write(f"**Rejection Confidence:** {1-prob:.2%}")
    res_col2.progress(1-prob)

if prob < 0.60 and prob > 0.40:
    st.warning("⚠️ This is a borderline case. Manual review is recommended.")