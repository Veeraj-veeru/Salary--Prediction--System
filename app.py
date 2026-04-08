import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("pipeline.pkl", "rb"))

# 👉 Add your model metrics (from training)
MODEL_ACCURACY = 99.18   # R2 %
MODEL_ERROR = 4.89       # MAE %

st.title("💼 Salary Prediction App")

years_experience = st.slider("Years of Experience", 0, 15)

job_title = st.selectbox("Job Title", ["Data Scientist", "ML Engineer", "AI Engineer"])
company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

skills_python = st.selectbox("Python", [0, 1])
skills_ml = st.selectbox("ML", [0, 1])

if st.button("Predict Salary"):

    input_data = pd.DataFrame([{
        'years_experience': years_experience,
        'job_title': job_title,
        'company_size': company_size,
        'company_industry': "Tech",
        'country': "India",
        'remote_type': "Remote",
        'experience_level': "Mid",
        'education_level': "Bachelor",
        'skills_python': skills_python,
        'skills_sql': 1,
        'skills_ml': skills_ml,
        'skills_deep_learning': 0,
        'skills_cloud': 1,
        'job_posting_month': 6,
        'job_posting_year': 2025,
        'hiring_urgency': "Medium",
        'job_openings': 3,
        'total_skills': skills_python + skills_ml + 2,
        'exp_squared': years_experience ** 2
    }])

    result = model.predict(input_data)

    st.success(f"💰 Predicted Salary: {result[0]:,.2f}")

    # ✅ Show percentage
    st.info(f"📊 Model Accuracy: {MODEL_ACCURACY}%")
    st.warning(f"⚠️ Error Margin: ±{MODEL_ERROR}%")