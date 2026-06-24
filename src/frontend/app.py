from pathlib import Path

import requests
import streamlit as st
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()
API_URL = config["api_url"]

st.title("Employee Attrition Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
        joining_year = st.number_input(
            "Joining Year", min_value=2000, max_value=2030, value=2016
        )
        city = st.selectbox("City", ["Bangalore", "New Delhi", "Pune"])
        payment_tier = st.selectbox("Payment Tier", [1, 2, 3], index=2)

    with col2:
        age = st.number_input("Age", min_value=18, max_value=65, value=24)
        gender = st.selectbox("Gender", ["Female", "Male"])
        ever_benched = st.selectbox("Ever Benched", ["No", "Yes"])
        experience = st.number_input(
            "Experience in Current Domain", min_value=0, max_value=20, value=2
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "features": [
            [education, joining_year, city, payment_tier, age, gender, ever_benched, experience]
        ]
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        prediction = result["predictions"][0]
        if prediction == 1:
            st.error("Prediction: **Employee is likely to leave**")
        else:
            st.success("Prediction: **Employee is likely to stay**")
    except requests.ConnectionError:
        st.error(f"Could not connect to the API at {API_URL}")
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.text}")
