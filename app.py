import json
import os
import urllib.error
import urllib.request

import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Diabetes Risk Prediction", page_icon="+", layout="centered"
)
st.title("Diabetes Risk Prediction")
st.caption(
    "Enter the clinical measurements below to request a prediction from the FastAPI service."
)


def call_api(payload: dict[str, int | float]) -> dict[str, int | float | str]:
    request = urllib.request.Request(
        f"{API_URL}/diabetes_prediction",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


with st.form("prediction_form"):
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1, step=1
    )
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input(
        "Blood pressure", min_value=0.0, max_value=200.0, value=70.0
    )
    skin_thickness = st.number_input(
        "Skin thickness", min_value=0.0, max_value=100.0, value=20.0
    )
    insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=30.0)
    pedigree = st.number_input(
        "Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }
    try:
        result = call_api(payload)
        if result["prediction"] == 1:
            st.error(result["result"])
        else:
            st.success(result["result"])
        st.metric("Estimated probability", f"{float(result['probability']):.1%}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8")
        st.error(f"API rejected the request ({error.code}): {detail}")
    except (urllib.error.URLError, TimeoutError) as error:
        st.error(f"Could not reach the prediction API at {API_URL}: {error}")
