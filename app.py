
import streamlit as st
import pandas as pd
import joblib

st.title("Student Exam Score Predictor")

model = joblib.load("exam_model.joblib")
features = joblib.load("features.joblib")

hours = st.slider("Hours Studied", 0, 12, 5)
attendance = st.slider("Attendance", 0, 100, 75)
sleep = st.slider("Sleep Hours", 0, 12, 7)

data = {
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Sleep_Hours": sleep
}

input_df = pd.DataFrame([data])

for col in features:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[features]

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")
