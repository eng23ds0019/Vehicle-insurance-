
import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Vehicle Insurance Fraud Detection")

uploaded_file = st.file_uploader("Upload Insurance Data (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    display_data = data.copy()
    if "Is_Fraud" in display_data.columns:
        display_data = display_data.drop("Is_Fraud", axis=1)
    st.write("Uploaded Data:", display_data.head())
    if "Vehicle_Damage" in data.columns:
        data["Vehicle_Damage"] = data["Vehicle_Damage"].map({"Yes": 1, "No": 0})
    if "Is_Fraud" in data.columns:
        data = data.drop("Is_Fraud", axis=1)
    prediction = model.predict(data)
    result_df = pd.DataFrame(prediction, columns=["Prediction"])
    result_df["Prediction"] = result_df["Prediction"].map({1: "Fraudulent", 0: "Genuine"})
    st.write("Prediction Results:", result_df)
