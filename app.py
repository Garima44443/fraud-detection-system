import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from model import train_models, ensemble_predict, risk_label

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("🚨 AI-Powered Real-Time Fraud Detection System")

file = st.file_uploader("Upload Transaction Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # 🔥 SPEED OPTIMIZATION
    df = df.sample(15000)   # reduce data size

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Threshold slider
    threshold = st.slider("Fraud Detection Threshold", 0.1, 0.9, 0.5)

    # 🔥 TRAIN WITH LOADING
    st.subheader("🧠 Training Models...")
    with st.spinner("Training models... please wait ⏳"):
        rf_model, xgb_model = train_models(df)

    st.success("Models trained successfully!")

    # 🔥 REAL-TIME SIMULATION
    st.subheader("📡 Real-Time Monitoring")

    placeholder = st.empty()

    for i in range(3):  # reduced loops for speed
        sample_data = df.sample(10)

        predictions, probabilities = ensemble_predict(
            rf_model, xgb_model, sample_data, threshold
        )

        sample_data["Prediction"] = predictions
        sample_data["Risk Score"] = probabilities
        sample_data["Risk Level"] = [risk_label(p) for p in probabilities]

        placeholder.dataframe(sample_data)

        time.sleep(1)

    # Metrics
    fraud_count = predictions.count("Fraud")
    normal_count = predictions.count("Normal")

    col1, col2, col3 = st.columns(3)

    col1.metric("🚨 Fraud", fraud_count)
    col2.metric("✅ Normal", normal_count)
    col3.metric("📊 Fraud %", round((fraud_count/len(predictions))*100, 2))

    # Alerts
    if fraud_count > 3:
        st.error("🚨 HIGH FRAUD ACTIVITY!")
    elif fraud_count > 0:
        st.warning("⚠️ Fraud Detected")
    else:
        st.success("✅ Safe")

    # Filter
    show_only_fraud = st.checkbox("Show only Fraud")

    if show_only_fraud:
        sample_data = sample_data[sample_data["Prediction"] == "Fraud"]

    st.subheader("📋 Final Data")
    st.dataframe(sample_data)

    # Chart
    st.subheader("📊 Fraud vs Normal")

    fig, ax = plt.subplots()
    ax.bar(["Fraud", "Normal"], [fraud_count, normal_count])
    st.pyplot(fig)