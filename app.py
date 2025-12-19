import streamlit as st
import torch
import numpy as np
from src.fetch_api import fetch_live_data
from src.preprocessing import scale_live_data
from src.model_lstm import LSTMModel

st.set_page_config(page_title="Delhi Air Quality Prediction")

st.title("🌫️ Real-Time Air Quality Prediction (Delhi)")

st.write("Predicts **next-hour CO concentration** using LSTM")

if st.button("Fetch Live Data & Predict"):
    df_live = fetch_live_data()
    st.subheader("Live Data")
    st.dataframe(df_live)

    scaled = scale_live_data(df_live)
    sequence = np.tile(scaled, (1, 24, 1))
    sequence = torch.tensor(sequence, dtype=torch.float32)

    model = LSTMModel(input_size=scaled.shape[1])
    model.load_state_dict(torch.load("models/lstm_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        prediction = model(sequence)

    st.success(f"Predicted CO next hour: **{prediction[0][0].item():.2f}**")
