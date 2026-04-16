import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
@st.cache_resource
def load_all():
    model = load_model("model.keras")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("historical_data.csv", index_col=0, parse_dates=True)
    return model, scaler, df

model, scaler, df_filled = load_all()

FEATURES = ['discharge','pH','conductance','temp','do']
SEQ_LEN = 120
HORIZON = 30

# -----------------------------
# UI
# -----------------------------
st.title("Water Quality Forecast (30 Days)")
st.write("Model: GRU + Attention (direct multi-step)")

feature = st.selectbox("Select parameter", FEATURES)

# -----------------------------
# FORECAST FUNCTION
# -----------------------------
def forecast_30_days(df, model, scaler):
    last_120 = df[FEATURES].iloc[-SEQ_LEN:]
    last_scaled = scaler.transform(last_120)

    X_input = last_scaled.reshape(1, SEQ_LEN, len(FEATURES))
    future_scaled = model.predict(X_input, verbose=0)

    future_2d = future_scaled.reshape(-1, len(FEATURES))
    future_actual = scaler.inverse_transform(future_2d)

    future_dates = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        periods=HORIZON
    )

    forecast_df = pd.DataFrame(
        future_actual,
        index=future_dates,
        columns=FEATURES
    )
    return forecast_df

# -----------------------------
# RUN
# -----------------------------
if st.button("Generate 30-Day Forecast"):
    forecast_df = forecast_30_days(df_filled, model, scaler)

    st.subheader("Forecast Table")
    st.dataframe(forecast_df)

    st.subheader(f"{feature} (History vs Forecast)")
    hist = df_filled[-60:]

    chart_df = pd.concat([
        hist[[feature]].rename(columns={feature: "History"}),
        forecast_df[[feature]].rename(columns={feature: "Forecast"})
    ])

    st.line_chart(chart_df)