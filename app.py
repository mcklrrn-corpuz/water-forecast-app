import streamlit as st
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort

# LOAD ARTIFACTS
@st.cache_resource
def load_all():
    session_gru  = ort.InferenceSession("gru_model.onnx")
    session_lstm = ort.InferenceSession("lstm_model.onnx")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("historical_data.csv", index_col=0, parse_dates=True)
    return session_gru, session_lstm, scaler, df

session_gru, session_lstm, scaler, df_filled = load_all()

FEATURES = [
 'discharge',
 'pH',
 'conductance',
 'water_temperature',
 'dissolved_oxygen'
]
SEQ_LEN = 120
HORIZON = 30

# UI
st.title("Water Quality Forecast (30 Days)")

st.markdown("""
Select a model to generate forecasts:

- **LSTM (Baseline)** — stable predictions  
- **GRU + Attention (Premium)** — improved accuracy and feature focus  
""")

model_choice = st.selectbox(
    "Choose Model",
    ["GRU + Attention (Premium)", "LSTM (Baseline)"]
)

if "GRU" in model_choice:
    st.success("Premium model selected: higher accuracy across most variables.")
else:
    st.info("Baseline model selected: simpler and stable predictions.")

feature = st.selectbox("Select parameter", FEATURES)

# FORECAST FUNCTION
def forecast_30_days(df, session, scaler):
    last_120 = df[FEATURES].iloc[-SEQ_LEN:]
    last_scaled = scaler.transform(last_120.values)    
    
    X_input = last_scaled.reshape(1, SEQ_LEN, len(FEATURES)).astype(np.float32)

    outputs = session.run(None, {"input": X_input})
    future_scaled = outputs[0]

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

# RUN
if st.button("Generate 30-Day Forecast"):

    # select model
    if "GRU" in model_choice:
        session = session_gru
        model_label = "GRU + Attention (Premium)"
    else:
        session = session_lstm
        model_label = "LSTM (Baseline)"

    with st.spinner("Generating forecast..."):
        forecast_df = forecast_30_days(df_filled, session, scaler)

    st.subheader(f"Forecast Table — {model_label}")
    st.dataframe(forecast_df)

    st.subheader(f"{feature} (History vs Forecast) — {model_label}")
    hist = df_filled[-60:]

    chart_df = pd.concat([
        hist[[feature]].rename(columns={feature: "History"}),
        forecast_df[[feature]].rename(columns={feature: "Forecast"})
    ])

    st.line_chart(chart_df)