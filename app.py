import streamlit as st
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
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

# -----------------------------
# UI
# -----------------------------
st.title("Water Quality Forecast (30 Days)")

st.markdown("""
Select a model to generate forecasts:

- **LSTM (Baseline)** — stable predictions  
- **GRU + Attention (Premium)** — improved accuracy  
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

# -----------------------------
# WQI FUNCTION
# -----------------------------
def compute_wqi(df):
    weights = {
        'pH': 0.2,
        'dissolved_oxygen': 0.3,
        'water_temperature': 0.2,
        'conductance': 0.2,
        'discharge': 0.1
    }

    q = pd.DataFrame(index=df.index)

    q['pH'] = 100 - abs(df['pH'] - 7) * 20
    q['dissolved_oxygen'] = df['dissolved_oxygen'] * 10
    q['water_temperature'] = 100 - abs(df['water_temperature'] - 25) * 2
    q['conductance'] = 100 - (df['conductance'] / df['conductance'].max()) * 100
    q['discharge'] = 100 - (df['discharge'] / df['discharge'].max()) * 100

    q = q.clip(0, 100)

    wqi = sum(q[col] * weights[col] for col in weights) / sum(weights.values())
    return wqi

def classify_wqi(wqi):
    if wqi >= 90:
        return "Excellent"
    elif wqi >= 70:
        return "Good"
    elif wqi >= 50:
        return "Medium"
    elif wqi >= 25:
        return "Poor"
    else:
        return "Very Poor"

# -----------------------------
# FORECAST FUNCTION
# -----------------------------
def forecast_30_days(df, session, scaler):
    last_120 = df[FEATURES].iloc[-SEQ_LEN:]

    # 🔥 FIX: avoid feature name mismatch
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

    return pd.DataFrame(
        future_actual,
        index=future_dates,
        columns=FEATURES
    )

# -----------------------------
# RUN
# -----------------------------
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

    # -----------------------------
    # ADD WQI
    # -----------------------------
    forecast_df['WQI'] = compute_wqi(forecast_df)

    # -----------------------------
    # DISPLAY TABLE
    # -----------------------------
    st.subheader(f"Forecast Table — {model_label}")
    st.dataframe(forecast_df)

    # -----------------------------
    # PARAMETER GRAPH
    # -----------------------------
    st.subheader(f"{feature} (History vs Forecast)")
    hist = df_filled[-60:]

    chart_df = pd.concat([
        hist[[feature]].rename(columns={feature: "History"}),
        forecast_df[[feature]].rename(columns={feature: "Forecast"})
    ])

    st.line_chart(chart_df)

    # -----------------------------
    # WQI GRAPH
    # -----------------------------
    st.subheader("Water Quality Index (WQI)")
    st.line_chart(forecast_df['WQI'])

    # -----------------------------
    # WQI INTERPRETATION
    # -----------------------------
    latest_wqi = forecast_df['WQI'].iloc[-1]
    st.write(f"Latest WQI: {latest_wqi:.2f}")
    st.write("Water Quality Status:", classify_wqi(latest_wqi))