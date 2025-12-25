"""
Sales Demand Forecasting - Streamlit App
Enhanced with ARIMA, LSTM and RMSE
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Sales Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Sales Demand Forecasting")
st.markdown("---")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Settings")

# --------------------------------------------------
# File Upload
# --------------------------------------------------
st.sidebar.subheader("1. Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload your sales data (CSV)",
    type=["csv"]
)

# --------------------------------------------------
# Sample Data
# --------------------------------------------------
@st.cache_data
def load_sample_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    n = len(dates)
    data = {
        "Date": dates,
        "Sales_Quantity": np.maximum(
            0,
            100 + np.sin(np.arange(n)/14)*20 + np.random.normal(0, 5, n)
        ).astype(int)
    }
    return pd.DataFrame(data)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.trained = False

# --------------------------------------------------
# Load Data
# --------------------------------------------------
if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.sidebar.success("Data loaded successfully!")

else:
    if st.sidebar.button("Load Sample Data"):
        st.session_state.df = load_sample_data()
        st.sidebar.success("Sample data loaded!")

# --------------------------------------------------
# Main App
# --------------------------------------------------
if st.session_state.df is not None:

    df = st.session_state.df.copy()

    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Column selection
    date_col = st.selectbox("Select Date Column", df.columns)
    target_col = st.selectbox("Select Target Column", df.columns)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, target_col]].dropna()
    df.set_index(date_col, inplace=True)

    # Stats
    st.subheader("Basic Statistics")
    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(df))
    col2.metric(
        "Date Range",
        f"{df.index.min().date()} to {df.index.max().date()}"
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    st.subheader("Sales Over Time")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df, x=df.index, y=target_col, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --------------------------------------------------
    # Model Settings
    # --------------------------------------------------
    st.sidebar.subheader("2. Model Settings")
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["ARIMA", "LSTM", "Compare (RMSE)"]
    )

    forecast_days = st.sidebar.slider(
        "Forecast Period (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )

    # --------------------------------------------------
    # ARIMA
    # --------------------------------------------------
    def run_arima(series, steps):
        train = series[:-steps]
        test = series[-steps:]

        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=steps)
        rmse = np.sqrt(mean_squared_error(test, forecast))

        return forecast, rmse

    # --------------------------------------------------
    # LSTM
    # --------------------------------------------------
    def run_lstm(series, steps):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1))

        lookback = 14
        X, y = [], []

        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i])
            y.append(scaled[i])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, activation="relu", input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=15, verbose=0)

        preds_scaled = model.predict(X[-steps:])
        preds = scaler.inverse_transform(preds_scaled)

        test = series[-steps:].values
        rmse = np.sqrt(mean_squared_error(test, preds))

        return preds.flatten(), rmse

    # --------------------------------------------------
    # Train Button
    # --------------------------------------------------
    if st.sidebar.button("Train & Forecast"):
        st.session_state.trained = True

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    if st.session_state.trained:
        future_dates = pd.date_range(
            df.index[-1], periods=forecast_days+1, freq="D"
        )[1:]

        if model_type == "ARIMA":
            forecast, rmse = run_arima(df[target_col], forecast_days)
            result_df = pd.DataFrame({"Forecast": forecast.values}, index=future_dates)

        elif model_type == "LSTM":
            forecast, rmse = run_lstm(df[target_col], forecast_days)
            result_df = pd.DataFrame({"Forecast": forecast}, index=future_dates)

        else:
            arima_f, arima_rmse = run_arima(df[target_col], forecast_days)
            lstm_f, lstm_rmse = run_lstm(df[target_col], forecast_days)

            result_df = pd.DataFrame({
                "ARIMA": arima_f.values,
                "LSTM": lstm_f
            }, index=future_dates)

            st.metric("ARIMA RMSE", f"{arima_rmse:.2f}")
            st.metric("LSTM RMSE", f"{lstm_rmse:.2f}")

        # Plot
        st.subheader("Forecast")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df, x=df.index, y=target_col, label="Historical", ax=ax)
        sns.lineplot(data=result_df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Table + Download
        st.subheader("Forecast Data")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            csv,
            f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

else:
    st.info("👈 Upload data or load sample data to begin")

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown("""
<style>
.stButton>button {background-color:#4CAF50;color:white;}
</style>
""", unsafe_allow_html=True)
