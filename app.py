import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------------------
# App Title
# ----------------------------
st.title("📈 Sales Demand Forecasting App")

st.write("This app forecasts future sales using Time Series models.")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/sales_data.csv")
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)
    df = df.asfreq('MS')
    df['venda'] = df['venda'].ffill()
    return df

df = load_data()

# ----------------------------
# Show Raw Data
# ----------------------------
if st.checkbox("Show raw data"):
    st.write(df.head())

# ----------------------------
# Plot Historical Sales
# ----------------------------
st.subheader("📊 Historical Sales")
st.line_chart(df['venda'])

# ----------------------------
# Forecast Settings
# ----------------------------
n_periods = st.slider("Select forecast horizon (months)", 3, 24, 12)

# ----------------------------
# ARIMA Forecast
# ----------------------------
if st.button("Run ARIMA Forecast"):
    train = df['venda'][:-n_periods]

    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_periods)

    forecast_df = pd.DataFrame({
        "Date": pd.date_range(df.index[-1], periods=n_periods+1, freq='MS')[1:],
        "Predicted Sales": forecast.values
    }).set_index("Date")

    st.subheader("🔮 ARIMA Forecast")
    st.line_chart(forecast_df)

# ----------------------------
# LSTM Forecast
# ----------------------------
if st.button("Run LSTM Forecast"):
    sales = df[['venda']].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(sales)

    X, y = [], []
    for i in range(12, len(scaled)):
        X.append(scaled[i-12:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=15, verbose=0)

    pred_scaled = model.predict(X[-n_periods:])
    pred = scaler.inverse_transform(pred_scaled)

    forecast_df = pd.DataFrame({
        "Date": pd.date_range(df.index[-1], periods=n_periods+1, freq='MS')[1:],
        "Predicted Sales": pred.flatten()
    }).set_index("Date")

    st.subheader("🔮 LSTM Forecast")
    st.line_chart(forecast_df)

st.success("Forecast completed successfully!")
