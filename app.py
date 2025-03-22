import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# st.header("📈 Stock Market Predictor")
# Create two columns: Title (Left) | Logo (Right)
col1, col2 = st.columns([8, 1])  

with col1:
    st.header("📈 Stock Market Predictor")  # Main Title

# ✅ **Stock Dictionary with Logos**
stocks = {
    "Google (GOOG)": ("GOOG", "https://logo.clearbit.com/google.com"),
    "Apple (AAPL)": ("AAPL", "https://logo.clearbit.com/apple.com"),
    "Microsoft (MSFT)": ("MSFT", "https://logo.clearbit.com/microsoft.com"),
    "Amazon (AMZN)": ("AMZN", "https://logo.clearbit.com/amazon.com"),
    "Tesla (TSLA)": ("TSLA", "https://logo.clearbit.com/tesla.com"),
    "Facebook (META)": ("META", "https://logo.clearbit.com/meta.com"),
    "Netflix (NFLX)": ("NFLX", "https://logo.clearbit.com/netflix.com"),
    "JPMorgan Chase (JPM)": ("JPM", "https://logo.clearbit.com/jpmorgan.com"),
    "Goldman Sachs (GS)": ("GS", "https://logo.clearbit.com/goldmansachs.com"),
    "Morgan Stanley (MS)": ("MS", "https://logo.clearbit.com/morganstanley.com"),
    "Visa (V)": ("V", "https://logo.clearbit.com/visa.com"),
    "Mastercard (MA)": ("MA", "https://logo.clearbit.com/mastercard.com"),
    "Bank of America (BAC)": ("BAC", "https://logo.clearbit.com/bankofamerica.com"),
    "Toyota (TM)": ("TM", "https://logo.clearbit.com/toyota.com"),
    "Ford (F)": ("F", "https://logo.clearbit.com/ford.com"),
    "General Motors (GM)": ("GM", "https://logo.clearbit.com/gm.com"),
    "Volkswagen (VWAGY)": ("VWAGY", "https://logo.clearbit.com/volkswagen.com"),
    "BMW (BMWYY)": ("BMWYY", "https://logo.clearbit.com/bmw.com"),
    "Mercedes-Benz (MBGAF)": ("MBGAF", "https://logo.clearbit.com/mercedes-benz.com"),
    "Reliance (RELIANCE.NS)": ("RELIANCE.NS", "https://logo.clearbit.com/reliance.com"),
    "Tata Motors (TATAMOTORS.NS)": ("TATAMOTORS.NS", "https://logo.clearbit.com/tatamotors.com"),
    "Infosys (INFY.NS)": ("INFY.NS", "https://logo.clearbit.com/infosys.com"),
    "Wipro (WIPRO.NS)": ("WIPRO.NS", "https://logo.clearbit.com/wipro.com"),
}

# ✅ **Searchable Dropdown for Stock Selection**
selected_stock_name = st.selectbox("🔍 Search & Select a Stock:", list(stocks.keys()))
stock_symbol, stock_logo = stocks[selected_stock_name]
with col2:
    st.image(stock_logo, width=120)  # Display Logo on the Right
# # ✅ **Display Company Logo Below the Selection**
# st.image(stock_logo, width=120, caption=selected_stock_name)

start = "2018-01-01"  # Train on last 2 years for better accuracy
end = "2024-12-31"

# Fetch stock data
try:
    data = yf.download(stock_symbol, start, end)
    if data.empty:
        st.error("❌ No data available for the selected stock.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error fetching stock data: {e}")
    st.stop()

# ✅ **Show First & Last 5 Rows of Stock Data**
st.subheader(f"📊 {selected_stock_name} Stock Data")
st.write("🔹 **First 5 Records:**")
st.write(data.head())
st.write("🔹 **Last 5 Records:**")
st.write(data.tail())

# ✅ **Train New Model for Selected Company**
data_train = data["Close"][: int(len(data) * 0.80)]
data_test = data["Close"][int(len(data) * 0.80):]

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(np.array(data_train).reshape(-1, 1))

# Create sequences
def create_sequences(data, time_step=100):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(data_train_scaled)

# Reshape for LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(100, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)  # Train dynamically

# Prepare test data
past_100_days = data_train.tail(100)
data_test_scaled = scaler.transform(np.array(data_test).reshape(-1, 1))
x_test, y_test = create_sequences(data_test_scaled)

# Reshape test data
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Predict prices
predictions = model.predict(x_test)

# ✅ **Fix Scaling Issue**
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# ✅ **Recalculate Accuracy for Selected Stock**
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
accuracy = 100 - mape  # Higher is better

# ✅ **Display Model Accuracy**
st.subheader("📊 Model Accuracy (Updated for Selected Stock)")
st.write(f"🔹 **MSE:** {mse:.4f}")
st.write(f"🔹 **RMSE:** {rmse:.4f}")
st.write(f"🔹 **R² Score:** {r2:.4f}")
st.write(f"✅ **Model Accuracy:** {accuracy:.2f}%")

# ✅ **Graph for Visual Comparison**
st.subheader(f"📉 {selected_stock_name} - Original vs Predicted Price")
fig = plt.figure(figsize=(8, 6))
plt.plot(y_test, "r", label="Original Price")
plt.plot(predictions, "g", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)

# ✅ **Future Price Prediction**
def predict_future_price(days):
    last_100_closes = data["Close"].values[-100:]  # Use actual recent prices
    last_100_scaled = scaler.transform(last_100_closes.reshape(-1, 1))
    
    x_input = last_100_scaled.reshape(1, -1, 1)
    predicted_scaled = model.predict(x_input)[0][0]
    
    return scaler.inverse_transform([[predicted_scaled]])[0][0]

# Get Next Day & Next Month Price
st.subheader(f"🚀 Future {selected_stock_name} Price Predictions")
next_day = predict_future_price(1)
next_month = predict_future_price(30)

st.write(f"📆 **Predicted Next Day Closing Price:** ${next_day:.2f}")
st.write(f"📅 **Predicted Next Month Closing Price:** ${next_month:.2f}")
