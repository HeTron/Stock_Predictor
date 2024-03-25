import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler

model_path = '/Users/jocksolo/PycharmProjects/trading_advisor/models/trading_advisor_model.keras'
# Correct use of loading the model once
model = tf.keras.models.load_model(model_path)

# Function to fetch stock data with error handling
def fetch_stock_data(symbol, start_date='2012-01-01'):
    headers = {'Content-Type': 'application/json'}
    response = requests.get(f'https://api.tiingo.com/tiingo/daily/{symbol}/prices', params={'startDate': start_date, 'token': '3ab68f4b54224a51847d25a4dc930dfa87b55e2a'}, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        st.error("Failed to fetch data")
        return pd.DataFrame()

@st.cache_data
def preprocess_data(df, feature_list):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[feature_list])
    return scaled_features, scaler

def generate_future_dates(last_known_date, num_days=10):
    future_dates = [last_known_date]
    while len(future_dates) < num_days:
        next_day = future_dates[-1] + timedelta(days=1)
        if next_day.weekday() < 5:  # Check for weekdays
            future_dates.append(next_day)
    return future_dates[1:]

def predict_prices(model, scaled_data, window_size=60, num_days=10):
    x_to_predict = scaled_data[-window_size:].reshape(1, window_size, -1)
    predictions = model.predict(x_to_predict)[0]
    return predictions

# Streamlit app starts here
st.title("NVDA Stock Advisor")

# Fetch and preprocess the data
symbol = 'NVDA'
start_date = '2012-01-01'
feature_list = ['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume']

df = fetch_stock_data(symbol, start_date)
scaled_data, scaler = preprocess_data(df, feature_list)

# Predict the next 10 days' prices
last_known_date = df.index.max()
predictions = predict_prices(model, scaled_data)

# Inverse transform the predictions to original scale
dummy_array = np.zeros((len(predictions), len(feature_list)))
dummy_array[:, 0] = predictions  # assuming adjClose is the first feature
predicted_prices = scaler.inverse_transform(dummy_array)[:, 0]

# Generate future dates and display predictions
future_dates = generate_future_dates(last_known_date, num_days=10)
predictions_table = pd.DataFrame({'Date': future_dates, 'Predicted Adj Close': predicted_prices})

st.write("Predicted 'adjClose' prices for the next 10 days:")
st.write(predictions_table)

