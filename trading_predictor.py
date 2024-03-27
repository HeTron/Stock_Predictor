import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from joblib import dump, load


# Function to fetch metadata and determine start date
def get_start_date(symbol, token, max_years=15):
    headers = {'Content-Type': 'application/json'}
    meta_url = f'https://api.tiingo.com/tiingo/daily/{symbol}?token={token}'
    meta_response = requests.get(meta_url, headers=headers)
    meta_data = meta_response.json()
    print(meta_data)
    
    # Extract the start date from metadata
    start_date = datetime.strptime(meta_data['startDate'], '%Y-%m-%d')
    
    # Calculate date 15 years ago from today
    years_ago_date = datetime.now() - timedelta(days=max_years * 365)
    
    # Use the later of the two dates as the start date for fetching historical data
    optimal_start_date = max(start_date, years_ago_date).strftime('%Y-%m-%d')
    return optimal_start_date

# Your existing function for fetching data, updated to use the dynamic start date
def fetch_data(symbol, token):
    start_date = get_start_date(symbol, token)
    headers = {'Content-Type': 'application/json'}
    url = f'https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&token={token}'
    response = requests.get(url, headers=headers)
    return pd.DataFrame(response.json())

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['adjClose'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Token and stock symbol input
# token = "3ab68f4b54224a51847d25a4dc930dfa87b55e2a"
# stock_symbol = input("Enter Stock Symbol ").lower()

def call_api():
    pass
# Fetch data using the new dynamic start date logic
# stock_data = fetch_data(stock_symbol, token)
# index_data = fetch_data('spy', token)  # SPY as an example for the index
# vxx_data = fetch_data('vxx', token)  # VXX data

def preprocess_data(stock_data, index_data, vxx_data):

    stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')
    stock_data['date'] = stock_data['date'].dt.date
    stock_data.drop(['divCash', 'splitFactor'], axis=1, inplace=True)
    stock_data['RSI'] = calculate_rsi(stock_data).fillna(50)

    index_data['date'] = pd.to_datetime(index_data['date'], errors='coerce')
    index_data['date'] = index_data['date'].dt.date
    index_data.drop(['divCash', 'splitFactor'], axis=1, inplace=True)

    vxx_data['date'] = pd.to_datetime(vxx_data['date'], errors='coerce')
    vxx_data['date'] = vxx_data['date'].dt.date
    vxx_data.drop(['divCash', 'splitFactor'], axis=1, inplace=True)

    stock_data['price_ratio_to_index'] = stock_data['adjClose'] / index_data['adjClose']
    stock_data['price_ratio_to_vxx'] = stock_data['adjClose'] / vxx_data['adjClose']

    stock_data['price_diff_from_index'] = stock_data['adjClose'] - index_data['adjClose']
    stock_data['price_diff_from_vxx'] = stock_data['adjClose'] - vxx_data['adjClose']

    stock_data['log_returns'] = np.log(stock_data['adjClose'] / stock_data['adjClose'].shift(1))
    stock_data['volatility_adjusted_returns'] = stock_data['log_returns'] / vxx_data['adjClose']

    for window in [14, 30, 90]:
        stock_data[f'ma_{window}'] = stock_data['adjClose'].rolling(window=window).mean()
        stock_data[f'index_ma_{window}'] = index_data['adjClose'].rolling(window=window).mean()
        stock_data[f'stock_over_ma_{window}'] = stock_data['adjClose'] / stock_data[f'ma_{window}']
        stock_data[f'index_over_ma_{window}'] = index_data['adjClose'] / stock_data[f'index_ma_{window}']

    # Creating lag features for the 'adjClose' column
    stock_data['lag_1_day'] = stock_data['adjClose'].shift(1)   # 1-day lag
    stock_data['lag_5_days'] = stock_data['adjClose'].shift(5)  # 5-day lag
    stock_data['lag_30_days'] = stock_data['adjClose'].shift(30) # 30-day lag
    stock_data['lag_45_days'] = stock_data['adjClose'].shift(45) # 45-day lag

    stock_data = stock_data.dropna()

    return stock_data, index_data, vxx_data

def training_data_prep(stock_data):
    # Features and target
    # X = stock_data[['RSI', 'price_ratio_to_index', 'price_ratio_to_vxx', 'price_diff_from_index', 'price_diff_from_vxx', 'log_returns', 'volatility_adjusted_returns', 'lag_1_day', 'lag_5_days', 'lag_30_days', 'lag_60_days', 'ma_14', 'index_ma_14', 'stock_over_ma_14', 'index_over_ma_14', 'ma_30', 'index_ma_30', 'stock_over_ma_30', 'index_over_ma_30', 'ma_90', 'index_ma_90', 'stock_over_ma_90', 'index_over_ma_90']]
    X = stock_data[['RSI', 'price_ratio_to_index', 'price_ratio_to_vxx', 'price_diff_from_index', 'price_diff_from_vxx', 'log_returns', 'volatility_adjusted_returns', 'lag_1_day', 'lag_5_days', 'lag_30_days', 'lag_45_days']]
    y = stock_data['adjClose']

    X_train = np.array(X)
    y_train = np.array(y)

    return X_train, y_train

def model_operation(X_train, y_train, stock_data):
# Pipeline for Ridge regression with standard scaling
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    last_row = stock_data.iloc[-1]
    features_for_prediction = last_row[['RSI', 'price_ratio_to_index', 'price_ratio_to_vxx', 'price_diff_from_index', 'price_diff_from_vxx', 'log_returns', 'volatility_adjusted_returns', 'lag_1_day', 'lag_5_days', 'lag_30_days', 'lag_45_days']].values.reshape(1, -1)

    predictions = []

    last_date = pd.to_datetime(stock_data['date'].max())  # Ensure last_date is a datetime object
    future_dates = pd.date_range(last_date, periods=10, freq='B')

    for _ in range(10):
        # Predict the next day's price
        next_day_prediction = model.predict(features_for_prediction)[0]
        predictions.append(next_day_prediction)

        # For simplification, we'll manually shift the lag features and assume the rest remain constant
        # This approach may need adjustment based on how the non-lag features are supposed to evolve over time
        features_for_prediction[0][7] = next_day_prediction  # Update lag_1_day with the new prediction
        features_for_prediction[0][8] = features_for_prediction[0][7]  # Shift previous lag_1_day to lag_5_days
        features_for_prediction[0][9] = features_for_prediction[0][8]  # Shift previous lag_5_days to lag_30_days
        features_for_prediction[0][10] = features_for_prediction[0][9]  # Shift previous lag_30_days to lag_60_days
        # NOTE: The above logic is simplistic and assumes direct shifting which may not accurately represent your feature dynamics

    # Assuming `model.predict()` expects 2D input and returns a 1D array of predictions
    predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Adj Close': predictions})

    return predictions_df




