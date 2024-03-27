import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge


# Function to fetch metadata and determine start date
def get_start_date(stock_symbol, token, max_years=15):
    headers = {'Content-Type': 'application/json'}
    meta_url = f'https://api.tiingo.com/tiingo/daily/{stock_symbol}?token={token}'
    meta_response = requests.get(meta_url, headers=headers)
    meta_data = meta_response.json()

    if 'startDate' not in meta_data:
        print("startDate not found in response. Response was:", meta_data)
        return "2019-01-01"
    
    # Extract the start date from metadata
    start_date = datetime.strptime(meta_data['startDate'], '%Y-%m-%d')
    
    # Calculate date 15 years ago from today
    years_ago_date = datetime.now() - timedelta(days=max_years * 365)
    
    # Use the later of the two dates as the start date for fetching historical data
    optimal_start_date = max(start_date, years_ago_date).strftime('%Y-%m-%d')
    return optimal_start_date

# Your existing function for fetching data, updated to use the dynamic start date
def fetch_data(stock_symbol, optimal_start_date, token):
    # start_date = get_start_date(symbol, token)
    headers = {'Content-Type': 'application/json'}
    url = f'https://api.tiingo.com/tiingo/daily/{stock_symbol}/prices?startDate={optimal_start_date}&token={token}'
    response = requests.get(url, headers=headers)
    json_response = response.json()

    if isinstance(json_response, list):
        # Directly convert list of records to DataFrame
        return pd.DataFrame(json_response)
    elif isinstance(json_response, dict):
        # For a single dictionary, wrap it in a list before creating the DataFrame
        return pd.DataFrame([json_response])
    else:
        # Handle unexpected response types (e.g., empty response or error message)
        print("Unexpected JSON response format:", json_response)
        # Return an empty DataFrame or raise an error as appropriate
        return pd.DataFrame()


# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['adjClose'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


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
        # Predict the prices
        next_day_prediction = model.predict(features_for_prediction)[0]
        predictions.append(next_day_prediction)

        features_for_prediction[0][7] = next_day_prediction  # Update lag_1_day with the new prediction
        features_for_prediction[0][8] = features_for_prediction[0][7]  # Shift previous lag_1_day to lag_5_days
        features_for_prediction[0][9] = features_for_prediction[0][8]  # Shift previous lag_5_days to lag_30_days
        features_for_prediction[0][10] = features_for_prediction[0][9]  # Shift previous lag_30_days to lag_60_days

    predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Adj Close': predictions})
    predictions_df['Date'] = predictions_df['Date'].dt.date
    predictions_df.set_index('Date', inplace=True)

    return predictions_df





