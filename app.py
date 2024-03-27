import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_predictor import fetch_data, get_start_date, preprocess_data, training_data_prep, model_operation
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('TIINGO_API_KEY')
print(token)

# Streamlit app
st.title('Stock Price Prediction')
st.write('Enter Stock Symbol')

stock_symbol = st.text_input('Stock Symbol').upper()

if st.button('Predict'):
    if stock_symbol:
        # Get the start date for fetching data (placeholder logic)
        # start_date = get_start_date(stock_symbol, token)
        # start_date = datetime.now() - timedelta(days=365)

        # Fetch the historical data
        stock_data = fetch_data(stock_symbol, token)
        index_data = fetch_data('spy', token)
        vxx_data = fetch_data('vxx', token)

        stock_data, index_data, vxx_data = preprocess_data(stock_data, index_data, vxx_data)

        X_train, y_train = training_data_prep(stock_data)

        predictions_df = model_operation(X_train, y_train, stock_data)

        # Prepare the features for the last available data point (similar logic as your training)
        # Ensure the features are correctly ordered and match the model's expected input
        # last_row = stock_data.iloc[-1]
        # features_for_prediction = last_row.drop('date').values.reshape(1, -1)

        # Predict the next 10 days
        # (You'll need to implement a loop similar to the one you have, updating features for each prediction)
        # Placeholder for prediction logic
        # predictions = [model.predict(features_for_prediction)[0] for _ in range(10)]

        # # Display predictions
        # future_dates = pd.date_range(
        #     stock_data['date'].max() + pd.Timedelta(days=1), periods=10,
        #     freq='B')
        # predictions_df = pd.DataFrame(
        #     {'Date': future_dates, 'Predicted Adj Close': predictions})

        st.write(predictions_df)
    else:
        st.write("Please enter a valid stock symbol.")

