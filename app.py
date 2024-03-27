import streamlit as st
from trading_predictor import fetch_data, get_start_date, preprocess_data, training_data_prep, model_operation
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

load_dotenv()
token = os.getenv('TIINGO_API_KEY')

# Streamlit app
st.title('Stock Price Prediction')
st.write('Enter Stock Symbol')

stock_symbol = st.text_input('Stock Symbol').upper()

if st.button('Predict'):
    if stock_symbol:
        optimal_start_date = get_start_date(stock_symbol, token)

        # Fetch the historical data
        stock_data = fetch_data(stock_symbol, optimal_start_date, token)
        index_data = fetch_data('spy', optimal_start_date, token)
        vxx_data = fetch_data('vxx', optimal_start_date, token)

        stock_data, index_data, vxx_data = preprocess_data(stock_data, index_data, vxx_data)

        X_train, y_train = training_data_prep(stock_data)

        predictions_df = model_operation(X_train, y_train, stock_data)

        # Create a figure and axis for the plot
        fig, ax = plt.subplots()

        # Plot the predicted adjusted close prices
        ax.plot(predictions_df.index, predictions_df['Predicted Adj Close'], label='Predicted Price')

        # Rotate date labels and set format
        plt.xticks(rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

        # Optionally, add a title and labels
        ax.set_title('Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price Prediction')

        # Display legend
        ax.legend()

        col1, col2 = st.columns(2)

        with col1:
            st.write("Results:")
            st.dataframe(predictions_df)

        with col2:
            st.write("Visualization:")
            st.pyplot(fig)

    else:
        st.write("Please enter a valid stock symbol.")

