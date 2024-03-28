import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from stocknews import StockNews

# Sidebar for user inputs
ticker_symbol = st.sidebar.text_input('Ticker', value='AAPL')  # Default to 'AAPL'
start_date = st.sidebar.date_input('Start Date', value=date(2023, 3, 1))  # Default start date
end_date = st.sidebar.date_input('End Date', value=date(2024, 3, 1))  # Default end date
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Tabs
pricing_data, fundamental_data, news_tab = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    st.write(data)
    
    # Create a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], name='Adjusted Close'))
    fig.update_layout(title=f"{ticker_symbol} Stock Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)
    
    # Calculate and display metrics
    data2 = data
    data2['% Change'] = data2['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    annual_return = data2['% Change'].mean() * 252 * 100
    st.write('Annual Return is ', annual_return, '%')
    stdev = np.std(data2['% Change']) * np.sqrt(252)
    st.write('Standard Deviation is ', stdev * 100, '%')
    st.write('Risk Adj. Return is ', annual_return / (stdev * 100))

with fundamental_data:
    st.header('Fundamental Data')
    
    # Create a Ticker object for the entered symbol
    ticker = yf.Ticker(ticker_symbol)

    # Fetch and display the annual cash flow statement
    st.subheader('Annual Cash Flow')
    try:
        cashflow = ticker.cashflow
        st.dataframe(cashflow)
    except Exception as e:
        st.error(f"Failed to load annual cash flow data: {e}")

    # Fetch and display the annual balance sheet
    st.subheader('Annual Balance Sheet')
    try:
        balance_sheet = ticker.balance_sheet
        st.dataframe(balance_sheet)
    except Exception as e:
        st.error(f"Failed to load annual balance sheet data: {e}")

    # Fetch and display the annual income statement
    st.subheader('Annual Income Statement')
    try:
        income_statement = ticker.financials
        st.dataframe(income_statement)
    except Exception as e:
        st.error(f"Failed to load annual income statement data: {e}")

with news_tab:
    st.header(f'News of {ticker_symbol}')
    sn = StockNews(ticker_symbol, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'news {i + 1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment: {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment: {news_sentiment}')