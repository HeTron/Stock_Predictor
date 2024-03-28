import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Stock Prophet! 👋")

st.sidebar.success("Here you can find various tools to help you with your stock analysis.")

st.markdown(
    """
    StockProphet is a web app that allows you to analyze stock data and predict future stock prices.
    **👈 Select a tab from the sidebar** To begin your stock analysis and find out what StockProphet can do.
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)