import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Stock Prophet! 👋")

st.sidebar.success("Here you can find various tools to help you with your stock analysis.")

st.markdown(
    """
    StockProphet is a web app that allows you to analyze stock data and predict future stock prices.
    
    **👈 Select a tab from the sidebar** to begin your stock analysis and find out what StockProphet can do.
    
    ### Want to learn more?
    """
)

# Open the image file
image = Image.open(r'C:\Users\bryan\PythonWorkSpaces\StockAdvisorFinal\Stock_Predictor\assets\Logo.jpg')

# Get the original width and height of the image
original_width, original_height = image.size

# Calculate the aspect ratio
aspect_ratio = original_height / original_width

# Set the desired width
desired_width = 400

# Calculate the corresponding height based on the aspect ratio
desired_height = int(desired_width * aspect_ratio)

# Display the image with the desired width and height at the bottom of the page
st.image(image, caption='Company Logo', width=desired_width, output_format='PNG')