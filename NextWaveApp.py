import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text


# Function to perform sentiment analysis


# Function to generate word cloud for selected product (asin)


# Streamlit UI
st.title('Welcome to NextWave')

# File upload option for the user to upload a file containing product reviews



# Perform sentiment analysis when the user uploads a file
st.title('Welcome to NextWave.')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    
    # Generate word cloud for selected product
  






