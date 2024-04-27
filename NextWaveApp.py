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
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Function to perform sentiment analysis


# Function to generate word cloud for selected product (asin)


# Streamlit UI
st.title('Welcome to NextWave')

# File upload option for the user to upload a file containing product reviews



# Perform sentiment analysis when the user uploads a file

uploaded_file = st.file_uploader("Upload your Amazon review JSONL file", type='jsonl')

if uploaded_file is not None:
    
    def read_jsonl(file):
        data = read_jsonl(uploaded_file)




    
    # Generate word cloud for selected product
  






