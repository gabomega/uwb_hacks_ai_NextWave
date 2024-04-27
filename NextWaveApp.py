import streamlit as st
import pandas as pd
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Streamlit UI setup
st.title('Welcome to NextWave')

# Upload functionality
uploaded_file = st.file_uploader("Upload your Amazon review JSONL file", type='jsonl')

# Check if file is uploaded
if uploaded_file is not None:
    # Read data
    data = pd.DataFrame([json.loads(line) for line in uploaded_file])

    # Allow user to input ASIN code
    selected_product = st.text_input('Search for a product by ASIN number')

    # Check if the user has entered an ASIN
    if selected_product:
        # Function to preprocess text
        def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
            processed_text = ' '.join(lemmatized_tokens)
            return processed_text

        # Apply preprocessing to the review text
        data['cleaned_text'] = data['text'].apply(preprocess_text)

        # Sentiment analysis
        sia = SentimentIntensityAnalyzer()
        data['sentiments'] = data['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

        # Generate and display WordCloud
        wordcloud = WordCloud(width=800, height=400).generate(' '.join(data['cleaned_text']))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Display sentiment statistics
        st.write(data['sentiments'].describe())

        # Check if ASIN is valid
        if selected_product in data['asin'].unique():
            # Display selected product information
            st.write(f"Selected product: {selected_product}")
        else:
            st.write("Product not found. Please try again.")
