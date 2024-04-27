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
uploaded_file = st.file_uploader("Upload your customer reviews JSONL file", type='jsonl')

# Check if file is uploaded
if uploaded_file is not None:
    # Read data
    data = pd.DataFrame([json.loads(line) for line in uploaded_file])
    
    # Allow user to input ASIN code
    selected_product = st.text_input('Search for a product by ASIN number')
    
    # Check if the user has entered an ASIN
    if selected_product:
        # Filter data based on the entered ASIN
        filtered_data = data[data['asin'] == selected_product]
        
        # Check if ASIN is valid
        if not filtered_data.empty:
            # Display selected product information
            st.write(f"Selected product: {selected_product}")

            # Function to preprocess text and filter out the word "magazine"
            def preprocess_text(text):
                tokens = word_tokenize(text.lower())
                filtered_tokens = [token for token in tokens if token not in stopwords.words('english') and token != 'magazine']
                lemmatizer = WordNetLemmatizer()
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
                processed_text = ' '.join(lemmatized_tokens)
                return processed_text
                
            # Apply preprocessing to the review text
            filtered_data['cleaned_text'] = filtered_data['text'].apply(preprocess_text)
            
            # Sentiment analysis
            sia = SentimentIntensityAnalyzer()
            filtered_data['sentiments'] = filtered_data['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
            
     
    
            # Calculate average rating
            average_rating = filtered_data['rating'].mean()

            # Determine if reviews are mostly positive or negative
            positive_count = (filtered_data['sentiments'] > 0).sum()
            negative_count = (filtered_data['sentiments'] < 0).sum()
            if positive_count > negative_count:
                sentiment_summary = "Positive"
            elif negative_count > positive_count:
                sentiment_summary = "Negative"
            else:
                sentiment_summary = "Mixed"

            # Display summary
            st.markdown(f"<h3 style='color:{sentiment_color}; font-size:24px;'>Your customer feedback Summary (Sentiment): {sentiment_summary}</h3>", unsafe_allow_html=True)
            #st.write(f"Your customer feedback overall is {sentiment_summary}")
            st.write(f"Average Rating for {selected_product} : {average_rating:.2f}")
                   # Generate and display WordCloud
            wordcloud = WordCloud(width=800, height=400).generate(' '.join(filtered_data['cleaned_text']))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.write("Product not found. Please try again.")

