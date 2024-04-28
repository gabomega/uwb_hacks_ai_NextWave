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

# Define page functions
def upload_and_view_results():
    st.title('Welcome to NextWave')
    st.markdown(f"<h3 style='font-size:20px; margin: 0; padding: 0;'>Upload your customer reviews JSONL file:</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type='jsonl')
    #if uploaded_file is not None:
        #process_data(uploaded_file)

    def process_data(uploaded_file):
        # Read data
        data = pd.DataFrame([json.loads(line) for line in uploaded_file])
        
        # Allow user to input ASIN code
        st.markdown("<p style='font-size:20px; margin: 0; padding: 0;'>Search for a product by ASIN number:</p>", unsafe_allow_html=True)
        selected_product = st.text_input("")
    
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
                    sentiment_summary = "<span style='color:green;'>Positive</span>"
                elif negative_count >= positive_count:
                    sentiment_summary = "<span style='color:red;'>Negative</span>"
    
                # Display summary
                st.markdown(f"<h3 style='color:black; font-size:24px;'>Your customer feedback overall is {sentiment_summary}</h3>", unsafe_allow_html=True)
                st.write(f"<h3 style='color:black; font-size:24px; padding-bottom: 20px; margin-bottom: 20px;'>Average Rating for Product {selected_product}: {average_rating:.2f}</h3>", unsafe_allow_html=True)
                st.write(f"<h3 style='color:black; font-size:24px;'>Here is the Word Cloud summary of your customer experience:</h3>", unsafe_allow_html=True)
                # Generate and display WordCloud
                wordcloud = WordCloud(width=800, height=400).generate(' '.join(filtered_data['cleaned_text']))
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("Product not found. Please try again.")

def about_section():
    st.title("About NextWave", anchor=None)
    st.header('What is Sentiment Analysis?', anchor=None)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('path_to_logo_or_relevant_image.jpg', width=300)  # Adjust path and dimensions as needed
    with col2:
        st.markdown("""
        <p style="color:midnightblue;">
        Sentiment Analysis is a powerful tool from the field of Natural Language Processing (NLP) that interprets and classifies emotions expressed in textual data using machine learning techniques. By analyzing words and phrases within text, sentiment analysis helps determine whether the underlying sentiment is positive, negative, or neutral.
        </p>
        """, unsafe_allow_html=True)

    # Using expanders with colored titles
    st.header('How Sentiment Analysis is Used in Business', anchor=None)
    with st.expander("Enhancing Customer Experience", expanded=False):
        st.markdown("""
        <p style="color:darkgreen;">
        Companies use sentiment analysis to monitor social media, customer reviews, and feedback to gauge public opinion about their products and services. This real-time insight allows businesses to swiftly address concerns, improve product offerings, and enhance overall customer satisfaction.
        </p>
        """, unsafe_allow_html=True)

    with st.expander("Market Research and Analysis", expanded=False):
        st.markdown("""
        <p style="color:darkred;">
        Sentiment analysis provides an in-depth understanding of market trends and consumer preferences by analyzing vast amounts of data from blogs, forums, and news articles. This helps businesses to stay ahead of the curve, tailor marketing strategies, and meet consumer demands effectively.
        </p>
        """, unsafe_allow_html=True)

    with st.expander("Employee Feedback and Engagement", expanded=False):
        st.markdown("""
        <p style="color:purple;">
        Internally, sentiment analysis is employed to understand employee feedback, measure engagement levels, and foster a positive work environment. This can lead to more effective HR strategies, improved employee retention rates, and a better workplace culture.
        </p>
        """, unsafe_allow_html=True)

    with st.expander("Brand Monitoring and Management", expanded=False):
        st.markdown("""
        <p style="color:maroon;">
        By continuously monitoring online conversations, sentiment analysis helps businesses protect and enhance their brand image. Companies can identify and respond to negative sentiments swiftly, mitigating potential damage and reinforcing positive perceptions in the marketplace.
        </p>
        """, unsafe_allow_html=True)

    with st.expander("Competitive Analysis", expanded=False):
        st.markdown("""
        <p style="color:navy;">
        Sentiment analysis can provide insights into how consumers feel about competitors’ products and services. This intelligence is crucial for benchmarking, strategic planning, and maintaining competitiveness in the market.
        </p>
        """, unsafe_allow_html=True)

    st.markdown('By harnessing the power of sentiment analysis, businesses can derive actionable insights from unstructured text data, leading to informed decision-making and a significant competitive advantage.')

    # Add a footer
    st.markdown("""
    <hr style="border-top: 2px solid gray;">
    <p style="color:gray;">
    Thank you for visiting our page. For more information, feel free to contact us.
    </p>
    """, unsafe_allow_html=True)

    # Optional: Include a contact form or additional links
    # st.text_input("Your Name")
    # st.text_input("Your Email")
    # st.text_area("Your Message")
    # st.button("Submit")

# Streamlit UI setup
#st.title('Welcome to NextWave')
st.write(f"<h3 style='color:grey; font-size:20px;'>~ a Sentiment Analysis Tool for Product Manager</h3>", unsafe_allow_html=True)

# Create navigation menu with buttons
st.sidebar.header("Select a page")
page = st.sidebar.button("Upload & View Results")
about_button = st.sidebar.button("About")

# Display selected page
if page:
    upload_and_view_results()
elif about_button:
    about_section()
