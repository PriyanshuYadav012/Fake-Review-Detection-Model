import streamlit as st
import pickle
import numpy as np
from pathlib import Path
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Page configuration
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .header-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.1rem;
    }
    .genuine {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .fake {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .confidence {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model_path = Path("Model/fake_review_model.pkl")
    vectorizer_path = Path("Model/tfidf_vectorizer.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

@st.cache_resource
def get_lemmatizer():
    return WordNetLemmatizer()

def clean_text(text, lemmatizer, stop_words):
    """Clean and preprocess review text"""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_review(review_text, model, vectorizer, lemmatizer, stop_words):
    """Predict if a review is fake or genuine"""
    cleaned = clean_text(review_text, lemmatizer, stop_words)
    X = vectorizer.transform([cleaned])
    
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    return prediction, probabilities

# Main app
try:
    model, vectorizer = load_model()
    lemmatizer = get_lemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Header
    st.title("🔍 Fake Review Detection System")
    st.markdown("""
    An intelligent machine learning system that detects fake/deceptive product reviews 
    using advanced NLP techniques and classification algorithms.
    """)
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🎯 Detector", "📊 About Model", "💡 Examples"])
    
    # TAB 1: Main Detector
    with tab1:
        st.subheader("Enter a Review to Analyze")
        
        review_input = st.text_area(
            "Paste or type a review:",
            height=150,
            placeholder="Enter the review text here...",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analyze_button = st.button("🔎 Analyze Review", use_container_width=True, type="primary")
        
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if analyze_button and review_input.strip():
            prediction, probs = predict_review(review_input, model, vectorizer, lemmatizer, stop_words)
            
            # Display results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:  # Fake
                    st.markdown("""
                    <div class="prediction-box fake">
                        <strong>⚠️ Prediction: FAKE REVIEW</strong>
                        <div class="confidence">Confidence: {:.2f}%</div>
                    </div>
                    """.format(probs[1] * 100), unsafe_allow_html=True)
                else:  # Genuine
                    st.markdown("""
                    <div class="prediction-box genuine">
                        <strong>✓ Prediction: GENUINE REVIEW</strong>
                        <div class="confidence">Confidence: {:.2f}%</div>
                    </div>
                    """.format(probs[0] * 100), unsafe_allow_html=True)
            
            with col2:
                # Probability visualization
                st.metric("Genuine Score", f"{probs[0]:.2%}")
                st.metric("Fake Score", f"{probs[1]:.2%}")
            
            # Show cleaned text
            with st.expander("📝 View Cleaned Text"):
                cleaned = clean_text(review_input, lemmatizer, stop_words)
                st.code(cleaned, language="text")
        
        elif analyze_button and not review_input.strip():
            st.warning("⚠️ Please enter a review text to analyze")
    
    # TAB 2: About Model
    with tab2:
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Model Type:** Logistic Regression
            
            **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
            
            **Max Features:** 5,000
            """)
        
        with col2:
            st.success("""
            **Data Preprocessing:**
            - Text lowercasing
            - Number removal
            - Punctuation removal
            - Stopword filtering
            - Lemmatization
            """)
        
        st.markdown("---")
        
        st.subheader("How It Works")
        st.markdown("""
        1. **Text Input**: You provide a product review
        2. **Preprocessing**: The review is cleaned using NLP techniques
        3. **Vectorization**: Text is converted to numerical features using TF-IDF
        4. **Classification**: The model predicts if it's genuine or fake
        5. **Confidence Score**: Returns probability scores for both classes
        """)
    
    # TAB 3: Examples
    with tab3:
        st.subheader("Example Reviews")
        
        examples = {
            "Genuine Review (Positive)": "This product is absolutely amazing! I've been using it for a month now and it's been a game-changer. The quality is exceptional and it works exactly as described. Highly recommended!",
            "Genuine Review (Negative)": "Unfortunately, this product didn't meet my expectations. While the design is nice, it broke after just two weeks. Customer service was helpful though.",
            "Potentially Fake Review": "BEST PRODUCT EVER!!! You MUST buy this!!! I love it love it love it!!! Everyone should get one NOW!!! Only took 10 minutes to set up and it's perfect perfect perfect!!!",
        }
        
        for review_type, review_text in examples.items():
            with st.expander(f"📖 {review_type}"):
                st.write(review_text)
                
                if st.button(f"Analyze This Review", key=review_type):
                    prediction, probs = predict_review(review_text, model, vectorizer, lemmatizer, stop_words)
                    
                    if prediction == 1:
                        st.error(f"⚠️ Fake Review - Confidence: {probs[1]:.2%}")
                    else:
                        st.success(f"✓ Genuine Review - Confidence: {probs[0]:.2%}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; margin-top: 2rem;'>
        <small>Fake Review Detection System | Built with Streamlit & Machine Learning</small>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError as e:
    st.error(f"❌ Error: Could not load model files. {str(e)}")
    st.info("Make sure you're running this app from the project root directory.")
