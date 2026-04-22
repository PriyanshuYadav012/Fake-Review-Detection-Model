import streamlit as st
import joblib
import numpy as np
from pathlib import Path
import re
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Page configuration
st.set_page_config(
    page_title="Review AI | Fake Review Detector",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Download required NLTK data
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Custom CSS for Modern, Dark, Glassmorphism UI
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography and Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #1A1F2C 100%);
        color: #E2E8F0;
    }

    /* Hero Section */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1.5px;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 3rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        justify-content: center;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0;
        color: #94A3B8;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #EC4899 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #E2E8F0;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTextArea textarea:focus {
        border-color: #EC4899 !important;
        box-shadow: 0 0 15px rgba(236, 72, 153, 0.3), inset 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* Primary Button (Gradient) */
    .stButton>button[kind="primary"] {
        background: linear-gradient(90deg, #4F46E5, #EC4899) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.2);
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.4);
    }

    /* Secondary Button */
    .stButton>button[kind="secondary"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton>button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #FFFFFF !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }

    /* Result Card Styling (Glassmorphism) */
    .result-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.5rem;
        margin-top: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
    }
    .result-card.genuine::before { background: linear-gradient(90deg, #34D399, #10B981); }
    .result-card.fake::before { background: linear-gradient(90deg, #F87171, #EF4444); }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .prediction-title {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    .status-fake {
        color: #EF4444;
        font-size: 3rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 1rem;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
        margin: 1rem 0;
    }

    .status-genuine {
        color: #10B981;
        font-size: 3rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 1rem;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
        margin: 1rem 0;
    }

    .confidence-meter-bg {
        background: rgba(0,0,0,0.4);
        height: 10px;
        border-radius: 5px;
        margin-top: 1rem;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .confidence-meter-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = Path("Model/fake_review_model.pkl")
    vectorizer_path = Path("Model/tfidf_vectorizer.pkl")
    
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = joblib.load(f)
    
    return model, vectorizer

@st.cache_resource(show_spinner=False)
def get_lemmatizer():
    return WordNetLemmatizer()

def clean_text(text, lemmatizer, stop_words):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_review(review_text, model, vectorizer, lemmatizer, stop_words):
    cleaned = clean_text(review_text, lemmatizer, stop_words)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    return prediction, probabilities

# App Execution
try:
    with st.spinner("Initializing AI Models..."):
        model, vectorizer = load_model()
        lemmatizer = get_lemmatizer()
        stop_words = set(stopwords.words('english'))
        
    st.markdown("<div class='hero-title'>Review AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Detect deceptive and fake product reviews using cutting-edge natural language processing and statistical modeling.</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Detector", "📊 About Model", "💡 Examples"])
    
    with tab1:
        review_input = st.text_area(
            "Review",
            height=200,
            placeholder="Paste a product review here to analyze its authenticity...",
            label_visibility="collapsed"
        )
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_button = st.button("Analyze Review", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("Clear", use_container_width=True)
            
        if clear_button:
            st.rerun()
            
        if analyze_button:
            if not review_input.strip():
                st.warning("⚠️ Please provide a review to analyze.")
            else:
                with st.spinner("Analyzing linguistic metadata..."):
                    time.sleep(1) # Artificial delay for UX feel
                    prediction, probs = predict_review(review_input, model, vectorizer, lemmatizer, stop_words)
                
                # Setup visualization vars
                is_fake = (prediction == 1)
                conf_score = probs[1] if is_fake else probs[0]
                conf_pct = conf_score * 100
                
                status_class = "status-fake" if is_fake else "status-genuine"
                card_class = "fake" if is_fake else "genuine"
                icon = "⚠️" if is_fake else "✅"
                label = "FAKE REVIEW" if is_fake else "GENUINE REVIEW"
                bar_color = "linear-gradient(90deg, #F87171, #EF4444)" if is_fake else "linear-gradient(90deg, #34D399, #10B981)"

                html_card = f"""
                <div class="result-card {card_class}">
                    <div class="prediction-title">AI Prediction</div>
                    <div class="{status_class}">
                        {icon} {label}
                    </div>
                    <div style="margin-top: 2rem; display: flex; justify-content: space-between; color: #94A3B8; font-size: 1.1rem;">
                        <span>Confidence Score</span>
                        <span style="color: white; font-weight: 700;">{conf_pct:.1f}%</span>
                    </div>
                    <div class="confidence-meter-bg">
                        <div class="confidence-meter-fill" style="width: {conf_pct}%; background: {bar_color};"></div>
                    </div>
                </div>
                """
                st.markdown(html_card, unsafe_allow_html=True)
                
                    
    with tab2:
        st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.4); padding: 2rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05);'>
        <h3 style='margin-top: 0;'>Under the Hood</h3>
        <p style='color: #94A3B8; font-size: 1.1rem;'>
        <b>Review AI</b> uses a highly optimized Logistic Regression classifier combined with TF-IDF Vectorization.
        </p>
        <hr style='border-color: rgba(255,255,255,0.1); margin: 1.5rem 0;'/>
        <h4 style='color: #E2E8F0;'>NLP Pipeline Flow:</h4>
        <ul style='color: #94A3B8; font-size: 1.05rem; line-height: 1.8;'>
            <li><b>Text Cleansing:</b> Absolute lowercasing, aggressive punctuation and numeral stripping.</li>
            <li><b>Semantic Reduction:</b> Advanced NLTK Stopword filtering and contextual Lemmatization.</li>
            <li><b>Vectorization:</b> Extracting and mapping the 5,000 most significant vocabulary features via Term Frequency-Inverse Document Frequency.</li>
            <li><b>Classification:</b> High-dimensional statistical modeling predicting the probability of deception based on intricate linguistic markers over 1000 max iterations.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with tab3:
        st.markdown("<h3 style='margin-bottom: 2rem;'>Sample Reviews to Test</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
        <b style='color: #34D399; font-size: 1.1rem;'>✅ Genuine Target Sample:</b><br><br>
        <span style='color: #E2E8F0; font-style: italic;'>
        "I was skeptical at first, but this monitor exceeded my expectations. The colors are vibrant and the setup was a breeze. I had an issue with the stand but customer service shipped a replacement within two days. Definitely worth the price if you work from home."
        </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); padding: 1.5rem; border-radius: 12px;'>
        <b style='color: #F87171; font-size: 1.1rem;'>⚠️ Fake Target Sample:</b><br><br>
        <span style='color: #E2E8F0; font-style: italic;'>
        "WOW BEST PRODUCT IN THE WORLD!!! BUY IT NOW EVERYONE AMAZING AWESOME PERFECTION!!!!! I BOUGHT 10 OF THEM FOR MY WHOLE FAMILY AND I LOVE THEM ALL!!! CHEAPEST AND GREATEST ON THE MARKET DONT WAIT BUY BUY BUY!!!"
        </span>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Failed to load application modules or UI elements: {e}")
import streamlit as st
import joblib
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
        model = joblib.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = joblib.load(f)
    
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
