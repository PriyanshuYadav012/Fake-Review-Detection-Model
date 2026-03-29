# Fake Review Detection Model

A machine learning project that detects fake product reviews using Natural Language Processing (NLP) techniques and classification algorithms.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## 🎯 Overview

Fake reviews are a growing problem in e-commerce platforms, misleading consumers and damaging trust. This project aims to automatically detect fake/deceptive reviews using machine learning classification techniques.

The system takes a review text as input and classifies it as either **genuine** or **fake** based on linguistic patterns learned from training data.

## 📁 Project Structure

```
Minor Project/
├── Data Set/
│   └── kaggle.json          # Kaggle API credentials for dataset download
├── Model/
│   ├── fake_review_model.pkl    # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── Notebooks/
│   └── Minor_Project.ipynb      # Main Jupyter notebook with complete pipeline
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── research.tex                 # Research paper/documentation (LaTeX)
```

## 📊 Dataset

- **Source**: [Kaggle - Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)
- **Features**: 
  - `text_`: The review text content
  - `label`: Binary classification (genuine/fake)

The dataset is downloaded automatically using Kaggle API in the notebook.

## 🔬 Methodology

### 1. Data Preprocessing

The text preprocessing pipeline includes:
- **Lowercasing**: Converting all text to lowercase
- **Number Removal**: Removing numeric characters
- **Punctuation Removal**: Stripping all punctuation marks
- **Stopword Removal**: Removing common English stopwords using NLTK
- **Lemmatization**: Reducing words to their base form using WordNetLemmatizer

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)
```

### 2. Feature Extraction

- **TF-IDF Vectorization**: Converts text to numerical features
- `max_features=5000`: Limits vocabulary to top 5000 terms

### 3. Train-Test Split

- **Test Size**: 20%
- **Random State**: 42 (for reproducibility)

## 🤖 Models Implemented

Three classification algorithms were trained and compared:

| Model | Description |
|-------|-------------|
| **Naive Bayes** | Multinomial Naive Bayes classifier |
| **Logistic Regression** | Linear classifier with max_iter=1000 |
| **Support Vector Machine (SVM)** | SVC with default kernel |

The **Logistic Regression** model was selected as the final model and saved for deployment.

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Kaggle account (for dataset download)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Fake-Review-Detection-Model.git
   cd Fake-Review-Detection-Model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Kaggle API** (for dataset download)
   - Go to [Kaggle](https://www.kaggle.com/) → Account → Create New API Token
   - Place `kaggle.json` in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 🚀 Usage

### Using Pre-trained Model

```python
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved model and vectorizer
model = joblib.load("Model/fake_review_model.pkl")
tfidf = joblib.load("Model/tfidf_vectorizer.pkl")

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_review(review):
    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

# Example usage
review = "This product is amazing and works perfectly!"
result = predict_review(review)
print(f"Prediction: {'Genuine' if result == 'OR' else 'Fake'}")
```

### Training from Scratch

1. Open `Notebooks/Minor_Project.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. The notebook will:
   - Download the dataset from Kaggle
   - Preprocess the text data
   - Train multiple models
   - Evaluate and compare performance
   - Save the best model

## 📈 Results

The models were evaluated using accuracy score and classification metrics:

| Model | Key Metrics |
|-------|-------------|
| Naive Bayes | Baseline classifier |
| Logistic Regression | **Best performer** - saved for deployment |
| SVM | Support vector classification |

Visualization includes:
- Confusion Matrix for Logistic Regression
- Model comparison chart

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.x |
| **ML Libraries** | scikit-learn |
| **NLP** | NLTK |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | joblib |
| **Environment** | Google Colab / Jupyter Notebook |

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
joblib
kaggle
```

## 🔮 Future Improvements

- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add more text features (sentiment scores, review length, etc.)
- [ ] Create a web interface for easy predictions
- [ ] Implement real-time review classification API
- [ ] Expand dataset with more diverse review sources
- [ ] Add explainability features (LIME/SHAP)

## 👨‍💻 Author

**Priyanshu Yadav**

## 📄 License

This project is open-source and available for educational purposes.

---

⭐ If you found this project helpful, please consider giving it a star!
