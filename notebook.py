# Proyek sistem rekomendasi film Netflix menggunakan Content-Based Filtering
# Dataset: netflix_titles.csv
# Dibuat untuk memenuhi submission Dicoding

# ==================
# Import Library
# ==================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

# Download stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ==================
# Data Understanding
# ==================

# Load dataset
# Ganti path sesuai lokasi dataset Anda
df = pd.read_csv('/content/drive/MyDrive/netflix_titles.csv')

# Display basic info
print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# ==================
# Data Preparation
# ==================

# Handle missing values
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
df['date_added'] = df['date_added'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Unknown')
df['duration'] = df['duration'].fillna('Unknown')

# Verify no missing values remain
print("\nMissing Values After Handling:")
print(df.isnull().sum())

# Combine description and listed_in for richer features
df['combined_features'] = df['description'] + ' ' + df['listed_in']

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing
df['combined_features'] = df['combined_features'].apply(preprocess_text)

# ==================
# Modeling
# ==================

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim, df=df):
    try:
        idx = df[df['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Top 10 recommendations
        movie_indices = [i[0] for i in sim_scores]
        return df[['title', 'listed_in', 'description']].iloc[movie_indices]
    except:
        return "Title not found in dataset."

# ==================
# Evaluation
# ==================

# Test recommendations
print("\nRecommendations for 'Squid Game':")
print(get_recommendations('Squid Game'))

print("\nRecommendations for 'Stranger Things':")
print(get_recommendations('Stranger Things'))

# Additional Experiment: Using only description
tfidf_desc = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix_desc = tfidf_desc.fit_transform(df['description'])
cosine_sim_desc = cosine_similarity(tfidf_matrix_desc, tfidf_matrix_desc)

print("\nRecommendations for 'Squid Game' (using only description):")
def get_recommendations_desc(title, cosine_sim=cosine_sim_desc, df=df):
    try:
        idx = df[df['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return df[['title', 'listed_in', 'description']].iloc[mobile_indices]
    except:
        return "Title not found in dataset."

print(get_recommendations_desc('Squid Game'))

# ==================
# Testing
# ==================

# Test additional titles
print("\nRecommendations for 'The Queen's Gambit':")
print(get_recommendations("The Queen's Gambit"))

print("\nRecommendations for 'Breaking Bad':")
print(get_recommendations('Breaking Bad'))