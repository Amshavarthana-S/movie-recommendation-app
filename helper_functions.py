import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_resource
def load_models():
    dataframe = pd.read_csv(os.path.join(DATA_DIR, "movie_data.csv"))
    data = dataframe[dataframe['overview'].notna()].copy()
    data = data.reset_index(drop=True)
    
    tfv = joblib.load(os.path.join(DATA_DIR, "tfidf_vectorizer.pkl"))
    sig = joblib.load(os.path.join(DATA_DIR, "sigmoid_kernel.pkl"))
    
    return data, dataframe, tfv, sig

def give_recommendations(movie_title, data, dataframe, sig, num_recommendations=10):
    try:
        # Find movie in data
        mask = data['original_title'] == movie_title
        if not mask.any():
            return None
        
        idx = data[mask].index[0]
        
        # Get similarities
        similarities = sig[idx]
        
        # Get top N indices (skip first = itself)
        top_indices = similarities.argsort()[-num_recommendations-1:-1][::-1]
        
        # Get movie names
        recommendations = data['original_title'].iloc[top_indices].tolist()
        
        return recommendations
    except:
        return None

def get_all_movies(data):
    return sorted(data['original_title'].values.tolist())

def get_movie_info(movie_title, dataframe):
    try:
        movie_data = dataframe[dataframe['original_title'] == movie_title].iloc[0]
        return {
            'title': movie_data.get('original_title', 'N/A'),
            'rating': movie_data.get('vote_average', 0),
            'popularity': movie_data.get('popularity', 0),
            'votes': movie_data.get('vote_count', 0),
            'overview': movie_data.get('overview', 'No overview'),
            'genre': movie_data.get('genres', 'N/A'),
            'release_date': movie_data.get('release_date', 'N/A'),
            'runtime': movie_data.get('runtime', 0),
            'budget': movie_data.get('budget', 0)
        }
    except:
        return None