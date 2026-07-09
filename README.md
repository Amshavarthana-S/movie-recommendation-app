# 🎬 Movie Recommendation System

A modern, content-based Movie Recommendation Engine built with Python and Streamlit. This application suggests similar movies based on plot descriptions using Natural Language Processing (NLP) techniques.

## ✨ Features

- **Personalized Recommendations**: Uses TF-IDF vectorization and Cosine Similarity to find movies with similar plots.
- **Search Functionality**: Quickly search through the dataset of 4,800 movies.
- **Detailed Insights**: View movie metrics like ratings, popularity, vote counts, budget, revenue, and plot summaries.
- **Modern Dashboard UI**: A sleek, dark analytics dashboard interface built with custom CSS for Streamlit.

## 🛠️ Technical Stack

- **Language**: Python 3.x
- **Framework**: Streamlit
- **Machine Learning**: Scikit-Learn (TF-IDF, Cosine Similarity)
- **Data Manipulation**: Pandas
- **Dataset**: [TMDB 5000 Movies Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata)

## 🚀 How it Works

1. **Data Preprocessing**: Movie overviews (plots) are converted into a matrix of TF-IDF features.
2. **Similarity Scoring**: A sigmoid kernel / cosine similarity matrix is computed to measure the distance between all movie plots.
3. **Recommendation**: When a user selects a movie, the system retrieves the top 10 movies with the highest similarity scores and presents them in an elegant UI.

## 💻 Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Amshavarthana-S/movie-recommendation-app.git
   ```
2. Navigate to the project directory:
   ```bash
   cd movie-recommendation-app
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

*(Note: Data files like `sigmoid_kernel.pkl` and `movie_data.csv` must be generated or downloaded locally, as they exceed GitHub's file size limits and are intentionally ignored in `.gitignore`.)*

## 🧑‍💻 Developed By
**Amshavarthana-S**
