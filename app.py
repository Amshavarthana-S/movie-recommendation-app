import streamlit as st
import pandas as pd
import joblib
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Movie Recommender",
        layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }

    /* Header */
    .header-title {
        color: #38bdf8; /* Light blue accent */
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
    }
    
    .subheader {
        color: #94a3b8;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    /* Movie Cards */
    .dashboard-card {
        background-color: #1e293b; /* Slate 800 */
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155; /* Slate 700 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    
    /* Metrics and Data */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #2dd4bf !important; /* Teal accent for data */
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #1e293b;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #38bdf8;
        color: #0f172a;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 4px 14px 0 rgba(56, 189, 248, 0.39);
    }
    
    .stButton>button:hover {
        background-color: #7dd3fc;
        color: #0f172a;
        transform: translateY(-1px);
    }
    
    /* Info/Error Messages */
    .stAlert {
        background-color: #1e293b;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
        color: #f8fafc;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    
    hr {
        border-top-color: #334155 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_resource
def load_data():
    df = pd.read_csv("data/movie_data.csv")
    sig = joblib.load("data/sigmoid_kernel.pkl")
    return df, sig

df, sig = load_data()

# ==================== HELPER FUNCTIONS ====================
def get_movie_info(movie_title):
    """Get detailed info about a movie"""
    try:
        movie = df[df['original_title'] == movie_title].iloc[0]
        return {
            'title': movie.get('original_title', 'N/A'),
            'rating': movie.get('vote_average', 0),
            'popularity': movie.get('popularity', 0),
            'votes': movie.get('vote_count', 0),
            'overview': movie.get('overview', 'No overview'),
            'genre': movie.get('genres', 'N/A'),
            'release_date': movie.get('release_date', 'N/A'),
            'runtime': movie.get('runtime', 0),
            'budget': movie.get('budget', 0),
            'revenue': movie.get('revenue', 0)
        }
    except:
        return None

def get_recommendations(movie_title, num_recs=10):
    """Get recommendations for a movie"""
    try:
        idx = df[df['original_title'] == movie_title].index[0]
        sims = sig[idx]
        top_idx = sims.argsort()[-num_recs-1:-1][::-1]
        recommendations = df.iloc[top_idx]['original_title'].tolist()
        similarity_scores = sims[top_idx]
        return list(zip(recommendations, similarity_scores))
    except:
        return None

# ==================== HEADER ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="header-title">Movie Recommender</h1>', unsafe_allow_html=True)

st.markdown("---")
st.write("Get personalized movie recommendations based on plot similarity using TF-IDF & Cosine Similarity")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Settings")
    
    view_mode = st.radio(
        "Choose view:",
        ["Get Recommendations", "Search Movies", "Dataset Info"]
    )
    
    st.divider()
    
    if view_mode == "Get Recommendations":
        num_recs = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
    
    st.divider()
    
    st.markdown("""
        ### How it Works:
        
        1. **Select a movie** you like
        2. **Click "Get Recommendations"**
        3. See **similar movies** based on plot
        
        **Algorithm:** TF-IDF + Cosine Similarity
        
        **Total Movies:** 4,800
    """)

# ==================== MAIN CONTENT ====================

if view_mode == "Get Recommendations":
    st.markdown('<h2 class="subheader">Select a Movie & Get Recommendations</h2>', unsafe_allow_html=True)
    
    # Movie selection
    movie_list = sorted(df['original_title'].unique().tolist())
    selected_movie = st.selectbox(
        "Choose a movie you like:",
        movie_list,
        key="movie_selector"
    )
    
    # Get button
    col1, col2 = st.columns([3, 1])
    with col2:
        get_recs_button = st.button("Get Recommendations", key="get_recs")
    
    if get_recs_button:
        # Get movie info
        movie_info = get_movie_info(selected_movie)
        
        if movie_info:
            # Display selected movie info
            st.markdown(f'<h3 class="subheader">You Selected: {selected_movie}</h3>', unsafe_allow_html=True)
            
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            # Movie metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Rating", f"{movie_info['rating']:.1f}/10")
            with col2:
                st.metric("Popularity", f"{movie_info['popularity']:.0f}")
            with col3:
                st.metric("Votes", f"{int(movie_info['votes']):,}")
            with col4:
                year = movie_info['release_date'][:4] if movie_info['release_date'] else "N/A"
                st.metric("Year", year)
            with col5:
                budget_m = f"${movie_info['budget']/1e6:.1f}M" if movie_info['budget'] > 0 else "N/A"
                st.metric("Budget", budget_m)
            
            st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
            
            # Overview
            st.markdown("**Plot Summary:**")
            st.info(movie_info['overview'])
            
            # Additional info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**Genre:** {movie_info['genre']}")
            with col2:
                st.caption(f"**Runtime:** {int(movie_info['runtime'])} mins")
            with col3:
                revenue_m = f"${movie_info['revenue']/1e6:.1f}M" if movie_info['revenue'] > 0 else "N/A"
                st.caption(f"**Revenue:** {revenue_m}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Get recommendations
            recommendations = get_recommendations(selected_movie, num_recs)
            
            if recommendations:
                st.markdown(f'<h3 class="subheader">Top {len(recommendations)} Similar Movies</h3>', unsafe_allow_html=True)
                
                # Display recommendations in columns
                cols = st.columns(2)
                
                for idx, (movie, score) in enumerate(recommendations):
                    with cols[idx % 2]:
                        movie_rec_info = get_movie_info(movie)
                        
                        if movie_rec_info:
                            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                            st.markdown(f"#### {idx+1}. {movie}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rating", f"{movie_rec_info['rating']:.1f}/10")
                            with col2:
                                st.metric("Pop.", f"{movie_rec_info['popularity']:.0f}")
                            with col3:
                                st.metric("Votes", f"{int(movie_rec_info['votes']):,}")
                            
                            # Similarity score
                            st.caption(f"**Similarity:** {score*100:.1f}%")
                            
                            # Plot preview
                            plot_preview = movie_rec_info['overview']
                            if len(plot_preview) > 130:
                                plot_preview = plot_preview[:130] + "..."
                            st.caption(plot_preview)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Could not get recommendations for this movie")

elif view_mode == "Search Movies":
    st.markdown('<h2 class="subheader">Search for Movies</h2>', unsafe_allow_html=True)
    
    search_query = st.text_input(
        "Enter movie name or partial name:",
        placeholder="e.g., Avatar, Inception, Titanic..."
    )
    
    if search_query:
        # Search
        matches = df[df['original_title'].str.contains(search_query, case=False, na=False)]['original_title'].tolist()
        
        if matches:
            st.success(f"Found {len(matches)} movies matching '{search_query}'")
            
            # Display in columns
            for i, movie in enumerate(matches[:20], 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{i}. {movie}")
                with col2:
                    if st.button("Select", key=f"select_{i}"):
                        st.session_state.selected_movie = movie
        else:
            st.warning(f"No movies found matching '{search_query}'")
    else:
        st.info(f"Enter a movie name to search (Total: {len(df)} movies)")

elif view_mode == "Dataset Info":
    st.markdown('<h2 class="subheader">Dataset Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", len(df))
    with col2:
        st.metric("Avg Rating", f"{df['vote_average'].mean():.2f}")
    with col3:
        st.metric("Avg Popularity", f"{df['popularity'].mean():.1f}")
    with col4:
        st.metric("Total Votes", f"{int(df['vote_count'].sum()):,}")
    
    st.divider()
    
    # Top rated
    st.markdown("### Top 10 Rated Movies")
    top_rated = df.nlargest(10, 'vote_average')[['original_title', 'vote_average', 'vote_count']]
    
    for idx, (i, row) in enumerate(top_rated.iterrows(), 1):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{idx}. {row['original_title']}**")
        with col2:
            st.caption(f"Rating: {row['vote_average']:.1f}")
        with col3:
            st.caption(f"Votes: {int(row['vote_count']):,}")
    
    st.divider()
    
    # Most popular
    st.markdown("### Top 10 Most Popular Movies")
    most_popular = df.nlargest(10, 'popularity')[['original_title', 'popularity', 'vote_average']]
    
    for idx, (i, row) in enumerate(most_popular.iterrows(), 1):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{idx}. {row['original_title']}**")
        with col2:
            st.caption(f"Popularity: {row['popularity']:.0f}")
        with col3:
            st.caption(f"Rating: {row['vote_average']:.1f}")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
    ---
    **Technical Stack:**
    - Language: Python 3.x
    - Framework: Streamlit
    - Algorithm: TF-IDF + Cosine Similarity
    - Data: TMDB 5000 Movies Dataset
    
    **How It Works:**
    1. Movie plots converted to TF-IDF vectors
    2. Cosine similarity computed between all movies
    3. Top 10 similar movies recommended
    
    ---
    Developed by Amshavarthana-S
""")