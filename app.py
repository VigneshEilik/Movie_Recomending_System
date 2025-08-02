import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
import os

st.set_page_config(page_title="Movie Recommender", layout="wide")

def check_login():
    def auth():
        if (st.session_state["username"] == "client" and
            st.session_state["password"] == "movie"):
            st.session_state["authenticated"] = True
            del st.session_state["username"]
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input("👤 Username", key="username")
        st.text_input("🔐 Password", type="password", key="password")
        st.button("Login", on_click=auth)
        st.stop()
    elif not st.session_state["authenticated"]:
        st.text_input("👤 Username", key="username")
        st.text_input("🔐 Password", type="password", key="password")
        st.button("Login", on_click=auth)
        st.error("❌ Invalid username or password")
        st.stop()

check_login()

st.title("🎬 Movie Recommendation System")

# Show full traceback if error
st.sidebar.subheader("📋 Debug Info")

# Load data
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv("data/movies.csv")
        ratings = pd.read_csv("data/ratings.csv")
        final_dataset = pd.read_csv("model/final_dataset.csv")
        knn = joblib.load("model/knn_model.pkl")
        return movies, ratings, final_dataset, knn
    except Exception as e:
        st.sidebar.error(f"Data Load Error: {e}")
        raise e

try:
    movies, ratings, final_dataset, knn = load_data()
    csr_data = csr_matrix(final_dataset.drop(columns='movieId').values)
    st.table(movies.head(10))  # Display first 5 rows of final_dataset for debugging
except Exception as e:
    st.stop()
# Recommendation Logic
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 20
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, na=False)]

    if len(movie_list):
        movie_id = movie_list.iloc[0]['movieId']
        try:
            movie_idx = final_dataset[final_dataset['movieId'] == movie_id].index[0]
        except IndexError:
            return "Movie not found in filtered dataset."

        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend+1)
        rec_movie_indices = sorted(
            list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
            key=lambda x: x[1]
        )[1:]

        recommend_frame = []
        for val in rec_movie_indices:
            rec_movie_id = final_dataset.iloc[val[0]]['movieId']
            title = movies[movies['movieId'] == rec_movie_id]['title'].values[0]
            recommend_frame.append({'Title': title, 'Distance': round(val[1], 4)})

        return pd.DataFrame(recommend_frame)
    else:
        return "No movies found. Please check your input."

# UI
movie_input = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if not movie_input.strip():
        st.warning("Please enter a movie name.")
    else:
        result = get_movie_recommendation(movie_input)
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        else:
            st.error(result)
