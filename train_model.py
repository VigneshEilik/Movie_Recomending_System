# train_model.py
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Pivot ratings
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

# Filter movies and users
no_user_voted = ratings.groupby('movieId')['rating'].count()
no_movies_voted = ratings.groupby('userId')['rating'].count()

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 30].index]

# Reset index & convert to sparse matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn.fit(csr_data)

# Save model & processed data
os.makedirs("model", exist_ok=True)
joblib.dump(knn, "model/knn_model.pkl")
final_dataset.to_csv("model/final_dataset.csv", index=False)
