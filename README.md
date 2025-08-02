# 🎬 Movie Recommending System

This is a **content-based movie recommendation system** built using collaborative filtering and KNN (k-Nearest Neighbors). It uses the **MovieLens dataset** and allows users to enter a movie title to receive similar movie recommendations.

---

## 🚀 Features
- Uses collaborative filtering with pivoted user ratings.
- KNN-based similarity search (via scikit-learn).
- Sparse matrix optimization for performance.
- Streamlit web interface for movie recommendations.

---

## 🧠 How It Works

1. Loads movie and rating data from `movies.csv` and `ratings.csv`.
2. Creates a sparse matrix of ratings.
3. Filters inactive users and unpopular movies.
4. Trains a KNN model to compute similarity between movies.
5. Takes a movie input and returns 10 similar recommendations.

---

## 💻 How to Run the App Locally

## install dependencies
pip install -r requirements.txt

## Structure

Movie_Recomending_System/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── model/
│   ├── knn_model.pkl
│   └── final_dataset.csv
│
├── .streamlit/
│   └── config.toml
│
├── movie_recommender_app.py
├── requirements.txt
└── README.md


##  📜 License

This project is licensed under the MIT License.

👤 Author
Vignesh 

GitHub: VigneshEilik 


