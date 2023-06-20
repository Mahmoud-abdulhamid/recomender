
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data
movies_df = pd.read_csv("movies.csv")

# Create a TF-IDF matrix for the movie descriptions
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df["plot"].values)

# Calculate the cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get the recommendations for a particular movie
movie_title = "The Shawshank Redemption"
movie_index = movies_df[movies_df["title"] == movie_title].index[0]

# Get the top 10 most similar movies
recommendations = []
for i in range(10):
    similar_movie_index = cosine_sim[movie_index].argsort()[-i - 1]
    similar_movie_title = movies_df.iloc[similar_movie_index]["title"]
    recommendations.append(similar_movie_title)

print("Recommendations for " + movie_title + ":")
print(recommendations)
