import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore


user_data = {
    'User': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
    'Movie1': [5, 4, 2, 4, 3],
    'Movie2': [4, 5, 1, 4, 2],
    'Movie3': [3, 3, 5, 1, 4],
    'Movie4': [2, 1, 4, 5, 5],
    'Movie5': [5, 5, 5, 2, 3]
}

movie_data = {
    'Movie': ['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5'],
    'Genre': ['Action, Comedy', 'Action, Drama', 'Drama, Romance', 'Comedy, Drama', 'Action, Thriller']
}

user_df = pd.DataFrame(user_data).set_index('User')
movie_df = pd.DataFrame(movie_data)

def collaborative_filtering(user_df, target_user):
    cosine_sim = cosine_similarity(user_df)
    similarity_df = pd.DataFrame(cosine_sim, index=user_df.index, columns=user_df.index)
    similar_users = similarity_df[target_user].sort_values(ascending=False)[1:].index  
    recommendations = {}

    
    for similar_user in similar_users:
        for movie, rating in user_df.loc[similar_user].items():
            if pd.isna(user_df.loc[target_user, movie]) and rating >= 4:
                recommendations[movie] = recommendations.get(movie, 0) + 1

  
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations


def content_based_filtering(movie_df, target_movie):
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(movie_df['Genre'])
    cosine_sim_content = cosine_similarity(genre_matrix)
    similarity_df_content = pd.DataFrame(cosine_sim_content, index=movie_df['Movie'], columns=movie_df['Movie'])
    similarity_scores = similarity_df_content[target_movie].sort_values(ascending=False)[1:] 
    recommendations = similarity_scores.index[:3] 
    return recommendations

def recommend_movies(user_df, movie_df, target_user, target_movie):
    print(f"\nCollaborative Filtering Recommendations for {target_user}:")
    collaborative_recommendations = collaborative_filtering(user_df, target_user)
    for movie, _ in collaborative_recommendations[:3]:  
        print(f" - {movie}")

    print(f"\nContent-Based Filtering Recommendations for {target_movie}:")
    content_recommendations = content_based_filtering(movie_df, target_movie)
    for movie in content_recommendations:
        print(f" - {movie}")


recommend_movies(user_df, movie_df, target_user='Alice', target_movie='Movie1')
