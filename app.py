import pandas as pd
import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
novel = pd.read_csv('novels_clean')
novels = novel.drop('Unnamed: 0', axis=1)

new_column_name = 'title_id'
novels.rename(columns={'Title id': new_column_name}, inplace=True)

def clean_titles(Title):
    return re.sub("[^a-zA-Z0-9 ]", "", Title)

novels["clean_titles"] = novels["Title"].apply(clean_titles)

# Create TF-IDF vectorizer
vec = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vec.fit_transform(novels["clean_titles"])

# Load and preprocess ratings data
rating = pd.read_csv('novel_rating_clean')
ratings = rating.drop("Unnamed: 0", axis=1)
ratings.rename(columns={'Title id': 'title_id', 'User id': 'user_id', 'Title': 'title'}, inplace=True)

# Define functions for finding similar novels
def search(title):
    title = clean_titles(title)
    q_vec = vec.transform([title])
    similarity = cosine_similarity(q_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = novels.iloc[indices][::-1]
    return results

def find_similar_novels(title_id):
    similar_user = ratings[(ratings["title_id"] ==title_id) & (ratings["Rating"] > 3)]['user_id'].unique()
    similar_user_rec = ratings[(ratings["user_id"].isin(similar_user)) & (ratings["Rating"] > 3)]["title_id"]
    
    similar_user_rec = similar_user_rec.value_counts() / len(similar_user)
    similar_user_rec = similar_user_rec[similar_user_rec>0.10]
    
    all_user = ratings[(ratings["title_id"].isin(similar_user_rec.index)) & (ratings["Rating"] > 3)]
    all_user_rec = all_user["title_id"].value_counts() / len(all_user['user_id'].unique())
    
    rec_percentages = pd.concat([similar_user_rec,all_user_rec], axis=1)
    rec_percentages.columns = ['similar','all']
    
    rec_percentages['scores'] = rec_percentages['similar'] / rec_percentages['all']
    
    rec_percentages.sort_values('scores', ascending=False)
    
    return rec_percentages.head(10).merge(novels, left_index=True, right_on="title_id")[['Title','Genre','Rating','Author','Published Year']]

# Streamlit app
def main():
    st.title("Novel Recommendation App")

    # Add interactive widgets
    novels_input = st.text_input("Enter a novel title:", "Search Here")

    if len(novels_input) > 3:
        results = search(novels_input)
        if not results.empty:
            title_id = results.iloc[0]['title_id']
            similar_novels = find_similar_novels(title_id)

            st.subheader("Recommended Similar Novels:")
            st.dataframe(similar_novels)
        else:
            st.warning("No matching results found.")

if __name__ == "__main__":
    main()
