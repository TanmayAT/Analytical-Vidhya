from scrapping import df 
import streamlit as st
import pandas as pd
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.strip()
    return text

# Apply preprocessing
df["Title_clean"] = df["Title"].apply(preprocess_text)

# Step 2: Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["Title_clean"])

# Step 3: Word2Vec Model - Tokenize the course titles
tokenized_titles = [title.split() for title in df["Title_clean"]]
w2v_model = Word2Vec(tokenized_titles, vector_size=100, window=5, min_count=1, workers=4)

# Average Word2Vec embedding for each course title
def get_w2v_embedding(text):
    tokens = text.split()
    embeddings = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
    if len(embeddings) == 0:
        return np.zeros(100)  # return a zero vector if no words are found in the model
    return np.mean(embeddings, axis=0)

df["W2V_Embedding"] = df["Title_clean"].apply(get_w2v_embedding)

# Function to search courses
def search_courses(query, top_n=3):
    query_clean = preprocess_text(query)

    # TF-IDF: Convert query to TF-IDF vector
    query_tfidf = tfidf.transform([query_clean])

    # Word2Vec: Get query embedding
    query_w2v = get_w2v_embedding(query_clean)

    # Compute the cosine similarity for both TF-IDF and Word2Vec
    tfidf_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    w2v_similarities = cosine_similarity([query_w2v], df["W2V_Embedding"].tolist()).flatten()

    # Combine both scores
    combined_scores = 0.5 * tfidf_similarities + 0.5 * w2v_similarities

    # Get top N most similar courses
    top_indices = combined_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# Streamlit UI
st.title("Course Search System")

# Search input
query = st.text_input("Enter your search query:")

if query:
    # Perform search
    top_courses = search_courses(query, top_n=3)

    # Display results
    st.subheader("Top Matching Courses")
    for index, row in top_courses.iterrows():
        st.write(f"**Title**: {row['Title']}")
        st.write(f"[Course URL]({row['Course URL']})")
        st.image(row["Image URL"], width=200)
        st.write("-" * 30)

# Instructions or Help
st.sidebar.header("How to Use")
st.sidebar.text("""
1. Enter a query (e.g., "Learn TensorFlow" or "Data Science").
2. The system will display the most relevant courses.
3. Click on the course URL to visit the course page.
""")


