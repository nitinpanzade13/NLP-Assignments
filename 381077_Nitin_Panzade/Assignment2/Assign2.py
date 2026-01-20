import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Dataset

documents = [
    "machine learning is fun",
    "machine learning is powerful",
    "learning NLP is interesting",
    "machine and NLP are related"
]

print("\nORIGINAL DOCUMENTS:")
for doc in documents:
    print("-", doc)

# BAG OF WORDS (COUNT)

count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(documents)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=count_vectorizer.get_feature_names_out()
)

print("\nBAG OF WORDS – COUNT OCCURRENCE:")
print(bow_df)

# NORMALIZED BAG OF WORDS

bow_normalized = bow_df.div(bow_df.sum(axis=1), axis=0)

print("\nNORMALIZED BAG OF WORDS:")
print(bow_normalized)

# TF-IDF

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\nTF-IDF MATRIX:")
print(tfidf_df)

# WORD2VEC EMBEDDINGS

tokenized_docs = [doc.split() for doc in documents]

word2vec_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=50,
    window=3,
    min_count=1,
    workers=4
)

print("\nWORD2VEC VECTOR FOR WORD 'machine':")
print(word2vec_model.wv["machine"])
