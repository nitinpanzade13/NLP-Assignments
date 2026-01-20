import nltk
import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Dataset

data = {
    "text": [
        "I love Machine Learning!",
        "Machine learning is AMAZING",
        "I enjoy learning NLP",
        "NLP and machine learning are related"
    ],
    "label": ["positive", "positive", "positive", "neutral"]
}

df = pd.DataFrame(data)

print("\nORIGINAL DATA:")
print(df)

# TEXT CLEANING

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

print("\nCLEANED TEXT:")
print(df["clean_text"])

# STOPWORD REMOVAL + LEMMATIZATION

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df["processed_text"] = df["clean_text"].apply(preprocess)

print("\nPROCESSED TEXT:")
print(df["processed_text"])

# LABEL ENCODING

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

print("\nLABEL ENCODING:")
print(df[["label", "label_encoded"]])

# TF-IDF REPRESENTATION

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["processed_text"])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\nTF-IDF MATRIX:")
print(tfidf_df)

# SAVE OUTPUTS

df.to_csv("processed_text_code3.csv", index=False)
tfidf_df.to_csv("tfidf_code3.csv", index=False)

print("\nFILES SAVED SUCCESSFULLY!")