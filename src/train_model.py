import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


print("Loading datasets...")

# Load datasets
fake1 = pd.read_csv(
    "data/processed/cleaned_fake.csv",
    encoding="latin1",
    on_bad_lines="skip"
)

fake2 = pd.read_csv(
    "data/processed/cleaned_fake_news.csv",
    encoding="latin1",
    on_bad_lines="skip"
)

true1 = pd.read_csv(
    "data/processed/cleaned_true.csv",
    encoding="latin1",
    on_bad_lines="skip"
)


print("Fake1 shape:", fake1.shape)
print("Fake2 shape:", fake2.shape)
print("True shape:", true1.shape)


# Combine datasets
df = pd.concat([fake1, fake2, true1], ignore_index=True)

print("Combined dataset shape:", df.shape)


# Keep only required columns
df = df[["text", "label"]]

# Remove empty rows
df = df.dropna()

print("Dataset after cleaning:", df.shape)


# Lowercase text
df["text"] = df["text"].str.lower()


# Features and labels
X = df["text"]
y = df["label"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)


# Train model
print("Training model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)


# Save model
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model trained successfully!")
