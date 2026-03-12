import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_text(text):
    text = text.lower()
  
    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'\d+', '', text)
 
    text = text.translate(str.maketrans('', '', string.punctuation))

    text = text.strip()

    return text
df = pd.read_csv("data/raw/fake_news_dataset.csv")

df["text"] = df["text"].apply(clean_text)

df.to_csv("data/processed/cleaned_news.csv", index=False)

print("Clean dataset saved to data/processed/")

def preprocess_data(df):

    df["text"] = df["text"].apply(clean_text)
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    df = load_data("data/fake_news_dataset.csv")

    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
