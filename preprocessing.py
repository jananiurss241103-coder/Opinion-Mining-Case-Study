
from pathlib import Path
from html import unescape
import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

input_path = Path("C:/Users/Janani/Desktop/Opinion_Mining_Case_Study/Reviews.csv")
output_path = Path("data/cleaned_reviews.csv")

if not input_path.exists():
    raise FileNotFoundError(
        "Reviews.csv was not found inside the data folder."
    )

df = pd.read_csv(
    input_path,
    usecols=["Id", "Score", "Summary", "Text"],
    nrows=10000
)

df["Text"] = df["Text"].fillna("").astype(str)
df["Summary"] = df["Summary"].fillna("").astype(str)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = unescape(text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\bbr\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(cleaned_tokens)


df["Cleaned_Text"] = df["Text"].apply(preprocess_text)

df = df[df["Cleaned_Text"].str.strip().ne("")].copy()

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("TEXT PREPROCESSING COMPLETED")
print("Reviews processed:", len(df))
print("Output saved to:", output_path)

print("\nOriginal review:")
print(df["Text"].iloc[0])

print("\nCleaned review:")
print(df["Cleaned_Text"].iloc[0])

print("\nFirst five cleaned reviews:")
print(df[["Score", "Cleaned_Text"]].head().to_string(index=False))

br_count = df["Cleaned_Text"].str.findall(
    r"\bbr\b", flags=re.IGNORECASE
).str.len().sum()

print("\nRemaining standalone 'br' tokens:", br_count)
