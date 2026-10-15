
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

input_path = Path("data/cleaned_reviews.csv")
output_path = Path("data")

if not input_path.exists():
    raise FileNotFoundError("Run preprocessing.py first.")

df = pd.read_csv(input_path)
texts = df["Cleaned_Text"].fillna("").astype(str)

vectorizer = CountVectorizer(max_features=5000)
bow_matrix = vectorizer.fit_transform(texts)

terms = vectorizer.get_feature_names_out()
word_counts = bow_matrix.sum(axis=0).A1

frequency_df = pd.DataFrame({
    "Word": terms,
    "Frequency": word_counts
}).sort_values("Frequency", ascending=False)

frequency_df.to_csv(
    output_path / "bow_word_frequencies.csv",
    index=False
)

print("BAG-OF-WORDS ANALYSIS")
print("Number of reviews:", len(texts))
print("Vocabulary size:", len(terms))
print("BoW matrix shape:", bow_matrix.shape)
print("\nTop 20 most frequent words:")
print(frequency_df.head(20).to_string(index=False))

plt.figure(figsize=(12, 6))
top_words = frequency_df.head(20).sort_values("Frequency")
plt.barh(top_words["Word"], top_words["Frequency"])
plt.title("Top 20 Most Frequent Words in Amazon Reviews")
plt.xlabel("Word Frequency")
plt.ylabel("Word")
plt.tight_layout()
plt.savefig("bow_top_words.png", dpi=300)
plt.show()

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate_from_frequencies(
    dict(zip(frequency_df["Word"], frequency_df["Frequency"]))
)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Bag-of-Words Word Cloud")
plt.tight_layout()
plt.savefig("bow_wordcloud.png", dpi=300)
plt.show()

print("\nSaved:")
print("data/bow_word_frequencies.csv")
print("bow_top_words.png")
print("bow_wordcloud.png")
print("\nBag-of-Words analysis completed.")
