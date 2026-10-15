
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = Path("C:/Users/Janani/Desktop/Opinion_Mining_Case_Study/Reviews.csv")

if not file_path.exists():
    raise FileNotFoundError("Reviews.csv not found in the data folder.")

df = pd.read_csv(file_path)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("Total reviews:", len(df))
print("Total columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTotal duplicate rows:", df.duplicated().sum())

print("\nStar rating distribution:")
print(df["Score"].value_counts().sort_index())

print("\nEmpty review texts:", df["Text"].fillna("").str.strip().eq("").sum())
print("Empty review summaries:", df["Summary"].fillna("").str.strip().eq("").sum())

df["Text_Length"] = df["Text"].fillna("").str.len()

print("\nReview text length statistics:")
print(df["Text_Length"].describe())

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Score")
plt.title("Distribution of Amazon Review Star Ratings")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("rating_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["Text_Length"], bins=50)
plt.title("Distribution of Review Text Lengths")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("review_length_distribution.png", dpi=300)
plt.show()

print("\nEDA completed successfully.")
print("Saved rating_distribution.png")
print("Saved review_length_distribution.png")
