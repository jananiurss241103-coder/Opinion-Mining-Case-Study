
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

input_path = Path("data/cleaned_reviews.csv")

if not input_path.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run preprocessing.py first."
    )

df = pd.read_csv(input_path)
texts = df["Cleaned_Text"].fillna("").astype(str)

vectorizer = TfidfVectorizer(
    max_features=5000,
    max_df=0.95,
    min_df=2,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(texts)

svd = TruncatedSVD(
    n_components=2,
    random_state=42,
    n_iter=7
)

reduced_data = svd.fit_transform(tfidf_matrix)

df["SVD_Component_1"] = reduced_data[:, 0]
df["SVD_Component_2"] = reduced_data[:, 1]

print("DIMENSIONALITY REDUCTION")
print("Original TF-IDF shape:", tfidf_matrix.shape)
print("Reduced data shape:", reduced_data.shape)
print(
    "Explained variance ratio:",
    svd.explained_variance_ratio_
)
print(
    "Total explained variance:",
    svd.explained_variance_ratio_.sum()
)

df.to_csv(
    "data/reviews_svd_coordinates.csv",
    index=False
)

plt.figure(figsize=(10, 7))

plt.scatter(
    reduced_data[:, 0],
    reduced_data[:, 1],
    s=10,
    alpha=0.5
)

plt.title("Amazon Reviews: TF-IDF Dimensionality Reduction using SVD")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.tight_layout()
plt.savefig("svd_review_scatter.png", dpi=300)
plt.show()

print("\nSaved: data/reviews_svd_coordinates.csv")
print("Saved: svd_review_scatter.png")
print("Dimensionality reduction completed.")
