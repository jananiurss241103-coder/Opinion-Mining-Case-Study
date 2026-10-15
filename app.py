
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="Amazon Review Opinion Mining",
    page_icon="📊",
    layout="wide"
)

st.title("Amazon Product Review Opinion Mining")
st.caption(
    "Text preprocessing, Bag of Words, TF-IDF, SVD, "
    "sentiment classification, and tag categorization"
)


@st.cache_data
def load_data():
    files = {
        "reviews": "cleaned_reviews.csv",
        "bow": "bow_word_frequencies.csv",
        "tfidf": "tfidf_word_scores.csv",
        "svd": "reviews_svd_coordinates.csv",
        "sentiment": "sentiment_results.csv",
        "tags": "ranked_tags.csv",
        "keywords": "refined_keywords.csv",
    }

    result = {}

    for key, filename in files.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required file is missing: {path}"
            )
        result[key] = pd.read_csv(path)

    return result


try:
    data = load_data()
except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.stop()

reviews = data["sentiment"].copy()
bow = data["bow"]
tfidf = data["tfidf"]
svd = data["svd"]
tags = data["tags"]
keywords = data["keywords"]

reviews["Text"] = reviews["Text"].fillna("").astype(str)
reviews["Cleaned_Text"] = (
    reviews["Cleaned_Text"].fillna("").astype(str)
)

if "Sentiment" not in reviews.columns:
    st.error("Sentiment column is missing.")
    st.stop()

if "Score" not in reviews.columns:
    st.error("Score column is missing.")
    st.stop()

total_reviews = len(reviews)
positive = int((reviews["Sentiment"] == "Positive").sum())
negative = int((reviews["Sentiment"] == "Negative").sum())
neutral = int((reviews["Sentiment"] == "Neutral").sum())

tab_overview, tab_text, tab_bow, tab_tfidf, tab_svd, tab_sentiment, tab_tags, tab_explorer = st.tabs([
    "Overview",
    "Text Processing",
    "Bag of Words",
    "TF-IDF",
    "Dimensionality Reduction",
    "Sentiment",
    "Tags",
    "Review Explorer"
])

with tab_overview:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{total_reviews:,}")
    c2.metric("Positive", f"{positive:,}")
    c3.metric("Negative", f"{negative:,}")
    c4.metric("Neutral", f"{neutral:,}")

    st.info(
        "The current analysis uses the first 10,000 records "
        "from the source dataset. Sentiment labels are derived "
        "from star ratings: 4–5 positive, 1–2 negative, "
        "and 3 neutral."
    )

    rating_counts = (
        reviews["Score"].value_counts()
        .sort_index()
        .rename_axis("Star Rating")
        .reset_index(name="Reviews")
    )

    fig = px.bar(
        rating_counts,
        x="Star Rating",
        y="Reviews",
        title="Star Rating Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Sample records")
    st.dataframe(
        reviews[["Score", "Summary", "Text", "Sentiment"]].head(10),
        use_container_width=True
    )

with tab_text:
    st.subheader("Text Preprocessing")
    st.write(
        "The pipeline converts text to lowercase, removes HTML "
        "artifacts and punctuation, removes English stop words, "
        "and applies lemmatization."
    )

    st.dataframe(
        reviews[["Text", "Cleaned_Text"]].head(20),
        use_container_width=True
    )

with tab_bow:
    st.subheader("Bag-of-Words Analysis")

    top_n = st.slider(
        "Number of frequent terms",
        min_value=5,
        max_value=30,
        value=20,
        key="bow_top_n"
    )

    top_bow = bow.sort_values(
        "Frequency", ascending=False
    ).head(top_n).sort_values("Frequency")

    fig = px.bar(
        top_bow,
        x="Frequency",
        y="Word",
        orientation="h",
        title=f"Top {top_n} Word Frequencies"
    )
    st.plotly_chart(fig, use_container_width=True)

    image_path = BASE_DIR / "bow_wordcloud.png"
    if image_path.exists():
        st.image(str(image_path), caption="Bag-of-Words Word Cloud")

    st.dataframe(bow.head(top_n), use_container_width=True)

with tab_tfidf:
    st.subheader("TF-IDF Feature Extraction")

    top_n_tfidf = st.slider(
        "Number of TF-IDF terms",
        min_value=5,
        max_value=30,
        value=20,
        key="tfidf_top_n"
    )

    top_tfidf = tfidf.sort_values(
        "Mean_TFIDF", ascending=False
    ).head(top_n_tfidf).sort_values("Mean_TFIDF")

    fig = px.bar(
        top_tfidf,
        x="Mean_TFIDF",
        y="Word",
        orientation="h",
        title="Terms Ranked by Mean TF-IDF"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        tfidf.sort_values("Mean_TFIDF", ascending=False).head(top_n_tfidf),
        use_container_width=True
    )

with tab_svd:
    st.subheader("Dimensionality Reduction using SVD")

    required = {"SVD_Component_1", "SVD_Component_2"}

    if required.issubset(svd.columns):
        plot_data = svd.copy()

        if "Sentiment" in plot_data.columns:
            color_column = "Sentiment"
        elif "Score" in plot_data.columns:
            color_column = "Score"
        else:
            color_column = None

        fig = px.scatter(
            plot_data,
            x="SVD_Component_1",
            y="SVD_Component_2",
            color=color_column,
            title="Review Distribution in Two SVD Components",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "The two-dimensional projection retains only a small "
            "fraction of the total variance in this run. "
            "Interpret it as a visualization, not a complete "
            "representation of review meaning."
        )
    else:
        st.error("Expected SVD coordinate columns were not found.")

with tab_sentiment:
    st.subheader("Sentiment Classification Results")

    sentiment_counts = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Reviews": [positive, negative, neutral]
    })

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Reviews",
        title="Rating-Derived Sentiment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    sentiment_counts["Percentage"] = (
        sentiment_counts["Reviews"] / total_reviews * 100
    ).round(2)

    st.dataframe(sentiment_counts, use_container_width=True)

    st.caption(
        "These labels are derived from star ratings, not from "
        "an independently trained text sentiment model."
    )

with tab_tags:
    st.subheader("Ranked, Refined and Enriched Tags")

    tag_count_column = "Review_Count"

    if {"Tag", tag_count_column}.issubset(tags.columns):
        ranked_tags = tags.sort_values(
            tag_count_column, ascending=False
        )

        fig = px.bar(
            ranked_tags.sort_values(tag_count_column),
            x=tag_count_column,
            y="Tag",
            orientation="h",
            title="Tags Ranked by Number of Reviews"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(ranked_tags, use_container_width=True)
    else:
        st.error("Expected tag columns were not found.")

    cloud_path = BASE_DIR / "ranked_tag_cloud.png"
    if cloud_path.exists():
        st.image(str(cloud_path), caption="Ranked Tag Cloud")

    st.subheader("Refined Keywords")
    st.dataframe(keywords.head(50), use_container_width=True)

with tab_explorer:
    st.subheader("Search and Filter Reviews")

    selected_sentiment = st.selectbox(
        "Filter by sentiment",
        ["All", "Positive", "Negative", "Neutral"]
    )

    query = st.text_input(
        "Search review text",
        placeholder="Enter a word or phrase"
    )

    filtered = reviews.copy()

    if selected_sentiment != "All":
        filtered = filtered[
            filtered["Sentiment"] == selected_sentiment
        ]

    if query.strip():
        filtered = filtered[
            filtered["Text"].str.contains(
                query.strip(),
                case=False,
                regex=False,
                na=False
            )
        ]

    st.write(f"Matching reviews: {len(filtered):,}")

    st.dataframe(
        filtered[["Score", "Sentiment", "Summary", "Text", "Cleaned_Text"]]
        .head(200),
        use_container_width=True
    )

st.divider()
st.caption(
    "Case Study: Opinion Mining and Tag-Based Categorization "
    "of Amazon Fine Food Reviews"
)
