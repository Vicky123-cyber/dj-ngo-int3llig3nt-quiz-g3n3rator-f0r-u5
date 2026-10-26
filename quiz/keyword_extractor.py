
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text, top_n=10):
    """Extract the highest-scoring terms using TF-IDF."""

    if not text or not text.strip():
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    try:
        matrix = vectorizer.fit_transform([text])
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]

    ranked_indices = scores.argsort()[::-1]

    keywords = []

    for index in ranked_indices:
        if scores[index] <= 0:
            break

        keywords.append({
            "word": feature_names[index],
            "score": round(float(scores[index]), 4),
        })

        if len(keywords) >= top_n:
            break

    return keywords
