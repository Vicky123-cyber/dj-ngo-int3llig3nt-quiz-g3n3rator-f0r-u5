
import re

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def clean_text(text):
    """Clean unnecessary characters and whitespace."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def process_text(text):
    """Split text into sentences and process useful tokens."""
    cleaned_text = clean_text(text)

    sentences = sent_tokenize(cleaned_text)

    stop_words = set(stopwords.words("english"))
    processed_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)

        useful_words = [
            word.lower()
            for word in words
            if word.isalpha()
            and word.lower() not in stop_words
        ]

        if useful_words:
            processed_sentences.append({
                "original": sentence,
                "tokens": useful_words,
            })

    return {
        "cleaned_text": cleaned_text,
        "sentences": sentences,
        "processed_sentences": processed_sentences,
    }
