"""
Extract candidate sentences from academic PDFs for manual labelling.

Usage:
    python ml/build_candidates.py path/to/pdf1.pdf path/to/pdf2.pdf ...

Output:
    ml/data/candidates.csv with columns: sentence, label
    label is blank for you to fill in (1 = question-worthy, 0 = not).
"""

import csv
import sys
from pathlib import Path

import pymupdf
from nltk.tokenize import sent_tokenize

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "candidates.csv"

MIN_WORDS = 5
MAX_WORDS = 60


def extract_sentences_from_pdf(pdf_path):
    sentences = []
    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            text = page.get_text().strip()
            if not text:
                continue
            for sentence in sent_tokenize(text):
                sentence = " ".join(sentence.split())
                word_count = len(sentence.split())
                if MIN_WORDS <= word_count <= MAX_WORDS:
                    sentences.append(sentence)
    return sentences


def main(pdf_paths):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_sentences = []
    seen = set()
    for path in pdf_paths:
        print(f"Reading {path} ...")
        for sentence in extract_sentences_from_pdf(path):
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            all_sentences.append(sentence)

    # Deterministic order so re-runs are stable, but no shuffle so you
    # can label in batches.
    all_sentences.sort()

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "label"])
        for sentence in all_sentences:
            writer.writerow([sentence, ""])

    print(f"\nWrote {len(all_sentences)} candidate sentences to {OUTPUT_FILE}")
    print("Open it in a spreadsheet and fill the 'label' column with 1 or 0.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ml/build_candidates.py <pdf1> [<pdf2> ...]")
        sys.exit(1)
    main(sys.argv[1:])