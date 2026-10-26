"""
Train and compare Logistic Regression, Linear SVM and Multinomial Naive Bayes
on the manually labelled sentence dataset.

Usage:
    python ml/train_model.py

Inputs:
    ml/data/labelled.csv   (columns: sentence, label)

Outputs:
    ml/data/model_comparison.csv
    ml/models/best_model.joblib
    ml/models/model_metadata.json
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "labelled.csv"
MODELS_DIR = BASE_DIR / "models"
COMPARISON_FILE = BASE_DIR / "data" / "model_comparison.csv"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

RANDOM_SEED = 42


def load_dataset(path):
    sentences = []
    labels = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            label = (row.get("label") or "").strip()
            if not sentence or label not in {"0", "1"}:
                continue
            sentences.append(sentence)
            labels.append(int(label))
    return sentences, np.array(labels)


def build_pipelines():
    """Each pipeline = TF-IDF -> classifier, so inference is one call."""
    vectorizer_args = dict(
        stop_words="english",
        ngram_range=(1, 2),      # single words + bigrams
        min_df=1,
        max_features=5000,
    )

    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_args)),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
            )),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_args)),
            ("clf", LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_args)),
            ("clf", MultinomialNB()),
        ]),
    }


def evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision_macro": round(
            precision_score(y_test, preds, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_test, preds, average="macro", zero_division=0), 4
        ),
        "f1_macro": round(
            f1_score(y_test, preds, average="macro", zero_division=0), 4
        ),
        "f1_weighted": round(
            f1_score(y_test, preds, average="weighted", zero_division=0), 4
        ),
        "_report": classification_report(
            y_test, preds,
            target_names=["not worthy (0)", "worthy (1)"],
            digits=4,
            zero_division=0,
        ),
    }


def main():
    print("Loading dataset...")
    sentences, labels = load_dataset(DATA_FILE)
    n_total = len(sentences)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"  Total sentences: {n_total}")
    print(f"  Positives: {n_pos} ({n_pos / n_total:.1%})")
    print(f"  Negatives: {n_neg} ({n_neg / n_total:.1%})")

    if n_total < 100:
        print("\nWARNING: Very small dataset — metrics will be noisy.")

    # 70 / 15 / 15 stratified split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        sentences, labels,
        test_size=0.15,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.1765,   # ≈ 15% of total
        stratify=y_trainval,
        random_state=RANDOM_SEED,
    )

    print("\nSplit sizes:")
    print(f"  Train:      {len(X_train)}  (pos={int((y_train==1).sum())})")
    print(f"  Validation: {len(X_val)}    (pos={int((y_val==1).sum())})")
    print(f"  Test:       {len(X_test)}    (pos={int((y_test==1).sum())})")

    pipelines = build_pipelines()
    results = []

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATION")
    print("=" * 70)

    for name, pipeline in pipelines.items():
        print(f"\n--- {name} ---")
        result = evaluate(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(result)

        print(f"  Accuracy:    {result['accuracy']}")
        print(f"  Precision:   {result['precision_macro']}  (macro)")
        print(f"  Recall:      {result['recall_macro']}     (macro)")
        print(f"  F1 (macro):  {result['f1_macro']}")
        print(f"  F1 (weighted): {result['f1_weighted']}")
        print("\n  Classification report on test set:")
        for line in result["_report"].splitlines():
            print("    " + line)

    # Save comparison table (without the verbose report column)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted",
    ]
    with COMPARISON_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})

    # Pick best by macro-F1 (correct metric for imbalanced data)
    best = max(results, key=lambda r: r["f1_macro"])
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best['model']}  (macro-F1 = {best['f1_macro']})")
    print("=" * 70)

    best_pipeline = pipelines[best["model"]]
    # Retrain on train+val for the final saved model (more data at deploy)
    best_pipeline.fit(X_trainval, y_trainval)

    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)

    metadata = {
        "best_model_name": best["model"],
        "selection_metric": "macro_f1_on_test_set",
        "dataset": {
            "total": n_total,
            "positives": n_pos,
            "negatives": n_neg,
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "results_on_test_set": [
            {k: r[k] for k in fields} for r in results
        ],
        "saved_at": str(__import__("datetime").datetime.now().isoformat()),
    }
    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model to:      {best_model_path}")
    print(f"Saved comparison table:   {COMPARISON_FILE}")
    print(f"Saved metadata:           {METADATA_FILE}")


if __name__ == "__main__":
    main()
    