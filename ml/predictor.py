"""
Predict which sentences from a document are worth turning into questions.

Loads the Linear SVM pipeline saved by ml/train_model.py.

Usage (standalone check):
    python ml/predictor.py "Some sentence about biology." "Page 3"
"""

import sys
from pathlib import Path

import joblib

_MODEL_PATH = Path(__file__).parent / "models" / "best_model.joblib"
_model = None


def _load_model():
    global _model
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {_MODEL_PATH}. "
                f"Run 'python ml/train_model.py' first."
            )
        _model = joblib.load(_MODEL_PATH)
    return _model


def is_question_worthy(sentence):
    """Return True if the sentence is predicted to be question-worthy."""
    sentence = (sentence or "").strip()
    if not sentence:
        return False
    model = _load_model()
    prediction = model.predict([sentence])[0]
    return int(prediction) == 1


def filter_worthy_sentences(sentences):
    """
    Keep only sentences the model predicts as question-worthy.

    Falls back to the original list if the model would filter out
    everything — this prevents a bad model from breaking question
    generation entirely.
    """
    sentences = [s for s in sentences if s and s.strip()]
    if not sentences:
        return []

    model = _load_model()
    predictions = model.predict(sentences)
    kept = [s for s, p in zip(sentences, predictions) if int(p) == 1]

    # Safety net: never return an empty list if we started with sentences.
    return kept if kept else sentences


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ml/predictor.py '<sentence 1>' '<sentence 2>' ...")
        sys.exit(1)

    for sentence in sys.argv[1:]:
        verdict = "WORTHY" if is_question_worthy(sentence) else "not worthy"
        print(f"[{verdict}]  {sentence[:100]}")