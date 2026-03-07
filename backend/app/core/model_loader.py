"""
backend/app/core/model_loader.py

Loads the trained TF-IDF vectorizer and Logistic Regression model
once at application startup, then reuses them across requests.
"""

import logging
from pathlib import Path

import joblib
import numpy as np

from backend.app.core.config import settings

log = logging.getLogger(__name__)

# Module-level singletons
_vectorizer = None
_model = None


def load_model():
    global _vectorizer, _model

    vec_path = Path(settings.VECTORIZER_PATH)
    model_path = Path(settings.MODEL_PATH)

    if not vec_path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at: {vec_path}\n"
            "Run: python scripts/train_model.py"
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Run: python scripts/train_model.py"
        )

    log.info(f"Loading vectorizer from {vec_path}")
    _vectorizer = joblib.load(vec_path)

    log.info(f"Loading model from {model_path}")
    _model = joblib.load(model_path)

    log.info("Model and vectorizer loaded successfully.")


def predict(text: str) -> dict:
    """
    Run inference on a single news article text.

    Returns:
        {
            "prediction": "Fake" | "Real",
            "confidence": float (0.0 - 1.0),
            "label": 0 | 1
        }
    """
    if _vectorizer is None or _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    X = _vectorizer.transform([text])
    label = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    return {
        "prediction": "Real" if label == 1 else "Fake",
        "confidence": round(confidence, 4),
        "label": label,
    }
