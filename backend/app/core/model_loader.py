"""
backend/app/core/model_loader.py

Loads the active model at startup and exposes a unified predict() interface.
Supports two backends:
  - "tfidf"      : TF-IDF + Logistic Regression (Phase 2)
  - "distilbert" : Fine-tuned DistilBERT (Phase 6)

Set ACTIVE_MODEL in .env to switch between them.
"""

import logging
from pathlib import Path

import joblib
import numpy as np

from backend.app.core.config import settings

log = logging.getLogger(__name__)

# Module-level singletons
_vectorizer   = None    # TF-IDF only
_tfidf_model  = None    # TF-IDF only
_bert_model   = None    # DistilBERT only
_bert_tokenizer = None  # DistilBERT only
_active_model = None    # "tfidf" or "distilbert"


def load_model():
    global _vectorizer, _tfidf_model, _bert_model, _bert_tokenizer, _active_model

    _active_model = settings.ACTIVE_MODEL.lower().strip()
    log.info(f"Active model: {_active_model}")

    if _active_model == "distilbert":
        _load_distilbert()
    else:
        _load_tfidf()


def _load_tfidf():
    global _vectorizer, _tfidf_model

    vec_path   = Path(settings.VECTORIZER_PATH)
    model_path = Path(settings.MODEL_PATH)

    if not vec_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vec_path}\nRun: python scripts/train_model.py")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nRun: python scripts/train_model.py")

    log.info(f"Loading TF-IDF vectorizer from {vec_path}")
    _vectorizer = joblib.load(vec_path)

    log.info(f"Loading Logistic Regression model from {model_path}")
    _tfidf_model = joblib.load(model_path)

    log.info("TF-IDF model loaded successfully.")


def _load_distilbert():
    global _bert_model, _bert_tokenizer

    model_dir = Path(settings.DISTILBERT_MODEL_PATH)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"DistilBERT model not found: {model_dir}\n"
            "Run: python scripts/train_distilbert.py"
        )

    try:
        import torch
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )
    except ImportError:
        raise ImportError(
            "transformers/torch not installed.\n"
            "Run: pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "Then: pip install transformers accelerate"
        )

    log.info(f"Loading DistilBERT tokenizer from {model_dir}")
    _bert_tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))

    log.info(f"Loading DistilBERT model from {model_dir}")
    _bert_model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _bert_model.to(device)
    _bert_model.eval()

    log.info(f"DistilBERT model loaded on {device.upper()}.")


def predict(text: str) -> dict:
    """
    Unified prediction interface. Works for both TF-IDF and DistilBERT.

    Returns:
        {
            "prediction": "Fake" | "Real",
            "confidence": float,
            "label": 0 | 1,
            "model": "tfidf" | "distilbert"
        }
    """
    if _active_model == "distilbert":
        return _predict_distilbert(text)
    else:
        return _predict_tfidf(text)


def _predict_tfidf(text: str) -> dict:
    if _vectorizer is None or _tfidf_model is None:
        raise RuntimeError("TF-IDF model not loaded. Call load_model() first.")

    X     = _vectorizer.transform([text])
    label = int(_tfidf_model.predict(X)[0])
    proba = _tfidf_model.predict_proba(X)[0]

    return {
        "prediction": "Real" if label == 1 else "Fake",
        "confidence": round(float(np.max(proba)), 4),
        "label":      label,
        "model":      "tfidf",
    }


def _predict_distilbert(text: str) -> dict:
    if _bert_model is None or _bert_tokenizer is None:
        raise RuntimeError("DistilBERT model not loaded. Call load_model() first.")

    import torch

    device = next(_bert_model.parameters()).device

    inputs = _bert_tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _bert_model(**inputs)
        proba   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        label   = int(np.argmax(proba))

    return {
        "prediction": "Real" if label == 1 else "Fake",
        "confidence": round(float(np.max(proba)), 4),
        "label":      label,
        "model":      "distilbert",
    }


def get_active_model_name() -> str:
    return _active_model or "unknown"
