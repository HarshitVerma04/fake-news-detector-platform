"""
scripts/train_model.py

Phase 2: Model Training & Evaluation

Trains a TF-IDF + Logistic Regression model on the preprocessed data.
Saves the model and vectorizer to model/saved/.
Saves evaluation metrics and confusion matrix to model/evaluation/.

Run from project root:
  python scripts/train_model.py
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --- Config ---
PROCESSED_DIR = Path("data/processed")
SAVED_MODEL_DIR = Path("model/saved")
EVAL_DIR = Path("model/evaluation")
RANDOM_SEED = 42

# TF-IDF settings
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
TFIDF_MIN_DF = 2             # Ignore terms appearing in fewer than 2 docs
TFIDF_SUBLINEAR_TF = True    # Apply log normalization to term frequencies

# Logistic Regression settings
LR_C = 5.0                   # Inverse regularization strength
LR_MAX_ITER = 1000
LR_SOLVER = "lbfgs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_splits():
    for split in ["train", "val", "test"]:
        path = PROCESSED_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {path}\n"
                "Run: python scripts/preprocess.py"
            )

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    log.info(f"Loaded splits — train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")
    return train, val, test


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────

def build_tfidf(train_texts):
    log.info("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    )
    X_train = vectorizer.fit_transform(train_texts)
    log.info(f"Vocabulary size: {len(vectorizer.vocabulary_):,} features")
    return vectorizer, X_train


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_logistic_regression(X_train, y_train):
    log.info("Training Logistic Regression...")
    model = LogisticRegression(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        solver=LR_SOLVER,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    log.info("Training complete.")
    return model


def train_passive_aggressive(X_train, y_train):
    """
    PassiveAggressiveClassifier: fast, works well on text,
    useful as a comparison baseline.
    """
    log.info("Training PassiveAggressiveClassifier...")
    model = PassiveAggressiveClassifier(
        C=0.5,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    log.info("Training complete.")
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model, X, y_true, split_name: str) -> dict:
    y_pred = model.predict(X)

    metrics = {
        "split": split_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }

    log.info(
        f"[{split_name.upper()}] "
        f"Accuracy={metrics['accuracy']:.4f}  "
        f"Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}"
    )
    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Fake", "Real"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.tight_layout()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved: {out_path}")


def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=["Fake", "Real"])
    print("\n--- Classification Report (Test Set) ---")
    print(report)


# ─────────────────────────────────────────────
# Save artefacts
# ─────────────────────────────────────────────

def save_model(model, vectorizer, metrics: dict):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_MODEL_DIR / "tfidf_logreg.joblib"
    vec_path = SAVED_MODEL_DIR / "tfidf_vectorizer.joblib"
    metrics_path = EVAL_DIR / "metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Model saved:      {model_path}")
    log.info(f"Vectorizer saved: {vec_path}")
    log.info(f"Metrics saved:    {metrics_path}")


# ─────────────────────────────────────────────
# Top features inspection
# ─────────────────────────────────────────────

def show_top_features(model, vectorizer, n=20):
    """
    For Logistic Regression, the coefficients directly indicate
    which words most strongly predict Fake vs Real.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    top_fake_idx = np.argsort(coefs)[:n]
    top_real_idx = np.argsort(coefs)[-n:][::-1]

    print(f"\n--- Top {n} features predicting FAKE news ---")
    for idx in top_fake_idx:
        print(f"  {feature_names[idx]:<30}  coef: {coefs[idx]:.4f}")

    print(f"\n--- Top {n} features predicting REAL news ---")
    for idx in top_real_idx:
        print(f"  {feature_names[idx]:<30}  coef: {coefs[idx]:.4f}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train_df, val_df, test_df = load_splits()

    X_texts_train = train_df["clean_text"].fillna("")
    X_texts_val = val_df["clean_text"].fillna("")
    X_texts_test = test_df["clean_text"].fillna("")

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # Build TF-IDF features
    vectorizer, X_train = build_tfidf(X_texts_train)
    X_val = vectorizer.transform(X_texts_val)
    X_test = vectorizer.transform(X_texts_test)

    # Train primary model
    model = train_logistic_regression(X_train, y_train)

    # Evaluate
    train_metrics, _ = evaluate(model, X_train, y_train, "train")
    val_metrics, _ = evaluate(model, X_val, y_val, "val")
    test_metrics, y_pred_test = evaluate(model, X_test, y_test, "test")

    # Full classification report on test set
    print_classification_report(y_test, y_pred_test)

    # Confusion matrix plot
    plot_confusion_matrix(y_test, y_pred_test, "Logistic Regression")

    # Inspect top features
    show_top_features(model, vectorizer)

    # Save
    all_metrics = {
        "model": "TF-IDF + LogisticRegression",
        "tfidf_config": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "min_df": TFIDF_MIN_DF,
            "sublinear_tf": TFIDF_SUBLINEAR_TF,
        },
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_model(model, vectorizer, all_metrics)

    print("\nPhase 2 complete.")
    print("Next step: start the FastAPI backend with: uvicorn backend.app.main:app --reload")
