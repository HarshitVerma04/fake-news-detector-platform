"""
scripts/train_distilbert.py

Phase 6: DistilBERT Fine-tuning on RTX 3050

Fine-tunes distilbert-base-uncased on the WELFake dataset.
Saves the model to model/saved/distilbert/.

RTX 3050 notes:
  - 4GB VRAM: batch_size=8 with gradient_accumulation=4 (effective batch=32)
  - Training time: ~45-60 minutes for 3 epochs on 57k samples
  - Mixed precision (fp16) is enabled to reduce VRAM usage

Run from project root:
  python scripts/train_distilbert.py
"""

import logging
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --- Config ---
PROCESSED_DIR  = Path("data/processed")
SAVED_MODEL_DIR = Path("model/saved/distilbert")
EVAL_DIR       = Path("model/evaluation")

MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH       = 512      # Max tokens per article
BATCH_SIZE       = 8        # Per-device batch size (safe for 4GB VRAM)
GRAD_ACCUM       = 4        # Effective batch size = 8 * 4 = 32
EPOCHS           = 3
LEARNING_RATE    = 2e-5
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
RANDOM_SEED      = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# GPU check
# ─────────────────────────────────────────────

def check_gpu():
    if not torch.cuda.is_available():
        log.warning("CUDA not available. Training on CPU will be extremely slow.")
        log.warning("Make sure you installed PyTorch with CUDA support:")
        log.warning("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {device_name}  ({vram_gb:.1f} GB VRAM)")

    if vram_gb < 3.5:
        log.warning("Less than 4GB VRAM detected. Reducing batch size to 4.")
        global BATCH_SIZE
        BATCH_SIZE = 4

    return True


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class NewsDataset(Dataset):
    """
    PyTorch Dataset that tokenizes text on the fly.
    Uses the RAW text (not preprocessed) for DistilBERT —
    transformers benefit from punctuation, casing, and stop words
    which were removed in Phase 1 preprocessing.
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_raw_data():
    """
    Load the ORIGINAL WELFake CSV (not the clean_text version).
    DistilBERT works better with original text.
    Falls back to clean_text from processed splits if raw isn't available.
    """
    raw_path = Path("data/raw/WELFake_Dataset.csv")
    processed_train = PROCESSED_DIR / "train.csv"

    if raw_path.exists():
        log.info("Loading raw WELFake dataset for DistilBERT...")
        df = pd.read_csv(raw_path)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(int)

        if "title" in df.columns:
            df["title"] = df["title"].fillna("")
            df["input_text"] = df["title"] + " " + df["text"]
        else:
            df["input_text"] = df["text"]

        # Use same split as Phase 1 for fair comparison
        from sklearn.model_selection import train_test_split
        train_val, test = train_test_split(df, test_size=0.10, random_state=RANDOM_SEED, stratify=df["label"])
        train, val      = train_test_split(train_val, test_size=0.10/0.90, random_state=RANDOM_SEED, stratify=train_val["label"])

        return (
            train["input_text"].tolist(), train["label"].tolist(),
            val["input_text"].tolist(),   val["label"].tolist(),
            test["input_text"].tolist(),  test["label"].tolist(),
        )
    else:
        log.info("Raw data not found. Using preprocessed clean_text splits.")
        train = pd.read_csv(PROCESSED_DIR / "train.csv")
        val   = pd.read_csv(PROCESSED_DIR / "val.csv")
        test  = pd.read_csv(PROCESSED_DIR / "test.csv")
        return (
            train["clean_text"].tolist(), train["label"].tolist(),
            val["clean_text"].tolist(),   val["label"].tolist(),
            test["clean_text"].tolist(),  test["label"].tolist(),
        )


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, zero_division=0), 4),
        "f1":        round(f1_score(labels, preds, zero_division=0), 4),
    }


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────

def full_evaluate(trainer, dataset, split_name: str):
    results = trainer.evaluate(dataset)
    log.info(
        f"[{split_name.upper()}] "
        f"Accuracy={results.get('eval_accuracy', 0):.4f}  "
        f"F1={results.get('eval_f1', 0):.4f}"
    )
    return results


def plot_confusion_matrix(trainer, dataset, y_true):
    preds_output = trainer.predict(dataset)
    y_pred = np.argmax(preds_output.predictions, axis=-1)

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Fake", "Real"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — DistilBERT", fontsize=13)
    plt.tight_layout()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "confusion_matrix_distilbert.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved: {out_path}")

    print("\n--- Classification Report (Test Set — DistilBERT) ---")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

    return y_pred


# ─────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────

def save_comparison(distilbert_metrics: dict):
    """
    Load the TF-IDF metrics and produce a side-by-side comparison JSON.
    """
    tfidf_metrics_path = EVAL_DIR / "metrics.json"
    comparison = {"distilbert": distilbert_metrics}

    if tfidf_metrics_path.exists():
        with open(tfidf_metrics_path) as f:
            tfidf = json.load(f)
        comparison["tfidf_logreg"] = tfidf.get("test", {})

        print("\n--- Model Comparison (Test Set) ---")
        print(f"{'Metric':<12} {'TF-IDF + LR':>14} {'DistilBERT':>14}")
        print("-" * 42)
        for metric in ["accuracy", "precision", "recall", "f1"]:
            tfidf_val  = tfidf.get("test", {}).get(metric, "N/A")
            bert_val   = distilbert_metrics.get(metric, "N/A")
            tfidf_str  = f"{tfidf_val:.4f}" if isinstance(tfidf_val, float) else str(tfidf_val)
            bert_str   = f"{bert_val:.4f}"  if isinstance(bert_val, float)  else str(bert_val)
            print(f"{metric.capitalize():<12} {tfidf_str:>14} {bert_str:>14}")

    out_path = EVAL_DIR / "model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info(f"Comparison saved: {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    check_gpu()

    # Load data
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test) = load_raw_data()

    log.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Tokenizer
    log.info(f"Loading tokenizer: {MODEL_CHECKPOINT}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)

    # Datasets
    train_dataset = NewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset   = NewsDataset(X_val,   y_val,   tokenizer, MAX_LENGTH)
    test_dataset  = NewsDataset(X_test,  y_test,  tokenizer, MAX_LENGTH)

    # Model
    log.info(f"Loading model: {MODEL_CHECKPOINT}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "Fake", 1: "Real"},
        label2id={"Fake": 0, "Real": 1},
    )

    # Training arguments tuned for RTX 3050 (4GB VRAM)
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(SAVED_MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),          # Mixed precision on GPU
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",                         # Disable wandb/tensorboard
        seed=RANDOM_SEED,
        dataloader_num_workers=0,                 # Avoids Windows multiprocessing issues
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    log.info("Starting training...")
    log.info(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    log.info(f"Epochs: {EPOCHS}  |  Learning rate: {LEARNING_RATE}")
    trainer.train()

    # Evaluate on test set
    log.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    y_pred = plot_confusion_matrix(trainer, test_dataset, y_test)

    test_metrics = {
        "accuracy":  round(test_results.get("eval_accuracy", 0), 4),
        "precision": round(test_results.get("eval_precision", 0), 4),
        "recall":    round(test_results.get("eval_recall", 0), 4),
        "f1":        round(test_results.get("eval_f1", 0), 4),
    }

    # Save comparison
    save_comparison(test_metrics)

    # Save tokenizer alongside model for inference
    tokenizer.save_pretrained(str(SAVED_MODEL_DIR))
    log.info(f"Model + tokenizer saved to: {SAVED_MODEL_DIR}")

    print("\nPhase 6 complete.")
    print(f"Model saved to: {SAVED_MODEL_DIR}")
    print("To use DistilBERT in the API, set ACTIVE_MODEL=distilbert in your .env file.")
