"""
scripts/preprocess.py

Phase 1: Dataset Preparation

Loads raw WELFake_Dataset.csv, cleans and preprocesses the text,
and saves train/validation/test splits to data/processed/.

Run from project root:
  python scripts/preprocess.py
"""

import re
import string
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# --- Config ---
RAW_FILE = Path("data/raw/WELFake_Dataset.csv")
PROCESSED_DIR = Path("data/processed")
RANDOM_SEED = 42

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# NLTK resource download
# ─────────────────────────────────────────────

def download_nltk_resources():
    resources = ["punkt", "stopwords", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            log.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


# ─────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────

STOP_WORDS = None  # Loaded once after NLTK download


def get_stop_words():
    global STOP_WORDS
    if STOP_WORDS is None:
        STOP_WORDS = set(stopwords.words("english"))
    return STOP_WORDS


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single article:
      1. Lowercase
      2. Remove URLs
      3. Remove HTML tags
      4. Remove non-ASCII characters
      5. Remove punctuation
      6. Tokenize
      7. Remove stopwords
      8. Remove short tokens (len < 2)
      9. Rejoin as clean string
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Remove non-ASCII characters (e.g. curly quotes, em-dashes)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # 5. Remove punctuation and digits
    text = re.sub(r"[^a-z\s]", " ", text)

    # 6. Tokenize
    tokens = word_tokenize(text)

    # 7. Remove stopwords and short tokens
    stop_words = get_stop_words()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    return " ".join(tokens)


# ─────────────────────────────────────────────
# Load & validate
# ─────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_FILE}\n"
            "Run: python scripts/download_dataset.py"
        )

    log.info(f"Loading {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)
    log.info(f"Loaded {len(df):,} rows")

    # Validate expected columns
    # WELFake columns: Unnamed: 0, title, text, label
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Expected columns 'text' and 'label'. Found: {list(df.columns)}"
        )

    return df


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Starting preprocessing...")

    # Drop unnamed index column if present
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    # Drop rows missing text or label
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    dropped = before - len(df)
    if dropped:
        log.info(f"Dropped {dropped:,} rows with missing text or label")

    # Ensure label is integer (0 = Fake, 1 = Real)
    df["label"] = df["label"].astype(int)

    # Combine title + text for richer signal (if title column present)
    if "title" in df.columns:
        df["title"] = df["title"].fillna("")
        df["combined_text"] = df["title"] + " " + df["text"]
        log.info("Combined title + text into 'combined_text'")
    else:
        df["combined_text"] = df["text"]

    # Clean text (this is the slow step — ~72k rows takes ~2-3 min)
    log.info("Cleaning text (this takes a few minutes)...")
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Drop rows where cleaning produced empty strings
    before = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 10]
    dropped = before - len(df)
    if dropped:
        log.info(f"Dropped {dropped:,} rows with empty/short clean text")

    log.info(f"Preprocessing complete. {len(df):,} rows remain.")

    # Show label distribution
    label_counts = df["label"].value_counts()
    total = len(df)
    for label, count in label_counts.items():
        label_name = "Real" if label == 1 else "Fake"
        log.info(f"  {label_name} (label={label}): {count:,} ({count/total*100:.1f}%)")

    return df


# ─────────────────────────────────────────────
# Train / Val / Test split
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """
    Stratified split to preserve label balance across splits.
    Returns: (train_df, val_df, test_df)
    """
    log.info(
        f"Splitting data: {TRAIN_RATIO*100:.0f}% train / "
        f"{VAL_RATIO*100:.0f}% val / "
        f"{TEST_RATIO*100:.0f}% test"
    )

    # First split off test set
    train_val, test = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    # Split remaining into train and val
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=RANDOM_SEED,
        stratify=train_val["label"]
    )

    log.info(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

def save_splits(train, val, test):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    splits = {"train": train, "val": val, "test": test}
    for name, df in splits.items():
        out_path = PROCESSED_DIR / f"{name}.csv"
        # Save only the columns needed downstream
        df[["clean_text", "label"]].to_csv(out_path, index=False)
        log.info(f"Saved {name} -> {out_path} ({len(df):,} rows)")


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

def sanity_check():
    """Print a few examples to verify the output looks correct."""
    train_path = PROCESSED_DIR / "train.csv"
    df = pd.read_csv(train_path)

    print("\n--- Sanity Check: 3 random samples from train set ---")
    for _, row in df.sample(3, random_state=1).iterrows():
        label_name = "REAL" if row["label"] == 1 else "FAKE"
        preview = row["clean_text"][:120]
        print(f"\n  [{label_name}]  {preview}...")

    print(f"\nTrain label distribution:")
    print(df["label"].value_counts(normalize=True).rename({0: "Fake", 1: "Real"}))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    download_nltk_resources()
    df = load_raw_data()
    df = preprocess(df)
    train, val, test = split_data(df)
    save_splits(train, val, test)
    sanity_check()

    print("\nPhase 1 complete.")
    print("Next step: python scripts/train_model.py")
