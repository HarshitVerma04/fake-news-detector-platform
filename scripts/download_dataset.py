"""
scripts/download_dataset.py

Downloads the WELFake dataset from Kaggle, which combines four news datasets:
  - Reuters
  - PolitiFact
  - GossipCop
  - WikiNews

Total: ~72,000 articles, balanced fake/real.

Prerequisites:
  1. Create a Kaggle account at https://www.kaggle.com
  2. Go to Account -> API -> Create New Token -> downloads kaggle.json
  3. Place kaggle.json at ~/.kaggle/kaggle.json
  4. Run: chmod 600 ~/.kaggle/kaggle.json

Then run this script:
  python scripts/download_dataset.py
"""

import subprocess
import sys
import os
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
DATASET_SLUG = "saurabhshahane/fake-news-classification"  # WELFake dataset


def check_kaggle_credentials():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: kaggle.json not found.")
        print()
        print("To set up Kaggle credentials:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Scroll to 'API' section -> 'Create New Token'")
        print("  3. This downloads kaggle.json")
        print("  4. Move it: mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
        print("  5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("Kaggle credentials found.")


def download():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET_SLUG}")
    print(f"Destination: {RAW_DATA_DIR.resolve()}")

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(RAW_DATA_DIR),
            "--unzip"
        ],
        capture_output=False
    )

    if result.returncode != 0:
        print("Download failed. Check your Kaggle credentials and internet connection.")
        sys.exit(1)

    files = list(RAW_DATA_DIR.iterdir())
    print(f"\nDownload complete. Files in {RAW_DATA_DIR}:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.1f} MB)")


def verify():
    """Check that the expected CSV file exists and has reasonable row count."""
    expected = RAW_DATA_DIR / "WELFake_Dataset.csv"
    if not expected.exists():
        print(f"WARNING: Expected file not found: {expected}")
        print("Files present:")
        for f in RAW_DATA_DIR.iterdir():
            print(f"  {f.name}")
        return

    import pandas as pd
    df = pd.read_csv(expected)
    print(f"\nDataset loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts())
    print("\n  0 = Fake, 1 = Real (WELFake convention)")


if __name__ == "__main__":
    check_kaggle_credentials()
    download()
    verify()
