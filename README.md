# Fake News Detection Platform

A full-stack application that detects fake news using NLP and machine learning.

## Stack

- **ML**: scikit-learn (baseline), DistilBERT (optional GPU upgrade)
- **Backend**: FastAPI + Python
- **Database**: PostgreSQL
- **Frontend**: React

## Project Structure

```
fake-news-detector/
│
├── data/
│   ├── raw/                            # Not committed to git
│   │   └── WELFake_Dataset.csv         # Added after: python scripts/download_dataset.py
│   └── processed/                      # Added after: python scripts/preprocess.py
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── model/
│   ├── saved/                          # Added after: python scripts/train_model.py
│   │   ├── tfidf_logreg.joblib
│   │   └── tfidf_vectorizer.joblib
│   └── evaluation/                     # Added after: python scripts/train_model.py
│       ├── metrics.json
│       └── confusion_matrix_logistic_regression.png
│
├── backend/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI app entry point
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py               # POST /predict, GET /history, GET /health
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Loads settings from .env
│   │   │   └── model_loader.py         # Loads .joblib model at startup
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── database.py             # SQLAlchemy engine and session
│   │   │   └── models.py               # news_analysis ORM model
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── news.py                 # Pydantic request/response schemas
│   └── tests/
│       ├── __init__.py
│       └── test_api.py                 # Endpoint tests
│
├── frontend/
│   └── src/                            # React app (built in Phase 5)
│
├── notebooks/                          # Jupyter notebooks for exploration
│
├── scripts/
│   ├── __init__.py
│   ├── download_dataset.py             # Phase 1: Download WELFake from Kaggle
│   ├── preprocess.py                   # Phase 1: Clean text, create train/val/test splits
│   └── train_model.py                  # Phase 2: Train TF-IDF + LogReg, save model
│
├── .env.example                        # Copy to .env and fill in your values
├── .gitignore
├── requirements.txt
└── README.md
```

## Build Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset preparation & preprocessing | - |
| 2 | Model training & evaluation | - |
| 3 | FastAPI backend | - |
| 4 | Database integration | - |
| 5 | Frontend | - |
| 6 | DistilBERT upgrade (optional) | - |

## Setup

See each phase's instructions in the docs or follow the phase scripts in `scripts/`.
