# SatyaParichay: A Fake News Detector

**Satya** (सत्य) — Truth | **Parichay** (परिचय) — Introduction

A full-stack AI-powered platform that detects fake news using NLP and deep learning. Built with FastAPI, PostgreSQL, and React. Supports two model backends — a classical TF-IDF + Logistic Regression baseline and a fine-tuned DistilBERT transformer.

---

## Stack

| Layer | Technology |
|-------|-----------|
| ML (baseline) | scikit-learn — TF-IDF + Logistic Regression |
| ML (advanced) | HuggingFace DistilBERT fine-tuned on WELFake |
| Backend | FastAPI + Python 3.13 |
| Database | PostgreSQL + SQLAlchemy |
| Auth | JWT (access + refresh tokens), bcrypt passwords |
| Frontend | React + Vite |
| Deployment | Nginx (HTTPS reverse proxy) |

---

## Project Structure

```
SatyaParichay/
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
│   ├── saved/                          # Added after training
│   │   ├── tfidf_logreg.joblib
│   │   ├── tfidf_vectorizer.joblib
│   │   └── distilbert/                 # Fine-tuned DistilBERT checkpoint
│   └── evaluation/
│       ├── metrics.json
│       ├── model_comparison.json
│       ├── confusion_matrix_logistic_regression.png
│       └── confusion_matrix_distilbert.png
│
├── backend/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI app entry point
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py               # predict, history, stats, fetch-url, models
│   │   │   └── auth_routes.py          # register, login, refresh, password reset
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Settings from .env
│   │   │   ├── auth.py                 # JWT logic, password hashing
│   │   │   └── model_loader.py         # Loads TF-IDF or DistilBERT at startup
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── database.py             # SQLAlchemy engine and session
│   │   │   └── models.py               # users, news_analysis ORM models
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── auth.py                 # Register, login, user schemas
│   │       └── news.py                 # Predict, history, stats schemas
│   └── tests/
│       ├── __init__.py
│       └── test_api.py
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx                     # Full UI — analyze, history, stats, models tabs
│       ├── api.js                      # All API calls
│       └── index.css                   # Dark editorial theme
│
├── scripts/
│   ├── __init__.py
│   ├── download_dataset.py             # Phase 1: Download WELFake from Kaggle
│   ├── preprocess.py                   # Phase 1: Clean text, train/val/test splits
│   ├── train_model.py                  # Phase 2: TF-IDF + LogReg training
│   └── train_distilbert.py             # Phase 6: DistilBERT fine-tuning (RTX 3050)
│
├── nginx/
│   └── satyaparichay.conf              # Nginx HTTPS reverse proxy config
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Build Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset download + preprocessing (WELFake, 72K articles) | Done |
| 2 | TF-IDF + Logistic Regression — 97.2% test accuracy | Done |
| 3 | FastAPI backend — predict, history, models endpoints | Done |
| 4 | PostgreSQL — stores all predictions with model used | Done |
| 5 | React frontend — analyze, history, stats, models tabs | Done |
| 6 | DistilBERT fine-tuned on RTX 3050 | Done |
| 7 | Stats dashboard, URL fetching, JWT authentication | Done |
| 8 | Rate limiting, password reset email, HTTPS (Nginx) | Done |

---

## Model Performance

| Metric | TF-IDF + LogReg | DistilBERT |
|--------|----------------|------------|
| Accuracy | 97.18% | — |
| Precision | 96.78% | — |
| Recall | 97.75% | — |
| F1 Score | 97.27% | — |

Training dataset: [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) — 72,134 articles combining Reuters, PolitiFact, GossipCop, and WikiNews.

---

## Known Limitations

### Satirical News (The Onion Problem)

The model classifies satirical news sites like **The Onion** as Real with high confidence. This is a known limitation of pattern-based NLP models. The Onion writes in clean, formal journalistic prose — grammatically identical to real news — so TF-IDF sees the word patterns and says Real.

**Why it happens:** The model learned writing style, not meaning. It has no concept of whether "Area Man Passionate Defender Of What He Imagines Constitution To Be" is plausible.

**Partial fixes implemented:**
- A domain blocklist that immediately flags known satire domains (theonion.com, babylonbee.com, etc.) before the model runs when using URL input
- DistilBERT performs better on absurdist sentences since it understands context, not just word frequency

**Full fix (future work):** Retrain with satire articles labeled as Fake, or add a dedicated satire classifier as a second-pass filter.

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd SatyaParichay
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your PostgreSQL password, email credentials, and a strong SECRET_KEY
```

### 3. Download dataset and train model

```bash
python scripts/download_dataset.py
python scripts/preprocess.py
python scripts/train_model.py
```

### 4. (Optional) Fine-tune DistilBERT

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python scripts/train_distilbert.py
# Then set ACTIVE_MODEL=distilbert in .env
```

### 5. Run

```bash
# Terminal 1 — Backend
uvicorn backend.app.main:app --reload

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` (or port 4000 if using `--port 4000`).

API docs available at `http://127.0.0.1:8000/docs`.

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | /api/v1/predict | Optional | Analyze article text |
| POST | /api/v1/fetch-url | Optional | Fetch and analyze from URL |
| GET | /api/v1/history | Optional | Prediction history |
| GET | /api/v1/stats | No | Aggregate statistics |
| GET | /api/v1/models | No | Model info and metrics |
| POST | /api/v1/auth/register | No | Create account |
| POST | /api/v1/auth/login | No | Get JWT tokens |
| POST | /api/v1/auth/refresh | No | Refresh access token |
| POST | /api/v1/auth/password-reset/request | No | Request password reset |
| POST | /api/v1/auth/password-reset/confirm | No | Confirm password reset |
| GET | /api/v1/auth/me | Yes | Get current user |

---

## Built With

- [FastAPI](https://fastapi.tiangolo.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [scikit-learn](https://scikit-learn.org/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [React](https://react.dev/)
- [Vite](https://vitejs.dev/)
- [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
