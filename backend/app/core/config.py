"""
backend/app/core/config.py

Loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_ENV: str = "development"
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/fakenews_db"

    # Phase 1-5 model paths
    MODEL_PATH: str = "model/saved/tfidf_logreg.joblib"
    VECTORIZER_PATH: str = "model/saved/tfidf_vectorizer.joblib"

    # Phase 6: set to "distilbert" to use the transformer model
    # Set to "tfidf" to use the baseline model
    ACTIVE_MODEL: str = "tfidf"

    # DistilBERT model directory (contains config.json, pytorch_model.bin, tokenizer files)
    DISTILBERT_MODEL_PATH: str = "model/saved/distilbert"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
