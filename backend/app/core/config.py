"""
backend/app/core/config.py

Loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_ENV: str = "development"
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/fakenews_db"
    MODEL_PATH: str = "model/saved/tfidf_logreg.joblib"
    VECTORIZER_PATH: str = "model/saved/tfidf_vectorizer.joblib"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
