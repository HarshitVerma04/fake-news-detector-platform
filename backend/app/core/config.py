"""
backend/app/core/config.py
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_ENV: str = "development"
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/fakenews_db"

    # ML models
    MODEL_PATH: str = "model/saved/tfidf_logreg.joblib"
    VECTORIZER_PATH: str = "model/saved/tfidf_vectorizer.joblib"
    ACTIVE_MODEL: str = "tfidf"
    DISTILBERT_MODEL_PATH: str = "model/saved/distilbert"

    # JWT Auth
    SECRET_KEY: str = "change-this-to-a-long-random-secret-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24        # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
