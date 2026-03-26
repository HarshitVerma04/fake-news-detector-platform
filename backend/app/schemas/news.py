"""
backend/app/schemas/news.py

Pydantic models for request validation and response serialization.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=50,
        description="The news article text to analyze. Minimum 50 characters.",
        examples=["Scientists discover new species in the Amazon rainforest..."]
    )


class PredictResponse(BaseModel):
    id: int
    prediction: str          # "Fake" or "Real"
    confidence: float        # 0.0 - 1.0
    model_used: str          # "tfidf" or "distilbert"
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryItem(BaseModel):
    id: int
    news_text: str
    prediction: str
    confidence: float
    model_used: str
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    total: int
    items: list[HistoryItem]


class ModelInfo(BaseModel):
    active_model: str
    available_models: list[str]
    tfidf_metrics: dict | None = None
    distilbert_metrics: dict | None = None
