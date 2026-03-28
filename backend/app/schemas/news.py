"""
backend/app/schemas/news.py
"""

from datetime import datetime
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=50)


class FetchUrlRequest(BaseModel):
    url: str = Field(..., min_length=10)


class PredictResponse(BaseModel):
    id:          int
    prediction:  str
    confidence:  float
    model_used:  str
    source_url:  str | None = None
    created_at:  datetime

    model_config = {"from_attributes": True}


class HistoryItem(BaseModel):
    id:          int
    news_text:   str
    prediction:  str
    confidence:  float
    model_used:  str
    source_url:  str | None = None
    created_at:  datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    total: int
    items: list[HistoryItem]


class StatsResponse(BaseModel):
    total_analyses:     int
    fake_count:         int
    real_count:         int
    fake_pct:           float
    real_pct:           float
    avg_confidence:     float
    model_breakdown:    dict        # {"tfidf": N, "distilbert": N}
    daily_counts:       list[dict]  # [{"date": "2026-03-01", "fake": N, "real": N}]
    confidence_buckets: list[dict]  # [{"range": "90-100%", "count": N}]


class ModelInfo(BaseModel):
    active_model:        str
    available_models:    list[str]
    tfidf_metrics:       dict | None = None
    distilbert_metrics:  dict | None = None
