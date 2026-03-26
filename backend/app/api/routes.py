"""
backend/app/api/routes.py

FastAPI route handlers.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.core import model_loader
from backend.app.db.database import get_db
from backend.app.db.models import NewsAnalysis
from backend.app.schemas.news import (
    PredictRequest, PredictResponse,
    HistoryResponse, ModelInfo,
)

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, summary="Predict fake or real news")
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Accepts a news article and returns a Fake/Real prediction with confidence score.
    The result is stored in the database.
    """
    try:
        result = model_loader.predict(request.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    record = NewsAnalysis(
        news_text=request.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_used=result["model"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return record


@router.get("/history", response_model=HistoryResponse, summary="Get prediction history")
def get_history(
    limit:  int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    total = db.query(NewsAnalysis).count()
    items = (
        db.query(NewsAnalysis)
        .order_by(NewsAnalysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return HistoryResponse(total=total, items=items)


@router.get("/models", response_model=ModelInfo, summary="Get model info and comparison metrics")
def get_models():
    """
    Returns the currently active model and evaluation metrics for both models
    if available (metrics.json and model_comparison.json must exist).
    """
    eval_dir = Path("model/evaluation")
    tfidf_metrics = None
    distilbert_metrics = None

    tfidf_path = eval_dir / "metrics.json"
    if tfidf_path.exists():
        with open(tfidf_path) as f:
            data = json.load(f)
            tfidf_metrics = data.get("test")

    comparison_path = eval_dir / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            data = json.load(f)
            distilbert_metrics = data.get("distilbert")

    return ModelInfo(
        active_model=model_loader.get_active_model_name(),
        available_models=["tfidf", "distilbert"],
        tfidf_metrics=tfidf_metrics,
        distilbert_metrics=distilbert_metrics,
    )


@router.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "active_model": model_loader.get_active_model_name(),
    }
