"""
backend/app/api/routes.py

FastAPI route handlers for prediction and history endpoints.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.core import model_loader
from backend.app.db.database import get_db
from backend.app.db.models import NewsAnalysis
from backend.app.schemas.news import PredictRequest, PredictResponse, HistoryResponse

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
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    # Persist to database
    record = NewsAnalysis(
        news_text=request.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return record


@router.get("/history", response_model=HistoryResponse, summary="Get prediction history")
def get_history(
    limit: int = Query(default=20, ge=1, le=100, description="Number of records to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
):
    """
    Returns previously analyzed news articles, most recent first.
    """
    total = db.query(NewsAnalysis).count()
    items = (
        db.query(NewsAnalysis)
        .order_by(NewsAnalysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return HistoryResponse(total=total, items=items)


@router.get("/health", summary="Health check")
def health():
    return {"status": "ok"}
