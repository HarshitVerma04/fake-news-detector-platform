"""
backend/app/api/routes.py

Main API routes: predict, history, stats, fetch-url, models, health.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, cast, Date
from sqlalchemy.orm import Session

from backend.app.core import model_loader
from backend.app.core.auth import get_current_user, get_optional_user
from backend.app.db.database import get_db
from backend.app.db.models import NewsAnalysis, User
from backend.app.schemas.news import (
    PredictRequest, FetchUrlRequest, PredictResponse,
    HistoryResponse, StatsResponse, ModelInfo,
)

log    = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
def predict(
    request:      PredictRequest,
    db:           Session       = Depends(get_db),
    current_user: User | None   = Depends(get_optional_user),
):
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
        user_id=current_user.id if current_user else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ─────────────────────────────────────────────
# Fetch URL
# ─────────────────────────────────────────────

@router.post("/fetch-url", response_model=PredictResponse)
def fetch_url(
    request:      FetchUrlRequest,
    db:           Session      = Depends(get_db),
    current_user: User | None  = Depends(get_optional_user),
):
    """
    Scrapes the article text from a URL and runs prediction on it.
    """
    try:
        import newspaper
        article = newspaper.Article(request.url)
        article.download()
        article.parse()
        text = article.text
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch article from URL. Make sure it's a public news page. ({e})"
        )

    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Could not extract enough text from this URL. Try copying the article text manually."
        )

    try:
        result = model_loader.predict(text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    record = NewsAnalysis(
        news_text=text[:5000],   # Cap stored text at 5000 chars
        source_url=request.url,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_used=result["model"],
        user_id=current_user.id if current_user else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ─────────────────────────────────────────────
# History
# ─────────────────────────────────────────────

@router.get("/history", response_model=HistoryResponse)
def get_history(
    limit:        int        = Query(default=20, ge=1, le=100),
    offset:       int        = Query(default=0, ge=0),
    db:           Session    = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    """
    If authenticated: returns only the current user's history.
    If not authenticated: returns all public history.
    """
    query = db.query(NewsAnalysis)
    if current_user:
        query = query.filter(NewsAnalysis.user_id == current_user.id)

    total = query.count()
    items = query.order_by(NewsAnalysis.created_at.desc()).offset(offset).limit(limit).all()
    return HistoryResponse(total=total, items=items)


# ─────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    total = db.query(NewsAnalysis).count()

    if total == 0:
        return StatsResponse(
            total_analyses=0, fake_count=0, real_count=0,
            fake_pct=0.0, real_pct=0.0, avg_confidence=0.0,
            model_breakdown={}, daily_counts=[], confidence_buckets=[],
        )

    fake_count = db.query(NewsAnalysis).filter(NewsAnalysis.prediction == "Fake").count()
    real_count = total - fake_count
    avg_conf   = db.query(func.avg(NewsAnalysis.confidence)).scalar() or 0.0

    # Model breakdown
    model_rows = db.query(NewsAnalysis.model_used, func.count()).group_by(NewsAnalysis.model_used).all()
    model_breakdown = {row[0]: row[1] for row in model_rows}

    # Daily counts — last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    daily_rows = (
        db.query(
            cast(NewsAnalysis.created_at, Date).label("date"),
            NewsAnalysis.prediction,
            func.count().label("count"),
        )
        .filter(NewsAnalysis.created_at >= thirty_days_ago)
        .group_by(cast(NewsAnalysis.created_at, Date), NewsAnalysis.prediction)
        .order_by(cast(NewsAnalysis.created_at, Date))
        .all()
    )

    # Reshape daily rows into [{date, fake, real}]
    daily_map: dict = {}
    for row in daily_rows:
        d = str(row.date)
        if d not in daily_map:
            daily_map[d] = {"date": d, "fake": 0, "real": 0}
        daily_map[d][row.prediction.lower()] = row.count
    daily_counts = list(daily_map.values())

    # Confidence buckets
    buckets = [
        ("50–60%", 0.50, 0.60), ("60–70%", 0.60, 0.70),
        ("70–80%", 0.70, 0.80), ("80–90%", 0.80, 0.90),
        ("90–100%", 0.90, 1.01),
    ]
    confidence_buckets = []
    for label, low, high in buckets:
        count = db.query(NewsAnalysis).filter(
            NewsAnalysis.confidence >= low,
            NewsAnalysis.confidence < high,
        ).count()
        confidence_buckets.append({"range": label, "count": count})

    return StatsResponse(
        total_analyses=total,
        fake_count=fake_count,
        real_count=real_count,
        fake_pct=round(fake_count / total * 100, 1),
        real_pct=round(real_count / total * 100, 1),
        avg_confidence=round(float(avg_conf) * 100, 1),
        model_breakdown=model_breakdown,
        daily_counts=daily_counts,
        confidence_buckets=confidence_buckets,
    )


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

@router.get("/models", response_model=ModelInfo)
def get_models():
    eval_dir           = Path("model/evaluation")
    tfidf_metrics      = None
    distilbert_metrics = None

    tfidf_path = eval_dir / "metrics.json"
    if tfidf_path.exists():
        with open(tfidf_path) as f:
            tfidf_metrics = json.load(f).get("test")

    comparison_path = eval_dir / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            distilbert_metrics = json.load(f).get("distilbert")

    return ModelInfo(
        active_model=model_loader.get_active_model_name(),
        available_models=["tfidf", "distilbert"],
        tfidf_metrics=tfidf_metrics,
        distilbert_metrics=distilbert_metrics,
    )


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "active_model": model_loader.get_active_model_name()}
