"""
backend/app/api/routes.py

Main API routes with rate limiting and satire domain detection.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, cast, Date
from sqlalchemy.orm import Session

from backend.app.core import model_loader
from backend.app.core.auth import get_current_user, get_optional_user
from backend.app.core.limiter import limiter
from backend.app.core.satire_domains import is_satire_domain
from backend.app.db.database import get_db
from backend.app.db.models import NewsAnalysis, User
from backend.app.schemas.news import (
    PredictRequest, FetchUrlRequest, PredictResponse,
    HistoryResponse, StatsResponse, ModelInfo,
)

log    = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
async def predict(
    request:      Request,
    req:          PredictRequest,
    db:           Session      = Depends(get_db),
    current_user: User | None  = Depends(get_optional_user),
):
    try:
        result = model_loader.predict(req.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    record = NewsAnalysis(
        news_text=req.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_used=result["model"],
        user_id=current_user.id if current_user else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.post("/fetch-url", response_model=PredictResponse)
@limiter.limit("30/minute")
async def fetch_url(
    request:      Request,
    req:          FetchUrlRequest,
    db:           Session      = Depends(get_db),
    current_user: User | None  = Depends(get_optional_user),
):
    # Satire domain check — before any ML inference
    if is_satire_domain(req.url):
        record = NewsAnalysis(
            news_text=f"[Satire domain detected] {req.url}",
            source_url=req.url,
            prediction="Fake",
            confidence=0.99,
            model_used="domain-blocklist",
            user_id=current_user.id if current_user else None,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    try:
        import newspaper
        article = newspaper.Article(req.url)
        article.download()
        article.parse()
        text = article.text
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch article from URL. Make sure it is a public news page. ({e})"
        )

    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Could not extract enough text from this URL. Try pasting the article text manually."
        )

    try:
        result = model_loader.predict(text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    record = NewsAnalysis(
        news_text=text[:5000],
        source_url=req.url,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_used=result["model"],
        user_id=current_user.id if current_user else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    limit:        int        = Query(default=20, ge=1, le=100),
    offset:       int        = Query(default=0, ge=0),
    db:           Session    = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    query = db.query(NewsAnalysis)
    if current_user:
        query = query.filter(NewsAnalysis.user_id == current_user.id)

    total = query.count()
    items = query.order_by(NewsAnalysis.created_at.desc()).offset(offset).limit(limit).all()
    return HistoryResponse(total=total, items=items)


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
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

    model_rows      = db.query(NewsAnalysis.model_used, func.count()).group_by(NewsAnalysis.model_used).all()
    model_breakdown = {row[0]: row[1] for row in model_rows}

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

    daily_map: dict = {}
    for row in daily_rows:
        d = str(row.date)
        if d not in daily_map:
            daily_map[d] = {"date": d, "fake": 0, "real": 0}
        daily_map[d][row.prediction.lower()] = row.count
    daily_counts = list(daily_map.values())

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


@router.get("/models", response_model=ModelInfo)
async def get_models():
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


@router.get("/health")
async def health():
    return {"status": "ok", "active_model": model_loader.get_active_model_name()}
