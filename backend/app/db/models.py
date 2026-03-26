"""
backend/app/db/models.py

SQLAlchemy ORM models.
"""

from datetime import datetime
from sqlalchemy import Integer, Text, String, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.database import Base


class NewsAnalysis(Base):
    __tablename__ = "news_analysis"

    id: Mapped[int]       = mapped_column(Integer, primary_key=True, index=True)
    news_text: Mapped[str]   = mapped_column(Text, nullable=False)
    prediction: Mapped[str]  = mapped_column(String(10), nullable=False)   # "Fake" | "Real"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_used: Mapped[str]  = mapped_column(String(20), nullable=False, default="tfidf")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
