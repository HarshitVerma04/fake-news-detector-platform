"""
backend/app/db/models.py

SQLAlchemy ORM models.
Tables: users, news_analysis
"""

from datetime import datetime
from sqlalchemy import Integer, Text, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int]           = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str]     = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str]        = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool]   = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Password reset
    reset_token: Mapped[str | None]          = mapped_column(String(255), nullable=True)
    reset_token_expires: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    analyses: Mapped[list["NewsAnalysis"]] = relationship("NewsAnalysis", back_populates="user")


class NewsAnalysis(Base):
    __tablename__ = "news_analysis"

    id: Mapped[int]           = mapped_column(Integer, primary_key=True, index=True)
    news_text: Mapped[str]    = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    prediction: Mapped[str]   = mapped_column(String(10), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_used: Mapped[str]   = mapped_column(String(20), nullable=False, default="tfidf")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Optional foreign key — null for unauthenticated predictions
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    user: Mapped["User | None"] = relationship("User", back_populates="analyses")
