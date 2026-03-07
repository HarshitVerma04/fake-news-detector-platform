"""
backend/app/db/database.py

Database connection setup using SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.app.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,   # Reconnect on stale connections
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
