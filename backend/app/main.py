"""
backend/app/main.py

FastAPI application entry point.

Run with:
  uvicorn backend.app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.model_loader import load_model
from backend.app.db.database import engine
from backend.app.db.models import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log.info("Creating database tables if not exist...")
    Base.metadata.create_all(bind=engine)

    log.info("Loading ML model...")
    load_model()

    yield

    # Shutdown (nothing to clean up for now)
    log.info("Shutting down.")


app = FastAPI(
    title="Fake News Detection API",
    description="Detect whether a news article is Fake or Real using NLP.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow React dev server to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
