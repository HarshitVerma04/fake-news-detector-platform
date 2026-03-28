"""
backend/app/main.py
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from backend.app.api.routes import router
from backend.app.api.auth_routes import router as auth_router
from backend.app.core.limiter import limiter
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
    log.info("Creating database tables if not exist...")
    Base.metadata.create_all(bind=engine)
    log.info("Loading ML model...")
    load_model()
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="SatyaParichay — Fake News Detection API",
    description="Satya (सत्य) — Truth | Parichay (परिचय) — Introduction",
    version="3.0.0",
    lifespan=lifespan,
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:4000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router,      prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
