"""
backend/app/main.py
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.api.auth_routes import router as auth_router
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
    title="Fake News Detection API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:4000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router,      prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
