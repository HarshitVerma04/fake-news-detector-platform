"""
backend/tests/test_api.py

Integration tests for the FastAPI endpoints.
Uses httpx.AsyncClient with a test database.

Run with:
  pytest backend/tests/ -v
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

from backend.app.main import app


SAMPLE_ARTICLE = (
    "Scientists at NASA have confirmed the discovery of water ice on the lunar surface. "
    "The findings were published in the journal Nature and confirmed by multiple independent "
    "research teams across three continents. The discovery opens new possibilities for future "
    "human missions to the Moon and potential permanent lunar bases."
)


@pytest.fixture
def mock_predict():
    with patch("backend.app.api.routes.model_loader.predict") as mock:
        mock.return_value = {
            "prediction": "Real",
            "confidence": 0.91,
            "label": 1,
        }
        yield mock


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_predict_valid_text(mock_predict):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            json={"text": SAMPLE_ARTICLE}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("Fake", "Real")
    assert 0.0 <= data["confidence"] <= 1.0
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_predict_text_too_short():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            json={"text": "short text"}
        )
    assert response.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_history_returns_list():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/history")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)
