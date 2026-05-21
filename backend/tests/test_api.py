"""
FastAPI Backend API Unit Tests
==============================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Implements unit tests using FastAPI's TestClient to verify health checks,
    routes validation, error handling, and model information endpoints.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Adjust sys.path to resolve app modules correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify that the health check endpoint returns 200 and valid JSON properties."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "cuda_available" in data


def test_root_endpoint():
    """Verify root endpoint returns basic application identity."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "ESRGAN Image Enhancement API" in data["message"]


def test_model_info_endpoint():
    """Verify the model information API is structured correctly."""
    response = client.get("/api/v1/models/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "ESRGAN (RRDBNet)"
    assert 4 in data["scale_factors"]
    assert "supported_formats" in data


def test_nonexistent_task_status_returns_404():
    """Verify that polling a fake task ID returns a proper 404 response."""
    response = client.get("/api/v1/task/fake-task-uuid-12345")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"]
