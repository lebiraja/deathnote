"""
Test configuration and fixtures.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.core.config import get_settings


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def settings():
    """Application settings."""
    return get_settings()


@pytest.fixture
def sample_prediction_input():
    """Sample valid prediction input."""
    return {
        "age": 45,
        "gender": "Male",
        "bmi": 24.5,
        "cholesterol": 180.0,
        "blood_pressure": "Normal",
        "diabetes": False,
        "hypertension": False,
        "heart_disease": False,
        "asthma": False,
        "smoking_status": "Never",
        "physical_activity": "High",
        "diet": "Good"
    }


@pytest.fixture
def sample_high_risk_input():
    """Sample high-risk prediction input."""
    return {
        "age": 65,
        "gender": "Male",
        "bmi": 32.0,
        "cholesterol": 250.0,
        "blood_pressure": "High",
        "diabetes": True,
        "hypertension": True,
        "heart_disease": True,
        "asthma": False,
        "smoking_status": "Current",
        "physical_activity": "Low",
        "diet": "Poor"
    }
