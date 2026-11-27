from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def profile() -> dict[str, Any]:
    return {
        "age": 30, "sex": "male", "height_cm": 172, "weight_kg": 65,
        "bedtime": "23:00", "wake_time": "06:30", "smoking": "never", "chews_tobacco": False,
        "alcohol": "never", "moderate_minutes": 200, "vigorous_minutes": 0, "diet": "good",
        "diabetes": False, "hypertension": False, "heart_disease": False, "asthma": False,
    }


HABITS = ("weight_kg", "bedtime", "wake_time", "smoking", "chews_tobacco", "alcohol",
          "moderate_minutes", "vigorous_minutes", "diet")
