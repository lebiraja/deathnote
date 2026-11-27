from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.schemas.prediction import hours_asleep
from tests.conftest import HABITS

PREDICT = "/api/v1/predictions/predict"
WHAT_IF = "/api/v1/predictions/what-if"


def what_if(client: TestClient, profile: dict[str, Any], **changes: Any):
    habits = {key: profile[key] for key in HABITS} | changes
    return client.post(WHAT_IF, json={"profile": profile, "changes": habits})


def median(client: TestClient, profile: dict[str, Any]) -> float:
    return client.post(PREDICT, json=profile).json()["estimate"]["median_age"]


def test_health(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {"status": "healthy", "model_version": "2.0.0"}


@pytest.mark.parametrize(
    ("bed", "wake", "hours"), [("23:00", "06:30", 7.5), ("01:00", "09:00", 8), ("13:00", "20:00", 7)]
)
def test_sleep_hours_wrap_midnight(bed: str, wake: str, hours: float) -> None:
    assert hours_asleep(bed, wake) == hours


def test_prediction_is_consistent(client: TestClient, profile: dict[str, Any]) -> None:
    body = client.post(PREDICT, json=profile).json()
    estimate = body["estimate"]
    assert profile["age"] < estimate["lower_age"] < estimate["median_age"] < estimate["upper_age"] <= 110
    assert body["sleep_hours"] == 7.5
    survival = [point["you"] for point in body["curve"]]
    assert survival[0] == 1 and survival == sorted(survival, reverse=True)


def test_clock_time_does_not_change_estimate(client: TestClient, profile: dict[str, Any]) -> None:
    shifted = profile | {"bedtime": "03:00", "wake_time": "10:30"}
    assert median(client, shifted) == median(client, profile)


@pytest.mark.parametrize(
    "worse",
    [
        {"smoking": "former"}, {"smoking": "current"}, {"chews_tobacco": True}, {"alcohol": "light"},
        {"alcohol": "heavy"}, {"diet": "poor"}, {"moderate_minutes": 0}, {"weight_kg": 105},
        {"diabetes": True}, {"hypertension": True}, {"heart_disease": True}, {"asthma": True},
        {"systolic_bp": 180}, {"total_cholesterol": 300}, {"bedtime": "02:30"},
    ],
)
def test_worse_answers_never_add_years(
    client: TestClient, profile: dict[str, Any], worse: dict[str, Any]
) -> None:
    assert median(client, profile | worse) <= median(client, profile)


@pytest.mark.parametrize(
    "bad",
    [
        {"age": 19}, {"age": 80}, {"sex": "x"}, {"alcohol": "High"}, {"bedtime": "25:00"},
        {"bedtime": "06:00"}, {"height_cm": 40}, {"extra": 1},
    ],
)
def test_invalid_input_is_rejected(
    client: TestClient, profile: dict[str, Any], bad: dict[str, Any]
) -> None:
    assert client.post(PREDICT, json=profile | bad).status_code == 422


def test_what_if_quitting_smoking_helps(client: TestClient, profile: dict[str, Any]) -> None:
    response = what_if(client, profile | {"smoking": "current"}, smoking="former")
    assert response.status_code == 200 and response.json()["change_years"] > 0


def test_what_if_without_changes_is_zero(client: TestClient, profile: dict[str, Any]) -> None:
    assert what_if(client, profile, ).json()["change_years"] == 0


@pytest.mark.parametrize(
    ("start", "change"),
    [
        ({"smoking": "current"}, {"smoking": "never"}),
        ({}, {"smoking": "former"}),
        ({"alcohol": "heavy"}, {"alcohol": "never"}),
        ({"alcohol": "heavy"}, {"alcohol": "former"}),
    ],
)
def test_what_if_rejects_unanswerable_changes(
    client: TestClient, profile: dict[str, Any], start: dict[str, Any], change: dict[str, Any]
) -> None:
    assert what_if(client, profile | start, **change).status_code == 422


def test_report_is_a_pdf(client: TestClient, profile: dict[str, Any]) -> None:
    response = client.post("/api/v1/report", json=profile)
    assert response.status_code == 200 and response.content.startswith(b"%PDF")


def test_factors_are_listed_even_when_risk_is_clipped(
    client: TestClient, profile: dict[str, Any]
) -> None:
    sick = profile | {
        "age": 60, "smoking": "current", "diabetes": True, "hypertension": True,
        "heart_disease": True, "systolic_bp": 200, "diet": "poor", "moderate_minutes": 0,
    }
    factors = client.post(PREDICT, json=sick).json()["factors"]
    assert factors and all(factor["years"] < 0 for factor in factors if factor["key"] != "sleep")
