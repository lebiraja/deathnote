"""Endpoints initialization."""
from app.api.v1.endpoints.prediction import router as prediction_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.metrics import router as metrics_router
from app.api.v1.endpoints.report import router as report_router

__all__ = [
    "prediction_router",
    "health_router",
    "metrics_router",
    "report_router",
]
