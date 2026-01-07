"""
API v1 router.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    prediction_router,
    health_router,
    metrics_router,
    report_router
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    prediction_router,
    prefix="/predictions",
    tags=["Predictions"]
)

api_router.include_router(
    health_router,
    tags=["Health"]
)

api_router.include_router(
    metrics_router,
    prefix="/metrics",
    tags=["Metrics"]
)

api_router.include_router(
    report_router,
    tags=["Reports"]
)
