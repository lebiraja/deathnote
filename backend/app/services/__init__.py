"""Service initialization."""
from app.services.ml_service import MLService, get_ml_service
from app.services.recommendation_service import RecommendationService

__all__ = [
    "MLService",
    "get_ml_service",
    "RecommendationService",
]
