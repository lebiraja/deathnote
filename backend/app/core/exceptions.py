"""
Custom exception classes for the application.
"""
from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Base exception for all API exceptions."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class ValidationException(BaseAPIException):
    """Exception raised for validation errors."""
    
    def __init__(self, detail: str = "Validation error"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )


class ModelNotFoundException(BaseAPIException):
    """Exception raised when ML model is not found."""
    
    def __init__(self, detail: str = "ML model not found"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class PredictionException(BaseAPIException):
    """Exception raised during prediction errors."""
    
    def __init__(self, detail: str = "Prediction failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class RateLimitException(BaseAPIException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail
        )


class CacheException(BaseAPIException):
    """Exception raised for caching errors."""
    
    def __init__(self, detail: str = "Cache operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class DataProcessingException(BaseAPIException):
    """Exception raised for data processing errors."""
    
    def __init__(self, detail: str = "Data processing failed"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )
