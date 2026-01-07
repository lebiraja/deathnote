"""
Main FastAPI application.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import logging

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.exceptions import BaseAPIException
from app.api.v1.router import api_router
from app.ml.model_manager import get_model_manager

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    
    Startup: Preload ML models
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting Life Expectancy Prediction API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"API Version: {settings.API_VERSION}")
    
    try:
        # Preload ML models
        model_manager = get_model_manager()
        model_info = model_manager.get_model_info()
        logger.info(f"Model loaded: {model_info['model_type']}")
        logger.info(f"Model features: {model_info['n_features']}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Don't fail startup, but log the error
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered life expectancy prediction API with health insights and recommendations",
    version=settings.API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Add rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions."""
    logger.error(f"API Exception: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    errors = exc.errors()
    logger.error(f"Validation error: {errors}")
    
    # Convert errors to JSON-serializable format
    serializable_errors = []
    for error in errors:
        serializable_errors.append({
            "loc": list(error.get("loc", [])),
            "msg": str(error.get("msg", "")),
            "type": str(error.get("type", ""))
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request data",
                "details": serializable_errors
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc) if settings.DEBUG else None
            }
        }
    )


# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/api/docs",
        "health": f"{settings.API_V1_PREFIX}/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
