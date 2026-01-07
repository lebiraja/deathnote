"""
Health check endpoints.
"""
from fastapi import APIRouter, status
from datetime import datetime
import psutil
import logging

from app.models.health import HealthCheck, DetailedHealthCheck, ServiceStatus
from app.ml.model_manager import get_model_manager
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheck,
    status_code=status.HTTP_200_OK,
    summary="Basic health check",
    description="Check if the service is running",
    tags=["Health"]
)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns a simple status indicating the service is operational.
    """
    settings = get_settings()
    
    # Check if model is loaded
    model_loaded = False
    try:
        model_manager = get_model_manager()
        model_manager.get_model_info()
        model_loaded = True
    except Exception as e:
        logger.warning(f"Model not loaded: {e}")
    
    # Check cache availability (Redis)
    cache_available = False
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        r.ping()
        cache_available = True
    except Exception as e:
        logger.warning(f"Cache not available: {e}")
    
    return HealthCheck(
        status="healthy",
        version=settings.API_VERSION,
        environment=settings.ENVIRONMENT,
        model_loaded=model_loaded,
        cache_available=cache_available,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@router.get(
    "/health/detailed",
    response_model=DetailedHealthCheck,
    status_code=status.HTTP_200_OK,
    summary="Detailed health check",
    description="Get detailed health status including model and system metrics",
    tags=["Health"]
)
async def detailed_health_check():
    """
    Detailed health check with system metrics.
    
    Returns:
    - Overall service status
    - Model loading status
    - System resource usage (CPU, memory, disk)
    - Service uptime
    - Environment information
    """
    try:
        settings = get_settings()
        model_manager = get_model_manager()
        
        # Check model status
        try:
            model_info = model_manager.get_model_info()
            model_status = ServiceStatus(
                status="healthy",
                message="Model loaded successfully",
                details={
                    "model_type": model_info["model_type"],
                    "model_version": settings.MODEL_VERSION,
                    "features": model_info["n_features"]
                }
            )
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            model_status = ServiceStatus(
                status="unhealthy",
                message=f"Model error: {str(e)}"
            )
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_status = ServiceStatus(
            status="healthy" if cpu_percent < 90 and memory.percent < 90 else "degraded",
            message="System resources normal",
            details={
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(memory.percent, 2),
                "disk_percent": round(disk.percent, 2)
            }
        )
        
        # Overall status
        overall_status = "healthy"
        if model_status.status != "healthy":
            overall_status = "unhealthy"
        elif system_status.status == "degraded":
            overall_status = "degraded"
        
        return DetailedHealthCheck(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            version=settings.API_VERSION,
            environment=settings.ENVIRONMENT,
            model=model_status,
            system=system_status
        )
        
    except Exception as e:
        logger.exception(f"Detailed health check failed: {e}")
        return DetailedHealthCheck(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat() + "Z",
            version=settings.API_VERSION,
            environment=settings.ENVIRONMENT,
            model=ServiceStatus(
                status="unknown",
                message="Health check failed"
            ),
            system=ServiceStatus(
                status="unknown",
                message=str(e)
            )
        )
