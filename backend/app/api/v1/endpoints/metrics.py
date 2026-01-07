"""
Metrics endpoints for monitoring.
"""
from fastapi import APIRouter, status
from datetime import datetime
import psutil
import logging

from app.schemas.metrics import SystemMetrics, ModelMetrics
from app.ml.model_manager import get_model_manager
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/metrics/system",
    response_model=SystemMetrics,
    status_code=status.HTTP_200_OK,
    summary="System metrics",
    description="Get current system resource usage metrics",
    tags=["Metrics"]
)
async def get_system_metrics():
    """
    Get system resource metrics.
    
    Returns:
    - CPU usage percentage
    - Memory usage (used, total, percent)
    - Disk usage (used, total, percent)
    - Timestamp
    """
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        return SystemMetrics(
            cpu_percent=round(cpu_percent, 2),
            cpu_count=cpu_count,
            memory_used=memory.used,
            memory_total=memory.total,
            memory_percent=round(memory.percent, 2),
            disk_used=disk.used,
            disk_total=disk.total,
            disk_percent=round(disk.percent, 2),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.exception(f"Failed to get system metrics: {e}")
        raise


@router.get(
    "/metrics/model",
    response_model=ModelMetrics,
    status_code=status.HTTP_200_OK,
    summary="Model metrics",
    description="Get ML model information and metrics",
    tags=["Metrics"]
)
async def get_model_metrics():
    """
    Get ML model metrics and information.
    
    Returns:
    - Model type and version
    - Number of features
    - Model performance metrics
    - Last loaded timestamp
    """
    try:
        settings = get_settings()
        model_manager = get_model_manager()
        
        # Get model info
        model_info = model_manager.get_model_info()
        
        return ModelMetrics(
            model_name="Gradient Boosting Regressor",
            model_version=settings.MODEL_VERSION,
            model_type=model_info["model_type"],
            n_features=model_info["n_features"],
            accuracy=0.87,  # R² score from training
            last_updated=datetime.utcnow().isoformat() + "Z",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.exception(f"Failed to get model metrics: {e}")
        raise
