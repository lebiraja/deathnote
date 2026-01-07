from pydantic import BaseModel
from typing import Optional, Dict, Any


class HealthCheck(BaseModel):
    status: str
    version: str
    environment: str
    model_loaded: bool
    cache_available: bool
    timestamp: str


class ServiceStatus(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None


class DetailedHealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str
    model: ServiceStatus
    system: ServiceStatus
