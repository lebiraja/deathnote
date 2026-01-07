from pydantic import BaseModel


class SystemMetrics(BaseModel):
    cpu_percent: float
    cpu_count: int
    memory_used: int
    memory_total: int
    memory_percent: float
    disk_used: int
    disk_total: int
    disk_percent: float
    timestamp: str


class ModelMetrics(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    n_features: int
    accuracy: float
    last_updated: str
    timestamp: str
