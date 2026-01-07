"""
Core configuration management for the Life Expectancy Prediction API.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Info
    APP_NAME: str = "Life Expectancy Prediction API"
    PROJECT_NAME: str = "Life Expectancy Prediction API"
    APP_VERSION: str = "1.0.0"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    
    # Security
    SECRET_KEY: str = Field(default="change-me-in-production")
    ALLOWED_HOSTS: List[str] = Field(default=["*"])
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://localhost:8000"
        ]
    )
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    API_RATE_LIMIT: str = "100/hour"
    MAX_REQUEST_SIZE: int = 10485760  # 10MB
    
    # Paths
    BASE_DIR: Path = Path("/app")
    MODEL_DIR: Path = Path("/app/models")
    DATA_DIR: Path = Path("/app/data")
    LOG_DIR: Path = Path("/app/logs")
    
    # ML Model Settings
    MODEL_PATH: Path = MODEL_DIR / "gradient_boosting_model.pkl"
    SCALER_PATH: Path = MODEL_DIR / "scaler.pkl"
    PREPROCESSOR_PATH: Path = MODEL_DIR / "preprocessor.pkl"
    MODEL_VERSION: str = "1.0.0"
    
    # Redis Settings
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 300  # 5 minutes
    
    # Database Settings (optional)
    DATABASE_URL: Optional[str] = None
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = LOG_DIR / "app.log"
    LOG_MAX_SIZE: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5
    LOG_FORMAT: str = "json"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Sentry (Error Tracking)
    SENTRY_DSN: Optional[str] = None
    SENTRY_ENVIRONMENT: str = "production"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()
