"""
Logging configuration for the application.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger

from app.core.config import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Setup application logging with file rotation and JSON formatting.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Log format ('json' or 'standard')
    """
    settings = get_settings()
    
    level = log_level or settings.LOG_LEVEL
    file_path = log_file or settings.LOG_FILE
    fmt = log_format or settings.LOG_FORMAT
    
    # Ensure log directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create formatters
    if fmt == "json":
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=file_path,
        maxBytes=settings.LOG_MAX_SIZE,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    logging.info(f"Logging configured: level={level}, file={file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
