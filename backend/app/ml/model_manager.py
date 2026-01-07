"""
Model manager for loading and caching ML models.
"""
import joblib
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple
import logging

from app.core.config import get_settings
from app.core.exceptions import ModelNotFoundException

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML model loading and caching."""
    
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._scaler = None
        self._preprocessor = None
        self._model_loaded_at = None
    
    @lru_cache(maxsize=1)
    def load_model(self) -> Any:
        """
        Load the ML model with caching.
        
        Returns:
            Loaded ML model
            
        Raises:
            ModelNotFoundException: If model file not found
        """
        if self._model is not None:
            return self._model
        
        model_path = self.settings.MODEL_PATH
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            raise ModelNotFoundException(
                f"Model file not found: {model_path}"
            )
        
        try:
            logger.info(f"Loading model from {model_path}")
            self._model = joblib.load(model_path)
            from datetime import datetime
            self._model_loaded_at = datetime.utcnow().isoformat()
            logger.info("Model loaded successfully")
            return self._model
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise ModelNotFoundException(
                f"Failed to load model: {str(e)}"
            )
    
    @lru_cache(maxsize=1)
    def load_scaler(self) -> Any:
        """
        Load the feature scaler with caching.
        
        Returns:
            Loaded scaler
            
        Raises:
            ModelNotFoundException: If scaler file not found
        """
        if self._scaler is not None:
            return self._scaler
        
        scaler_path = self.settings.SCALER_PATH
        
        if not scaler_path.exists():
            logger.error(f"Scaler not found at {scaler_path}")
            raise ModelNotFoundException(
                f"Scaler file not found: {scaler_path}"
            )
        
        try:
            logger.info(f"Loading scaler from {scaler_path}")
            self._scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
            return self._scaler
        except Exception as e:
            logger.exception(f"Failed to load scaler: {e}")
            raise ModelNotFoundException(
                f"Failed to load scaler: {str(e)}"
            )
    
    @lru_cache(maxsize=1)
    def load_preprocessor(self) -> Any:
        """
        Load the preprocessor with caching.
        
        Returns:
            Loaded preprocessor
            
        Raises:
            ModelNotFoundException: If preprocessor file not found
        """
        if self._preprocessor is not None:
            return self._preprocessor
        
        preprocessor_path = self.settings.PREPROCESSOR_PATH
        
        if not preprocessor_path.exists():
            logger.error(f"Preprocessor not found at {preprocessor_path}")
            raise ModelNotFoundException(
                f"Preprocessor file not found: {preprocessor_path}"
            )
        
        try:
            logger.info(f"Loading preprocessor from {preprocessor_path}")
            self._preprocessor = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded successfully")
            return self._preprocessor
        except Exception as e:
            logger.exception(f"Failed to load preprocessor: {e}")
            raise ModelNotFoundException(
                f"Failed to load preprocessor: {str(e)}"
            )
    
    def load_all(self) -> Tuple[Any, Any, Any]:
        """
        Load all model artifacts.
        
        Returns:
            Tuple of (model, scaler, preprocessor)
        """
        model = self.load_model()
        scaler = self.load_scaler()
        preprocessor = self.load_preprocessor()
        return model, scaler, preprocessor
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model is not None
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        model_type = "unknown"
        n_features = 0
        
        if self._model is not None:
            model_type = type(self._model).__name__
            if hasattr(self._model, 'n_features_in_'):
                n_features = self._model.n_features_in_
        
        return {
            "version": self.settings.MODEL_VERSION,
            "path": str(self.settings.MODEL_PATH),
            "loaded": self.is_loaded(),
            "loaded_at": self._model_loaded_at,
            "model_type": model_type,
            "n_features": n_features
        }


# Global model manager instance
@lru_cache()
def get_model_manager() -> ModelManager:
    """
    Get the global model manager instance.
    
    Returns:
        ModelManager instance
    """
    return ModelManager()
