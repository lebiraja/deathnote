"""
Prediction endpoints.
"""
from fastapi import APIRouter, HTTPException, status, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

from app.models.prediction import PredictionInput, PredictionResponse
from app.services import get_ml_service
from app.services.ml_service import MLService
from app.core.exceptions import PredictionException, DataProcessingException

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict life expectancy",
    description="Predict life expectancy based on health and lifestyle factors",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "prediction": 78.5,
                        "confidence": 0.87,
                        "profile": {
                            "bmi": 24.2,
                            "bmi_category": "Normal",
                            "cholesterol": 180.5,
                            "cholesterol_status": "Desirable",
                            "activity": "High",
                            "smoking": "Never",
                            "risk_factors": 0
                        },
                        "insights": [
                            {
                                "category": "Body Composition",
                                "message": "Your BMI is in the healthy range",
                                "severity": "info"
                            }
                        ],
                        "recommendations": [
                            {
                                "title": "Maintain Healthy Lifestyle",
                                "description": "Continue your excellent health practices",
                                "priority": "medium"
                            }
                        ],
                        "model_version": "1.0.0",
                        "timestamp": "2025-01-09T12:00:00Z"
                    }
                }
            }
        },
        400: {"description": "Invalid input data"},
        422: {"description": "Validation error"},
        500: {"description": "Server error during prediction"}
    }
)
@limiter.limit("10/minute")
async def predict_life_expectancy(
    request: Request,
    data: PredictionInput,
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Predict life expectancy based on health metrics.
    
    **Input Parameters:**
    - age: Age in years (18-100)
    - gender: Biological sex (Male/Female)
    - bmi: Body Mass Index (10.0-60.0)
    - cholesterol: Total cholesterol mg/dL (100.0-400.0)
    - blood_pressure: Category (Low/Normal/High)
    - diabetes: Diabetes diagnosis (True/False)
    - hypertension: Hypertension diagnosis (True/False)
    - heart_disease: Heart disease diagnosis (True/False)
    - asthma: Asthma diagnosis (True/False)
    - smoking_status: Smoking status (Never/Former/Current)
    - physical_activity: Activity level (Low/Medium/High)
    - diet: Diet quality (Poor/Average/Good)
    
    **Returns:**
    - Predicted life expectancy in years
    - Confidence score
    - Health profile summary
    - Personalized insights
    - Health recommendations
    """
    try:
        logger.info(f"Prediction request received for gender={data.gender}, bmi={data.bmi}")
        
        # Make prediction
        result = ml_service.predict(data)
        
        logger.info(f"Prediction successful: {result.prediction} years")
        return result
        
    except DataProcessingException as e:
        logger.error(f"Data processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PredictionException as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction"
        )
