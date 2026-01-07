"""
ML Service for handling predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
import logging

from app.ml.model_manager import get_model_manager
from app.core.exceptions import PredictionException, DataProcessingException
from app.models.prediction import (
    PredictionInput,
    PredictionResponse,
    PredictionProfile,
    HealthInsight,
    HealthRecommendation
)
from app.services.recommendation_service import RecommendationService
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class MLService:
    """Service for ML predictions."""
    
    def __init__(self):
        self.model_manager = get_model_manager()
        self.recommendation_service = RecommendationService()
        self.settings = get_settings()
    
    def prepare_input(self, data: PredictionInput) -> np.ndarray:
        """
        Prepare input data for prediction.
        
        Args:
            data: Validated prediction input
            
        Returns:
            Prepared numpy array
            
        Raises:
            DataProcessingException: If preprocessing fails
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data.model_dump()])
            
            # Rename columns to match training data format (Title_Case with underscores)
            column_mapping = {
                'gender': 'Gender',
                'height': 'Height',
                'weight': 'Weight',
                'bmi': 'BMI',
                'physical_activity': 'Physical_Activity',
                'smoking_status': 'Smoking_Status',
                'alcohol_consumption': 'Alcohol_Consumption',
                'diet': 'Diet',
                'blood_pressure': 'Blood_Pressure',
                'cholesterol': 'Cholesterol',
                'diabetes': 'Diabetes',
                'hypertension': 'Hypertension',
                'heart_disease': 'Heart_Disease',
                'asthma': 'Asthma'
            }
            df = df.rename(columns=column_mapping)
            
            # Load preprocessor
            preprocessor = self.model_manager.load_preprocessor()
            
            # Encode categorical features
            for col, le in preprocessor.items():
                if col in df.columns:
                    try:
                        df[col] = le.transform(df[col])
                    except ValueError as e:
                        logger.error(f"Error encoding {col}: {e}")
                        raise DataProcessingException(
                            f"Invalid value for {col}: {df[col].iloc[0]}"
                        )
            
            # Load scaler and scale features
            scaler = self.model_manager.load_scaler()
            scaled_data = scaler.transform(df)
            
            return scaled_data
            
        except DataProcessingException:
            raise
        except Exception as e:
            logger.exception(f"Data processing failed: {e}")
            raise DataProcessingException(
                f"Failed to process input data: {str(e)}"
            )
    
    def predict(self, data: PredictionInput) -> PredictionResponse:
        """
        Make a prediction for life expectancy.
        
        Args:
            data: Validated prediction input
            
        Returns:
            Prediction response with insights and recommendations
            
        Raises:
            PredictionException: If prediction fails
        """
        try:
            logger.info("Starting prediction")
            
            # Prepare input
            prepared_data = self.prepare_input(data)
            
            # Load model and make prediction
            model = self.model_manager.load_model()
            prediction = model.predict(prepared_data)[0]
            
            # Round prediction to 1 decimal place
            prediction = round(float(prediction), 1)
            
            logger.info(f"Prediction: {prediction} years")
            
            # Generate profile
            profile = self._generate_profile(data)
            
            # Generate insights
            insights = self._generate_insights(data, prediction)
            
            # Generate recommendations
            recommendations = self.recommendation_service.generate_recommendations(
                data,
                prediction
            )
            
            # Create response
            response = PredictionResponse(
                success=True,
                prediction=prediction,
                confidence=0.87,  # Model R² score
                profile=profile,
                insights=insights,
                recommendations=recommendations,
                model_version=self.settings.MODEL_VERSION,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
            logger.info("Prediction completed successfully")
            return response
            
        except (DataProcessingException, PredictionException):
            raise
        except Exception as e:
            logger.exception(f"Prediction failed: {e}")
            raise PredictionException(
                f"Prediction failed: {str(e)}"
            )
    
    def _generate_profile(self, data: PredictionInput) -> PredictionProfile:
        """Generate user health profile summary."""
        
        # BMI category
        bmi_category = "Normal"
        if data.bmi < 18.5:
            bmi_category = "Underweight"
        elif 18.5 <= data.bmi < 25:
            bmi_category = "Normal"
        elif 25 <= data.bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        
        # Cholesterol status
        chol_status = "Optimal"
        if data.cholesterol < 200:
            chol_status = "Desirable"
        elif 200 <= data.cholesterol < 240:
            chol_status = "Borderline High"
        else:
            chol_status = "High"
        
        # Count risk factors
        risk_factors = sum([
            data.diabetes,
            data.hypertension,
            data.heart_disease,
            data.asthma,
            1 if data.smoking_status == "Current" else 0,
            1 if data.bmi >= 30 else 0,
            1 if data.cholesterol >= 240 else 0
        ])
        
        return PredictionProfile(
            bmi=round(data.bmi, 1),
            bmi_category=bmi_category,
            cholesterol=round(data.cholesterol, 1),
            cholesterol_status=chol_status,
            activity=data.physical_activity,
            smoking=data.smoking_status,
            risk_factors=risk_factors
        )
    
    def _generate_insights(
        self,
        data: PredictionInput,
        prediction: float
    ) -> list[HealthInsight]:
        """Generate health insights based on user data."""
        
        insights = []
        
        # BMI insight
        if data.bmi < 18.5:
            insights.append(HealthInsight(
                category="Body Composition",
                message="Your BMI suggests you may be underweight. Consider consulting a healthcare provider.",
                severity="warning"
            ))
        elif 18.5 <= data.bmi < 25:
            insights.append(HealthInsight(
                category="Body Composition",
                message="Your BMI is in the healthy range, which is excellent for longevity.",
                severity="info"
            ))
        elif 25 <= data.bmi < 30:
            insights.append(HealthInsight(
                category="Body Composition",
                message="Your BMI indicates you're overweight. Weight reduction could improve life expectancy.",
                severity="warning"
            ))
        else:
            insights.append(HealthInsight(
                category="Body Composition",
                message="Your BMI suggests obesity. Weight management is crucial for improving health outcomes.",
                severity="critical"
            ))
        
        # Physical activity insight
        if data.physical_activity == "High":
            insights.append(HealthInsight(
                category="Lifestyle",
                message="Your high physical activity level positively impacts your longevity prediction.",
                severity="info"
            ))
        elif data.physical_activity == "Low":
            insights.append(HealthInsight(
                category="Lifestyle",
                message="Low physical activity is a risk factor. Increasing exercise could add years to your life.",
                severity="warning"
            ))
        
        # Smoking insight
        if data.smoking_status == "Current":
            insights.append(HealthInsight(
                category="Lifestyle",
                message="Smoking significantly reduces life expectancy. Quitting is the single best health decision.",
                severity="critical"
            ))
        elif data.smoking_status == "Never":
            insights.append(HealthInsight(
                category="Lifestyle",
                message="Never smoking is a major protective factor for your health and longevity.",
                severity="info"
            ))
        
        # Cholesterol insight
        if data.cholesterol >= 240:
            insights.append(HealthInsight(
                category="Health Indicators",
                message="High cholesterol increases cardiovascular risk. Consider dietary changes and medication if needed.",
                severity="warning"
            ))
        
        # Chronic conditions insight
        conditions = []
        if data.diabetes:
            conditions.append("diabetes")
        if data.hypertension:
            conditions.append("hypertension")
        if data.heart_disease:
            conditions.append("heart disease")
        
        if conditions:
            insights.append(HealthInsight(
                category="Chronic Conditions",
                message=f"Managing your {', '.join(conditions)} is crucial for optimal health outcomes.",
                severity="warning"
            ))
        
        return insights


# Global service instance
_ml_service_instance = None


def get_ml_service() -> MLService:
    """
    Get the global ML service instance.
    
    Returns:
        MLService instance
    """
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLService()
    return _ml_service_instance
