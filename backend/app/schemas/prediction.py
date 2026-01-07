from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Insight(BaseModel):
    category: str
    message: str
    severity: str


class Recommendation(BaseModel):
    title: str
    description: str
    priority: str


class HealthProfile(BaseModel):
    bmi: float
    bmi_category: str
    cholesterol: float
    cholesterol_status: str
    activity: str
    smoking: str
    risk_factors: int


# Aliases for compatibility
HealthInsight = Insight
HealthRecommendation = Recommendation
PredictionProfile = HealthProfile


class PredictionInput(BaseModel):
    gender: str
    height: float
    weight: float
    bmi: float
    cholesterol: float
    blood_pressure: str
    diabetes: int
    hypertension: int
    heart_disease: int
    asthma: int
    smoking_status: str
    physical_activity: str
    alcohol_consumption: str
    diet: str


class PredictionResponse(BaseModel):
    success: bool
    prediction: float
    confidence: float
    profile: HealthProfile
    insights: List[Insight]
    recommendations: List[Recommendation]
    model_version: str
    timestamp: str
