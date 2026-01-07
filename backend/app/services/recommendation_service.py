"""
Recommendation service for generating health recommendations.
"""
from typing import List
import logging

from app.models.prediction import (
    PredictionInput,
    HealthRecommendation
)

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating health recommendations."""
    
    def generate_recommendations(
        self,
        data: PredictionInput,
        prediction: float
    ) -> List[HealthRecommendation]:
        """
        Generate personalized health recommendations.
        
        Args:
            data: User health data
            prediction: Predicted life expectancy
            
        Returns:
            List of health recommendations
        """
        recommendations = []
        
        # BMI recommendations
        if data.bmi >= 30:
            recommendations.append(HealthRecommendation(
                title="Weight Management Program",
                description="Consider enrolling in a structured weight loss program. Aim for gradual weight reduction through balanced nutrition and regular exercise.",
                priority="high"
            ))
        elif data.bmi >= 25:
            recommendations.append(HealthRecommendation(
                title="Healthy Weight Maintenance",
                description="Focus on portion control and regular physical activity to return to a healthy weight range.",
                priority="medium"
            ))
        
        # Physical activity recommendations
        if data.physical_activity == "Low":
            recommendations.append(HealthRecommendation(
                title="Increase Physical Activity",
                description="Aim for at least 150 minutes of moderate aerobic activity per week. Start with walking and gradually increase intensity.",
                priority="high"
            ))
        elif data.physical_activity == "Medium":
            recommendations.append(HealthRecommendation(
                title="Enhance Exercise Routine",
                description="Consider adding strength training exercises twice a week to complement your aerobic activities.",
                priority="medium"
            ))
        
        # Smoking recommendations
        if data.smoking_status == "Current":
            recommendations.append(HealthRecommendation(
                title="Smoking Cessation Program",
                description="Quitting smoking is crucial. Consider nicotine replacement therapy, counseling, or prescription medications to help you quit.",
                priority="high"
            ))
        elif data.smoking_status == "Former":
            recommendations.append(HealthRecommendation(
                title="Maintain Smoke-Free Status",
                description="Congratulations on quitting! Continue to avoid tobacco and secondhand smoke exposure.",
                priority="medium"
            ))
        
        # Diet recommendations
        if data.diet == "Poor":
            recommendations.append(HealthRecommendation(
                title="Nutrition Improvement",
                description="Adopt a Mediterranean-style diet rich in fruits, vegetables, whole grains, and lean proteins. Consider consulting a nutritionist.",
                priority="high"
            ))
        elif data.diet == "Average":
            recommendations.append(HealthRecommendation(
                title="Dietary Enhancement",
                description="Focus on increasing vegetable and fruit intake while reducing processed foods and added sugars.",
                priority="medium"
            ))
        
        # Cholesterol recommendations
        if data.cholesterol >= 240:
            recommendations.append(HealthRecommendation(
                title="Cholesterol Management",
                description="Work with your doctor to manage cholesterol through diet, exercise, and potentially medication. Reduce saturated fats and increase fiber intake.",
                priority="high"
            ))
        elif data.cholesterol >= 200:
            recommendations.append(HealthRecommendation(
                title="Monitor Cholesterol Levels",
                description="Keep cholesterol in check with a heart-healthy diet and regular exercise. Get regular checkups.",
                priority="medium"
            ))
        
        # Blood pressure recommendations
        if data.blood_pressure == "High" or data.hypertension:
            recommendations.append(HealthRecommendation(
                title="Blood Pressure Control",
                description="Monitor blood pressure regularly. Reduce sodium intake, maintain healthy weight, and take prescribed medications consistently.",
                priority="high"
            ))
        
        # Diabetes management
        if data.diabetes:
            recommendations.append(HealthRecommendation(
                title="Diabetes Management",
                description="Maintain stable blood sugar through diet, exercise, and medication adherence. Regular monitoring and doctor visits are essential.",
                priority="high"
            ))
        
        # Heart disease management
        if data.heart_disease:
            recommendations.append(HealthRecommendation(
                title="Cardiovascular Health",
                description="Follow your cardiologist's treatment plan. Cardiac rehabilitation and lifestyle modifications are crucial for heart health.",
                priority="high"
            ))
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append(HealthRecommendation(
                title="Maintain Healthy Lifestyle",
                description="Continue your excellent health practices. Regular checkups and preventive screenings are important.",
                priority="medium"
            ))
        
        # Always add preventive care
        recommendations.append(HealthRecommendation(
            title="Regular Health Screenings",
            description="Schedule regular checkups and age-appropriate screenings for cancer, heart disease, and other conditions.",
            priority="medium"
        ))
        
        # Stress management
        recommendations.append(HealthRecommendation(
            title="Stress Management",
            description="Practice stress-reduction techniques such as meditation, yoga, or deep breathing exercises to improve overall well-being.",
            priority="low"
        ))
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
