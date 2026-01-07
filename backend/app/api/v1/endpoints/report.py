"""
Report generation endpoint.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class ReportRequest(BaseModel):
    """Request model for report generation."""
    formData: Dict[str, Any]
    prediction: float
    insights: List[str]
    recommendations: List[str]


@router.post("/report", response_class=Response)
async def generate_report(request: ReportRequest):
    """
    Generate PDF report for life expectancy prediction.
    
    Args:
        request: Report request data
        
    Returns:
        PDF file as bytes
    """
    try:
        logger.info("Generating PDF report")
        
        # For now, generate a simple text-based report
        # In production, you'd use a library like reportlab or weasyprint
        report_content = generate_text_report(request)
        
        logger.info("PDF report generated successfully")
        
        return Response(
            content=report_content.encode('utf-8'),
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=life-expectancy-report-{datetime.now().strftime('%Y%m%d')}.txt"
            }
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


def generate_text_report(request: ReportRequest) -> str:
    """
    Generate a text-based report.
    
    Args:
        request: Report request data
        
    Returns:
        Formatted text report
    """
    form_data = request.formData
    
    report = f"""
================================================================================
                    LIFE EXPECTANCY PREDICTION REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--------------------------------------------------------------------------------
PREDICTION SUMMARY
--------------------------------------------------------------------------------

Predicted Life Expectancy: {request.prediction:.1f} years


--------------------------------------------------------------------------------
PERSONAL INFORMATION
--------------------------------------------------------------------------------

Gender:               {form_data.get('gender', 'N/A')}
Height:               {form_data.get('height', 'N/A')} cm
Weight:               {form_data.get('weight', 'N/A')} kg
BMI:                  {form_data.get('bmi', 'N/A')}


--------------------------------------------------------------------------------
LIFESTYLE FACTORS
--------------------------------------------------------------------------------

Physical Activity:    {form_data.get('physical_activity', 'N/A')}
Smoking Status:       {form_data.get('smoking_status', 'N/A')}
Alcohol Consumption:  {form_data.get('alcohol_consumption', 'N/A')}
Diet:                 {form_data.get('diet', 'N/A')}


--------------------------------------------------------------------------------
HEALTH INDICATORS
--------------------------------------------------------------------------------

Blood Pressure:       {form_data.get('blood_pressure', 'N/A')}
Cholesterol:          {form_data.get('cholesterol', 'N/A')} mg/dL

Medical Conditions:
  - Diabetes:         {'Yes' if form_data.get('diabetes') else 'No'}
  - Hypertension:     {'Yes' if form_data.get('hypertension') else 'No'}
  - Heart Disease:    {'Yes' if form_data.get('heart_disease') else 'No'}
  - Asthma:           {'Yes' if form_data.get('asthma') else 'No'}


--------------------------------------------------------------------------------
HEALTH INSIGHTS
--------------------------------------------------------------------------------

"""
    
    for i, insight in enumerate(request.insights, 1):
        report += f"{i}. {insight}\n\n"
    
    report += """
--------------------------------------------------------------------------------
RECOMMENDATIONS
--------------------------------------------------------------------------------

"""
    
    for i, recommendation in enumerate(request.recommendations, 1):
        report += f"{i}. {recommendation}\n\n"
    
    report += """
================================================================================
DISCLAIMER
================================================================================

This prediction is for informational purposes only and should not be considered
as medical advice. Please consult with healthcare professionals for personalized
medical guidance and health assessments.

The prediction is based on statistical models and general health factors. 
Individual circumstances, genetics, environmental factors, and access to 
healthcare can significantly impact actual life expectancy.

================================================================================
"""
    
    return report
