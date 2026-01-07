"""
Report generation endpoint.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import logging
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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
        
        pdf_content = generate_pdf_report(request)
        
        logger.info("PDF report generated successfully")
        
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=life-expectancy-report-{datetime.now().strftime('%Y%m%d')}.pdf"
            }
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


def generate_pdf_report(request: ReportRequest) -> bytes:
    """
    Generate a PDF report.
    
    Args:
        request: Report request data
        
    Returns:
        PDF content as bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    normal_style = styles['Normal']
    
    # Title
    story.append(Paragraph("LIFE EXPECTANCY PREDICTION REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Prediction Summary
    story.append(Paragraph("PREDICTION SUMMARY", heading_style))
    prediction_data = [[Paragraph("<b>Predicted Life Expectancy</b>", normal_style), 
                        Paragraph(f"<b>{request.prediction:.1f} years</b>", normal_style)]]
    prediction_table = Table(prediction_data, colWidths=[3*inch, 3*inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e0f2fe')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#0284c7'))
    ]))
    story.append(prediction_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Personal Information
    form_data = request.formData
    story.append(Paragraph("PERSONAL INFORMATION", heading_style))
    personal_data = [
        ["Gender:", str(form_data.get('gender', 'N/A'))],
        ["Height:", f"{form_data.get('height', 'N/A')} cm"],
        ["Weight:", f"{form_data.get('weight', 'N/A')} kg"],
        ["BMI:", str(form_data.get('bmi', 'N/A'))]
    ]
    personal_table = Table(personal_data, colWidths=[2*inch, 4*inch])
    personal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f9ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(personal_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Lifestyle Factors
    story.append(Paragraph("LIFESTYLE FACTORS", heading_style))
    lifestyle_data = [
        ["Physical Activity:", str(form_data.get('physical_activity', 'N/A'))],
        ["Smoking Status:", str(form_data.get('smoking_status', 'N/A'))],
        ["Alcohol Consumption:", str(form_data.get('alcohol_consumption', 'N/A'))],
        ["Diet:", str(form_data.get('diet', 'N/A'))]
    ]
    lifestyle_table = Table(lifestyle_data, colWidths=[2*inch, 4*inch])
    lifestyle_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f9ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(lifestyle_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Health Indicators
    story.append(Paragraph("HEALTH INDICATORS", heading_style))
    health_data = [
        ["Blood Pressure:", str(form_data.get('blood_pressure', 'N/A'))],
        ["Cholesterol:", f"{form_data.get('cholesterol', 'N/A')} mg/dL"],
        ["", ""],
        ["Medical Conditions:", ""],
        ["Diabetes:", 'Yes' if form_data.get('diabetes') else 'No'],
        ["Hypertension:", 'Yes' if form_data.get('hypertension') else 'No'],
        ["Heart Disease:", 'Yes' if form_data.get('heart_disease') else 'No'],
        ["Asthma:", 'Yes' if form_data.get('asthma') else 'No']
    ]
    health_table = Table(health_data, colWidths=[2*inch, 4*inch])
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f9ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('SPAN', (0, 2), (-1, 2)),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#dbeafe')),
        ('FONTNAME', (0, 3), (-1, 3), 'Helvetica-Bold')
    ]))
    story.append(health_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Health Insights
    story.append(Paragraph("HEALTH INSIGHTS", heading_style))
    for i, insight in enumerate(request.insights, 1):
        story.append(Paragraph(f"{i}. {insight}", normal_style))
        story.append(Spacer(1, 0.1*inch))
    story.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    story.append(Paragraph("RECOMMENDATIONS", heading_style))
    for i, recommendation in enumerate(request.recommendations, 1):
        story.append(Paragraph(f"{i}. {recommendation}", normal_style))
        story.append(Spacer(1, 0.1*inch))
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(Paragraph("DISCLAIMER", heading_style))
    disclaimer_text = """
    This prediction is for informational purposes only and should not be considered
    as medical advice. Please consult with healthcare professionals for personalized
    medical guidance and health assessments. The prediction is based on statistical 
    models and general health factors. Individual circumstances, genetics, environmental 
    factors, and access to healthcare can significantly impact actual life expectancy.
    """
    story.append(Paragraph(disclaimer_text, normal_style))
    
    doc.build(story)
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content
