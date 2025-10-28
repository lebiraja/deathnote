"""
Flask app for life expectancy prediction
Modern, animated web interface with HTML, CSS, and JavaScript
"""
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
from preprocessing import DataPreprocessor

import io
from datetime import datetime

# PDF generation (optional - ReportLab)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global variables for model artifacts
model = None
scaler = None
preprocessor = None
feature_names = None


def load_artifacts():
    """Load trained model, scaler, and preprocessor"""
    global model, scaler, preprocessor, feature_names
    
    try:
        model = joblib.load('models/gradient_boosting_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        
        # Feature names in the order they're used
        feature_names = ['Gender', 'Height', 'Weight', 'BMI', 'Physical_Activity', 
                        'Smoking_Status', 'Alcohol_Consumption', 'Diet', 'Blood_Pressure',
                        'Cholesterol', 'Diabetes', 'Hypertension', 'Heart_Disease', 'Asthma']
        
        return True
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        return False


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        # Extract and validate input
        user_input = {
            'Gender': data.get('gender'),
            'Height': float(data.get('height', 170)),
            'Weight': float(data.get('weight', 70)),
            'BMI': float(data.get('bmi', 24)),
            'Physical_Activity': data.get('physical_activity'),
            'Smoking_Status': data.get('smoking_status'),
            'Alcohol_Consumption': data.get('alcohol_consumption'),
            'Diet': data.get('diet'),
            'Blood_Pressure': data.get('blood_pressure'),
            'Cholesterol': float(data.get('cholesterol', 200)),
            'Diabetes': int(data.get('diabetes', 0)),
            'Hypertension': int(data.get('hypertension', 0)),
            'Heart_Disease': int(data.get('heart_disease', 0)),
            'Asthma': int(data.get('asthma', 0))
        }
        
        # Prepare input for prediction
        df_user = pd.DataFrame([user_input])
        
        # Encode categorical features
        for col, le in preprocessor.items():
            if col in df_user.columns:
                try:
                    df_user[col] = le.transform(df_user[col])
                except ValueError as e:
                    return jsonify({'error': f'Invalid value for {col}'}), 400
        
        # Scale features
        df_scaled = scaler.transform(df_user)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        
        # Generate insights
        insights = generate_insights(user_input)
        recommendations = generate_recommendations(user_input)
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 1),
            'insights': insights,
            'recommendations': recommendations,
            'profile': {
                'bmi': round(user_input['BMI'], 1),
                'cholesterol': user_input['Cholesterol'],
                'smoking': user_input['Smoking_Status'],
                'activity': user_input['Physical_Activity'],
                'diet': user_input['Diet']
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report', methods=['POST'])
def generate_pdf_report():
    """Generate a PDF report and return it as attachment."""
    if not REPORTLAB_AVAILABLE:
        return jsonify({'error': 'ReportLab library is not installed on the server.'}), 500

    try:
        data = request.json or {}
        form = data.get('formData', {})
        prediction = data.get('prediction', '')
        insights = data.get('insights', []) or []
        recommendations = data.get('recommendations', []) or []

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph('Life Expectancy Prediction Report', styles['Title']))
        elements.append(Paragraph(f'Generated: {timestamp}', styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(f'Predicted Life Expectancy: <b>{prediction} years</b>', styles['Heading2']))
        elements.append(Spacer(1, 12))

        # Profile table
        profile_data = [
            ['Field', 'Value'],
            ['Gender', form.get('gender', '')],
            ['Height (cm)', str(form.get('height', ''))],
            ['Weight (kg)', str(form.get('weight', ''))],
            ['BMI', str(form.get('bmi', ''))],
            ['Physical Activity', form.get('physical_activity', '')],
            ['Smoking Status', form.get('smoking_status', '')],
            ['Alcohol Consumption', form.get('alcohol_consumption', '')],
            ['Diet', form.get('diet', '')],
            ['Blood Pressure', form.get('blood_pressure', '')],
            ['Cholesterol (mg/dL)', str(form.get('cholesterol', ''))],
            ['Diabetes', 'Yes' if form.get('diabetes') else 'No'],
            ['Hypertension', 'Yes' if form.get('hypertension') else 'No'],
            ['Heart Disease', 'Yes' if form.get('heart_disease') else 'No'],
            ['Asthma', 'Yes' if form.get('asthma') else 'No']
        ]

        table = Table(profile_data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

        if insights:
            elements.append(Paragraph('Health Insights', styles['Heading2']))
            for ins in insights:
                elements.append(Paragraph('- ' + ins, styles['Normal']))
            elements.append(Spacer(1, 12))

        if recommendations:
            elements.append(Paragraph('Recommendations', styles['Heading2']))
            for rec in recommendations:
                elements.append(Paragraph('- ' + rec, styles['Normal']))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph('Disclaimer: This report is for informational purposes only and should not replace professional medical advice.', styles['Italic']))

        doc.build(elements)
        buffer.seek(0)

        filename = f"life-expectancy-report-{int(datetime.now().timestamp())}.pdf"
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_insights(profile):
    """Generate health insights based on profile"""
    insights = []
    bmi = profile['BMI']
    
    if bmi < 18.5:
        insights.append({'type': 'info', 'text': '⚠️ BMI indicates underweight - consider nutrition consultation'})
    elif 18.5 <= bmi < 25:
        insights.append({'type': 'success', 'text': '✓ BMI is in healthy range'})
    elif 25 <= bmi < 30:
        insights.append({'type': 'warning', 'text': '⚠️ BMI indicates overweight - consider diet and exercise'})
    else:
        insights.append({'type': 'danger', 'text': '❌ BMI indicates obesity - consult healthcare provider'})
    
    if profile['Cholesterol'] > 240:
        insights.append({'type': 'danger', 'text': '❌ High cholesterol - monitor and consult doctor'})
    elif profile['Cholesterol'] > 200:
        insights.append({'type': 'warning', 'text': '⚠️ Borderline cholesterol - keep monitoring'})
    else:
        insights.append({'type': 'success', 'text': '✓ Cholesterol levels are healthy'})
    
    if profile['Smoking_Status'] != 'Never':
        insights.append({'type': 'danger', 'text': '❌ Smoking significantly impacts life expectancy'})
    else:
        insights.append({'type': 'success', 'text': '✓ Non-smoking status is beneficial'})
    
    if profile['Physical_Activity'] == 'Low':
        insights.append({'type': 'warning', 'text': '⚠️ Low activity - aim for 30 mins daily exercise'})
    elif profile['Physical_Activity'] == 'High':
        insights.append({'type': 'success', 'text': '✓ Good physical activity level'})
    
    return insights


def generate_recommendations(profile):
    """Generate personalized health recommendations"""
    recommendations = []
    
    bmi = profile['BMI']
    if bmi >= 25:
        recommendations.append('💪 Maintain a healthy weight through regular exercise and balanced diet')
    
    if profile['Physical_Activity'] == 'Low':
        recommendations.append('🏃 Increase physical activity - aim for at least 30 minutes of exercise daily')
    
    if profile['Smoking_Status'] != 'Never':
        recommendations.append('🚭 Consider quitting smoking - it significantly impacts life expectancy')
    
    if profile['Alcohol_Consumption'] == 'High':
        recommendations.append('🍷 Reduce alcohol consumption for better health')
    
    if profile['Diabetes'] or profile['Hypertension'] or profile['Heart_Disease']:
        recommendations.append('❤️ Regular health checkups and medication adherence are crucial')
    
    if profile['Cholesterol'] > 240:
        recommendations.append('🧬 Monitor cholesterol levels and consult with a healthcare provider')
    
    if profile['Diet'] == 'Poor':
        recommendations.append('🥗 Improve diet quality - focus on whole foods and balanced nutrition')
    
    if not recommendations:
        recommendations.append('✨ Keep up the great lifestyle habits!')
    
    return recommendations


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Life Expectancy Prediction API is running'
    }), 200


if __name__ == '__main__':
    # Load model artifacts
    if load_artifacts():
        print("✓ Model artifacts loaded successfully")
        print("✓ Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("✗ Failed to load model artifacts")
        print("Please run 'python train.py' first")
