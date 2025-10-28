"""
Streamlit app for life expectancy prediction
Interactive user interface for making predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from preprocessing import DataPreprocessor


# Page configuration
st.set_page_config(
    page_title="Life Expectancy Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
        .main {
            padding: 20px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load trained model, scaler, and preprocessor"""
    try:
        model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        return model, scaler, preprocessor
    except FileNotFoundError:
        st.error("❌ Model artifacts not found. Please run train.py first.")
        st.stop()


def prepare_input(user_data, preprocessor, scaler):
    """Prepare user input for prediction"""
    # Create a DataFrame with user input
    df_user = pd.DataFrame([user_data])
    
    # Encode categorical features using the preprocessor's label encoders
    for col, le in preprocessor.items():
        if col in df_user.columns:
            try:
                df_user[col] = le.transform(df_user[col])
            except ValueError as e:
                st.error(f"Error encoding {col}: {str(e)}")
                return None
    
    # Scale features
    df_scaled = scaler.transform(df_user)
    
    return df_scaled


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #667eea;'>🏥 Life Expectancy Predictor</h1>
        <p style='text-align: center; color: #666;'>Predict life expectancy based on health and lifestyle factors</p>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Load model and preprocessor
    model, scaler, preprocessor = load_artifacts()
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.slider("Height (cm)", min_value=140, max_value=220, value=170, step=1)
        weight = st.slider("Weight (kg)", min_value=40, max_value=150, value=70, step=1)
        
        st.subheader("💪 Lifestyle Factors")
        physical_activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Consumption", ["Moderate", "High", "None"])
        diet = st.selectbox("Diet Type", ["Poor", "Average", "Healthy"])
    
    with col2:
        st.subheader("❤️ Health Indicators")
        blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
        cholesterol = st.slider("Cholesterol (mg/dL)", min_value=100, max_value=300, value=200, step=1)
        
        st.subheader("🏥 Medical History")
        diabetes = st.checkbox("Diabetes", value=False)
        hypertension = st.checkbox("Hypertension", value=False)
        heart_disease = st.checkbox("Heart Disease", value=False)
        asthma = st.checkbox("Asthma", value=False)
    
    st.divider()
    
    # Calculate BMI
    bmi = weight / (height / 100) ** 2
    
    # Prepare prediction input
    user_data = {
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'BMI': bmi,
        'Physical_Activity': physical_activity,
        'Smoking_Status': smoking,
        'Alcohol_Consumption': alcohol,
        'Diet': diet,
        'Blood_Pressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Diabetes': int(diabetes),
        'Hypertension': int(hypertension),
        'Heart_Disease': int(heart_disease),
        'Asthma': int(asthma)
    }
    
    # Display input summary
    st.subheader("📊 Your Profile Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gender", gender)
    with col2:
        st.metric("Height", f"{height} cm")
    with col3:
        st.metric("Weight", f"{weight} kg")
    with col4:
        st.metric("BMI", f"{bmi:.1f}")
    
    # Prediction button
    if st.button("🔮 Predict Life Expectancy", use_container_width=True):
        with st.spinner("Analyzing your data..."):
            # Prepare input
            user_input = prepare_input(user_data, preprocessor, scaler)
            
            if user_input is not None:
                # Make prediction
                prediction = model.predict(user_input)[0]
                
                # Display result
                st.divider()
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    border-radius: 15px; color: white;'>
                            <h2>Predicted Life Expectancy</h2>
                            <h1 style='font-size: 48px; margin: 20px 0;'>{prediction:.1f} years</h1>
                            <p style='font-size: 16px; margin-top: 20px;'>Based on your health profile</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # Additional insights
                st.subheader("💡 Health Insights")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **Your Profile:**
                    - BMI: {bmi:.1f} {"(Healthy)" if 18.5 <= bmi < 25 else "(Monitor your weight)"}
                    - Smoking: {smoking}
                    - Physical Activity: {physical_activity}
                    - Diet: {diet}
                    """)
                
                with col2:
                    st.warning(f"""
                    **Medical Conditions:**
                    - Diabetes: {"Yes ⚠️" if diabetes else "No ✓"}
                    - Hypertension: {"Yes ⚠️" if hypertension else "No ✓"}
                    - Heart Disease: {"Yes ⚠️" if heart_disease else "No ✓"}
                    - Asthma: {"Yes ⚠️" if asthma else "No ✓"}
                    """)
                
                st.divider()
                
                # Recommendations
                st.subheader("🎯 Recommendations")
                recommendations = []
                
                if bmi >= 25:
                    recommendations.append("💪 Maintain a healthy weight through regular exercise and balanced diet")
                if physical_activity == "Low":
                    recommendations.append("🏃 Increase physical activity - aim for at least 30 minutes of exercise daily")
                if smoking != "Never":
                    recommendations.append("🚭 Consider quitting smoking - it significantly impacts life expectancy")
                if alcohol == "High":
                    recommendations.append("🍷 Reduce alcohol consumption for better health")
                if diabetes or hypertension or heart_disease:
                    recommendations.append("❤️ Regular health checkups and medication adherence are crucial")
                if cholesterol > 240:
                    recommendations.append("🧬 Monitor cholesterol levels and consult with a healthcare provider")
                
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("✨ Keep up the great lifestyle habits!")


if __name__ == "__main__":
    main()
