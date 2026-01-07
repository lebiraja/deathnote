#!/usr/bin/env python3
"""Quick training script to generate model files"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib
import os

# Generate sample data
np.random.seed(42)
n_samples = 5000

data = {
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Height': np.random.randint(150, 200, n_samples),
    'Weight': np.random.randint(45, 120, n_samples),
    'BMI': np.random.uniform(15, 40, n_samples),
    'Physical_Activity': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Smoking_Status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
    'Alcohol_Consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples),
    'Diet': np.random.choice(['Poor', 'Average', 'Good'], n_samples),
    'Blood_Pressure': np.random.choice(['Low', 'Normal', 'High'], n_samples),
    'Cholesterol': np.random.uniform(120, 300, n_samples),
    'Diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'Hypertension': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    'Heart_Disease': np.random.choice([0, 1], n_samples, p=[0.90, 0.10]),
    'Asthma': np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
}

df = pd.DataFrame(data)

# Calculate life expectancy based on factors
base_life_expectancy = 75
df['Life_Expectancy'] = base_life_expectancy
df['Life_Expectancy'] += np.where(df['Gender'] == 'Female', 5, 0)
df['Life_Expectancy'] -= (df['BMI'] - 22) * 0.3
df['Life_Expectancy'] += {'Low': -5, 'Medium': 0, 'High': 5}[df['Physical_Activity'].iloc[0]] * np.ones(n_samples)
df.loc[df['Physical_Activity'] == 'Low', 'Life_Expectancy'] -= 5
df.loc[df['Physical_Activity'] == 'High', 'Life_Expectancy'] += 5
df.loc[df['Smoking_Status'] == 'Current', 'Life_Expectancy'] -= 10
df.loc[df['Smoking_Status'] == 'Former', 'Life_Expectancy'] -= 3
df.loc[df['Blood_Pressure'] == 'High', 'Life_Expectancy'] -= 5
df.loc[df['Blood_Pressure'] == 'Low', 'Life_Expectancy'] -= 2
df['Life_Expectancy'] -= (df['Cholesterol'] - 180) * 0.02
df['Life_Expectancy'] -= df['Diabetes'] * 8
df['Life_Expectancy'] -= df['Hypertension'] * 5
df['Life_Expectancy'] -= df['Heart_Disease'] * 12
df['Life_Expectancy'] -= df['Asthma'] * 2
df['Life_Expectancy'] = df['Life_Expectancy'].clip(45, 95)

# Add some noise
df['Life_Expectancy'] += np.random.normal(0, 2, n_samples)

print(f"Generated {len(df)} training samples")
print(f"Life expectancy range: {df['Life_Expectancy'].min():.1f} - {df['Life_Expectancy'].max():.1f}")

# Prepare features and target
feature_cols = ['Gender', 'Height', 'Weight', 'BMI', 'Physical_Activity', 'Smoking_Status', 
                'Alcohol_Consumption', 'Diet', 'Blood_Pressure', 'Cholesterol', 
                'Diabetes', 'Hypertension', 'Heart_Disease', 'Asthma']
X = df[feature_cols]
y = df['Life_Expectancy']

# Create preprocessors
categorical_cols = ['Gender', 'Physical_Activity', 'Smoking_Status', 'Alcohol_Consumption', 'Diet', 'Blood_Pressure']
preprocessor = {}

X_copy = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_copy[col] = le.fit_transform(X_copy[col])
    preprocessor[col] = le

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_copy)

# Train models
print("Training Gradient Boosting model...")
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_scaled, y)

print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_scaled, y)

# Save models
os.makedirs('/tmp/models', exist_ok=True)
preprocessor['_feature_names'] = list(X_copy.columns)
joblib.dump(preprocessor, '/tmp/models/preprocessor.pkl')
joblib.dump(scaler, '/tmp/models/scaler.pkl')
joblib.dump(gb_model, '/tmp/models/gradient_boosting_model.pkl')
joblib.dump(lr_model, '/tmp/models/linear_model.pkl')

print("\n✅ Models saved successfully!")
print(f"Files saved to /tmp/models/:")
for f in os.listdir('/tmp/models'):
    size = os.path.getsize(f'/tmp/models/{f}')
    print(f"  - {f}: {size/1024:.1f} KB")
