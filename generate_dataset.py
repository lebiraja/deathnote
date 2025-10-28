"""
Enhanced dataset creation for improved accuracy
Generates synthetic data based on real-world health patterns
"""
import pandas as pd
import numpy as np
import os

def generate_enhanced_dataset(output_file='life-expectancy-enhanced.csv', n_samples=50000):
    """
    Generate an enhanced, realistic dataset for life expectancy prediction
    Based on real-world health correlations and medical research
    
    Args:
        output_file (str): Output filename
        n_samples (int): Number of samples to generate
    """
    np.random.seed(42)
    
    print(f"Generating {n_samples:,} samples of enhanced health data...")
    
    # Generate base features
    data = {}
    
    # Gender distribution
    data['Gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.51, 0.49])
    
    # Height (cm) - realistic distribution
    male_height = np.random.normal(177, 7, n_samples // 2)
    female_height = np.random.normal(163, 6, n_samples // 2)
    data['Height'] = np.concatenate([
        np.clip(male_height, 150, 210),
        np.clip(female_height, 140, 195)
    ])
    
    # Weight (kg) - correlated with height
    data['Weight'] = []
    for i, (gender, height) in enumerate(zip(data['Gender'], data['Height'])):
        base_bmi = np.random.normal(24.5, 4)
        weight = (height / 100) ** 2 * base_bmi
        data['Weight'].append(np.clip(weight, 40, 150))
    data['Weight'] = np.array(data['Weight'])
    
    # BMI (calculated)
    data['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2
    
    # Physical Activity - affects life expectancy significantly
    data['Physical_Activity'] = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.30, 0.40, 0.30])
    
    # Smoking Status - major impact
    data['Smoking_Status'] = np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.60, 0.25, 0.15])
    
    # Alcohol Consumption
    data['Alcohol_Consumption'] = np.random.choice(['None', 'Moderate', 'High'], n_samples, p=[0.40, 0.45, 0.15])
    
    # Diet - strongly correlated with life expectancy
    data['Diet'] = np.random.choice(['Poor', 'Average', 'Healthy'], n_samples, p=[0.20, 0.50, 0.30])
    
    # Blood Pressure - health indicator
    data['Blood_Pressure'] = np.random.choice(['Low', 'Normal', 'High'], n_samples, p=[0.10, 0.70, 0.20])
    
    # Cholesterol (mg/dL) - continuous
    data['Cholesterol'] = np.random.normal(200, 40, n_samples)
    data['Cholesterol'] = np.clip(data['Cholesterol'], 100, 300)
    
    # Medical conditions - independent probabilities
    data['Diabetes'] = np.random.binomial(1, 0.08, n_samples)
    data['Hypertension'] = np.random.binomial(1, 0.18, n_samples)
    data['Heart_Disease'] = np.random.binomial(1, 0.11, n_samples)
    data['Asthma'] = np.random.binomial(1, 0.07, n_samples)
    
    # Generate Life Expectancy (Age) based on factors
    print("Calculating life expectancy based on health factors...")
    ages = []
    
    for i in range(n_samples):
        # Base age (world average life expectancy)
        age = 73.0
        
        # BMI impact (-0.178 correlation)
        bmi = data['BMI'][i]
        if bmi < 18.5:
            age -= 2
        elif 18.5 <= bmi < 25:
            age += 1
        elif 25 <= bmi < 30:
            age -= 1
        else:
            age -= 5
        
        # Physical Activity impact (+0.3 correlation)
        if data['Physical_Activity'][i] == 'High':
            age += 4
        elif data['Physical_Activity'][i] == 'Low':
            age -= 3
        
        # Smoking Status impact (-0.4 correlation)
        if data['Smoking_Status'][i] == 'Never':
            age += 3
        elif data['Smoking_Status'][i] == 'Former':
            age += 1
        else:  # Current
            age -= 7
        
        # Diet impact (+0.25 correlation)
        if data['Diet'][i] == 'Healthy':
            age += 3
        elif data['Diet'][i] == 'Poor':
            age -= 3
        
        # Alcohol impact
        if data['Alcohol_Consumption'][i] == 'None':
            age += 0.5
        elif data['Alcohol_Consumption'][i] == 'High':
            age -= 3
        
        # Blood Pressure impact
        if data['Blood_Pressure'][i] == 'Low':
            age -= 1
        elif data['Blood_Pressure'][i] == 'High':
            age -= 4
        
        # Cholesterol impact (-0.26 correlation)
        chol = data['Cholesterol'][i]
        if chol > 240:
            age -= 4
        elif chol > 200:
            age -= 2
        else:
            age += 1
        
        # Medical conditions impact
        if data['Diabetes'][i]:
            age -= 6
        if data['Hypertension'][i]:
            age -= 5
        if data['Heart_Disease'][i]:
            age -= 8
        if data['Asthma'][i]:
            age -= 2
        
        # Add some random variation
        age += np.random.normal(0, 2)
        
        # Gender adjustment (females tend to live longer)
        if data['Gender'][i] == 'Female':
            age += 3.5
        
        # Clip age to realistic range
        age = np.clip(age, 40, 100)
        ages.append(age)
    
    data['Age'] = np.array(ages)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Enhanced dataset created: {output_file}")
    print(f"  Samples: {n_samples:,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Life Expectancy Range: {df['Age'].min():.1f} - {df['Age'].max():.1f} years")
    print(f"  Mean Life Expectancy: {df['Age'].mean():.1f} years")
    print(f"  Std Dev: {df['Age'].std():.2f}")
    
    return df


def merge_with_original(original_file='life-expectancy.csv', 
                        enhanced_file='life-expectancy-enhanced.csv',
                        output_file='life-expectancy-merged.csv'):
    """
    Merge original and enhanced datasets for better diversity
    """
    print("\nMerging original and enhanced datasets...")
    
    df_original = pd.read_csv(original_file)
    df_enhanced = pd.read_csv(enhanced_file)
    
    # Combine datasets
    df_merged = pd.concat([df_original, df_enhanced], ignore_index=True)
    
    # Shuffle
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df_merged.to_csv(output_file, index=False)
    
    print(f"✓ Merged dataset created: {output_file}")
    print(f"  Total Samples: {len(df_merged):,}")
    print(f"  Original: {len(df_original):,}")
    print(f"  Enhanced: {len(df_enhanced):,}")
    print(f"  Life Expectancy Range: {df_merged['Age'].min():.1f} - {df_merged['Age'].max():.1f} years")
    
    return df_merged


if __name__ == "__main__":
    # Generate enhanced dataset
    df_enhanced = generate_enhanced_dataset(n_samples=40000)
    
    # Merge with original dataset
    df_merged = merge_with_original()
    
    print("\n" + "="*60)
    print("Dataset Summary Statistics")
    print("="*60)
    print(df_merged.describe())
