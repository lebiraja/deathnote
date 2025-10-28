"""
Training script for life expectancy prediction model
Runs EDA, trains model, and saves artifacts
"""
import sys
import os
from data_loader import load_data
from eda import EDA
from preprocessing import DataPreprocessor
from model import LifeExpectancyModel


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("LIFE EXPECTANCY PREDICTION MODEL - TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    print("\n[STEP 1/5] Loading data...")
    df = load_data('life-expectancy.csv')
    
    # Step 2: Exploratory Data Analysis
    print("\n[STEP 2/5] Performing EDA...")
    eda = EDA(df, output_dir='eda_outputs')
    eda.run_full_eda()
    
    # Step 3: Preprocessing
    print("\n[STEP 3/5] Preprocessing data...")
    preprocessor = DataPreprocessor(random_state=42)
    
    # Separate features and target
    X = df.drop('Age', axis=1)
    y = df['Age']
    
    # Preprocess
    X_processed, cat_cols, num_cols = preprocessor.preprocess(X, fit=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_test_split_data(
        X_processed, y, test_size=0.2, val_size=0.1
    )
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocessor.scale_features(
        X_train, X_val, X_test, fit=True
    )
    
    # Save preprocessing artifacts
    os.makedirs('models', exist_ok=True)
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Step 4: Train models
    print("\n[STEP 4/5] Training models...")
    models_to_train = ['linear', 'random_forest', 'gradient_boosting']
    results = {}
    
    for model_type in models_to_train:
        print(f"\n--- Training {model_type} model ---")
        
        # Initialize and train model
        le_model = LifeExpectancyModel(model_type=model_type, random_state=42)
        le_model.train(X_train_scaled, y_train)
        
        # Evaluate on all sets
        train_metrics = le_model.evaluate(X_train_scaled, y_train, set_name='Train')
        val_metrics = le_model.evaluate(X_val_scaled, y_val, set_name='Validation')
        test_metrics = le_model.evaluate(X_test_scaled, y_test, set_name='Test')
        
        results[model_type] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        
        # Save model
        model_path = f'models/{model_type}_model.pkl'
        le_model.save_model(model_path)
    
    # Save scaler
    scaler_path = 'models/scaler.pkl'
    from joblib import dump
    dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Step 5: Summary
    print("\n[STEP 5/5] Training Summary...")
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    for model_type, metrics in results.items():
        print(f"\n{model_type.upper()} Model:")
        print(f"  Train R²: {metrics['train']['r2']:.4f}, RMSE: {metrics['train']['rmse']:.4f}")
        print(f"  Val R²:   {metrics['val']['r2']:.4f}, RMSE: {metrics['val']['rmse']:.4f}")
        print(f"  Test R²:  {metrics['test']['r2']:.4f}, RMSE: {metrics['test']['rmse']:.4f}")
    
    # Select best model based on test R² score
    best_model = max(results.items(), key=lambda x: x[1]['test']['r2'])
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model[0].upper()}")
    print(f"Test R² Score: {best_model[1]['test']['r2']:.4f}")
    print(f"Test RMSE: {best_model[1]['test']['rmse']:.4f}")
    print(f"{'='*70}")
    
    print("\n✓ Training completed successfully!")
    print("  - Models saved in 'models/' directory")
    print("  - EDA plots saved in 'eda_outputs/' directory")
    print("  - Ready to use app.py for predictions")


if __name__ == "__main__":
    main()
