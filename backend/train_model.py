#!/usr/bin/env python3
"""
Model training script for Docker container
Trains the life expectancy prediction model using available data
"""
import sys
import os
from pathlib import Path
from data_loader import load_data
from preprocessing import DataPreprocessor
from model import LifeExpectancyModel
import joblib

def main():
    """Training pipeline for Docker container"""
    print("\n" + "="*70)
    print("LIFE EXPECTANCY MODEL TRAINING - DOCKER CONTAINER")
    print("="*70)
    
    # Ensure output directories exist
    models_dir = Path("/app/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load dataset
    print("\n[STEP 1/5] Loading dataset...")
    data_paths = [
        '/app/data/life-expectancy-merged.csv',
        '/app/data/life-expectancy-enhanced.csv',
        '/app/data/life-expectancy.csv'
    ]
    
    df = None
    for data_path in data_paths:
        if os.path.exists(data_path):
            try:
                df = load_data(data_path)
                print(f"✓ Loaded dataset from {data_path} ({len(df):,} samples)")
                break
            except Exception as e:
                print(f"⚠ Failed to load {data_path}: {e}")
                continue
    
    if df is None:
        print("❌ ERROR: No dataset found!")
        sys.exit(1)
    
    # Step 2: Preprocessing
    print("\n[STEP 2/5] Preprocessing data...")
    preprocessor = DataPreprocessor(random_state=42)
    
    X = df.drop('Age', axis=1)
    y = df['Age']
    
    X_processed, cat_cols, num_cols = preprocessor.preprocess(X, fit=True)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_test_split_data(
        X_processed, y, test_size=0.15, val_size=0.1
    )
    
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocessor.scale_features(
        X_train, X_val, X_test
    )
    
    print(f"✓ Train set: {len(X_train_scaled):,} samples")
    print(f"✓ Validation set: {len(X_val_scaled):,} samples")
    print(f"✓ Test set: {len(X_test_scaled):,} samples")
    
    # Step 3: Train models
    print("\n[STEP 3/5] Training models...")
    model_trainer = LifeExpectancyModel()
    
    results = model_trainer.train_multiple_models(
        X_train_scaled, y_train,
        X_val_scaled, y_val
    )
    
    # Step 4: Evaluate
    print("\n[STEP 4/5] Evaluating models on test set...")
    test_results = model_trainer.evaluate_on_test_set(X_test_scaled, y_test)
    
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    for name, metrics in test_results.items():
        print(f"\n{name}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
    
    # Step 5: Save models
    print("\n[STEP 5/5] Saving models and preprocessor...")
    
    # Save best model
    best_model_name = model_trainer.get_best_model_name(results)
    best_model = model_trainer.models[best_model_name]
    
    model_path = models_dir / 'best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"✓ Saved best model ({best_model_name}) to {model_path}")
    
    # Save all models
    all_models_path = models_dir / 'all_models.pkl'
    joblib.dump(model_trainer.models, all_models_path)
    print(f"✓ Saved all models to {all_models_path}")
    
    # Save preprocessor components
    preprocessor_data = {
        'scaler': scaler,
        'categorical_columns': cat_cols,
        'numerical_columns': num_cols,
        'preprocessor': preprocessor
    }
    preprocessor_path = models_dir / 'preprocessor.pkl'
    joblib.dump(preprocessor_data, preprocessor_path)
    print(f"✓ Saved preprocessor to {preprocessor_path}")
    
    # Save metadata
    metadata = {
        'best_model': best_model_name,
        'test_results': test_results,
        'train_samples': len(X_train_scaled),
        'val_samples': len(X_val_scaled),
        'test_samples': len(X_test_scaled),
        'feature_count': X_train_scaled.shape[1]
    }
    metadata_path = models_dir / 'model_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"✓ Saved metadata to {metadata_path}")
    
    print("\n" + "="*70)
    print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Best Model: {best_model_name}")
    print(f"Test R² Score: {test_results[best_model_name]['r2']:.4f}")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
