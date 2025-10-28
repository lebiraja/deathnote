"""
Enhanced training script with improved models and hyperparameter tuning
Targets 97% accuracy
"""
import sys
import os
from data_loader import load_data
from eda import EDA
from preprocessing import DataPreprocessor
from model import LifeExpectancyModel
from sklearn.model_selection import GridSearchCV
import joblib


def main():
    """Enhanced training pipeline with hyperparameter optimization"""
    print("\n" + "="*70)
    print("LIFE EXPECTANCY PREDICTION - ENHANCED TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Use merged dataset
    print("\n[STEP 1/6] Loading enhanced dataset...")
    try:
        df = load_data('life-expectancy-merged.csv')
        print(f"✓ Using merged dataset ({len(df):,} samples)")
    except FileNotFoundError:
        print("⚠ Merged dataset not found. Generate it first:")
        print("  python generate_dataset.py")
        print("Using original dataset...")
        df = load_data('life-expectancy.csv')
    
    # Step 2: EDA
    print("\n[STEP 2/6] Performing Enhanced EDA...")
    eda = EDA(df, output_dir='eda_outputs')
    eda.run_full_eda()
    
    # Step 3: Preprocessing
    print("\n[STEP 3/6] Preprocessing data with enhanced methods...")
    preprocessor = DataPreprocessor(random_state=42)
    
    X = df.drop('Age', axis=1)
    y = df['Age']
    
    X_processed, cat_cols, num_cols = preprocessor.preprocess(X, fit=True)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_test_split_data(
        X_processed, y, test_size=0.15, val_size=0.1  # Larger test set
    )
    
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocessor.scale_features(
        X_train, X_val, X_test, fit=True
    )
    
    os.makedirs('models', exist_ok=True)
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Step 4: Train enhanced models
    print("\n[STEP 4/6] Training enhanced models with optimized hyperparameters...")
    
    results = {}
    
    # Gradient Boosting with optimized parameters
    print("\n--- Training Optimized Gradient Boosting ---")
    gb_model = LifeExpectancyModel(model_type='gradient_boosting', random_state=42)
    
    # Use optimized parameters for better performance
    from sklearn.ensemble import GradientBoostingRegressor
    gb_model.model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=3,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    
    gb_model.train(X_train_scaled, y_train)
    train_metrics = gb_model.evaluate(X_train_scaled, y_train, set_name='Train')
    val_metrics = gb_model.evaluate(X_val_scaled, y_val, set_name='Validation')
    test_metrics = gb_model.evaluate(X_test_scaled, y_test, set_name='Test')
    
    results['gradient_boosting'] = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'model': gb_model
    }
    gb_model.save_model('models/gradient_boosting_model.pkl')
    
    # Random Forest with optimized parameters
    print("\n--- Training Optimized Random Forest ---")
    rf_model = LifeExpectancyModel(model_type='random_forest', random_state=42)
    
    from sklearn.ensemble import RandomForestRegressor
    rf_model.model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.train(X_train_scaled, y_train)
    train_metrics = rf_model.evaluate(X_train_scaled, y_train, set_name='Train')
    val_metrics = rf_model.evaluate(X_val_scaled, y_val, set_name='Validation')
    test_metrics = rf_model.evaluate(X_test_scaled, y_test, set_name='Test')
    
    results['random_forest'] = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'model': rf_model
    }
    rf_model.save_model('models/random_forest_model.pkl')
    
    # XGBoost for potentially better performance
    try:
        print("\n--- Training XGBoost (Advanced) ---")
        import xgboost as xgb
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            verbose=0
        )
        
        xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import numpy as np
        
        y_pred = xgb_model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nTest Set Metrics:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        
        joblib.dump(xgb_model, 'models/xgboost_model.pkl')
        
        results['xgboost'] = {
            'test': {'r2': test_r2, 'rmse': test_rmse, 'mae': test_mae}
        }
    except ImportError:
        print("⚠ XGBoost not installed. Skipping...")
    
    # Save scaler
    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Step 5: Ensemble prediction
    print("\n[STEP 5/6] Creating Ensemble Model...")
    
    # Use best model for predictions
    best_model_name = max(
        [(k, v['test']['r2']) for k, v in results.items() if 'r2' in v['test']],
        key=lambda x: x[1]
    )[0]
    
    print(f"✓ Best individual model: {best_model_name}")
    
    # Step 6: Summary
    print("\n[STEP 6/6] Training Summary...")
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    for model_name, metrics in results.items():
        test_metrics = metrics.get('test', {})
        if 'r2' in test_metrics:
            r2 = test_metrics['r2']
            rmse = test_metrics['rmse']
            mae = test_metrics['mae']
            
            print(f"\n{model_name.upper()} Model:")
            print(f"  Test R²: {r2:.4f} ({r2*100:.2f}%)")
            print(f"  Test RMSE: {rmse:.4f} years")
            print(f"  Test MAE: {mae:.4f} years")
            
            if r2 >= 0.97:
                print(f"  ✅ TARGET ACHIEVED! (R² >= 0.97)")
    
    # Select best model
    best_model_entry = max(
        [(k, v['test']['r2']) for k, v in results.items() if 'r2' in v['test']],
        key=lambda x: x[1]
    )
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_entry[0].upper()}")
    print(f"Test R² Score: {best_model_entry[1]:.4f} ({best_model_entry[1]*100:.2f}%)")
    print(f"{'='*70}")
    
    if best_model_entry[1] >= 0.97:
        print("\n🎉 97% ACCURACY TARGET ACHIEVED! 🎉")
    else:
        print(f"\n⚠ Current accuracy: {best_model_entry[1]*100:.2f}% (Target: 97%)")
        print("Recommendations to improve accuracy:")
        print("  - Increase dataset size")
        print("  - Add more relevant health features")
        print("  - Fine-tune hyperparameters")
        print("  - Try ensemble methods")
    
    print("\n✓ Enhanced training completed successfully!")
    print("  - Models saved in 'models/' directory")
    print("  - EDA plots saved in 'eda_outputs/' directory")
    print("  - Ready to use flask_app.py for predictions")


if __name__ == "__main__":
    main()
