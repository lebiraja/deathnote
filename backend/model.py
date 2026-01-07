"""
Model training module for life expectancy prediction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os


class LifeExpectancyModel:
    """Train and manage life expectancy prediction models"""
    
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize the model
        
        Args:
            model_type (str): Type of model to use ('linear', 'random_forest', 'gradient_boosting')
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on model_type"""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train (np.ndarray or pd.DataFrame): Training features
            y_train (np.ndarray or pd.Series): Training labels
        """
        print(f"\n{'='*50}")
        print(f"Training {self.model_type} model...")
        print(f"{'='*50}")
        
        self.model.fit(X_train, y_train)
        
        # Print feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            print(f"\nTop 10 Important Features:")
            feature_importance = self.model.feature_importances_
            # Note: You need to track feature names externally
            for i, importance in enumerate(sorted(enumerate(feature_importance), 
                                                 key=lambda x: x[1], reverse=True)[:10]):
                print(f"  Feature {importance[0]}: {importance[1]:.4f}")
    
    def evaluate(self, X_test, y_test, set_name='Test'):
        """
        Evaluate the model
        
        Args:
            X_test (np.ndarray or pd.DataFrame): Test features
            y_test (np.ndarray or pd.Series): Test labels
            set_name (str): Name of the set (for logging)
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{set_name} Set Metrics:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.ndarray or pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def save_model(self, file_path):
        """
        Save the model
        
        Args:
            file_path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        """
        Load a trained model
        
        Args:
            file_path (str): Path to load the model from
        """
        self.model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
    
    def save_scaler(self, scaler, file_path):
        """
        Save the scaler
        
        Args:
            scaler: The scaler object
            file_path (str): Path to save the scaler
        """
        joblib.dump(scaler, file_path)
        print(f"Scaler saved to {file_path}")
    
    def load_scaler(self, file_path):
        """
        Load the scaler
        
        Args:
            file_path (str): Path to load the scaler from
            
        Returns:
            The loaded scaler object
        """
        scaler = joblib.load(file_path)
        self.scaler = scaler
        print(f"Scaler loaded from {file_path}")
        return scaler
