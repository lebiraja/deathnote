"""
Preprocessing module for life expectancy prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import os


class DataPreprocessor:
    """Handle data preprocessing and feature engineering"""
    
    def __init__(self, random_state=42):
        """
        Initialize the preprocessor
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        self.label_encoders = {}
    
    def preprocess(self, df, fit=True):
        """
        Preprocess the dataset

        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the preprocessor or just transform

        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df_processed = df.copy()

        # Handle missing values in categorical columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            df_processed[col] = df_processed[col].fillna('None')

        # Identify categorical and numerical columns (excluding target)
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

        if 'Age' in numerical_cols:
            numerical_cols.remove('Age')

        print(f"\nCategorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")

        # Encode categorical features
        if fit:
            self.label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                print(f"\n{col} encoding:")
                for i, class_name in enumerate(le.classes_):
                    print(f"  {class_name}: {i}")
        else:
            for col in categorical_cols:
                if col in self.label_encoders:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])

        return df_processed, categorical_cols, numerical_cols
    
    def train_test_split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test set size
            val_size (float): Validation set size (as fraction of remaining after test split)
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=self.random_state
        )
        
        print(f"\nData split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test, fit=True):
        """
        Scale numerical features
        
        Args:
            X_train (pd.DataFrame): Training features
            X_val (pd.DataFrame): Validation features
            X_test (pd.DataFrame): Test features
            fit (bool): Whether to fit the scaler
            
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
        """
        scaler = StandardScaler()
        
        if fit:
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            X_train_scaled = scaler.transform(X_train)
        
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def save_preprocessor(self, file_path):
        """Save the preprocessor"""
        joblib.dump(self.label_encoders, file_path)
        print(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path):
        """Load the preprocessor"""
        self.label_encoders = joblib.load(file_path)
        print(f"Preprocessor loaded from {file_path}")
