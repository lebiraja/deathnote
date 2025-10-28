"""
Data loading module for life expectancy prediction
"""
import pandas as pd
import os


def load_data(file_path=None):
    """
    Load the life expectancy dataset
    
    Args:
        file_path (str): Path to the CSV file. If None, uses default path.
        
    Returns:
        pd.DataFrame: The loaded dataset
    """
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), 'life-expectancy.csv')
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df


def get_dataset_info(df):
    """
    Print basic information about the dataset
    
    Args:
        df (pd.DataFrame): The dataset
    """
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nBasic statistics:\n{df.describe()}")
