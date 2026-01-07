"""
EDA (Exploratory Data Analysis) module for life expectancy prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class EDA:
    """Perform exploratory data analysis on the dataset"""
    
    def __init__(self, df, output_dir='eda_outputs'):
        """
        Initialize EDA
        
        Args:
            df (pd.DataFrame): The dataset
            output_dir (str): Directory to save EDA plots
        """
        self.df = df.copy()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def analyze_target_variable(self):
        """Analyze the target variable (Age/Life Expectancy)"""
        print("\n" + "="*50)
        print("TARGET VARIABLE ANALYSIS (Age/Life Expectancy)")
        print("="*50)
        
        age = self.df['Age']
        print(f"Mean: {age.mean():.2f}")
        print(f"Median: {age.median():.2f}")
        print(f"Std Dev: {age.std():.2f}")
        print(f"Min: {age.min():.2f}")
        print(f"Max: {age.max():.2f}")
        
        # Plot distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(age, bins=50, edgecolor='black', color='skyblue')
        axes[0].set_xlabel('Age (Life Expectancy)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Life Expectancy')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot(age)
        axes[1].set_ylabel('Age')
        axes[1].set_title('Boxplot of Life Expectancy')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_target_distribution.png'), dpi=100)
        plt.close()
    
    def analyze_categorical_features(self):
        """Analyze categorical features"""
        print("\n" + "="*50)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*50)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print(f"Categorical columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
        
        # Create visualizations for categorical features
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(categorical_cols):
            if idx < len(axes):
                sns.boxplot(data=self.df, x=col, y='Age', ax=axes[idx])
                axes[idx].set_title(f'Age by {col}')
                axes[idx].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_categorical_analysis.png'), dpi=100)
        plt.close()
    
    def analyze_numerical_features(self):
        """Analyze numerical features"""
        print("\n" + "="*50)
        print("NUMERICAL FEATURES ANALYSIS")
        print("="*50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Age' in numerical_cols:
            numerical_cols.remove('Age')
        
        print(f"Numerical columns: {numerical_cols}")
        print(f"\nStatistics:\n{self.df[numerical_cols].describe()}")
        
        # Correlation with target
        print(f"\nCorrelation with Age (Life Expectancy):")
        correlations = self.df[numerical_cols + ['Age']].corr()['Age'].sort_values(ascending=False)
        print(correlations)
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = self.df[numerical_cols + ['Age']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_correlation_heatmap.png'), dpi=100)
        plt.close()
        
        # Distribution plots for numerical features
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols[:6]):
            axes[idx].scatter(self.df[col], self.df['Age'], alpha=0.5)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Age')
            axes[idx].set_title(f'Age vs {col}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_numerical_scatter.png'), dpi=100)
        plt.close()
    
    def analyze_missing_values(self):
        """Analyze missing values"""
        print("\n" + "="*50)
        print("MISSING VALUES ANALYSIS")
        print("="*50)
        
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])
    
    def run_full_eda(self):
        """Run complete EDA"""
        print("\n" + "="*70)
        print("STARTING EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        self.analyze_missing_values()
        self.analyze_target_variable()
        self.analyze_categorical_features()
        self.analyze_numerical_features()
        
        print("\n" + "="*70)
        print("EDA COMPLETED - Plots saved to:", self.output_dir)
        print("="*70)
