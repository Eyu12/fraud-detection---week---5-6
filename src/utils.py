"""
Utility functions for fraud detection project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def save_dataframe(df, path, index=False):
    """Save dataframe to file."""
    if path.endswith('.csv'):
        df.to_csv(path, index=index)
    elif path.endswith('.parquet'):
        df.to_parquet(path, index=index)
    elif path.endswith('.pkl'):
        df.to_pickle(path)
    else:
        raise ValueError("Unsupported file format. Use .csv, .parquet, or .pkl")


def load_dataframe(path):
    """Load dataframe from file."""
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.pkl'):
        return pd.read_pickle(path)
    else:
        raise ValueError("Unsupported file format. Use .csv, .parquet, or .pkl")


def save_model(model, path):
    """Save trained model to file."""
    if path.endswith('.pkl') or path.endswith('.joblib'):
        joblib.dump(model, path)
    elif path.endswith('.json'):
        # For tree-based models with JSON export
        if hasattr(model, 'save_model'):
            model.save_model(path)
        else:
            raise ValueError("Model doesn't support JSON export")
    else:
        raise ValueError("Unsupported file format. Use .pkl, .joblib, or .json")


def load_model(path):
    """Load trained model from file."""
    if path.endswith('.pkl') or path.endswith('.joblib'):
        return joblib.load(path)
    elif path.endswith('.json'):
        # For tree-based models
        import xgboost as xgb
        return xgb.Booster(model_file=path)
    else:
        raise ValueError("Unsupported file format. Use .pkl, .joblib, or .json")


def plot_class_distribution(y, title="Class Distribution"):
    """Plot class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    class_counts = pd.Series(y).value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[0])
    axes[0].set_title(f'{title} - Count')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    
    # Add percentage labels
    total = len(y)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + total*0.01, f'{v}\n({v/total*100:.1f}%)', 
                    ha='center', va='bottom')
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=['Legitimate', 'Fraud'], 
               autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])
    axes[1].set_title(f'{title} - Percentage')
    
    plt.tight_layout()
    return fig


def create_summary_statistics(df):
    """Create summary statistics for dataframe."""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median())
        }
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_values': int(df[col].nunique()),
            'top_value': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None,
            'top_frequency': int(df[col].value_counts().iloc[0]) if len(df[col]) > 0 else 0
        }
    
    return summary


def save_summary(summary, path):
    """Save summary statistics to JSON file."""
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


def timer_decorator(func):
    """Decorator to measure function execution time."""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper