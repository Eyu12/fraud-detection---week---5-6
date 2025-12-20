"""
Data preprocessing functions for fraud detection.
Handles data cleaning, missing values, and basic transformations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocesses fraud detection datasets."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.numeric_imputer = None
        self.categorical_imputer = None
        
    def preprocess_fraud_data(self, df):
        """Preprocess e-commerce fraud data."""
        df_clean = df.copy()
        
        # Convert timestamps
        if 'signup_time' in df_clean.columns:
            df_clean['signup_time'] = pd.to_datetime(df_clean['signup_time'])
        if 'purchase_time' in df_clean.columns:
            df_clean['purchase_time'] = pd.to_datetime(df_clean['purchase_time'])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        return df_clean
    
    def preprocess_creditcard_data(self, df):
        """Preprocess credit card fraud data."""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # No missing values in this dataset typically
        if df_clean.isnull().sum().sum() > 0:
            df_clean = self._handle_missing_values(df_clean)
        
        return df_clean
    
    def _handle_missing_values(self, df):
        """Handle missing values in dataframe."""
        df_clean = df.copy()
        
        # Numeric columns
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] 
                                                   if len(df_clean[col].mode()) > 0 
                                                   else 'unknown')
        
        return df_clean
    
    def convert_ip_to_int(self, ip_address):
        """Convert IP address to integer for range lookup."""
        if pd.isna(ip_address) or ip_address == 'unknown':
            return None
        try:
            octets = list(map(int, str(ip_address).split('.')))
            return (octets[0] << 24) + (octets[1] << 16) + (octets[2] << 8) + octets[3]
        except:
            return None


def merge_ip_country(fraud_df, ip_country_df):
    """
    Merge fraud data with IP address to country mapping.
    Uses binary search for efficient lookup.
    """
    if 'ip_address' not in fraud_df.columns:
        return fraud_df
    
    # Sort IP country data for binary search
    ip_country_sorted = ip_country_df.sort_values('lower_bound_ip_address').reset_index(drop=True)
    
    def find_country(ip_int):
        """Find country using binary search."""
        if ip_int is None:
            return 'Unknown'
        
        low, high = 0, len(ip_country_sorted) - 1
        while low <= high:
            mid = (low + high) // 2
            lower_bound = ip_country_sorted.loc[mid, 'lower_bound_ip_address']
            upper_bound = ip_country_sorted.loc[mid, 'upper_bound_ip_address']
            
            if lower_bound <= ip_int <= upper_bound:
                return ip_country_sorted.loc[mid, 'country']
            elif ip_int < lower_bound:
                high = mid - 1
            else:
                low = mid + 1
        
        return 'Unknown'
    
    # Convert IPs and find countries for a sample
    fraud_sample = fraud_df.copy()
    fraud_sample['ip_int'] = fraud_sample['ip_address'].apply(
        lambda x: DataPreprocessor().convert_ip_to_int(x)
    )
    fraud_sample['country'] = fraud_sample['ip_int'].apply(find_country)
    
    return fraud_sample