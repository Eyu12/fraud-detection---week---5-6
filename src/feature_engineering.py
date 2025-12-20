"""
Feature engineering functions for fraud detection.
Creates time-based, behavioral, and risk features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FeatureEngineer:
    """Creates features for fraud detection models."""
    
    def __init__(self):
        self.feature_stats = {}
        
    def engineer_fraud_features(self, df):
        """Create features for e-commerce fraud data."""
        df_featured = df.copy()
        
        # Time-based features
        df_featured = self._create_time_features(df_featured)
        
        # User behavior features
        df_featured = self._create_behavior_features(df_featured)
        
        # Transaction features
        df_featured = self._create_transaction_features(df_featured)
        
        # Risk score features
        df_featured = self._create_risk_features(df_featured)
        
        # Store feature statistics
        self._store_feature_stats(df_featured)
        
        return df_featured
    
    def engineer_creditcard_features(self, df):
        """Create features for credit card fraud data."""
        df_featured = df.copy()
        
        # Time-based features
        if 'Time' in df_featured.columns:
            df_featured['transaction_hour'] = (df_featured['Time'] // 3600) % 24
            df_featured['is_night'] = ((df_featured['transaction_hour'] >= 22) | 
                                      (df_featured['transaction_hour'] <= 6)).astype(int)
        
        # Amount features
        if 'Amount' in df_featured.columns:
            df_featured['amount_log'] = np.log1p(df_featured['Amount'])
            df_featured['amount_scaled'] = (df_featured['Amount'] - df_featured['Amount'].mean()) / df_featured['Amount'].std()
            df_featured['high_amount'] = (df_featured['Amount'] > df_featured['Amount'].quantile(0.95)).astype(int)
        
        # Time since last transaction
        if 'Time' in df_featured.columns:
            df_featured = df_featured.sort_values('Time')
            df_featured['time_since_last'] = df_featured['Time'].diff().fillna(0)
            df_featured['rapid_transaction'] = (df_featured['time_since_last'] < 60).astype(int)
        
        # PCA component interactions
        pca_cols = [f'V{i}' for i in range(1, 29) if f'V{i}' in df_featured.columns]
        if len(pca_cols) >= 2:
            for i in range(min(3, len(pca_cols))):
                for j in range(i+1, min(4, len(pca_cols))):
                    col1, col2 = pca_cols[i], pca_cols[j]
                    df_featured[f'{col1}_{col2}_interaction'] = df_featured[col1] * df_featured[col2]
        
        return df_featured
    
    def _create_time_features(self, df):
        """Create time-based features."""
        df_featured = df.copy()
        
        if 'purchase_time' in df_featured.columns:
            # Purchase time features
            df_featured['purchase_hour'] = df_featured['purchase_time'].dt.hour
            df_featured['purchase_day'] = df_featured['purchase_time'].dt.dayofweek
            df_featured['is_weekend'] = df_featured['purchase_day'].isin([5, 6]).astype(int)
            df_featured['is_night'] = ((df_featured['purchase_hour'] >= 22) | 
                                      (df_featured['purchase_hour'] <= 6)).astype(int)
        
        if 'signup_time' in df_featured.columns and 'purchase_time' in df_featured.columns:
            # Time since signup
            df_featured['time_since_signup'] = (df_featured['purchase_time'] - 
                                               df_featured['signup_time']).dt.total_seconds() / 3600
            df_featured['same_day_signup'] = (df_featured['purchase_time'].dt.date == 
                                             df_featured['signup_time'].dt.date).astype(int)
            df_featured['within_1h_signup'] = (df_featured['time_since_signup'] <= 1).astype(int)
        
        return df_featured
    
    def _create_behavior_features(self, df):
        """Create user behavior features."""
        df_featured = df.copy()
        
        if 'user_id' in df_featured.columns:
            # User transaction frequency
            user_counts = df_featured['user_id'].value_counts()
            df_featured['user_transaction_count'] = df_featured['user_id'].map(user_counts)
        
        if 'device_id' in df_featured.columns:
            # Device usage
            device_counts = df_featured['device_id'].value_counts()
            df_featured['device_transaction_count'] = df_featured['device_id'].map(device_counts)
            df_featured['device_uniqueness'] = 1 / df_featured['device_transaction_count'].replace(0, 1)
        
        return df_featured
    
    def _create_transaction_features(self, df):
        """Create transaction-based features."""
        df_featured = df.copy()
        
        if 'purchase_value' in df_featured.columns:
            # Value features
            df_featured['purchase_value_log'] = np.log1p(df_featured['purchase_value'])
            df_featured['high_value'] = (df_featured['purchase_value'] > 
                                        df_featured['purchase_value'].quantile(0.95)).astype(int)
            df_featured['low_value'] = (df_featured['purchase_value'] < 
                                       df_featured['purchase_value'].quantile(0.05)).astype(int)
        
        return df_featured
    
    def _create_risk_features(self, df):
        """Create risk score features."""
        df_featured = df.copy()
        
        # Browser risk
        if 'browser' in df_featured.columns and 'class' in df_featured.columns:
            browser_risk = df_featured.groupby('browser')['class'].mean()
            df_featured['browser_risk_score'] = df_featured['browser'].map(browser_risk).fillna(0)
        
        # Source risk
        if 'source' in df_featured.columns and 'class' in df_featured.columns:
            source_risk = df_featured.groupby('source')['class'].mean()
            df_featured['source_risk_score'] = df_featured['source'].map(source_risk).fillna(0)
        
        # Composite risk score
        risk_components = []
        if 'browser_risk_score' in df_featured.columns:
            risk_components.append(df_featured['browser_risk_score'])
        if 'source_risk_score' in df_featured.columns:
            risk_components.append(df_featured['source_risk_score'])
        if 'within_1h_signup' in df_featured.columns:
            risk_components.append(df_featured['within_1h_signup'])
        if 'high_value' in df_featured.columns:
            risk_components.append(df_featured['high_value'])
        
        if risk_components:
            df_featured['composite_risk_score'] = sum(comp * (1/len(risk_components)) 
                                                     for comp in risk_components)
        
        return df_featured
    
    def _store_feature_stats(self, df):
        """Store feature statistics for later use."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }