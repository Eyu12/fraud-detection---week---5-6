"""
Tests for feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def setup_method(self):
        """Setup test data."""
        self.engineer = FeatureEngineer()
        
        # Create test fraud data
        self.fraud_data = pd.DataFrame({
            'user_id': [1, 2, 3, 1, 2],
            'device_id': ['A', 'B', 'C', 'A', 'B'],
            'signup_time': pd.date_range('2023-01-01', periods=5, freq='H'),
            'purchase_time': pd.date_range('2023-01-01 00:30:00', periods=5, freq='H'),
            'purchase_value': [100, 200, 300, 400, 500],
            'browser': ['Chrome', 'Firefox', 'Chrome', 'Safari', 'Chrome'],
            'source': ['SEO', 'Ads', 'Direct', 'SEO', 'Ads'],
            'class': [0, 1, 0, 1, 0]
        })
        
        # Create test credit card data
        self.credit_data = pd.DataFrame({
            'Time': [0, 3600, 7200, 10800, 14400],
            'V1': [1.0, -1.0, 0.5, -0.5, 0.0],
            'V2': [0.5, -0.5, 1.0, -1.0, 0.0],
            'V3': [0.2, -0.2, 0.8, -0.8, 0.0],
            'Amount': [10.0, 1000.0, 20.0, 2000.0, 30.0],
            'Class': [0, 1, 0, 1, 0]
        })
    
    def test_engineer_fraud_features(self):
        """Test fraud feature engineering."""
        result = self.engineer.engineer_fraud_features(self.fraud_data)
        
        # Check time features added
        assert 'purchase_hour' in result.columns
        assert 'purchase_day' in result.columns
        assert 'time_since_signup' in result.columns
        
        # Check behavior features added
        assert 'user_transaction_count' in result.columns
        assert 'device_transaction_count' in result.columns
        
        # Check transaction features added
        assert 'purchase_value_log' in result.columns
        assert 'high_value' in result.columns
        
        # Check risk features added
        assert 'browser_risk_score' in result.columns
        assert 'source_risk_score' in result.columns
        assert 'composite_risk_score' in result.columns
    
    def test_engineer_creditcard_features(self):
        """Test credit card feature engineering."""
        result = self.engineer.engineer_creditcard_features(self.credit_data)
        
        # Check time features added
        assert 'transaction_hour' in result.columns
        assert 'is_night' in result.columns
        
        # Check amount features added
        assert 'amount_log' in result.columns
        assert 'amount_scaled' in result.columns
        assert 'high_amount' in result.columns
        
        # Check time difference features
        assert 'time_since_last' in result.columns
        assert 'rapid_transaction' in result.columns
        
        # Check interaction features
        assert 'V1_V2_interaction' in result.columns
    
    def test_feature_stats_stored(self):
        """Test feature statistics storage."""
        result = self.engineer.engineer_fraud_features(self.fraud_data)
        
        # Check stats were stored
        assert len(self.engineer.feature_stats) > 0
        
        # Check stats structure
        for col, stats in self.engineer.feature_stats.items():
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats