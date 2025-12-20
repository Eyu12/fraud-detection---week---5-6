"""
Tests for data preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor, merge_ip_country


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def setup_method(self):
        """Setup test data."""
        self.preprocessor = DataPreprocessor()
        
        # Create test fraud data
        self.fraud_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', 
                           '2023-01-01 12:00:00', '2023-01-01 13:00:00', 
                           '2023-01-01 14:00:00'],
            'purchase_time': ['2023-01-01 10:30:00', '2023-01-01 11:30:00',
                             '2023-01-01 12:30:00', '2023-01-01 13:30:00',
                             '2023-01-01 14:30:00'],
            'purchase_value': [100, 200, np.nan, 400, 500],
            'age': [25, 30, 35, np.nan, 45],
            'class': [0, 1, 0, 1, 0]
        })
        
        # Create test credit card data
        self.credit_data = pd.DataFrame({
            'Time': [0, 86400, 172800, 259200, 345600],
            'V1': [1.0, -1.0, 0.5, -0.5, 0.0],
            'V2': [0.5, -0.5, 1.0, -1.0, 0.0],
            'Amount': [10.0, 20.0, 30.0, 40.0, 50.0],
            'Class': [0, 1, 0, 1, 0]
        })
    
    def test_preprocess_fraud_data(self):
        """Test fraud data preprocessing."""
        result = self.preprocessor.preprocess_fraud_data(self.fraud_data)
        
        # Check shape
        assert result.shape[0] == 5
        
        # Check missing values handled
        assert result['purchase_value'].isnull().sum() == 0
        assert result['age'].isnull().sum() == 0
        
        # Check timestamp conversion
        assert pd.api.types.is_datetime64_any_dtype(result['signup_time'])
        assert pd.api.types.is_datetime64_any_dtype(result['purchase_time'])
    
    def test_preprocess_creditcard_data(self):
        """Test credit card data preprocessing."""
        result = self.preprocessor.preprocess_creditcard_data(self.credit_data)
        
        # Check shape
        assert result.shape[0] == 5
        
        # Check no duplicates (none in test data)
        assert not result.duplicated().any()
    
    def test_convert_ip_to_int(self):
        """Test IP address to integer conversion."""
        # Valid IP
        result = self.preprocessor.convert_ip_to_int('192.168.1.1')
        assert result == 3232235777
        
        # Invalid IP
        result = self.preprocessor.convert_ip_to_int('invalid')
        assert result is None
        
        # Unknown IP
        result = self.preprocessor.convert_ip_to_int('unknown')
        assert result is None


class TestMergeIPCountry:
    """Test IP country merging."""
    
    def setup_method(self):
        """Setup test data."""
        self.fraud_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1']
        })
        
        self.ip_country_data = pd.DataFrame({
            'lower_bound_ip_address': [3232235776, 167772160, 2886729728],
            'upper_bound_ip_address': [3232236031, 184549375, 2886737919],
            'country': ['Private', 'Private', 'Private']
        })
    
    def test_merge_ip_country(self):
        """Test IP country merging."""
        result = merge_ip_country(self.fraud_data, self.ip_country_data)
        
        # Check new columns added
        assert 'ip_int' in result.columns
        assert 'country' in result.columns
        
        # Check country values
        assert all(result['country'] == 'Private')