"""
Tests for data transformation functions.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
import pandas as pd
import numpy as np
from src.data_transformation import DataTransformer, split_data_stratified


class TestDataTransformer:
    """Test DataTransformer class."""
    
    def setup_method(self):
        """Setup test data."""
        self.transformer = DataTransformer(random_state=42)
        
        # Create test data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0] * 90 + [1] * 10)  # Imbalanced data
    
    def test_prepare_features(self):
        """Test feature preparation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'target': [0, 1, 0, 1, 0]
        })
        
        X, y = self.transformer.prepare_features(df, 'target')
        
        # Check shapes
        assert X.shape == (5, 2)
        assert y.shape == (5,)
        
        # Check columns
        assert 'feature1' in X.columns
        assert 'feature2' in X.columns
        assert 'target' not in X.columns
    
    def test_scale_features(self):
        """Test feature scaling."""
        X_train_scaled = self.transformer.scale_features(self.X)
        
        # Check scaling
        assert X_train_scaled.shape == self.X.shape
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_handle_imbalance_smote(self):
        """Test SMOTE imbalance handling."""
        X_resampled, y_resampled = self.transformer.handle_imbalance_smote(
            self.X, self.y, sampling_strategy=0.5
        )
        
        # Check resampling
        assert X_resampled.shape[0] > self.X.shape[0]
        assert y_resampled.shape[0] > self.y.shape[0]
        
        # Check class distribution improved
        unique, counts = np.unique(y_resampled, return_counts=True)
        class_counts = dict(zip(unique, counts))
        ratio = class_counts[1] / class_counts[0]
        assert 0.4 <= ratio <= 0.6  # Should be close to 0.5
    
    def test_get_class_weights(self):
        """Test class weight calculation."""
        weights = self.transformer.get_class_weights(self.y)
        
        # Check weights structure
        assert 0 in weights
        assert 1 in weights
        
        # Check weight values
        assert weights[1] > weights[0]  # Minority class should have higher weight


class TestSplitDataStratified:
    """Test stratified data splitting."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0] * 70 + [1] * 30)
    
    def test_split_data_stratified(self):
        """Test stratified splitting."""
        X_train, X_test, y_train, y_test = split_data_stratified(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Check shapes
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
        
        # Check stratification
        train_ratio = np.sum(y_train == 1) / len(y_train)
        test_ratio = np.sum(y_test == 1) / len(y_test)
        original_ratio = np.sum(self.y == 1) / len(self.y)
        
        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.05