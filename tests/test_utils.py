"""
Tests for utility functions.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.utils import (
    save_dataframe,
    load_dataframe,
    save_model,
    load_model,
    plot_class_distribution,
    create_summary_statistics,
    save_summary,
    timer_decorator
)


class TestDataFrameIO:
    """Test dataframe I/O functions."""
    
    def test_save_load_csv(self):
        """Test save/load dataframe as CSV."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save
            save_dataframe(df, tmp_path)
            
            # Load
            loaded_df = load_dataframe(tmp_path)
            
            # Check equality
            pd.testing.assert_frame_equal(df, loaded_df)
        finally:
            os.unlink(tmp_path)
    
    def test_save_load_pickle(self):
        """Test save/load dataframe as pickle."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save
            save_dataframe(df, tmp_path)
            
            # Load
            loaded_df = load_dataframe(tmp_path)
            
            # Check equality
            pd.testing.assert_frame_equal(df, loaded_df)
        finally:
            os.unlink(tmp_path)


class TestModelIO:
    """Test model I/O functions."""
    
    def test_save_load_model(self):
        """Test save/load model."""
        from sklearn.linear_model import LogisticRegression
        
        # Create dummy model
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        model = LogisticRegression()
        model.fit(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save
            save_model(model, tmp_path)
            
            # Load
            loaded_model = load_model(tmp_path)
            
            # Check predictions match
            predictions_original = model.predict(X)
            predictions_loaded = loaded_model.predict(X)
            
            np.testing.assert_array_equal(predictions_original, predictions_loaded)
        finally:
            os.unlink(tmp_path)


class TestSummaryFunctions:
    """Test summary functions."""
    
    def test_create_summary_statistics(self):
        """Test summary statistics creation."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0, 4.0, 5.0],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        
        summary = create_summary_statistics(df)
        
        # Check structure
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'numeric_stats' in summary
        assert 'categorical_stats' in summary
        
        # Check values
        assert summary['shape'] == (5, 2)
        assert 'numeric' in summary['numeric_stats']
        assert 'categorical' in summary['categorical_stats']
    
    def test_save_summary(self):
        """Test summary saving."""
        summary = {'test': 'data'}
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save
            save_summary(summary, tmp_path)
            
            # Check file exists
            assert os.path.exists(tmp_path)
            
            # Check content
            with open(tmp_path, 'r') as f:
                loaded_summary = f.read()
            assert 'test' in loaded_summary
        finally:
            os.unlink(tmp_path)


class TestTimerDecorator:
    """Test timer decorator."""
    
    def test_timer_decorator(self):
        """Test timer decorator."""
        
        @timer_decorator
        def dummy_function():
            import time
            time.sleep(0.1)
            return "done"
        
        # Should print execution time
        result = dummy_function()
        assert result == "done"