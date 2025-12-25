"""
Data transformation functions for fraud detection.
Handles scaling, encoding, and imbalance handling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


class DataTransformer:
    """Transforms data for modeling."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.smote = SMOTE(random_state=random_state)
        self.undersampler = RandomUnderSampler(random_state=random_state)
        
    def prepare_features(self, df, target_col):
        """Prepare features and target for modeling."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def handle_imbalance_smote(self, X, y, sampling_strategy='auto'):
        """Apply SMOTE to handle class imbalance."""
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def handle_imbalance_combined(self, X, y, over_strategy=0.1, under_strategy=0.5):
        """Apply combined SMOTE and undersampling."""
        # Define pipeline
        pipeline = Pipeline([
            ('over', SMOTE(sampling_strategy=over_strategy, random_state=self.random_state)),
            ('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=self.random_state))
        ])
        
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def get_class_weights(self, y):
        """Calculate class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        return class_weights


def split_data_stratified(X, y, test_size=0.2, random_state=42):
    """Split data with stratification to preserve class distribution."""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test