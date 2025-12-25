
"""
Model training script for fraud detection.
"""

import argparse
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           f1_score, precision_score, recall_score, 
                           accuracy_score, confusion_matrix)


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(X_train_path, X_test_path, y_train_path, y_test_path):
    """Load training and test data."""
    logger = setup_logging()
    
    logger.info(f"Loading training data from {X_train_path}")
    X_train = pd.read_csv(X_train_path)
    
    logger.info(f"Loading test data from {X_test_path}")
    X_test = pd.read_csv(X_test_path)
    
    logger.info(f"Loading training labels from {y_train_path}")
    y_train = pd.read_csv(y_train_path).squeeze()
    
    logger.info(f"Loading test labels from {y_test_path}")
    y_test = pd.read_csv(y_test_path).squeeze()
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, class_weight='balanced'):
    """Train logistic regression model."""
    logger = setup_logging()
    logger.info("Training Logistic Regression model...")
    
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=42,
        max_iter=1000,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training completed")
    
    return model


def train_random_forest(X_train, y_train, n_estimators=100, class_weight='balanced_subsample'):
    """Train random forest model."""
    logger = setup_logging()
    logger.info("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("Random Forest training completed")
    
    return model


def train_xgboost(X_train, y_train, n_estimators=100, scale_pos_weight=None):
    """Train XGBoost model."""
    logger = setup_logging()
    logger.info("Training XGBoost model...")
    
    if scale_pos_weight is None:
        # Calculate scale_pos_weight from data
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    
    model.fit(X_train, y_train)
    logger.info("XGBoost training completed")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    logger = setup_logging()
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
    
    logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


def save_model_and_metrics(model, metrics, model_path, metrics_path):
    """Save model and metrics to files."""
    logger = setup_logging()
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    
    # Save metrics
    logger.info(f"Saving metrics to {metrics_path}")
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    """Main function to train model."""
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['logistic', 'random_forest', 'xgboost'],
                       help='Model to train')
    parser.add_argument('--X-train', type=str, required=True,
                       help='Path to training features CSV')
    parser.add_argument('--X-test', type=str, required=True,
                       help='Path to test features CSV')
    parser.add_argument('--y-train', type=str, required=True,
                       help='Path to training labels CSV')
    parser.add_argument('--y-test', type=str, required=True,
                       help='Path to test labels CSV')
    parser.add_argument('--output-dir', type=str, default='models/',
                       help='Output directory for model and metrics')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of estimators for tree-based models')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting model training for {args.model}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data(
            args.X_train, args.X_test, args.y_train, args.y_test
        )
        
        # Train model
        if args.model == 'logistic':
            model = train_logistic_regression(X_train, y_train)
        elif args.model == 'random_forest':
            model = train_random_forest(X_train, y_train, args.n_estimators)
        else:  # xgboost
            model = train_xgboost(X_train, y_train, args.n_estimators)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model and metrics
        model_path = os.path.join(args.output_dir, f'{args.model}_model.pkl')
        metrics_path = os.path.join(args.output_dir, f'{args.model}_metrics.json')
        save_model_and_metrics(model, metrics, model_path, metrics_path)
        
        logger.info(f"Model training completed successfully!")
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Print metrics
        print("\nModel Metrics:")
        for key, value in metrics.items():
            if key not in ['confusion_matrix', 'tn', 'fp', 'fn', 'tp']:
                print(f"{key}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()