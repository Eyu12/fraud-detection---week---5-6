

"""
Complete fraud detection pipeline runner.
Runs data preprocessing, feature engineering, and model training.
"""

import argparse
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.data_transformation import DataTransformer, split_data_stratified


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_fraud_pipeline(data_path, output_dir='data/processed/'):
    """Run pipeline for e-commerce fraud data."""
    logger = setup_logging()
    
    try:
        logger.info("Starting fraud detection pipeline...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        fraud_data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(fraud_data)} rows")
        
        # Step 1: Data preprocessing
        logger.info("Step 1: Data preprocessing")
        preprocessor = DataPreprocessor()
        fraud_clean = preprocessor.preprocess_fraud_data(fraud_data)
        
        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering")
        engineer = FeatureEngineer()
        fraud_features = engineer.engineer_fraud_features(fraud_clean)
        
        # Step 3: Save processed data
        logger.info("Step 3: Saving processed data")
        output_path = os.path.join(output_dir, 'fraud_processed.csv')
        fraud_features.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Step 4: Prepare for modeling
        logger.info("Step 4: Preparing for modeling")
        transformer = DataTransformer()
        X, y = transformer.prepare_features(fraud_features, 'class')
        
        # Step 5: Split data
        logger.info("Step 5: Splitting data")
        X_train, X_test, y_train, y_test = split_data_stratified(X, y, test_size=0.2)
        
        # Save splits
        pd.DataFrame(X_train).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        
        logger.info("Pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return False


def run_creditcard_pipeline(data_path, output_dir='data/processed/'):
    """Run pipeline for credit card fraud data."""
    logger = setup_logging()
    
    try:
        logger.info("Starting credit card fraud detection pipeline...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        credit_data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(credit_data)} rows")
        
        # Step 1: Data preprocessing
        logger.info("Step 1: Data preprocessing")
        preprocessor = DataPreprocessor()
        credit_clean = preprocessor.preprocess_creditcard_data(credit_data)
        
        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering")
        engineer = FeatureEngineer()
        credit_features = engineer.engineer_creditcard_features(credit_clean)
        
        # Step 3: Save processed data
        logger.info("Step 3: Saving processed data")
        output_path = os.path.join(output_dir, 'creditcard_processed.csv')
        credit_features.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Step 4: Prepare for modeling
        logger.info("Step 4: Preparing for modeling")
        transformer = DataTransformer()
        X, y = transformer.prepare_features(credit_features, 'Class')
        
        # Step 5: Split data
        logger.info("Step 5: Splitting data")
        X_train, X_test, y_train, y_test = split_data_stratified(X, y, test_size=0.2)
        
        # Save splits
        pd.DataFrame(X_train).to_csv(os.path.join(output_dir, 'X_train_credit.csv'), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(output_dir, 'X_test_credit.csv'), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(output_dir, 'y_train_credit.csv'), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(output_dir, 'y_test_credit.csv'), index=False)
        
        logger.info("Pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return False


def main():
    """Main function to run pipeline."""
    parser = argparse.ArgumentParser(description='Run fraud detection pipeline')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['fraud', 'creditcard'],
                       help='Dataset to process: fraud or creditcard')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--output-dir', type=str, default='data/processed/',
                       help='Output directory for processed data')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    # Run appropriate pipeline
    if args.dataset == 'fraud':
        success = run_fraud_pipeline(args.data_path, args.output_dir)
    else:
        success = run_creditcard_pipeline(args.data_path, args.output_dir)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()