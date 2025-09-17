"""
Data preprocessing module for ASOS A/B testing dataset.
Handles data type conversion and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing and validation for ASOS experiments dataset."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.expected_schema = {
            'experiment_id': 'str',
            'variant_id': 'int64',
            'metric_id': 'int64',
            'time_since_start': 'float64',
            'count_c': 'int64',
            'mean_c': 'float64',
            'variance_c': 'float64',
            'count_t': 'int64',
            'mean_t': 'float64',
            'variance_t': 'float64'
        }
        
        self.required_columns = [
            'experiment_id', 'variant_id', 'metric_id', 'time_since_start',
            'count_c', 'mean_c', 'variance_c', 'count_t', 'mean_t', 'variance_t'
        ]
        
        self.nullable_columns = ['mean_c', 'variance_c', 'count_t', 'mean_t', 'variance_t']
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the dataset schema according to the expected format.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        for column, expected_dtype in self.expected_schema.items():
            if column in df.columns:
                actual_dtype = str(df[column].dtype)
                if expected_dtype not in actual_dtype:
                    validation_results['warnings'].append(
                        f"Column '{column}' has dtype {actual_dtype}, expected {expected_dtype}"
                    )
        
        # Check for null values in non-nullable columns
        non_nullable_columns = set(self.required_columns) - set(self.nullable_columns)
        for column in non_nullable_columns:
            if column in df.columns:
                null_count = df[column].isnull().sum()
                if null_count > 0:
                    validation_results['errors'].append(
                        f"Column '{column}' has {null_count} null values but should not be nullable"
                    )
                    validation_results['is_valid'] = False
        
        # Check metric_id values
        if 'metric_id' in df.columns:
            valid_metrics = {1, 2, 3, 4}
            invalid_metrics = set(df['metric_id'].unique()) - valid_metrics
            if invalid_metrics:
                validation_results['errors'].append(
                    f"Invalid metric_id values found: {invalid_metrics}. Valid values are: {valid_metrics}"
                )
                validation_results['is_valid'] = False
        
        # Check variant_id values
        if 'variant_id' in df.columns:
            if df['variant_id'].min() < 0:
                validation_results['errors'].append(
                    "variant_id should be non-negative integers"
                )
                validation_results['is_valid'] = False
        
        # Check time_since_start values
        if 'time_since_start' in df.columns:
            if df['time_since_start'].min() < 0:
                validation_results['errors'].append(
                    "time_since_start should be non-negative"
                )
                validation_results['is_valid'] = False
        
        return validation_results
    
    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types according to the expected schema.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            DataFrame with converted data types
        """
        df_converted = df.copy()
        
        # Convert experiment_id to string
        if 'experiment_id' in df_converted.columns:
            df_converted['experiment_id'] = df_converted['experiment_id'].astype(str)
        
        # Convert variant_id and metric_id to int64
        for col in ['variant_id', 'metric_id']:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype('int64')
        
        # Convert time_since_start to float64
        if 'time_since_start' in df_converted.columns:
            df_converted['time_since_start'] = df_converted['time_since_start'].astype('float64')
        
        # Convert count columns to int64
        for col in ['count_c', 'count_t']:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype('int64')
        
        # Convert mean and variance columns to float64
        for col in ['mean_c', 'variance_c', 'mean_t', 'variance_t']:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype('float64')
        
        logger.info("Data types converted successfully")
        return df_converted
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for the dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Validate schema first
        validation_results = self.validate_schema(df)
        
        if not validation_results['is_valid']:
            logger.error("Schema validation failed:")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
            raise ValueError("Data validation failed. Please check the dataset.")
        
        if validation_results['warnings']:
            logger.warning("Schema validation warnings:")
            for warning in validation_results['warnings']:
                logger.warning(f"  - {warning}")
        
        # Convert data types
        df_processed = self.convert_data_types(df)
        logger.info("Data preprocessing completed successfully")
        return df_processed
    
    def get_metric_type(self, metric_id: int) -> str:
        """
        Get the type of metric based on metric_id.
        
        Args:
            metric_id: The metric identifier
            
        Returns:
            str: Type of metric ('binary', 'count', 'continuous')
        """
        metric_types = {
            1: 'binary',
            2: 'count',
            3: 'count',
            4: 'continuous'
        }
        
        if metric_id not in metric_types:
            raise ValueError(f"Invalid metric_id: {metric_id}. Valid values are 1, 2, 3, 4")
        
        return metric_types[metric_id]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to preprocess the ASOS dataset.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_data(df)


if __name__ == "__main__":
    # Test the preprocessing
    from data_collection import collect_data
    
    df = collect_data()
    if df is not None:
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess_data(df)
        print("Preprocessing completed successfully!")
        print(f"Processed shape: {df_processed.shape}")
        print(f"Data types:")
        print(df_processed.dtypes)
    else:
        print("Failed to load dataset for preprocessing test")
