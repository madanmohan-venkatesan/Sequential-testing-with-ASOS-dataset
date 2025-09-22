"""
Data generation module for ASOS A/B testing dataset.
Handles count generation and assignment probability calculations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Handles data generation for count-based metrics and assignment probabilities."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_counts(self, mu: float, var: float, n: int) -> int:
        """
        Generate count values using normal distribution approximation.
        
        Args:
            mu: Mean value
            var: Variance value
            n: Sample size
            
        Returns:
            int: Generated count value
        """
        agg_mu = n * mu
        agg_std = np.sqrt(n * var)
        count = np.random.normal(agg_mu, agg_std)
        return max(0, int(count))  # Ensure non-negative integer
    
    def calculate_assignment_probability(self, count_control: int, count_treatment: int) -> float:
        """
        Calculate assignment probability for treatment group.
        
        Args:
            count_control: Count for control group
            count_treatment: Count for treatment group
            
        Returns:
            float: Assignment probability for treatment group
        """
        total_count = count_control + count_treatment
        if total_count == 0:
            return 0.5  # Default to 50/50 if no data
        
        return count_treatment / total_count
    
    def generate_counts_for_metric(self, df: pd.DataFrame, metric_id: int) -> pd.DataFrame:
        """
        Generate count values for a specific metric type.
        
        Args:
            df: DataFrame with metric data
            metric_id: Metric identifier (1, 2, or 3 for count-based metrics)
            
        Returns:
            DataFrame with generated counts and assignment probabilities
        """
        if metric_id not in [1, 2, 3]:
            raise ValueError(f"Count generation only supported for metric_id 1, 2, 3. Got: {metric_id}")
        
        df_generated = df.copy()
        
        # Generate counts for control group
        df_generated['generated_count_c'] = df_generated.apply(
            lambda row: self.generate_counts(row['mean_c'], row['variance_c'], row['count_c']),
            axis=1
        )
        
        # Generate counts for treatment group
        df_generated['generated_count_t'] = df_generated.apply(
            lambda row: self.generate_counts(row['mean_t'], row['variance_t'], row['count_t']),
            axis=1
        )
        
        # Calculate assignment probabilities
        df_generated['assignment_probability'] = df_generated.apply(
            lambda row: self.calculate_assignment_probability(
                row['generated_count_c'], row['generated_count_t']
            ),
            axis=1
        )
        
        logger.info(f"Generated counts and assignment probabilities for metric_id {metric_id}")
        return df_generated
    
    def generate_binary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate binary data for metric_id = 1 (conversion rates).
        
        Args:
            df: DataFrame with metric data
            
        Returns:
            DataFrame with generated binary data
        """
        df_binary = df.copy()
        
        # For binary metrics, we generate success counts based on conversion rates
        # mean_c and mean_t represent conversion rates (0-1)
        
        # Generate success counts for control group
        df_binary['success_count_c'] = df_binary.apply(
            lambda row: np.random.binomial(row['count_c'], row['mean_c']),
            axis=1
        )
        
        # Generate success counts for treatment group
        df_binary['success_count_t'] = df_binary.apply(
            lambda row: np.random.binomial(row['count_t'], row['mean_t']),
            axis=1
        )
        
        # Calculate assignment probabilities
        df_binary['assignment_probability'] = df_binary.apply(
            lambda row: self.calculate_assignment_probability(
                row['success_count_c'], row['success_count_t']
            ),
            axis=1
        )
        
        logger.info("Generated binary data for metric_id = 1")
        return df_binary
    
    def generate_count_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate count data for metric_id = 2 or 3.
        
        Args:
            df: DataFrame with metric data
            
        Returns:
            DataFrame with generated count data
        """
        df_count = df.copy()
        
        # For count metrics, we use the count generation method
        df_count = self.generate_counts_for_metric(df_count, df_count['metric_id'].iloc[0])
        
        logger.info(f"Generated count data for metric_id = {df_count['metric_id'].iloc[0]}")
        return df_count
    
    def generate_data_for_metric(self, df: pd.DataFrame, metric_id: int) -> pd.DataFrame:
        """
        Generate appropriate data based on metric type.
        
        Args:
            df: DataFrame with metric data
            metric_id: Metric identifier
            
        Returns:
            DataFrame with generated data
        """
        if metric_id == 1:
            return self.generate_binary_data(df)
        elif metric_id in [2, 3]:
            return self.generate_count_data(df)
        else:
            raise ValueError(f"Data generation not supported for metric_id {metric_id}")
    
    def get_metric_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the generated data.
        
        Args:
            df: DataFrame with generated data
            
        Returns:
            Dict with summary statistics
        """
        summary = {
            'total_rows': len(df),
            'metric_id': df['metric_id'].iloc[0] if len(df) > 0 else None,
            'experiment_id': df['experiment_id'].iloc[0] if len(df) > 0 else None,
            'variant_id': df['variant_id'].iloc[0] if len(df) > 0 else None,
        }
        
        if 'assignment_probability' in df.columns:
            summary['avg_assignment_probability'] = df['assignment_probability'].mean()
            summary['assignment_probability_range'] = (
                df['assignment_probability'].min(),
                df['assignment_probability'].max()
            )
        
        if 'generated_count_c' in df.columns:
            summary['total_generated_count_c'] = df['generated_count_c'].sum()
            summary['total_generated_count_t'] = df['generated_count_t'].sum()
        
        if 'success_count_c' in df.columns:
            summary['total_success_count_c'] = df['success_count_c'].sum()
            summary['total_success_count_t'] = df['success_count_t'].sum()
        
        return summary


def generate_data_for_metric(df: pd.DataFrame, metric_id: int) -> pd.DataFrame:
    """
    Convenience function to generate data for a specific metric.
    
    Args:
        df: DataFrame with metric data
        metric_id: Metric identifier
        
    Returns:
        DataFrame with generated data
    """
    generator = DataGenerator()
    return generator.generate_data_for_metric(df, metric_id)


if __name__ == "__main__":
    # Test the data generation
    from data_collection import collect_data
    from data_preprocessing import preprocess_data
    
    df = collect_data()
    if df is not None:
        df_processed = preprocess_data(df)
        
        # Test with metric_id = 1
        test_df = df_processed[df_processed['metric_id'] == 1].head(10)
        if len(test_df) > 0:
            generator = DataGenerator()
            df_generated = generator.generate_data_for_metric(test_df, 1)
            summary = generator.get_metric_summary(df_generated)
            
            print("Data generation test completed!")
            print(f"Summary: {summary}")
            print(f"Generated columns: {[col for col in df_generated.columns if 'generated' in col or 'success' in col or 'assignment' in col]}")
        else:
            print("No data found for metric_id = 1")
    else:
        print("Failed to load dataset for data generation test")
