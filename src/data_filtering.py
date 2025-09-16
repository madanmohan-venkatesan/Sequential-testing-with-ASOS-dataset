"""
Data filtering module for ASOS A/B testing dataset.
Handles filtering and extraction of specific experiment data.
"""

import pandas as pd
from typing import Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFilter:
    """Handles data filtering and extraction for specific experiments."""
    
    def __init__(self):
        """Initialize the data filter."""
        pass
    
    def filter_data(self, df: pd.DataFrame, experiment_id: str, variant_id: int, metric_id: int) -> pd.DataFrame:
        """
        Filter data for specific experiment, variant, and metric.
        
        Args:
            df: Full dataset
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            
        Returns:
            Filtered DataFrame
        """
        # Apply filters
        filtered_df = df[
            (df['experiment_id'] == experiment_id) &
            (df['variant_id'] == variant_id) &
            (df['metric_id'] == metric_id)
        ].copy()
        
        if len(filtered_df) == 0:
            logger.warning(f"No data found for experiment_id={experiment_id}, variant_id={variant_id}, metric_id={metric_id}")
            return pd.DataFrame()
        
        # Sort by time_since_start in ascending order
        filtered_df = filtered_df.sort_values('time_since_start').reset_index(drop=True)
        
        logger.info(f"Filtered data: {len(filtered_df)} rows for experiment_id={experiment_id}, variant_id={variant_id}, metric_id={metric_id}")
        return filtered_df
    
    def get_available_experiments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get list of available experiments in the dataset.
        
        Args:
            df: Full dataset
            
        Returns:
            DataFrame with unique experiment combinations
        """
        available = df[['experiment_id', 'variant_id', 'metric_id']].drop_duplicates().sort_values(
            ['experiment_id', 'variant_id', 'metric_id']
        )
        
        logger.info(f"Found {len(available)} unique experiment/variant/metric combinations")
        return available
    
    def get_experiment_summary(self, df: pd.DataFrame, experiment_id: str, variant_id: int, metric_id: int) -> dict:
        """
        Get summary information for a specific experiment.
        
        Args:
            df: Full dataset
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            
        Returns:
            Dict with experiment summary
        """
        filtered_df = self.filter_data(df, experiment_id, variant_id, metric_id)
        
        if len(filtered_df) == 0:
            return {
                'experiment_id': experiment_id,
                'variant_id': variant_id,
                'metric_id': metric_id,
                'data_points': 0,
                'time_range': None,
                'has_data': False
            }
        
        summary = {
            'experiment_id': experiment_id,
            'variant_id': variant_id,
            'metric_id': metric_id,
            'data_points': len(filtered_df),
            'time_range': {
                'start': filtered_df['time_since_start'].min(),
                'end': filtered_df['time_since_start'].max()
            },
            'has_data': True,
            'control_stats': {
                'mean_count': filtered_df['count_c'].mean(),
                'mean_mean': filtered_df['mean_c'].mean(),
                'mean_variance': filtered_df['variance_c'].mean()
            },
            'treatment_stats': {
                'mean_count': filtered_df['count_t'].mean(),
                'mean_mean': filtered_df['mean_t'].mean(),
                'mean_variance': filtered_df['variance_t'].mean()
            }
        }
        
        return summary
    
    def validate_experiment_exists(self, df: pd.DataFrame, experiment_id: str, variant_id: int, metric_id: int) -> bool:
        """
        Check if the specified experiment exists in the dataset.
        
        Args:
            df: Full dataset
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            
        Returns:
            bool: True if experiment exists
        """
        filtered_df = df[
            (df['experiment_id'] == experiment_id) &
            (df['variant_id'] == variant_id) &
            (df['metric_id'] == metric_id)
        ]
        
        return len(filtered_df) > 0
    
    def get_time_points(self, df: pd.DataFrame, experiment_id: str, variant_id: int, metric_id: int) -> list:
        """
        Get sorted time points for a specific experiment.
        
        Args:
            df: Full dataset
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            
        Returns:
            List of time points
        """
        filtered_df = self.filter_data(df, experiment_id, variant_id, metric_id)
        
        if len(filtered_df) == 0:
            return []
        
        return sorted(filtered_df['time_since_start'].tolist())
    
    def get_data_at_time_point(self, df: pd.DataFrame, experiment_id: str, variant_id: int, 
                              metric_id: int, time_point: float) -> Optional[pd.Series]:
        """
        Get data for a specific time point.
        
        Args:
            df: Full dataset
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            time_point: Specific time point
            
        Returns:
            Series with data for the time point or None if not found
        """
        filtered_df = self.filter_data(df, experiment_id, variant_id, metric_id)
        
        time_data = filtered_df[filtered_df['time_since_start'] == time_point]
        
        if len(time_data) == 0:
            logger.warning(f"No data found for time_point={time_point}")
            return None
        
        return time_data.iloc[0]


def filter_data(df: pd.DataFrame, experiment_id: str, variant_id: int, metric_id: int) -> pd.DataFrame:
    """
    Convenience function to filter data for specific experiment parameters.
    
    Args:
        df: Full dataset
        experiment_id: Experiment identifier
        variant_id: Variant identifier
        metric_id: Metric identifier
        
    Returns:
        Filtered DataFrame
    """
    filter_obj = DataFilter()
    return filter_obj.filter_data(df, experiment_id, variant_id, metric_id)


if __name__ == "__main__":
    # Test the data filtering
    from data_collection import collect_data
    from data_preprocessing import preprocess_data
    
    df = collect_data()
    if df is not None:
        df_processed = preprocess_data(df)
        
        filter_obj = DataFilter()
        
        # Get available experiments
        available = filter_obj.get_available_experiments(df_processed)
        print(f"Available experiments: {len(available)}")
        print(available.head())
        
        if len(available) > 0:
            # Test filtering with first available experiment
            first_exp = available.iloc[0]
            filtered_df = filter_obj.filter_data(
                df_processed,
                first_exp['experiment_id'],
                first_exp['variant_id'],
                first_exp['metric_id']
            )
            
            print(f"\nFiltered data shape: {filtered_df.shape}")
            print(f"Time points: {filter_obj.get_time_points(df_processed, first_exp['experiment_id'], first_exp['variant_id'], first_exp['metric_id'])}")
            
            summary = filter_obj.get_experiment_summary(
                df_processed,
                first_exp['experiment_id'],
                first_exp['variant_id'],
                first_exp['metric_id']
            )
            print(f"\nExperiment summary: {summary}")
    else:
        print("Failed to load dataset for filtering test")
