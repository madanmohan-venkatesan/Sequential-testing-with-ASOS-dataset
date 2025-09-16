"""
Data collection module for ASOS A/B testing dataset.
Handles downloading and initial data loading.
"""

import os
import pandas as pd
import requests
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Handles data collection and downloading for ASOS experiments dataset."""
    
    def __init__(self, working_folder: str = "."):
        """
        Initialize the data collector.
        
        Args:
            working_folder: Base directory for the project
        """
        self.working_folder = working_folder
        self.figures_folder = os.path.join(working_folder, 'figures')
        self.data_folder = os.path.join(working_folder, 'data')
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Dataset paths
        self.abtest_metrics_local_path = os.path.join(
            self.data_folder, 'asos_digital_experiments_dataset.parquet'
        )
        self.abtest_metrics_remote_path = "https://osf.io/62t7f/download"
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for folder in [self.figures_folder, self.data_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder, exist_ok=True)
                logger.info(f"Created directory: {folder}")
    
    def download_dataset(self) -> bool:
        """
        Download the ASOS dataset if it doesn't already exist.
        
        Returns:
            bool: True if download was successful or file already exists
        """
        if os.path.exists(self.abtest_metrics_local_path):
            logger.info("Dataset already exists locally")
            return True
        
        try:
            logger.info("Downloading ASOS dataset...")
            response = requests.get(self.abtest_metrics_remote_path, stream=True)
            response.raise_for_status()
            
            with open(self.abtest_metrics_local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Dataset downloaded successfully to: {self.abtest_metrics_local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            return False
    
    def load_dataset(self) -> Optional[pd.DataFrame]:
        """
        Load the ASOS dataset from local file.
        
        Returns:
            pd.DataFrame: Loaded dataset or None if loading failed
        """
        if not os.path.exists(self.abtest_metrics_local_path):
            logger.error("Dataset file not found. Please download it first.")
            return None
        
        try:
            logger.info("Loading ASOS dataset...")
            df = pd.read_parquet(self.abtest_metrics_local_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return None
    
    def get_dataset(self) -> Optional[pd.DataFrame]:
        """
        Get the dataset, downloading it if necessary.
        
        Returns:
            pd.DataFrame: Loaded dataset or None if failed
        """
        if not self.download_dataset():
            return None
        
        return self.load_dataset()


def collect_data(working_folder: str = ".") -> Optional[pd.DataFrame]:
    """
    Convenience function to collect and load the ASOS dataset.
    
    Args:
        working_folder: Base directory for the project
        
    Returns:
        pd.DataFrame: Loaded dataset or None if failed
    """
    collector = DataCollector(working_folder)
    return collector.get_dataset()


if __name__ == "__main__":
    # Test the data collection
    df = collect_data()
    if df is not None:
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
    else:
        print("Failed to load dataset")
