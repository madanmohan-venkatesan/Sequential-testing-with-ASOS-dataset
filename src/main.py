"""
Main orchestration module for A/B testing sequential analysis.
Coordinates all components and provides high-level interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from datetime import datetime

# Import our modules
from data_collection import DataCollector
from data_preprocessing import DataPreprocessor
from data_generation import DataGenerator
from data_filtering import DataFilter
from inference_engine import SequentialTester
from visualization import ExperimentVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABTestOrchestrator:
    """Main orchestrator for A/B testing sequential analysis."""
    
    def __init__(self, working_folder: str = "."):
        """
        Initialize the orchestrator.
        
        Args:
            working_folder: Base directory for the project
        """
        self.working_folder = working_folder
        
        # Initialize components
        self.data_collector = DataCollector(working_folder)
        self.data_preprocessor = DataPreprocessor()
        self.data_generator = DataGenerator()
        self.data_filter = DataFilter()
        self.sequential_tester = SequentialTester()
        self.visualizer = ExperimentVisualizer()
        
        # Cache for processed data
        self._processed_data = None
        self._last_load_time = None
    
    def load_and_preprocess_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and preprocess the dataset.
        
        Args:
            force_reload: Force reload even if data is cached
            
        Returns:
            Preprocessed DataFrame
        """
        if self._processed_data is not None and not force_reload:
            logger.info("Using cached processed data")
            return self._processed_data
        
        logger.info("Loading and preprocessing data...")
        
        # Load raw data
        raw_data = self.data_collector.get_dataset()
        if raw_data is None:
            raise ValueError("Failed to load dataset")
        
        # Preprocess data
        self._processed_data = self.data_preprocessor.preprocess_data(raw_data)
        self._last_load_time = datetime.now()
        
        logger.info(f"Data loaded and preprocessed successfully. Shape: {self._processed_data.shape}")
        return self._processed_data
    
    def get_available_experiments(self) -> List[Dict[str, Any]]:
        """
        Get list of available experiments.
        
        Returns:
            List of experiment summaries
        """
        if self._processed_data is None:
            self.load_and_preprocess_data()
        
        available_df = self.data_filter.get_available_experiments(self._processed_data)
        
        experiments = []
        for _, row in available_df.iterrows():
            summary = self.data_filter.get_experiment_summary(
                self._processed_data,
                row['experiment_id'],
                row['variant_id'],
                row['metric_id']
            )
            experiments.append(summary)
        
        return experiments
    
    def validate_experiment_parameters(self, experiment_id: str, variant_id: int, 
                                     metric_id: int, hypothesis: str) -> Tuple[bool, str]:
        """
        Validate experiment parameters.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            hypothesis: Hypothesis to test
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate hypothesis
        valid_hypotheses = ['a>=b', 'a<=b', 'a=b']
        if hypothesis not in valid_hypotheses:
            return False, f"Invalid hypothesis. Must be one of: {valid_hypotheses}"
        
        # Validate metric_id
        if metric_id not in [1, 2, 3, 4]:
            return False, "Invalid metric_id. Must be 1, 2, 3, or 4"
        
        # Validate variant_id
        if variant_id < 0:
            return False, "variant_id must be a non-negative integer"
        
        # Check if experiment exists
        if self._processed_data is None:
            self.load_and_preprocess_data()
        
        if not self.data_filter.validate_experiment_exists(
            self._processed_data, experiment_id, variant_id, metric_id
        ):
            return False, f"Experiment not found: {experiment_id}, variant {variant_id}, metric {metric_id}"
        
        return True, ""
    
    def run_complete_analysis(self, experiment_id: str, variant_id: int, 
                             metric_id: int, hypothesis: str,
                             create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Run complete sequential analysis for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            hypothesis: Hypothesis to test
            create_visualizations: Whether to create visualizations
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        logger.info(f"Starting complete analysis for experiment {experiment_id}")
        
        # Validate parameters
        is_valid, error_msg = self.validate_experiment_parameters(
            experiment_id, variant_id, metric_id, hypothesis
        )
        if not is_valid:
            raise ValueError(error_msg)
        
        # Load data if needed
        if self._processed_data is None:
            self.load_and_preprocess_data()
        
        # Filter data for the experiment
        filtered_data = self.data_filter.filter_data(
            self._processed_data, experiment_id, variant_id, metric_id
        )
        
        if len(filtered_data) == 0:
            raise ValueError("No data found for the specified experiment")
        
        # Get metric type
        metric_type = self.data_preprocessor.get_metric_type(metric_id)
        
        # Generate data if needed for count-based metrics
        if metric_id in [1, 2, 3]:
            logger.info(f"Generating data for metric type: {metric_type}")
            filtered_data = self.data_generator.generate_data_for_metric(filtered_data, metric_id)
        
        # Run sequential test
        logger.info("Running sequential test...")
        results = self.sequential_tester.run_sequential_experiment(
            filtered_data, experiment_id, variant_id, metric_id, hypothesis
        )
        
        # Create visualizations if requested
        plots = {}
        if create_visualizations:
            logger.info("Creating visualizations...")
            plots = self.visualizer.save_all_plots(
                filtered_data, results, experiment_id, variant_id, metric_id
            )
        
        # Compile final results
        analysis_time = time.time() - start_time
        
        final_results = {
            'experiment_id': experiment_id,
            'variant_id': variant_id,
            'metric_id': metric_id,
            'hypothesis': hypothesis,
            'metric_type': metric_type,
            'analysis_time': analysis_time,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'plots': plots,
            'data_summary': {
                'total_data_points': len(filtered_data),
                'time_range': {
                    'start': filtered_data['time_since_start'].min(),
                    'end': filtered_data['time_since_start'].max()
                },
                'control_stats': {
                    'mean_count': filtered_data['count_c'].mean(),
                    'mean_mean': filtered_data['mean_c'].mean(),
                    'mean_variance': filtered_data['variance_c'].mean()
                },
                'treatment_stats': {
                    'mean_count': filtered_data['count_t'].mean(),
                    'mean_mean': filtered_data['mean_t'].mean(),
                    'mean_variance': filtered_data['variance_t'].mean()
                }
            }
        }
        
        logger.info(f"Complete analysis finished in {analysis_time:.2f} seconds")
        return final_results
    
    def get_experiment_summary(self, experiment_id: str, variant_id: int, metric_id: int) -> Dict[str, Any]:
        """
        Get summary information for a specific experiment.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            
        Returns:
            Experiment summary
        """
        if self._processed_data is None:
            self.load_and_preprocess_data()
        
        return self.data_filter.get_experiment_summary(
            self._processed_data, experiment_id, variant_id, metric_id
        )
    
    def get_time_points(self, experiment_id: str, variant_id: int, metric_id: int) -> List[float]:
        """
        Get time points for a specific experiment.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            
        Returns:
            List of time points
        """
        if self._processed_data is None:
            self.load_and_preprocess_data()
        
        return self.data_filter.get_time_points(
            self._processed_data, experiment_id, variant_id, metric_id
        )


def run_analysis(experiment_id: str, variant_id: int, metric_id: int, hypothesis: str) -> Dict[str, Any]:
    """
    Convenience function to run complete analysis.
    
    Args:
        experiment_id: Experiment identifier
        variant_id: Variant identifier
        metric_id: Metric identifier
        hypothesis: Hypothesis to test
        
    Returns:
        Complete analysis results
    """
    orchestrator = ABTestOrchestrator()
    return orchestrator.run_complete_analysis(experiment_id, variant_id, metric_id, hypothesis)


if __name__ == "__main__":
    # Example usage
    orchestrator = ABTestOrchestrator()
    
    # Get available experiments
    experiments = orchestrator.get_available_experiments()
    print(f"Found {len(experiments)} available experiments")
    
    if len(experiments) > 0:
        # Run analysis on first experiment
        first_exp = experiments[0]
        print(f"Running analysis on: {first_exp}")
        
        try:
            results = orchestrator.run_complete_analysis(
                first_exp['experiment_id'],
                first_exp['variant_id'],
                first_exp['metric_id'],
                'a=b'
            )
            
            print("Analysis completed successfully!")
            print(f"Results: {results['results']}")
            print(f"Plots created: {list(results['plots'].keys())}")
            
        except Exception as e:
            print(f"Analysis failed: {e}")
    else:
        print("No experiments available for testing")
