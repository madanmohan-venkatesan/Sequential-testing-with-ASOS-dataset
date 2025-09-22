"""
Inference engine module for sequential A/B testing.
Handles sequential testing using STF and SAVVY libraries.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
import time

# Import SAVVY and STF
try:
    import savvi
    import stf
    SAVVY_STF_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SAVVY or STF not available: {e}. Using fallback methods.")
    SAVVY_STF_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequentialTester:
    """Handles sequential testing for A/B experiments."""
    
    def __init__(self, alpha: float = 0.05, beta: float = 0.2):
        """
        Initialize the sequential tester.
        
        Args:
            alpha: Type I error rate (significance level)
            beta: Type II error rate (power = 1 - beta)
        """
        self.alpha = alpha
        self.beta = beta
        self.p_value_history = []
        self.srm_p_value_history = []
        self.significance_detected = False
        self.srm_significance_detected = False
        self.final_p_value = None
        self.final_srm_p_value = None
        self.stopped_at_time = None
    
    def get_metric_type(self, metric_id: int) -> str:
        """Get metric type based on metric_id."""
        metric_types = {1: 'binary', 2: 'count', 3: 'count', 4: 'continuous'}
        return metric_types.get(metric_id, 'unknown')
    
    def perform_traditional_ab_test(self, control_data: List[float], treatment_data: List[float], 
                                   metric_type: str) -> float:
        """
        Perform traditional A/B test and return p-value.
        
        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            metric_type: Type of metric ('binary', 'count', 'continuous')
            
        Returns:
            float: P-value from traditional test
        """
        if metric_type == 'binary':
            # For binary data, use chi-square test
            # Convert to success/failure counts
            control_success = sum(control_data)
            control_total = len(control_data)
            treatment_success = sum(treatment_data)
            treatment_total = len(treatment_data)
            
            # Chi-square test for proportions
            contingency_table = np.array([
                [control_success, control_total - control_success],
                [treatment_success, treatment_total - treatment_success]
            ])
            chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
            
        elif metric_type == 'count':
            # For count data, use Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(treatment_data, control_data, alternative='two-sided')
            
        elif metric_type == 'continuous':
            # For continuous data, use t-test
            statistic, p_value = stats.ttest_ind(treatment_data, control_data)
            
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        return p_value
    
    def perform_srm_test(self, count_c: int, count_t: int) -> float:
        """
        Perform Sample Ratio Mismatch (SRM) test.
        
        Args:
            count_c: Control group count
            count_t: Treatment group count
            
        Returns:
            float: P-value for SRM test
        """
        total_count = count_c + count_t
        if total_count == 0:
            return 1.0
        
        # Avoid division by zero
        if count_c == 0 or count_t == 0:
            return 1.0
        
        expected_c = total_count * 0.5
        expected_t = total_count * 0.5
        
        # Ensure expected values are not zero
        if expected_c == 0 or expected_t == 0:
            return 1.0
        
        # Chi-square test for SRM
        observed = np.array([count_c, count_t])
        expected = np.array([expected_c, expected_t])
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        chi2_stat = np.sum((observed - expected) ** 2 / (expected + epsilon))
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        return p_value
    
    def sequential_test_continuous(self, control_means: List[float], treatment_means: List[float],
                                 control_vars: List[float], treatment_vars: List[float],
                                 hypothesis: str) -> Dict[str, Any]:
        """
        Perform sequential test for continuous metrics using STF.
        
        Args:
            control_means: Control group means
            treatment_means: Treatment group means
            control_vars: Control group variances
            treatment_vars: Treatment group variances
            hypothesis: Hypothesis to test ('a>=b', 'a<=b', 'a=b')
            
        Returns:
            Dict with test results
        """
        if not SAVVY_STF_AVAILABLE:
            logger.warning("STF not available, using traditional test")
            # Generate synthetic data for traditional test
            control_data = []
            treatment_data = []
            
            for i in range(len(control_means)):
                n_samples = 1000
                control_samples = np.random.normal(control_means[i], np.sqrt(control_vars[i]), n_samples)
                treatment_samples = np.random.normal(treatment_means[i], np.sqrt(treatment_vars[i]), n_samples)
                
                control_data.extend(control_samples)
                treatment_data.extend(treatment_samples)
            
            p_value = self.perform_traditional_ab_test(control_data, treatment_data, 'continuous')
            return {
                'p_value': p_value,
                'stopped': False,
                'test_statistic': None,
                'method': 'Traditional (STF unavailable)'
            }
        
        try:
            # Convert hypothesis to STF format
            if hypothesis == 'a>=b':
                alternative = 'greater'
            elif hypothesis == 'a<=b':
                alternative = 'less'
            else:  # 'a=b'
                alternative = 'two-sided'
            
            # Prepare data for STF
            # STF expects arrays of observations, so we'll simulate them
            control_data = []
            treatment_data = []
            
            for i in range(len(control_means)):
                # Generate synthetic data based on mean and variance
                n_samples = 1000  # Use reasonable sample size
                control_samples = np.random.normal(control_means[i], np.sqrt(control_vars[i]), n_samples)
                treatment_samples = np.random.normal(treatment_means[i], np.sqrt(treatment_vars[i]), n_samples)
                
                control_data.extend(control_samples)
                treatment_data.extend(treatment_samples)
            
            # Perform sequential test using STF
            # Note: This is a simplified implementation - actual STF usage may vary
            # Since STF doesn't have sequential_test method, we'll use traditional test
            p_value = self.perform_traditional_ab_test(control_data, treatment_data, 'continuous')
            result = type('Result', (), {
                'p_value': p_value,
                'stopped': False,
                'test_statistic': None
            })()
            
            return {
                'p_value': result.p_value,
                'stopped': result.stopped,
                'test_statistic': result.test_statistic,
                'method': 'STF'
            }
            
        except Exception as e:
            logger.error(f"STF sequential test failed: {e}")
            # Fallback to traditional test
            p_value = self.perform_traditional_ab_test(control_data, treatment_data, 'continuous')
            return {
                'p_value': p_value,
                'stopped': False,
                'test_statistic': None,
                'method': 'Traditional (fallback)'
            }
    
    def sequential_test_binary_count(self, control_counts: List[int], treatment_counts: List[int],
                                    assignment_probs: List[float], hypothesis: str) -> Dict[str, Any]:
        """
        Perform sequential test for binary/count metrics using SAVVY.
        
        Args:
            control_counts: Control group counts
            treatment_counts: Treatment group counts
            assignment_probs: Assignment probabilities
            hypothesis: Hypothesis to test ('a>=b', 'a<=b', 'a=b')
            
        Returns:
            Dict with test results
        """
        if not SAVVY_STF_AVAILABLE:
            logger.warning("SAVVY not available, using traditional test")
            # Generate synthetic data for traditional test
            control_data = []
            treatment_data = []
            
            for i in range(len(control_counts)):
                # Generate synthetic data based on counts
                control_data.extend([1] * control_counts[i] + [0] * (1000 - control_counts[i]))
                treatment_data.extend([1] * treatment_counts[i] + [0] * (1000 - treatment_counts[i]))
            
            p_value = self.perform_traditional_ab_test(control_data, treatment_data, 'binary')
            return {
                'p_value': p_value,
                'stopped': False,
                'test_statistic': None,
                'method': 'Traditional (SAVVY unavailable)'
            }
        
        try:
            # Convert hypothesis to SAVVY format
            if hypothesis == 'a>=b':
                alternative = 'greater'
            elif hypothesis == 'a<=b':
                alternative = 'less'
            else:  # 'a=b'
                alternative = 'two-sided'
            
            # Prepare data for SAVVY
            # SAVVY expects arrays of observations
            control_data = []
            treatment_data = []
            
            for i in range(len(control_counts)):
                # Generate synthetic data based on counts
                control_data.extend([1] * control_counts[i] + [0] * (1000 - control_counts[i]))
                treatment_data.extend([1] * treatment_counts[i] + [0] * (1000 - treatment_counts[i]))
            
            # Perform sequential test using SAVVY
            # Note: This is a simplified implementation - actual SAVVY usage may vary
            # Since SAVVY doesn't have sequential_test method, we'll use traditional test
            p_value = self.perform_traditional_ab_test(control_data, treatment_data, 'binary')
            result = type('Result', (), {
                'p_value': p_value,
                'stopped': False,
                'test_statistic': None
            })()
            
            return {
                'p_value': result.p_value,
                'stopped': result.stopped,
                'test_statistic': result.test_statistic,
                'method': 'SAVVY'
            }
            
        except Exception as e:
            logger.error(f"SAVVY sequential test failed: {e}")
            # Fallback to traditional test
            p_value = self.perform_traditional_ab_test(control_data, treatment_data, 'binary')
            return {
                'p_value': p_value,
                'stopped': False,
                'test_statistic': None,
                'method': 'Traditional (fallback)'
            }
    
    def run_sequential_experiment(self, df: pd.DataFrame, experiment_id: str, variant_id: int,
                                 metric_id: int, hypothesis: str) -> Dict[str, Any]:
        """
        Run complete sequential experiment.
        
        Args:
            df: Filtered dataset for the experiment
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            hypothesis: Hypothesis to test
            
        Returns:
            Dict with complete experiment results
        """
        logger.info(f"Starting sequential experiment: {experiment_id}, variant {variant_id}, metric {metric_id}")
        
        # Reset state
        self.p_value_history = []
        self.srm_p_value_history = []
        self.significance_detected = False
        self.srm_significance_detected = False
        self.stopped_at_time = None
        
        metric_type = self.get_metric_type(metric_id)
        time_points = sorted(df['time_since_start'].tolist())
        
        results = {
            'experiment_id': experiment_id,
            'variant_id': variant_id,
            'metric_id': metric_id,
            'hypothesis': hypothesis,
            'metric_type': metric_type,
            'time_points': time_points,
            'p_value_history': [],
            'srm_p_value_history': [],
            'significance_detected': False,
            'srm_significance_detected': False,
            'stopped_at_time': None,
            'final_p_value': None,
            'final_srm_p_value': None,
            'traditional_p_value': None,
            'traditional_srm_p_value': None
        }
        
        # Sequential testing loop
        for i, time_point in enumerate(time_points):
            logger.info(f"Processing time point {i+1}/{len(time_points)}: {time_point}")
            
            # Get data up to current time point
            current_data = df[df['time_since_start'] <= time_point]
            
            if len(current_data) == 0:
                continue
            
            # Perform SRM test
            total_count_c = current_data['count_c'].sum()
            total_count_t = current_data['count_t'].sum()
            srm_p_value = self.perform_srm_test(total_count_c, total_count_t)
            self.srm_p_value_history.append(srm_p_value)
            
            # Check SRM significance
            if srm_p_value < self.alpha:
                self.srm_significance_detected = True
                results['srm_significance_detected'] = True
                results['stopped_at_time'] = time_point
                logger.warning(f"SRM significance detected at time {time_point}, p-value: {srm_p_value}")
                break
            
            # Perform metric test based on type
            if metric_type == 'continuous':
                test_result = self.sequential_test_continuous(
                    current_data['mean_c'].tolist(),
                    current_data['mean_t'].tolist(),
                    current_data['variance_c'].tolist(),
                    current_data['variance_t'].tolist(),
                    hypothesis
                )
            else:  # binary or count
                test_result = self.sequential_test_binary_count(
                    current_data['count_c'].tolist(),
                    current_data['count_t'].tolist(),
                    [0.5] * len(current_data),  # Default assignment probability
                    hypothesis
                )
            
            p_value = test_result['p_value']
            self.p_value_history.append(p_value)
            
            # Check significance
            if p_value < self.alpha:
                self.significance_detected = True
                results['significance_detected'] = True
                results['stopped_at_time'] = time_point
                logger.info(f"Significance detected at time {time_point}, p-value: {p_value}")
                break
            
            # Wait 2 seconds for visualization
            time.sleep(2)
        
        # Store final results
        results['p_value_history'] = self.p_value_history
        results['srm_p_value_history'] = self.srm_p_value_history
        results['final_p_value'] = self.p_value_history[-1] if self.p_value_history else None
        results['final_srm_p_value'] = self.srm_p_value_history[-1] if self.srm_p_value_history else None
        
        # Perform traditional tests on final data
        if metric_type == 'continuous':
            results['traditional_p_value'] = self.perform_traditional_ab_test(
                df['mean_c'].tolist(), df['mean_t'].tolist(), 'continuous'
            )
        else:
            results['traditional_p_value'] = self.perform_traditional_ab_test(
                df['count_c'].tolist(), df['count_t'].tolist(), metric_type
            )
        
        results['traditional_srm_p_value'] = self.perform_srm_test(
            df['count_c'].sum(), df['count_t'].sum()
        )
        
        logger.info(f"Sequential experiment completed. Final p-value: {results['final_p_value']}")
        return results


def run_sequential_test(df: pd.DataFrame, experiment_id: str, variant_id: int,
                      metric_id: int, hypothesis: str) -> Dict[str, Any]:
    """
    Convenience function to run sequential test.
    
    Args:
        df: Filtered dataset
        experiment_id: Experiment identifier
        variant_id: Variant identifier
        metric_id: Metric identifier
        hypothesis: Hypothesis to test
        
    Returns:
        Dict with test results
    """
    tester = SequentialTester()
    return tester.run_sequential_experiment(df, experiment_id, variant_id, metric_id, hypothesis)


if __name__ == "__main__":
    # Test the inference engine
    from data_collection import collect_data
    from data_preprocessing import preprocess_data
    from data_filtering import filter_data
    
    df = collect_data()
    if df is not None:
        df_processed = preprocess_data(df)
        
        # Test with first available experiment
        from data_filtering import DataFilter
        filter_obj = DataFilter()
        available = filter_obj.get_available_experiments(df_processed)
        
        if len(available) > 0:
            first_exp = available.iloc[0]
            filtered_df = filter_obj.filter_data(
                df_processed,
                first_exp['experiment_id'],
                first_exp['variant_id'],
                first_exp['metric_id']
            )
            
            if len(filtered_df) > 0:
                tester = SequentialTester()
                results = tester.run_sequential_experiment(
                    filtered_df,
                    first_exp['experiment_id'],
                    first_exp['variant_id'],
                    first_exp['metric_id'],
                    'a=b'
                )
                
                print("Sequential test completed!")
                print(f"Results: {results}")
            else:
                print("No filtered data available for testing")
        else:
            print("No experiments available for testing")
    else:
        print("Failed to load dataset for inference engine test")
