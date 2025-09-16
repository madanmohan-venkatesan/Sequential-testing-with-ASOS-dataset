"""
Visualization module for A/B testing results.
Handles plotting of distributions, p-value history, and experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ExperimentVisualizer:
    """Handles visualization of A/B testing experiments."""
    
    def __init__(self, figures_folder: str = "figures"):
        """
        Initialize the visualizer.
        
        Args:
            figures_folder: Directory to save figures
        """
        self.figures_folder = figures_folder
        self._create_figures_folder()
    
    def _create_figures_folder(self) -> None:
        """Create figures folder if it doesn't exist."""
        if not os.path.exists(self.figures_folder):
            os.makedirs(self.figures_folder, exist_ok=True)
            logger.info(f"Created figures folder: {self.figures_folder}")
    
    def plot_distribution_comparison(self, df: pd.DataFrame, experiment_id: str, 
                                   variant_id: int, metric_id: int, 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution comparison for control vs treatment groups.
        
        Args:
            df: Dataset with experiment data
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Distribution Analysis - Experiment: {experiment_id}, Variant: {variant_id}, Metric: {metric_id}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Mean comparison over time
        ax1 = axes[0, 0]
        ax1.plot(df['time_since_start'], df['mean_c'], 'b-', label='Control', linewidth=2, marker='o')
        ax1.plot(df['time_since_start'], df['mean_t'], 'r-', label='Treatment', linewidth=2, marker='s')
        ax1.set_xlabel('Time Since Start (days)')
        ax1.set_ylabel('Mean Value')
        ax1.set_title('Mean Values Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Variance comparison over time
        ax2 = axes[0, 1]
        ax2.plot(df['time_since_start'], df['variance_c'], 'b-', label='Control', linewidth=2, marker='o')
        ax2.plot(df['time_since_start'], df['variance_t'], 'r-', label='Treatment', linewidth=2, marker='s')
        ax2.set_xlabel('Time Since Start (days)')
        ax2.set_ylabel('Variance')
        ax2.set_title('Variance Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sample size comparison
        ax3 = axes[1, 0]
        ax3.plot(df['time_since_start'], df['count_c'], 'b-', label='Control', linewidth=2, marker='o')
        ax3.plot(df['time_since_start'], df['count_t'], 'r-', label='Treatment', linewidth=2, marker='s')
        ax3.set_xlabel('Time Since Start (days)')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Sample Sizes Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Difference in means
        ax4 = axes[1, 1]
        mean_diff = df['mean_t'] - df['mean_c']
        ax4.plot(df['time_since_start'], mean_diff, 'g-', linewidth=2, marker='D')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Time Since Start (days)')
        ax4.set_ylabel('Difference (Treatment - Control)')
        ax4.set_title('Difference in Means Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to: {save_path}")
        
        return fig
    
    def plot_p_value_history(self, results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot p-value history from sequential testing.
        
        Args:
            results: Results from sequential testing
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Sequential Testing Results - Experiment: {results["experiment_id"]}, '
                    f'Variant: {results["variant_id"]}, Metric: {results["metric_id"]}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Metric p-value history
        ax1 = axes[0]
        time_points = results['time_points'][:len(results['p_value_history'])]
        p_values = results['p_value_history']
        
        ax1.plot(time_points, p_values, 'b-', linewidth=2, marker='o', markersize=6)
        ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='α = 0.05')
        ax1.set_xlabel('Time Since Start (days)')
        ax1.set_ylabel('P-value')
        ax1.set_title('Sequential Test P-value History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(0.1, max(p_values) * 1.1))
        
        # Highlight significance
        if results['significance_detected']:
            sig_idx = next(i for i, p in enumerate(p_values) if p < 0.05)
            ax1.scatter(time_points[sig_idx], p_values[sig_idx], 
                       color='red', s=100, zorder=5, label='Significance Detected')
            ax1.legend()
        
        # Plot 2: SRM p-value history
        ax2 = axes[1]
        srm_time_points = results['time_points'][:len(results['srm_p_value_history'])]
        srm_p_values = results['srm_p_value_history']
        
        ax2.plot(srm_time_points, srm_p_values, 'g-', linewidth=2, marker='s', markersize=6)
        ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='α = 0.05')
        ax2.set_xlabel('Time Since Start (days)')
        ax2.set_ylabel('P-value')
        ax2.set_title('Sample Ratio Mismatch (SRM) P-value History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(0.1, max(srm_p_values) * 1.1))
        
        # Highlight SRM significance
        if results['srm_significance_detected']:
            sig_idx = next(i for i, p in enumerate(srm_p_values) if p < 0.05)
            ax2.scatter(srm_time_points[sig_idx], srm_p_values[sig_idx], 
                       color='red', s=100, zorder=5, label='SRM Significance Detected')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"P-value history plot saved to: {save_path}")
        
        return fig
    
    def plot_final_comparison(self, results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot final comparison between sequential and traditional tests.
        
        Args:
            results: Results from sequential testing
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Final Test Comparison - Experiment: {results["experiment_id"]}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: P-value comparison
        ax1 = axes[0]
        test_types = ['Sequential\n(Metric)', 'Traditional\n(Metric)', 'Sequential\n(SRM)', 'Traditional\n(SRM)']
        p_values = [
            results['final_p_value'] or 1.0,
            results['traditional_p_value'] or 1.0,
            results['final_srm_p_value'] or 1.0,
            results['traditional_srm_p_value'] or 1.0
        ]
        
        colors = ['blue', 'lightblue', 'green', 'lightgreen']
        bars = ax1.bar(test_types, p_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        ax1.set_ylabel('P-value')
        ax1.set_title('P-value Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, p_val in zip(bars, p_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{p_val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Significance summary
        ax2 = axes[1]
        significance_data = {
            'Sequential Metric': results['significance_detected'],
            'Traditional Metric': (results['traditional_p_value'] or 1.0) < 0.05,
            'Sequential SRM': results['srm_significance_detected'],
            'Traditional SRM': (results['traditional_srm_p_value'] or 1.0) < 0.05
        }
        
        categories = list(significance_data.keys())
        values = [1 if sig else 0 for sig in significance_data.values()]
        colors = ['red' if sig else 'green' for sig in significance_data.values()]
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Significance Detected')
        ax2.set_title('Significance Detection Summary')
        ax2.set_ylim(0, 1.2)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No', 'Yes'])
        
        # Add labels
        for bar, sig in zip(bars, significance_data.values()):
            height = bar.get_height()
            label = 'Significant' if sig else 'Not Significant'
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    label, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Final comparison plot saved to: {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, df: pd.DataFrame, results: Dict[str, Any]) -> plt.Figure:
        """
        Create comprehensive visualization report.
        
        Args:
            df: Dataset with experiment data
            results: Results from sequential testing
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Comprehensive A/B Test Report\n'
                    f'Experiment: {results["experiment_id"]} | '
                    f'Variant: {results["variant_id"]} | '
                    f'Metric: {results["metric_id"]} | '
                    f'Hypothesis: {results["hypothesis"]}', 
                    fontsize=18, fontweight='bold')
        
        # Plot 1: Mean comparison over time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['time_since_start'], df['mean_c'], 'b-', label='Control', linewidth=2, marker='o')
        ax1.plot(df['time_since_start'], df['mean_t'], 'r-', label='Treatment', linewidth=2, marker='s')
        ax1.set_xlabel('Time Since Start (days)')
        ax1.set_ylabel('Mean Value')
        ax1.set_title('Mean Values Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample sizes over time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['time_since_start'], df['count_c'], 'b-', label='Control', linewidth=2, marker='o')
        ax2.plot(df['time_since_start'], df['count_t'], 'r-', label='Treatment', linewidth=2, marker='s')
        ax2.set_xlabel('Time Since Start (days)')
        ax2.set_ylabel('Sample Count')
        ax2.set_title('Sample Sizes Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Difference in means
        ax3 = fig.add_subplot(gs[0, 2])
        mean_diff = df['mean_t'] - df['mean_c']
        ax3.plot(df['time_since_start'], mean_diff, 'g-', linewidth=2, marker='D')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time Since Start (days)')
        ax3.set_ylabel('Difference (Treatment - Control)')
        ax3.set_title('Difference in Means Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sequential p-value history
        ax4 = fig.add_subplot(gs[1, 0])
        time_points = results['time_points'][:len(results['p_value_history'])]
        p_values = results['p_value_history']
        
        ax4.plot(time_points, p_values, 'b-', linewidth=2, marker='o', markersize=6)
        ax4.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='α = 0.05')
        ax4.set_xlabel('Time Since Start (days)')
        ax4.set_ylabel('P-value')
        ax4.set_title('Sequential Test P-value History')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, max(0.1, max(p_values) * 1.1))
        
        # Plot 5: SRM p-value history
        ax5 = fig.add_subplot(gs[1, 1])
        srm_time_points = results['time_points'][:len(results['srm_p_value_history'])]
        srm_p_values = results['srm_p_value_history']
        
        ax5.plot(srm_time_points, srm_p_values, 'g-', linewidth=2, marker='s', markersize=6)
        ax5.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='α = 0.05')
        ax5.set_xlabel('Time Since Start (days)')
        ax5.set_ylabel('P-value')
        ax5.set_title('SRM P-value History')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, max(0.1, max(srm_p_values) * 1.1))
        
        # Plot 6: Final p-value comparison
        ax6 = fig.add_subplot(gs[1, 2])
        test_types = ['Sequential\n(Metric)', 'Traditional\n(Metric)', 'Sequential\n(SRM)', 'Traditional\n(SRM)']
        p_values = [
            results['final_p_value'] or 1.0,
            results['traditional_p_value'] or 1.0,
            results['final_srm_p_value'] or 1.0,
            results['traditional_srm_p_value'] or 1.0
        ]
        
        colors = ['blue', 'lightblue', 'green', 'lightgreen']
        bars = ax6.bar(test_types, p_values, color=colors, alpha=0.7, edgecolor='black')
        ax6.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        ax6.set_ylabel('P-value')
        ax6.set_title('Final P-value Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, p_val in zip(bars, p_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{p_val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 7: Summary statistics table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create summary table
        summary_data = [
            ['Metric Type', results['metric_type']],
            ['Hypothesis', results['hypothesis']],
            ['Significance Detected', 'Yes' if results['significance_detected'] else 'No'],
            ['SRM Significance Detected', 'Yes' if results['srm_significance_detected'] else 'No'],
            ['Stopped at Time', f"{results['stopped_at_time']:.2f} days" if results['stopped_at_time'] else 'Experiment completed'],
            ['Final Sequential P-value', f"{results['final_p_value']:.6f}" if results['final_p_value'] else 'N/A'],
            ['Traditional P-value', f"{results['traditional_p_value']:.6f}" if results['traditional_p_value'] else 'N/A'],
            ['Final SRM P-value', f"{results['final_srm_p_value']:.6f}" if results['final_srm_p_value'] else 'N/A'],
            ['Traditional SRM P-value', f"{results['traditional_srm_p_value']:.6f}" if results['traditional_srm_p_value'] else 'N/A']
        ]
        
        table = ax7.table(cellText=summary_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax7.set_title('Experiment Summary', fontsize=14, fontweight='bold', pad=20)
        
        return fig
    
    def save_all_plots(self, df: pd.DataFrame, results: Dict[str, Any], 
                      experiment_id: str, variant_id: int, metric_id: int) -> Dict[str, str]:
        """
        Save all plots for an experiment.
        
        Args:
            df: Dataset with experiment data
            results: Results from sequential testing
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_id: Metric identifier
            
        Returns:
            Dict with paths to saved plots
        """
        base_filename = f"{experiment_id}_variant{variant_id}_metric{metric_id}"
        
        saved_plots = {}
        
        # Save distribution plot
        dist_path = os.path.join(self.figures_folder, f"{base_filename}_distribution.png")
        self.plot_distribution_comparison(df, experiment_id, variant_id, metric_id, dist_path)
        saved_plots['distribution'] = dist_path
        
        # Save p-value history plot
        pval_path = os.path.join(self.figures_folder, f"{base_filename}_pvalue_history.png")
        self.plot_p_value_history(results, pval_path)
        saved_plots['pvalue_history'] = pval_path
        
        # Save final comparison plot
        final_path = os.path.join(self.figures_folder, f"{base_filename}_final_comparison.png")
        self.plot_final_comparison(results, final_path)
        saved_plots['final_comparison'] = final_path
        
        # Save comprehensive report
        report_path = os.path.join(self.figures_folder, f"{base_filename}_comprehensive_report.png")
        self.create_comprehensive_report(df, results)
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        saved_plots['comprehensive_report'] = report_path
        
        plt.close('all')  # Close all figures to free memory
        
        logger.info(f"All plots saved for experiment {experiment_id}")
        return saved_plots


def create_visualizations(df: pd.DataFrame, results: Dict[str, Any], 
                         experiment_id: str, variant_id: int, metric_id: int) -> Dict[str, str]:
    """
    Convenience function to create all visualizations.
    
    Args:
        df: Dataset with experiment data
        results: Results from sequential testing
        experiment_id: Experiment identifier
        variant_id: Variant identifier
        metric_id: Metric identifier
        
    Returns:
        Dict with paths to saved plots
    """
    visualizer = ExperimentVisualizer()
    return visualizer.save_all_plots(df, results, experiment_id, variant_id, metric_id)


if __name__ == "__main__":
    # Test the visualization module
    print("Visualization module loaded successfully!")
    print("This module provides comprehensive plotting capabilities for A/B testing results.")
