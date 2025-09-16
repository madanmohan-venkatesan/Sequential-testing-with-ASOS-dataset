"""
Streamlit frontend for A/B testing sequential analysis.
Provides interactive user interface for experiment analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="A/B Testing Sequential Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.api_base_url = API_BASE_URL
        self.session_state = st.session_state
    
    def check_api_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_experiments(self) -> Optional[List[Dict]]:
        """Get available experiments from API."""
        try:
            response = requests.get(f"{self.api_base_url}/experiments/available", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['experiments']
            return None
        except Exception as e:
            st.error(f"Error fetching experiments: {e}")
            return None
    
    def analyze_experiment(self, experiment_id: str, variant_id: int, 
                          metric_id: int, hypothesis: str) -> Optional[Dict]:
        """Analyze experiment using API."""
        try:
            payload = {
                "experiment_id": experiment_id,
                "variant_id": variant_id,
                "metric_id": metric_id,
                "hypothesis": hypothesis
            }
            
            response = requests.post(
                f"{self.api_base_url}/experiments/analyze",
                json=payload,
                timeout=300  # 5 minutes timeout for analysis
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error analyzing experiment: {e}")
            return None
    
    def create_p_value_history_plot(self, results: Dict[str, Any]) -> go.Figure:
        """Create p-value history plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sequential Test P-value History', 'SRM P-value History'),
            vertical_spacing=0.1
        )
        
        # Plot 1: Sequential p-value history
        time_points = results['time_points'][:len(results['p_value_history'])]
        p_values = results['p_value_history']
        
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=p_values,
                mode='lines+markers',
                name='Sequential P-value',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add significance line
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                     annotation_text="α = 0.05", row=1, col=1)
        
        # Highlight significance
        if results['significance_detected']:
            sig_idx = next(i for i, p in enumerate(p_values) if p < 0.05)
            fig.add_trace(
                go.Scatter(
                    x=[time_points[sig_idx]],
                    y=[p_values[sig_idx]],
                    mode='markers',
                    name='Significance Detected',
                    marker=dict(color='red', size=12, symbol='x')
                ),
                row=1, col=1
            )
        
        # Plot 2: SRM p-value history
        srm_time_points = results['time_points'][:len(results['srm_p_value_history'])]
        srm_p_values = results['srm_p_value_history']
        
        fig.add_trace(
            go.Scatter(
                x=srm_time_points,
                y=srm_p_values,
                mode='lines+markers',
                name='SRM P-value',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # Add significance line for SRM
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                     annotation_text="α = 0.05", row=2, col=1)
        
        # Highlight SRM significance
        if results['srm_significance_detected']:
            sig_idx = next(i for i, p in enumerate(srm_p_values) if p < 0.05)
            fig.add_trace(
                go.Scatter(
                    x=[srm_time_points[sig_idx]],
                    y=[srm_p_values[sig_idx]],
                    mode='markers',
                    name='SRM Significance Detected',
                    marker=dict(color='red', size=12, symbol='x')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=f"Sequential Testing Results - {results['experiment_id']}"
        )
        
        fig.update_xaxes(title_text="Time Since Start (days)", row=2, col=1)
        fig.update_yaxes(title_text="P-value", row=1, col=1)
        fig.update_yaxes(title_text="P-value", row=2, col=1)
        
        return fig
    
    def create_final_comparison_plot(self, results: Dict[str, Any]) -> go.Figure:
        """Create final comparison plot."""
        test_types = ['Sequential\n(Metric)', 'Traditional\n(Metric)', 'Sequential\n(SRM)', 'Traditional\n(SRM)']
        p_values = [
            results['final_p_value'] or 1.0,
            results['traditional_p_value'] or 1.0,
            results['final_srm_p_value'] or 1.0,
            results['traditional_srm_p_value'] or 1.0
        ]
        
        colors = ['blue', 'lightblue', 'green', 'lightgreen']
        
        fig = go.Figure(data=[
            go.Bar(
                x=test_types,
                y=p_values,
                marker_color=colors,
                text=[f'{p:.4f}' for p in p_values],
                textposition='auto',
            )
        ])
        
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                     annotation_text="α = 0.05")
        
        fig.update_layout(
            title="Final P-value Comparison",
            xaxis_title="Test Type",
            yaxis_title="P-value",
            height=400
        )
        
        return fig
    
    def create_summary_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary table."""
        summary_data = {
            'Metric': [
                'Experiment ID',
                'Variant ID',
                'Metric ID',
                'Metric Type',
                'Hypothesis',
                'Significance Detected',
                'SRM Significance Detected',
                'Stopped at Time',
                'Final Sequential P-value',
                'Traditional P-value',
                'Final SRM P-value',
                'Traditional SRM P-value'
            ],
            'Value': [
                results['experiment_id'],
                results['variant_id'],
                results['metric_id'],
                results['metric_type'],
                results['hypothesis'],
                'Yes' if results['significance_detected'] else 'No',
                'Yes' if results['srm_significance_detected'] else 'No',
                f"{results['stopped_at_time']:.2f} days" if results['stopped_at_time'] else 'Experiment completed',
                f"{results['final_p_value']:.6f}" if results['final_p_value'] else 'N/A',
                f"{results['traditional_p_value']:.6f}" if results['traditional_p_value'] else 'N/A',
                f"{results['final_srm_p_value']:.6f}" if results['final_srm_p_value'] else 'N/A',
                f"{results['traditional_srm_p_value']:.6f}" if results['traditional_srm_p_value'] else 'N/A'
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def run(self):
        """Run the Streamlit application."""
        # Header
        st.title("📊 A/B Testing Sequential Analysis")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("🔧 Configuration")
        
        # Check API health
        if not self.check_api_health():
            st.error("❌ API is not available. Please ensure the FastAPI server is running on port 8000.")
            st.stop()
        
        st.sidebar.success("✅ API is healthy")
        
        # Get available experiments
        with st.spinner("Loading available experiments..."):
            experiments = self.get_available_experiments()
        
        if not experiments:
            st.error("❌ No experiments available or API error.")
            st.stop()
        
        # User input section
        st.sidebar.subheader("📋 Experiment Parameters")
        
        # Create options for dropdowns
        experiment_options = {f"{exp['experiment_id']}": exp['experiment_id'] for exp in experiments}
        variant_options = {}
        metric_options = {}
        
        # Group by experiment
        for exp in experiments:
            exp_id = exp['experiment_id']
            if exp_id not in variant_options:
                variant_options[exp_id] = {}
            if exp_id not in metric_options:
                metric_options[exp_id] = {}
            
            variant_options[exp_id][f"Variant {exp['variant_id']}"] = exp['variant_id']
            metric_options[exp_id][f"Metric {exp['metric_id']} ({exp.get('metric_type', 'unknown')})"] = exp['metric_id']
        
        # Experiment selection
        selected_exp_key = st.sidebar.selectbox(
            "Select Experiment:",
            options=list(experiment_options.keys()),
            key="experiment_select"
        )
        selected_experiment = experiment_options[selected_exp_key]
        
        # Variant selection
        selected_var_key = st.sidebar.selectbox(
            "Select Variant:",
            options=list(variant_options[selected_experiment].keys()),
            key="variant_select"
        )
        selected_variant = variant_options[selected_experiment][selected_var_key]
        
        # Metric selection
        selected_metric_key = st.sidebar.selectbox(
            "Select Metric:",
            options=list(metric_options[selected_experiment].keys()),
            key="metric_select"
        )
        selected_metric = metric_options[selected_experiment][selected_metric_key]
        
        # Hypothesis selection
        hypothesis_options = {
            "Treatment ≥ Control (a≥b)": "a>=b",
            "Treatment ≤ Control (a≤b)": "a<=b",
            "Treatment = Control (a=b)": "a=b"
        }
        
        selected_hypothesis_key = st.sidebar.selectbox(
            "Select Hypothesis:",
            options=list(hypothesis_options.keys()),
            key="hypothesis_select"
        )
        selected_hypothesis = hypothesis_options[selected_hypothesis_key]
        
        # Display selected parameters
        st.sidebar.subheader("📊 Selected Parameters")
        st.sidebar.write(f"**Experiment:** {selected_experiment}")
        st.sidebar.write(f"**Variant:** {selected_variant}")
        st.sidebar.write(f"**Metric:** {selected_metric}")
        st.sidebar.write(f"**Hypothesis:** {selected_hypothesis}")
        
        # Analysis button
        if st.sidebar.button("🚀 Run Sequential Analysis", type="primary"):
            # Run analysis
            with st.spinner("Running sequential analysis... This may take a few minutes."):
                results = self.analyze_experiment(
                    selected_experiment,
                    selected_variant,
                    selected_metric,
                    selected_hypothesis
                )
            
            if results and results.get('status') == 'completed':
                st.success("✅ Analysis completed successfully!")
                
                # Store results in session state
                self.session_state.analysis_results = results
                
                # Display results
                self.display_results(results)
            else:
                st.error("❌ Analysis failed. Please check the parameters and try again.")
        
        # Display previous results if available
        if hasattr(self.session_state, 'analysis_results'):
            st.markdown("---")
            st.subheader("📈 Previous Analysis Results")
            self.display_results(self.session_state.analysis_results)
    
    def display_results(self, results: Dict[str, Any]):
        """Display analysis results."""
        if not results or not results.get('results'):
            st.warning("No results to display.")
            return
        
        analysis_results = results['results']
        
        # Summary section
        st.subheader("📋 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Significance Detected",
                value="✅ Yes" if analysis_results['significance_detected'] else "❌ No"
            )
        
        with col2:
            st.metric(
                label="SRM Significance",
                value="✅ Yes" if analysis_results['srm_significance_detected'] else "❌ No"
            )
        
        with col3:
            stopped_time = analysis_results.get('stopped_at_time')
            st.metric(
                label="Stopped at Time",
                value=f"{stopped_time:.2f} days" if stopped_time else "Completed"
            )
        
        with col4:
            final_p = analysis_results.get('final_p_value')
            st.metric(
                label="Final P-value",
                value=f"{final_p:.6f}" if final_p else "N/A"
            )
        
        # P-value history plot
        st.subheader("📊 P-value History")
        p_value_fig = self.create_p_value_history_plot(analysis_results)
        st.plotly_chart(p_value_fig, use_container_width=True, key="p_value_history")
        
        # Final comparison plot
        st.subheader("⚖️ Final Comparison")
        comparison_fig = self.create_final_comparison_plot(analysis_results)
        st.plotly_chart(comparison_fig, use_container_width=True, key="final_comparison")
        
        # Summary table
        st.subheader("📋 Detailed Summary")
        summary_df = self.create_summary_table(analysis_results)
        st.dataframe(summary_df, use_container_width=True)
        
        # Interpretation
        st.subheader("🔍 Interpretation")
        
        if analysis_results['significance_detected']:
            st.success("🎉 **Significant difference detected!** The sequential test found a statistically significant difference between treatment and control groups.")
        else:
            st.info("📊 **No significant difference detected.** The sequential test did not find a statistically significant difference.")
        
        if analysis_results['srm_significance_detected']:
            st.warning("⚠️ **Sample Ratio Mismatch detected!** There may be an issue with the experiment setup or data quality.")
        else:
            st.success("✅ **No Sample Ratio Mismatch.** The sample sizes are consistent with expected ratios.")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if analysis_results['significance_detected']:
            st.write("• Consider implementing the treatment variant if the business impact is positive")
            st.write("• Conduct additional validation tests to confirm the results")
        else:
            st.write("• Consider extending the experiment duration to collect more data")
            st.write("• Review the experimental design and metrics for potential improvements")
        
        if analysis_results['srm_significance_detected']:
            st.write("• Investigate the cause of the sample ratio mismatch")
            st.write("• Check for technical issues in the experiment implementation")


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
