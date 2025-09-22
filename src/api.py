"""
FastAPI backend for A/B testing sequential analysis.
Provides REST API endpoints for experiment analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import asyncio
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

# Initialize FastAPI app
app = FastAPI(
    title="A/B Testing Sequential Analysis API",
    description="API for sequential A/B testing analysis using STF and SAVVY",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
cached_data = None
cached_processed_data = None


# Pydantic models for request/response
class ExperimentRequest(BaseModel):
    experiment_id: str = Field(..., description="Experiment identifier")
    variant_id: int = Field(..., description="Variant identifier")
    metric_id: int = Field(..., description="Metric identifier (1-4)")
    hypothesis: str = Field(..., description="Hypothesis: 'a>=b', 'a<=b', or 'a=b'")


class ExperimentResponse(BaseModel):
    experiment_id: str
    variant_id: int
    metric_id: int
    hypothesis: str
    metric_type: str
    status: str
    results: Optional[Dict[str, Any]] = None
    plots: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class AvailableExperimentsResponse(BaseModel):
    experiments: List[Dict[str, Any]]
    total_count: int


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


# Initialize components
data_collector = DataCollector()
data_preprocessor = DataPreprocessor()
data_generator = DataGenerator()
data_filter = DataFilter()
sequential_tester = SequentialTester()
visualizer = ExperimentVisualizer()


@app.on_event("startup")
async def startup_event():
    """Initialize data on startup."""
    global cached_data, cached_processed_data
    
    logger.info("Starting up A/B Testing API...")
    
    try:
        # Load and preprocess data
        logger.info("Loading dataset...")
        cached_data = data_collector.get_dataset()
        
        if cached_data is not None:
            logger.info("Preprocessing dataset...")
            cached_processed_data = data_preprocessor.preprocess_data(cached_data)
            logger.info(f"Dataset loaded and preprocessed successfully. Shape: {cached_processed_data.shape}")
        else:
            logger.error("Failed to load dataset")
            
    except Exception as e:
        logger.error(f"Startup failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )


@app.get("/experiments/available", response_model=AvailableExperimentsResponse)
async def get_available_experiments():
    """Get list of available experiments."""
    if cached_processed_data is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    try:
        available_experiments = data_filter.get_available_experiments(cached_processed_data)
        
        experiments = []
        for _, row in available_experiments.iterrows():
            summary = data_filter.get_experiment_summary(
                cached_processed_data,
                row['experiment_id'],
                row['variant_id'],
                row['metric_id']
            )
            experiments.append(summary)
        
        return AvailableExperimentsResponse(
            experiments=experiments,
            total_count=len(experiments)
        )
        
    except Exception as e:
        logger.error(f"Error getting available experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/experiments/analyze", response_model=ExperimentResponse)
async def analyze_experiment(request: ExperimentRequest, background_tasks: BackgroundTasks):
    """Analyze a specific experiment with sequential testing."""
    if cached_processed_data is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    try:
        # Validate hypothesis
        valid_hypotheses = ['a>=b', 'a<=b', 'a=b']
        if request.hypothesis not in valid_hypotheses:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid hypothesis. Must be one of: {valid_hypotheses}"
            )
        
        # Validate metric_id
        if request.metric_id not in [1, 2, 3, 4]:
            raise HTTPException(
                status_code=400,
                detail="Invalid metric_id. Must be 1, 2, 3, or 4"
            )
        
        # Check if experiment exists
        if not data_filter.validate_experiment_exists(
            cached_processed_data,
            request.experiment_id,
            request.variant_id,
            request.metric_id
        ):
            raise HTTPException(
                status_code=404,
                detail=f"Experiment not found: {request.experiment_id}, variant {request.variant_id}, metric {request.metric_id}"
            )
        
        # Filter data for the experiment
        filtered_data = data_filter.filter_data(
            cached_processed_data,
            request.experiment_id,
            request.variant_id,
            request.metric_id
        )
        
        if len(filtered_data) == 0:
            raise HTTPException(
                status_code=404,
                detail="No data found for the specified experiment"
            )
        
        # Get metric type
        metric_type = data_preprocessor.get_metric_type(request.metric_id)
        
        # Generate data if needed for count-based metrics
        if request.metric_id in [1, 2, 3]:
            filtered_data = data_generator.generate_data_for_metric(filtered_data, request.metric_id)
        
        # Run sequential test
        logger.info(f"Starting sequential analysis for experiment {request.experiment_id}")
        results = sequential_tester.run_sequential_experiment(
            filtered_data,
            request.experiment_id,
            request.variant_id,
            request.metric_id,
            request.hypothesis
        )
        
        # Create visualizations in background
        background_tasks.add_task(
            create_visualizations_background,
            filtered_data,
            results,
            request.experiment_id,
            request.variant_id,
            request.metric_id
        )
        
        return ExperimentResponse(
            experiment_id=request.experiment_id,
            variant_id=request.variant_id,
            metric_id=request.metric_id,
            hypothesis=request.hypothesis,
            metric_type=metric_type,
            status="completed",
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/{experiment_id}/variants/{variant_id}/metrics/{metric_id}/summary")
async def get_experiment_summary(experiment_id: str, variant_id: int, metric_id: int):
    """Get summary information for a specific experiment."""
    if cached_processed_data is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    try:
        summary = data_filter.get_experiment_summary(
            cached_processed_data,
            experiment_id,
            variant_id,
            metric_id
        )
        
        if not summary['has_data']:
            raise HTTPException(
                status_code=404,
                detail="No data found for the specified experiment"
            )
        
        return summary
        
    # except HTTPException:
    #     raise
    except Exception as e:
        logger.error(f"Error getting experiment summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/{experiment_id}/variants/{variant_id}/metrics/{metric_id}/time-points")
async def get_time_points(experiment_id: str, variant_id: int, metric_id: int):
    """Get time points for a specific experiment."""
    if cached_processed_data is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    try:
        time_points = data_filter.get_time_points(
            cached_processed_data,
            experiment_id,
            variant_id,
            metric_id
        )
        
        return {
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "metric_id": metric_id,
            "time_points": time_points,
            "count": len(time_points)
        }
        
    except Exception as e:
        logger.error(f"Error getting time points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plots/{experiment_id}/variant{variant_id}/metric{metric_id}/{plot_type}")
async def get_plot(experiment_id: str, variant_id: int, metric_id: int, plot_type: str):
    """Get a specific plot for an experiment."""
    import os
    
    valid_plot_types = ['distribution', 'pvalue_history', 'final_comparison', 'comprehensive_report']
    if plot_type not in valid_plot_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plot type. Must be one of: {valid_plot_types}"
        )
    
    plot_path = os.path.join(
        visualizer.figures_folder,
        f"{experiment_id}_variant{variant_id}_metric{metric_id}_{plot_type}.png"
    )
    
    if not os.path.exists(plot_path):
        raise HTTPException(
            status_code=404,
            detail=f"Plot not found: {plot_type}"
        )
    
    from fastapi.responses import FileResponse
    return FileResponse(plot_path)


async def create_visualizations_background(filtered_data: Any, results: Dict[str, Any],
                                          experiment_id: str, variant_id: int, metric_id: int):
    """Create visualizations in background task."""
    try:
        logger.info(f"Creating visualizations for experiment {experiment_id}")
        visualizer.save_all_plots(filtered_data, results, experiment_id, variant_id, metric_id)
        logger.info(f"Visualizations created successfully for experiment {experiment_id}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
